"""Hardware-free regression tests for the high-level controller safety gates."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest import mock

import numpy as np


class _PosUnit(Enum):
    Turns = 0
    Radians = 1
    Degrees = 2


class _ServoChannel(Enum):
    POSITION = 0
    MIT = 1


class _ServoOptions:
    pass


class _NativeRobotState(Enum):
    DISCONNECTED = 0
    IDLE = 1


class _FinishMode(Enum):
    STOP = 0
    BRAKE = 1
    HOLD = 2


# The controller imports the native extension at module import time. These unit
# tests exercise Python policy only, so provide the smallest compatible module.
_fake_motor = types.ModuleType("fafu_motor")
_fake_motor.PosUnit = _PosUnit
_fake_motor.ServoChannel = _ServoChannel
_fake_motor.ServoOptions = _ServoOptions
_fake_motor.RobotState = _NativeRobotState
_fake_motor.FinishMode = _FinishMode
_fake_motor.gain_to_raw = lambda gain, _model: int(round(gain * 10.0 * 2.0 * np.pi))
_fake_motor.torques_to_raw = lambda values, _models, scale=1.0: [
    int(round(value * scale / 0.01)) for value in values
]
sys.modules["fafu_motor"] = _fake_motor

_controller_path = Path(__file__).resolve().parents[1] / "fafu_robot_controller.py"
_spec = importlib.util.spec_from_file_location(
    "fafu_robot_controller_under_test", _controller_path
)
assert _spec is not None and _spec.loader is not None
controller = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = controller
_spec.loader.exec_module(controller)


class _FakeDriver:
    def __init__(self, ages=None, responses=None):
        self.ages = dict(ages or {1: 0.0, 2: 0.0})
        self.responses = dict(responses or {1: object(), 2: object()})
        self.mit_calls = []
        self.pos_vel_tqe_calls = []
        self.open = True

    def is_async_rx(self):
        return True

    def is_polling(self):
        return False

    def get_state_age_ms(self, motor_id):
        return self.ages[motor_id]

    def read_motor_state(self, motor_id, timeout=0.1):
        del timeout
        return self.responses.get(motor_id)

    def get_cached_state(self, motor_id):
        return self.responses.get(motor_id)

    def set_many_mit(self, *args):
        self.mit_calls.append(args)
        return {}

    def set_pos_vel_tqe(self, *args):
        self.pos_vel_tqe_calls.append(args)

    def is_open(self):
        return self.open

    def close(self):
        self.open = False


def _make_arm(*, state=None, ages=None, responses=None):
    arm = controller.FafuRobotController.__new__(controller.FafuRobotController)
    arm._joint_motor_ids = [1, 2]
    arm._ht = _FakeDriver(ages=ages, responses=responses)
    arm._state_lock = threading.RLock()
    arm._state = state or controller.RobotState.IDLE
    arm._state_verbose = False
    arm._op_depth = 0
    arm._op_owner_thread_id = None
    arm._gravity_comp_active = False
    arm._gravity_comp_owner_thread_id = None
    arm._dead_rx_timeout_ms = 500.0
    arm._dead_reason = None
    arm._motor_last_seen_monotonic = {
        1: time.monotonic(),
        2: time.monotonic(),
    }
    arm._servo_active = False
    arm._last_cmd_turns = None
    arm._dyn_torque_scale = np.ones(2)
    arm._dyn_motor_models = None
    arm._use_group_mit = True
    arm._pin_model = None
    arm._has_gripper = True
    arm._gripper_motor_id = 7
    return arm


class ControllerSafetyTests(unittest.TestCase):
    def test_all_public_hardware_writers_are_guarded(self):
        for name in (
            "move_j",
            "move_MIT",
            "apply_compensation_torque",
            "gripper_control",
            "grasp",
            "reset_zero",
        ):
            with self.subTest(method=name):
                self.assertTrue(
                    hasattr(getattr(controller.FafuRobotController, name),
                            "__wrapped__"),
                    f"{name} must hold the per-controller writer lease",
                )

    def test_servo_defaults_to_calibration_free_position_channel(self):
        self.assertFalse(controller.ServoOpts().use_mit)

    def test_motor_model_configuration_rejects_missing_names(self):
        arm = _make_arm()
        for bad in ([], ["M5036_02"], ["M5036_02", ""]):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    arm.set_motor_models(bad)

    def test_torque_scale_rejects_unsafe_values(self):
        arm = _make_arm()
        for bad in (-1.0, np.nan, np.inf, [1.0, -0.1]):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "finite|non-negative"):
                    arm.set_torque_scale(bad)
        arm.set_torque_scale([1.0, 2.0])
        np.testing.assert_array_equal(
            arm._dyn_torque_scale, np.array([1.0, 2.0]))

    def test_joint_validation_rejects_non_finite_and_wrong_shape(self):
        arm = _make_arm()

        with self.assertRaisesRegex(ValueError, "1-D vector"):
            arm._validate_joint_angles([[0.0, 1.0]], is_radians=True)
        for bad in ([np.nan, 0.0], [np.inf, 0.0], [-np.inf, 0.0]):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "finite"):
                    arm._validate_joint_angles(bad, is_radians=True)

    def test_operation_nesting_is_same_thread_only(self):
        arm = _make_arm()
        owns = arm._enter_operation("outer", controller.RobotState.MOVING)
        errors = []

        def enter_from_another_thread():
            try:
                arm._enter_operation("other", controller.RobotState.MOVING)
            except Exception as exc:  # capture the exact exception for assertion
                errors.append(exc)

        worker = threading.Thread(target=enter_from_another_thread)
        worker.start()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], controller.RobotStateError)
        self.assertEqual(arm._op_depth, 1)

        arm._exit_operation(owns)
        self.assertIs(arm.state, controller.RobotState.IDLE)

    def test_partial_feedback_loss_latches_dead(self):
        arm = _make_arm(ages={1: 10.0, 2: 750.0})

        self.assertFalse(arm._stream_link_ok())
        self.assertIs(arm.state, controller.RobotState.DEAD)
        self.assertIn("M2=750ms", arm.dead_reason)

    def test_check_alive_requires_every_joint(self):
        arm = _make_arm(responses={1: object(), 2: None})

        self.assertFalse(arm.check_alive())
        self.assertIs(arm.state, controller.RobotState.DEAD)
        self.assertIn("[2]", arm.dead_reason)

    def test_move_mit_cannot_bypass_estop(self):
        arm = _make_arm(state=controller.RobotState.ESTOP)

        with self.assertRaises(controller.RobotStateError):
            arm.move_MIT([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        self.assertEqual(arm._ht.mit_calls, [])

    def test_move_mit_rejects_non_finite_values_without_sending(self):
        arm = _make_arm()

        with self.assertRaisesRegex(ValueError, "finite"):
            arm.move_MIT([np.nan, 0.0], [0.0, 0.0], [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            arm.move_MIT([0.0, 0.0], [0.0, 0.0], [np.inf, 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            arm.move_MIT(
                [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], kp=np.nan
            )
        self.assertEqual(arm._ht.mit_calls, [])

    def test_same_thread_composite_operation_may_stream_mit(self):
        arm = _make_arm()
        owns = arm._enter_operation("composite", controller.RobotState.MOVING)
        try:
            result = arm.move_MIT(
                [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], kp=0.0, kd=0.0
            )
        finally:
            arm._exit_operation(owns)

        self.assertEqual(result, {})
        self.assertEqual(len(arm._ht.mit_calls), 1)

    def test_move_j_rejects_unknown_style_without_sending(self):
        arm = _make_arm()

        with self.assertRaisesRegex(ValueError, "style must be one of"):
            arm.move_j([0.0, 0.0], style="typo")

        self.assertEqual(arm._ht.mit_calls, [])

    def test_servo_start_always_delegates_ownership_to_native_core(self):
        arm = _make_arm()
        core = mock.Mock()
        core.is_servoing = True
        arm._core = core
        arm._sync_state_from_core = mock.Mock()

        arm.servo_start()

        core.servo_start.assert_called_once()
        self.assertIsInstance(
            core.servo_start.call_args.args[0], _ServoOptions
        )

    def test_grasp_defaults_to_force_threshold_as_firmware_cap(self):
        arm = _make_arm()
        expected = controller.GraspResult(
            grasped=True,
            reason="detected_object_force",
            angle_rad=0.0,
            closed_deg=5.0,
            peak_torque_raw=321,
            duration_s=0.1,
        )
        arm._wait_until_gripper_done = mock.Mock(return_value=expected)

        result = arm.grasp(force_threshold=321)
        generic_result = arm.gripper_control(
            0.0, effort=500, effort_threshold=222
        )

        self.assertIs(result, expected)
        self.assertIs(generic_result, expected)
        self.assertEqual(len(arm._ht.pos_vel_tqe_calls), 2)
        self.assertEqual(arm._ht.pos_vel_tqe_calls[0][3], 321)
        self.assertEqual(arm._ht.pos_vel_tqe_calls[1][3], 222)

    def test_gripper_limits_are_validated_without_sending(self):
        arm = _make_arm()

        with self.assertRaisesRegex(ValueError, "force_threshold"):
            arm.grasp(force_threshold=0)
        with self.assertRaisesRegex(ValueError, "effort"):
            arm.grasp(force_threshold=100, effort=0)
        with self.assertRaisesRegex(ValueError, "effort_threshold"):
            arm.gripper_control(0.0, effort_threshold=-1)

        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])

    def test_fault_recovery_waits_for_inflight_operation_exit(self):
        arm = _make_arm()
        owns = arm._enter_operation("move", controller.RobotState.MOVING)
        arm._enable_impl = mock.Mock()

        try:
            arm._set_state(controller.RobotState.ESTOP)
            with self.assertRaisesRegex(controller.RobotStateError, "in-flight"):
                arm.enable()
            arm._enable_impl.assert_not_called()

            arm._set_state(controller.RobotState.DEAD)
            with self.assertRaisesRegex(controller.RobotStateError, "in-flight"):
                arm.recover(confirm=True)
        finally:
            arm._exit_operation(owns)

        self.assertEqual(arm._op_depth, 0)
        self.assertIsNone(arm._op_owner_thread_id)
        self.assertIs(arm.state, controller.RobotState.DEAD)

    def test_invalid_close_policy_has_no_side_effects(self):
        arm = _make_arm()
        driver = mock.Mock()
        arm._ht = driver
        arm._servo_active = True
        arm.servo_end = mock.Mock()

        with self.assertRaises(ValueError):
            arm.close_connection(joint_release="invalid")

        self.assertEqual(driver.mock_calls, [])
        arm.servo_end.assert_not_called()
        self.assertIs(arm.state, controller.RobotState.IDLE)

    def test_close_retries_driver_after_core_is_disconnected(self):
        arm = _make_arm()

        class FailOnceDriver(_FakeDriver):
            def __init__(self):
                super().__init__()
                self.close_attempts = 0

            def close(self):
                self.close_attempts += 1
                if self.close_attempts == 1:
                    raise OSError("close failed")
                self.open = False

        driver = FailOnceDriver()
        core = types.SimpleNamespace(
            state=_NativeRobotState.IDLE,
            dead_reason="",
            shutdown=mock.Mock(),
        )

        def shutdown(*args):
            del args
            core.state = _NativeRobotState.DISCONNECTED

        core.shutdown.side_effect = shutdown
        arm._ht = driver
        arm._core = core

        with self.assertRaisesRegex(OSError, "close failed"):
            arm.close_connection()
        arm.close_connection()

        core.shutdown.assert_called_once()
        self.assertEqual(driver.close_attempts, 2)
        self.assertFalse(driver.is_open())

    def test_mit_path_does_not_nest_enable_inside_moving(self):
        arm = _make_arm()
        mit_states = []

        def record_mit(*args, **kwargs):
            del args, kwargs
            mit_states.append(arm.state)
            return {}

        class FakeInterpolator:
            def interpolate_by_max_spdacc(self, **kwargs):
                return np.asarray(kwargs["path"], dtype=float)

        arm._enable_impl = mock.Mock()
        arm.enable = mock.Mock()
        arm.move_MIT = record_mit
        fake_pwp = types.SimpleNamespace(PiecewisePolyTOPPRA=FakeInterpolator)

        with mock.patch.object(controller, "_TOPPRA_EXIST", True), mock.patch.object(
            controller, "pwp", fake_pwp, create=True
        ), mock.patch.object(controller.time, "sleep"):
            arm.move_jntspace_path_mit(
                [[0.0, 0.0]],
                start_frame_id=0,
                control_frequency=0.005,
                gravity_ff=False,
            )

        arm._enable_impl.assert_not_called()
        arm.enable.assert_not_called()
        self.assertEqual(len(mit_states), 4)
        self.assertTrue(all(state is controller.RobotState.MOVING for state in mit_states))
        self.assertIs(arm.state, controller.RobotState.IDLE)


if __name__ == "__main__":
    unittest.main()
