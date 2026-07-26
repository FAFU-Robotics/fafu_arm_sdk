"""Hardware-free regression tests for the high-level controller safety gates."""

from __future__ import annotations
from contextlib import nullcontext

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
    DISABLED = 1
    BRAKED = 2
    IDLE = 3
    MOVING = 4
    SERVOING = 5
    GRASPING = 6
    GRAVITY_COMP = 7
    ESTOP = 8
    DEAD = 9


class _OperationKind(Enum):
    NONE = 0
    LIFECYCLE = 1
    JOINT_MOTION = 2
    SERVO = 3
    GRIPPER_MOTION = 4
    GRASP = 5
    GRAVITY_COMP = 6
    RAW_STREAM = 7


class _FinishMode(Enum):
    STOP = 0
    BRAKE = 1
    HOLD = 2


# The controller imports the native extension at module import time. These unit
# tests exercise Python policy only, so provide the smallest compatible module.
_fake_motor = types.ModuleType("fafu_motor")
_fake_motor.CORE_ABI_VERSION = 3
_fake_motor.PosUnit = _PosUnit
_fake_motor.ServoChannel = _ServoChannel
_fake_motor.ServoOptions = _ServoOptions
_fake_motor.OperationKind = _OperationKind
_fake_motor.RobotState = _NativeRobotState
_fake_motor.FinishMode = _FinishMode
_fake_motor.gain_to_raw = lambda gain, _model: int(round(gain * 10.0 * 2.0 * np.pi))
_fake_motor.torques_to_raw = lambda values, _models, scale=1.0: [
    int(round(value * scale / 0.01)) for value in values
]
sys.modules["fafu_motor"] = _fake_motor

_controller_path = Path(__file__).resolve().parents[1] / "fafu_robot_controller.py"
sys.path.insert(0, str(_controller_path.parent))
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
        self.pos_vel_acc_calls = []
        self.control_loop_result = 0
        self.control_loop_calls = []
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

    def set_pos_vel_acc(self, *args):
        self.pos_vel_acc_calls.append(args)

    def run_control_loop(self, *args, **kwargs):
        self.control_loop_calls.append((args, kwargs))
        return self.control_loop_result

    def is_open(self):
        return self.open

    def close(self):
        self.open = False


class _FakeServoCore:
    def __init__(self):
        self.state = _NativeRobotState.SERVOING
        self.active_operation = _OperationKind.SERVO
        self.dead_reason = ""
        self.cancel_requested = False
        self.is_servoing = True
        self._owner = threading.get_ident()
        self._depth = 1
        self._token = 41
        self.begin_calls = []
        self.end_calls = []
        self.servo_tick_calls = []

    def operation_owned_by_current_thread(self):
        return self._owner == threading.get_ident() and self._depth > 0

    def stream_link_ok(self):
        return True

    def begin_operation(self, kind):
        self.begin_calls.append((threading.get_ident(), kind))
        if not self.operation_owned_by_current_thread():
            raise RuntimeError("another thread owns the active control operation")
        if kind != self.active_operation:
            raise RuntimeError("nested control operations must have the same kind")
        self._depth += 1
        return self._token

    def end_operation(self, token):
        self.end_calls.append(token)
        if token != self._token:
            raise RuntimeError("operation token is stale")
        if not self.operation_owned_by_current_thread():
            raise RuntimeError("operation must be ended by its owner thread")
        self._depth -= 1

    def command_guard(self):
        if not self.operation_owned_by_current_thread():
            raise RuntimeError(
                "command send requires ownership of the active operation"
            )
        return nullcontext()

    def servo_tick(self, target_angles, torque_ff):
        if not self.operation_owned_by_current_thread():
            raise RuntimeError("another thread owns the active servo session")
        self.servo_tick_calls.append((target_angles, torque_ff))
        return types.SimpleNamespace(sent=True, message="", aborted=False)


def _make_arm(*, state=None, ages=None, responses=None):
    arm = controller.FafuRobotController.__new__(controller.FafuRobotController)
    arm._joint_motor_ids = [1, 2]
    arm._ht = _FakeDriver(ages=ages, responses=responses)
    arm._config_lock = threading.RLock()
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
    arm._servo_opts = None
    arm._servo_active = False
    arm._last_cmd_turns = None
    arm._dyn_torque_scale = np.ones(2)
    arm._dyn_motor_models = None
    arm._use_group_mit = True
    arm._dynamics = None
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

        arm.servo_start(
            controller.ServoOpts(position_error_deadband_rad=0.0123))

        core.servo_start.assert_called_once()
        native = core.servo_start.call_args.args[0]
        self.assertIsInstance(native, _ServoOptions)
        self.assertAlmostEqual(native.position_error_deadband_rad, 0.0123)

    def test_servo_owner_may_send_nonblocking_gripper(self):
        arm = _make_arm(state=controller.RobotState.SERVOING)
        core = _FakeServoCore()
        arm._core = core
        arm._servo_active = True

        result = arm.gripper_control(0.0, block=False)

        self.assertIsNone(result)
        self.assertEqual(len(arm._ht.pos_vel_acc_calls), 1)
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertEqual(core.begin_calls, [])
        self.assertEqual(core.end_calls, [])
        self.assertIs(core.active_operation, _OperationKind.SERVO)
        self.assertIs(core.state, _NativeRobotState.SERVOING)
        self.assertIs(arm.state, controller.RobotState.SERVOING)
        self.assertEqual(arm._op_depth, 0)

        self.assertTrue(arm.servo_j([0.0, 0.0]))
        self.assertEqual(len(core.servo_tick_calls), 1)
        self.assertIs(core.active_operation, _OperationKind.SERVO)

    def test_other_thread_cannot_borrow_servo_for_gripper(self):
        arm = _make_arm(state=controller.RobotState.SERVOING)
        core = _FakeServoCore()
        arm._core = core
        errors = []

        def command_gripper():
            try:
                arm.gripper_control(0.0, effort=123, block=False)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=command_gripper)
        worker.start()
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], controller.RobotStateError)
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertIs(core.active_operation, _OperationKind.SERVO)
        self.assertIs(core.state, _NativeRobotState.SERVOING)

    def test_servo_rejects_blocking_gripper_and_grasp(self):
        arm = _make_arm(state=controller.RobotState.SERVOING)
        core = _FakeServoCore()
        arm._core = core

        with self.assertRaises(controller.RobotStateError):
            arm.gripper_control(0.0, effort=123, block=True)
        with self.assertRaises(controller.RobotStateError):
            arm.grasp(force_threshold=123)

        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertIs(core.active_operation, _OperationKind.SERVO)
        self.assertIs(core.state, _NativeRobotState.SERVOING)

    def test_servo_rejects_new_joint_motion(self):
        arm = _make_arm(state=controller.RobotState.SERVOING)
        core = _FakeServoCore()
        arm._core = core

        with self.assertRaises(controller.RobotStateError):
            arm.move_j([0.0, 0.0], block=False)

        self.assertEqual(arm._ht.mit_calls, [])
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertIs(core.active_operation, _OperationKind.SERVO)
        self.assertIs(core.state, _NativeRobotState.SERVOING)

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

        core.shutdown.assert_called_once_with(
            _FinishMode.BRAKE,
            _FinishMode.BRAKE,
            5.0,
        )
        self.assertEqual(driver.close_attempts, 2)
        self.assertFalse(driver.is_open())

    def test_scurve_generic_abort_uses_native_brake_cleanup(self):
        for result, message in ((1, "abort_check"), (2, "send error")):
            with self.subTest(result=result):
                state = types.SimpleNamespace(position=0.0)
                arm = _make_arm(responses={1: state, 2: state})
                arm._cfg = types.SimpleNamespace(
                    control_rate_hz=100.0,
                    trajectory_dt_s=1.0,
                    motor_ids=[1, 2],
                    max_torque_raw=100,
                )
                arm._last_cmd_turns = {1: 0.0, 2: 0.0}
                arm._core = types.SimpleNamespace(
                    brake_active_operation=mock.Mock()
                )
                arm._ht.control_loop_result = result

                with self.assertRaisesRegex(RuntimeError, message):
                    arm._move_scurve({1: 0.1}, speed_pct=10)

                arm._core.brake_active_operation.assert_called_once_with()
                self.assertIsNone(arm._last_cmd_turns)
                self.assertEqual(len(arm._ht.control_loop_calls), 1)
                _, kwargs = arm._ht.control_loop_calls[0]
                self.assertFalse(kwargs["stop_on_abort"])

    def test_nonblocking_move_send_failure_brakes(self):
        arm = _make_arm()
        arm._cfg = types.SimpleNamespace(motor_ids=[1, 2])
        arm._core = types.SimpleNamespace(
            brake_active_operation=mock.Mock()
        )
        arm._validate_joint_angles = mock.Mock(return_value=[0.0, 0.0])
        arm._clamp_speed = mock.Mock(return_value=10)
        arm._build_many_cmds_holding_others = mock.Mock(
            return_value=[object()]
        )
        arm._command_guard = mock.Mock(return_value=nullcontext())
        arm._ht.set_many_pos_vel_tqe = mock.Mock(
            side_effect=OSError("write failed")
        )

        with self.assertRaisesRegex(OSError, "write failed"):
            controller.FafuRobotController.move_j.__wrapped__(
                arm, [0.0, 0.0], block=False
            )

        arm._core.brake_active_operation.assert_called_once_with()
        self.assertIsNone(arm._last_cmd_turns)

    def test_acc_partial_send_failure_brakes(self):
        arm = _make_arm()
        arm._cfg = types.SimpleNamespace(motor_ids=[1, 2])
        arm._core = types.SimpleNamespace(
            brake_active_operation=mock.Mock()
        )
        arm._command_guard = mock.Mock(return_value=nullcontext())
        arm._ht.set_pos_vel_acc = mock.Mock(
            side_effect=[None, OSError("second motor write failed")]
        )

        with self.assertRaisesRegex(OSError, "second motor write failed"):
            arm._move_acc_sync(
                {1: 0.1, 2: 0.2},
                speed_pct=10,
                block=False,
            )

        self.assertEqual(arm._ht.set_pos_vel_acc.call_count, 2)
        arm._core.brake_active_operation.assert_called_once_with()
        self.assertIsNone(arm._last_cmd_turns)

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


    def test_gravity_begin_failure_does_not_leave_active_flags(self):
        arm = _make_arm()
        arm._dynamics = object()
        begin = mock.Mock(side_effect=RuntimeError("native busy"))
        arm._core = types.SimpleNamespace(
            state=_NativeRobotState.IDLE,
            dead_reason="",
            stream_link_ok=lambda: True,
            begin_operation=begin,
        )

        with mock.patch.object(
            controller.FafuRobotController,
            "is_enabled",
            new_callable=mock.PropertyMock,
            return_value=True,
        ), self.assertRaisesRegex(RuntimeError, "native busy"):
            arm.start_gravity_compensation(duration=0.0)

        self.assertFalse(arm._gravity_comp_active)
        self.assertIsNone(arm._gravity_comp_owner_thread_id)
        self.assertIsNone(getattr(arm, "_gravity_core_token", None))


class ModuleLayoutTests(unittest.TestCase):
    def test_public_types_are_reexported_from_split_module(self):
        self.assertEqual(controller.ServoOpts.__module__, "_api_types")
        self.assertEqual(controller.FrictionParams.__module__, "_api_types")
        self.assertIs(controller.FafuRobotController.State, controller.RobotState)

    def test_servo_opts_preserve_legacy_positional_field_order(self):
        expected = [
            "watchdog_ms",
            "max_vel",
            "max_step_rad",
            "max_lag_rad",
            "is_radians",
            "rate_hz",
            "feedforward_vel",
            "lookahead_time",
            "lag_abort_consecutive",
            "use_mit",
            "mit_kp",
            "mit_kd",
            "motor_models",
            "mit_gravity_ff",
            "position_error_deadband_rad",
        ]
        self.assertEqual(
            list(controller.ServoOpts.__dataclass_fields__), expected
        )

    def test_friction_helper_is_hardware_independent(self):
        params = controller.FrictionParams(
            fc=np.array([1.0, 2.0]),
            fv=np.array([0.5, 0.25]),
            vel_threshold=0.1,
        )
        actual = controller.friction_compensation([0.05, -2.0], params)
        np.testing.assert_allclose(actual, [0.025, -2.5])

    def test_torque_scale_copies_caller_array(self):
        arm = _make_arm()
        scale = np.array([1.0, 2.0])

        arm.set_torque_scale(scale)
        scale[:] = -10.0

        np.testing.assert_allclose(arm._dyn_torque_scale, [1.0, 2.0])

    def test_vendored_urdf_resolves_from_package(self):
        package_dir = str(_controller_path.parent)
        resolved = controller.resolve_urdf_path(package_dir, None)
        self.assertIsNotNone(resolved)
        self.assertTrue(Path(resolved).is_file())
        self.assertEqual(Path(resolved).suffix, ".urdf")

    def test_mass_matrix_mirrors_pinocchio_upper_triangle(self):
        raw = np.array([[2.0, 0.5], [99.0, 3.0]])
        fake_pin = types.SimpleNamespace(
            crba=lambda _model, _data, _q: raw
        )
        dynamics_module = sys.modules[controller.DynamicsModel.__module__]
        model = controller.DynamicsModel.__new__(controller.DynamicsModel)
        model.model = object()
        model.data = object()
        model._lock = threading.RLock()

        with mock.patch.object(dynamics_module, "pin", fake_pin):
            matrix = model.mass_matrix([0.0, 0.0])

        raw[:] = 0.0
        np.testing.assert_allclose(matrix, [[2.0, 0.5], [0.5, 3.0]])

    def test_single_seed_ik_does_not_require_live_feedback(self):
        captured = {}

        class FakeDynamics:
            def inverse_kinematics(self, *args, **kwargs):
                del args
                captured.update(kwargs)
                return np.array([np.pi / 2.0, -np.pi / 2.0])

        arm = _make_arm()
        arm._dynamics = FakeDynamics()
        arm.get_joint_values = mock.Mock(
            side_effect=AssertionError("live feedback should not be read")
        )

        result = arm.inverse_kinematics(
            [0.0, 0.0, 0.0],
            is_radians=False,
            init_q=[90.0, -90.0],
            multi_init=False,
            clamp_limits=False,
        )

        np.testing.assert_allclose(result, [90.0, -90.0])
        np.testing.assert_allclose(
            captured["current_q"],
            [np.pi / 2.0, -np.pi / 2.0],
        )

    def test_move_p_converts_degree_ik_result_before_move_j(self):
        arm = _make_arm()
        arm._require_kinematics = mock.Mock()
        arm.inverse_kinematics = mock.Mock(
            return_value=np.array([90.0, -90.0])
        )
        arm.move_j = mock.Mock()

        result = arm.move_p(
            [0.1, 0.2, 0.3], is_radians=False, multi_init=False
        )

        np.testing.assert_allclose(result, [np.pi / 2.0, -np.pi / 2.0])
        commanded = arm.move_j.call_args.args[0]
        np.testing.assert_allclose(
            commanded, [np.pi / 2.0, -np.pi / 2.0]
        )
        self.assertTrue(arm.move_j.call_args.kwargs["is_radians"])

    def test_legacy_mode_constants_are_star_exports(self):
        for name in ("MODE_POSITION", "MODE_BRAKE", "MODE_STOP", "MODE_MIT"):
            self.assertIn(name, controller.__all__)

    def test_multi_seed_ik_ignores_explicit_seed_as_documented(self):
        captured = {}

        class FakeDynamics:
            def inverse_kinematics(self, *args, **kwargs):
                del args
                captured.update(kwargs)
                return np.zeros(2)

        arm = _make_arm()
        arm._dynamics = FakeDynamics()
        arm.get_joint_values = mock.Mock(return_value=np.array([0.1, 0.2]))

        arm.inverse_kinematics(
            [0.0, 0.0, 0.0],
            init_q=[1.0, 1.0],
            multi_init=True,
            clamp_limits=False,
        )

        np.testing.assert_allclose(captured["current_q"], [0.1, 0.2])

    def test_single_seed_ik_without_seed_propagates_feedback_error(self):
        arm = _make_arm()
        arm._dynamics = mock.Mock()
        arm.get_joint_values = mock.Mock(side_effect=OSError("feedback lost"))

        with self.assertRaisesRegex(OSError, "feedback lost"):
            arm.inverse_kinematics(
                [0.0, 0.0, 0.0],
                multi_init=False,
                clamp_limits=False,
            )
        arm._dynamics.inverse_kinematics.assert_not_called()

    def test_setup_dynamics_is_atomic_when_native_commit_fails(self):
        arm = _make_arm()
        old_dynamics = object()
        old_friction = controller.FrictionParams(
            fc=np.array([0.1, 0.1]),
            fv=np.array([0.2, 0.2]),
        )
        arm._dynamics = old_dynamics
        arm._dyn_motor_models = ["old-a", "old-b"]
        arm._dyn_tau_limit = np.array([1.0, 2.0])
        arm._dyn_torque_scale = np.array([3.0, 4.0])
        arm._friction_params = old_friction
        arm._core = types.SimpleNamespace(
            state=_NativeRobotState.IDLE,
            dead_reason="",
            set_joint_motor_models=mock.Mock(
                side_effect=RuntimeError("native busy")
            ),
        )
        loaded = types.SimpleNamespace(
            model=types.SimpleNamespace(nq=2),
            eef_frame_name="tool_link",
            gravity_vector=np.array([0.0, 0.0, -9.81]),
        )

        with mock.patch.object(
            controller, "resolve_urdf_path", return_value="robot.urdf"
        ), mock.patch.object(
            controller.DynamicsModel, "load", return_value=loaded
        ), self.assertRaisesRegex(RuntimeError, "native busy"):
            arm.setup_dynamics(
                motor_models=["new-a", "new-b"],
                tau_limit=[5.0, 6.0],
                torque_scale=[7.0, 8.0],
                friction=controller.FrictionParams(
                    fc=np.zeros(2),
                    fv=np.zeros(2),
                ),
            )

        self.assertIs(arm._dynamics, old_dynamics)
        self.assertEqual(arm._dyn_motor_models, ["old-a", "old-b"])
        np.testing.assert_allclose(arm._dyn_tau_limit, [1.0, 2.0])
        np.testing.assert_allclose(arm._dyn_torque_scale, [3.0, 4.0])
        self.assertIs(arm._friction_params, old_friction)

    def test_setup_dynamics_rechecks_busy_state_before_commit(self):
        arm = _make_arm()
        old_dynamics = object()
        arm._dynamics = old_dynamics
        loaded = types.SimpleNamespace(
            model=types.SimpleNamespace(nq=2),
            eef_frame_name="tool_link",
            gravity_vector=np.array([0.0, 0.0, -9.81]),
        )

        def load_and_start_operation(*_args, **_kwargs):
            arm._state = controller.RobotState.MOVING
            return loaded

        with mock.patch.object(
            controller, "resolve_urdf_path", return_value="robot.urdf"
        ), mock.patch.object(
            controller.DynamicsModel,
            "load",
            side_effect=load_and_start_operation,
        ), self.assertRaises(controller.RobotStateError):
            arm.setup_dynamics()

        self.assertIs(arm._dynamics, old_dynamics)

    def test_setup_dynamics_copies_torque_scale(self):
        arm = _make_arm()
        loaded = types.SimpleNamespace(
            model=types.SimpleNamespace(nq=2),
            eef_frame_name="tool_link",
            gravity_vector=np.array([0.0, 0.0, -9.81]),
        )
        scale = np.array([1.0, 2.0])

        with mock.patch.object(
            controller, "resolve_urdf_path", return_value="robot.urdf"
        ), mock.patch.object(
            controller.DynamicsModel, "load", return_value=loaded
        ):
            arm.setup_dynamics(torque_scale=scale)
        scale[:] = -10.0

        np.testing.assert_allclose(arm._dyn_torque_scale, [1.0, 2.0])

    def test_dynamics_rejects_non_finite_gravity(self):
        model = types.SimpleNamespace(createData=lambda: object())
        with self.assertRaisesRegex(ValueError, "finite"):
            controller.DynamicsModel(
                model,
                gravity_vector=[np.nan, 0.0, -9.81],
                eef_frame=None,
            )

    def test_dynamics_rejects_non_one_dof_joint_models(self):
        dynamics_module = sys.modules[controller.DynamicsModel.__module__]
        fake_model = types.SimpleNamespace(nq=2, nv=1)
        fake_pin = types.SimpleNamespace(
            buildModelFromUrdf=lambda _path: fake_model
        )
        with mock.patch.object(dynamics_module, "pin", fake_pin):
            with self.assertRaisesRegex(RuntimeError, "nq=2, nv=1"):
                controller.DynamicsModel.load(
                    "robot.urdf",
                    num_joints=2,
                    gravity_vector=[0.0, 0.0, -9.81],
                )


if __name__ == "__main__":
    unittest.main()
