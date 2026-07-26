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


class _MoveJOptions:
    def __init__(self):
        self.block = True
        self.max_velocity_rad_s = 1.0
        self.control_rate_hz = 100.0
        self.min_duration_s = 0.3
        self.tolerance_rad = 0.01
        self.settle_timeout_s = 1.0


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


class _StateError(RuntimeError):
    pass


class _BusyError(_StateError):
    pass


class _EnableOptions:
    def __init__(self):
        self.allow_motor_reset = True


# The controller imports the native extension at module import time. These unit
# tests exercise Python policy only, so provide the smallest compatible module.
_fake_motor = types.ModuleType("fafu_motor")
_fake_motor.CORE_ABI_VERSION = 4
_fake_motor.PosUnit = _PosUnit
_fake_motor.ServoChannel = _ServoChannel
_fake_motor.ServoOptions = _ServoOptions
_fake_motor.MoveJOptions = _MoveJOptions
_fake_motor.OperationKind = _OperationKind
_fake_motor.RobotState = _NativeRobotState
_fake_motor.FinishMode = _FinishMode
_fake_motor.StateError = _StateError
_fake_motor.BusyError = _BusyError
_fake_motor.EnableOptions = _EnableOptions
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


class _FakeRobotConfig:
    def __init__(self):
        self.motor_ids = [1, 2]
        self.control_rate_hz = 100.0
        self.trajectory_dt_s = 0.3
        self._limits = {}

    @property
    def limits(self):
        return dict(self._limits)

    @limits.setter
    def limits(self, value):
        self._limits = dict(value)


class _FakeDriver:
    def __init__(self, ages=None, responses=None):
        self.ages = dict(ages or {1: 0.0, 2: 0.0})
        self.responses = dict(responses or {1: object(), 2: object()})
        self.mit_calls = []
        self.pos_vel_tqe_calls = []
        self.pos_vel_acc_calls = []
        self.position_limit_calls = []
        self.position_limits = {}
        self.disable_limit_calls = []
        self.clear_limit_calls = 0
        self.mode_calls = []
        self.stop_calls = []
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

    def set_many_mit_rad(self, *args):
        self.mit_calls.append(args)
        return {}

    def set_pos_vel_tqe(self, *args):
        self.pos_vel_tqe_calls.append(args)

    def set_pos_vel_tqe_rad(self, *args):
        self.pos_vel_tqe_calls.append(args)

    def set_pos_vel_acc(self, *args):
        self.pos_vel_acc_calls.append(args)

    def set_pos_vel_acc_rad(self, *args):
        self.pos_vel_acc_calls.append(args)

    def enable_position_limit(self, motor_id, lo, hi, unit):
        self.position_limit_calls.append((motor_id, lo, hi, unit))
        scale = 2.0 * np.pi if unit is _PosUnit.Radians else 360.0
        self.position_limits[motor_id] = (lo / scale, hi / scale)

    def get_position_limit_turns(self, motor_id):
        return self.position_limits.get(motor_id)

    def disable_position_limit(self, motor_id):
        self.disable_limit_calls.append(motor_id)
        self.position_limits.pop(motor_id, None)

    def clear_all_position_limits(self):
        self.clear_limit_calls += 1
        self.position_limits.clear()

    def set_motor_mode(self, motor_id, mode):
        self.mode_calls.append((motor_id, mode))

    def stop(self, motor_id):
        self.stop_calls.append(motor_id)

    def is_open(self):
        return self.open

    def close(self):
        self.open = False


class _FakeRobotCore:
    _BUSY_STATE = {
        _OperationKind.JOINT_MOTION: _NativeRobotState.MOVING,
        _OperationKind.RAW_STREAM: _NativeRobotState.MOVING,
        _OperationKind.SERVO: _NativeRobotState.SERVOING,
        _OperationKind.GRIPPER_MOTION: _NativeRobotState.GRASPING,
        _OperationKind.GRASP: _NativeRobotState.GRASPING,
        _OperationKind.GRAVITY_COMP: _NativeRobotState.GRAVITY_COMP,
    }

    def __init__(
        self,
        *,
        state=_NativeRobotState.IDLE,
        link_ok=True,
        alive=True,
        dead_reason="",
    ):
        self.state = state
        self.active_operation = _OperationKind.NONE
        self.cancel_requested = False
        self.dead_reason = dead_reason
        self.is_servoing = False
        self.closing = False
        self.link_ok = bool(link_ok)
        self.alive = bool(alive)
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._owner = None
        self._depth = 0
        self._token = 0
        self.begin_calls = []
        self.end_calls = []
        self.servo_tick_calls = []
        self.servo_start_calls = []
        self.move_j_calls = []
        self.shutdown_calls = []
        self.motor_model_calls = []
        self.brake_active_calls = 0

        if state is _NativeRobotState.SERVOING:
            self.active_operation = _OperationKind.SERVO
            self.is_servoing = True
            self._owner = threading.get_ident()
            self._depth = 1
            self._token = 1

    def operation_owned_by_current_thread(self):
        with self._lock:
            return (
                self._depth > 0
                and self._owner == threading.get_ident()
            )

    def begin_operation(self, kind):
        caller = threading.get_ident()
        self.begin_calls.append((caller, kind))
        with self._lock:
            if self.closing:
                raise _StateError("controller is closing")
            if self._depth > 0:
                if self._owner != caller:
                    raise _BusyError(
                        "another thread owns the active control operation"
                    )
                if kind is not self.active_operation:
                    raise _BusyError(
                        "nested control operations must have the same kind"
                    )
                self._depth += 1
                return self._token

            if self.state is _NativeRobotState.DISCONNECTED:
                raise _StateError("controller is disconnected")
            if self.state is _NativeRobotState.DEAD:
                raise _StateError(f"controller is DEAD: {self.dead_reason}")
            if self.state is _NativeRobotState.ESTOP:
                raise _StateError("controller is in ESTOP")
            if (
                kind is not _OperationKind.LIFECYCLE
                and self.state is not _NativeRobotState.IDLE
            ):
                raise _StateError(
                    f"control operation requires IDLE; current state={self.state.name}"
                )

            self._token += 1
            self._owner = caller
            self._depth = 1
            self.active_operation = kind
            busy_state = self._BUSY_STATE.get(kind)
            if busy_state is not None:
                self.state = busy_state
            return self._token

    def end_operation(self, token):
        self.end_calls.append(token)
        with self._lock:
            if self._depth == 0:
                return
            if token != self._token:
                raise _StateError("operation token is stale")
            if self._owner != threading.get_ident():
                raise _StateError(
                    "operation must be ended by its owner thread"
                )

            self._depth -= 1
            if self._depth:
                return

            operation = self.active_operation
            self.active_operation = _OperationKind.NONE
            self._owner = None
            if (
                not self.closing
                and self.state is self._BUSY_STATE.get(operation)
            ):
                self.state = _NativeRobotState.IDLE
            self._idle.notify_all()

    def command_guard(self):
        if not self.operation_owned_by_current_thread():
            raise _BusyError(
                "command send requires ownership of the active operation"
            )
        if self.closing:
            raise _StateError("controller is closing")
        if self.state is _NativeRobotState.DISCONNECTED:
            raise _StateError("controller is disconnected")
        if self.state is _NativeRobotState.DEAD:
            raise _StateError(f"controller is DEAD: {self.dead_reason}")
        if self.state is _NativeRobotState.ESTOP:
            raise _StateError("controller is in ESTOP")
        return nullcontext()

    def _latch_dead(self, reason):
        if self.state not in (
            _NativeRobotState.DISCONNECTED,
            _NativeRobotState.ESTOP,
        ):
            self.dead_reason = str(reason)
            self.state = _NativeRobotState.DEAD
            self.cancel_requested = True

    def stream_link_ok(self):
        if not self.link_ok:
            self._latch_dead(
                self.dead_reason or "stale motor feedback"
            )
            return False
        return self.state not in (
            _NativeRobotState.DEAD,
            _NativeRobotState.DISCONNECTED,
        )

    def health(self):
        return types.SimpleNamespace(
            state=self.state,
            active_operation=self.active_operation,
            closing=self.closing,
            cancel_requested=self.cancel_requested,
            dead_reason=self.dead_reason,
            stale_motor_ids=[],
            link_ok=self.link_ok,
        )

    def check_alive(self, fresh=True, timeout=0.1):
        del fresh, timeout
        if not self.alive:
            self._latch_dead(
                self.dead_reason or "one or more motors did not respond"
            )
            return False
        return self.state is not _NativeRobotState.DISCONNECTED

    def recover(self, confirm, timeout=0.2):
        del timeout
        if not confirm:
            raise _StateError("recover requires confirmation")
        if self._depth:
            raise _BusyError("active operation has not stopped yet")
        if self.state is not _NativeRobotState.DEAD:
            return True
        if not self.alive:
            return False
        self.dead_reason = ""
        self.cancel_requested = False
        self.state = _NativeRobotState.BRAKED
        return True

    def transition(self, state):
        if (
            self.state in (_NativeRobotState.DEAD, _NativeRobotState.ESTOP)
            and state is not self.state
        ):
            raise _StateError("latched safety state requires explicit recovery")
        self.state = state

    def enable(self, options):
        del options
        token = self.begin_operation(_OperationKind.LIFECYCLE)
        try:
            self.state = _NativeRobotState.IDLE
            return types.SimpleNamespace(
                success=True,
                failed_motor_ids=[],
                message="",
            )
        finally:
            self.end_operation(token)

    def disable(self):
        token = self.begin_operation(_OperationKind.LIFECYCLE)
        try:
            self.state = _NativeRobotState.DISABLED
        finally:
            self.end_operation(token)

    def brake(self):
        token = self.begin_operation(_OperationKind.LIFECYCLE)
        try:
            self.state = _NativeRobotState.BRAKED
        finally:
            self.end_operation(token)

    def brake_active_operation(self):
        self.brake_active_calls += 1

    def emergency_stop(self):
        self.state = _NativeRobotState.ESTOP
        self.cancel_requested = True
        self.is_servoing = False

    def resume(self):
        if self.state is _NativeRobotState.DEAD or self._depth:
            return False
        self.state = _NativeRobotState.IDLE
        self.cancel_requested = False
        return True

    def shutdown(self, joint_mode, gripper_mode, timeout):
        self.shutdown_calls.append((joint_mode, gripper_mode, timeout))
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._idle:
            if (
                self._depth
                and self._owner == threading.get_ident()
                and self.active_operation is not _OperationKind.SERVO
            ):
                raise _BusyError(
                    "shutdown cannot be called by the active operation owner"
                )
            self.closing = True
            self.cancel_requested = True
            if self.active_operation is _OperationKind.SERVO:
                self.active_operation = _OperationKind.NONE
                self._depth = 0
                self._owner = None
                self.is_servoing = False
                self._idle.notify_all()
            while self._depth:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise _BusyError(
                        "timed out waiting for the active operation to stop"
                    )
                self._idle.wait(remaining)
            self.state = _NativeRobotState.DISCONNECTED
            self.active_operation = _OperationKind.NONE
            self.is_servoing = False
            self._owner = None

    def set_joint_motor_models(self, models):
        self.motor_model_calls.append(list(models))

    def move_j(self, joint_angles_rad, options):
        token = self.begin_operation(_OperationKind.JOINT_MOTION)
        try:
            self.move_j_calls.append(
                (list(joint_angles_rad), options)
            )
            return types.SimpleNamespace(
                sent=True,
                reached=bool(options.block),
                elapsed_s=0.0,
                max_error_rad=0.0,
            )
        finally:
            self.end_operation(token)

    def servo_start(self, options):
        if self.is_servoing:
            if self.operation_owned_by_current_thread():
                return
            raise _BusyError(
                "another thread owns the active servo session"
            )
        token = self.begin_operation(_OperationKind.SERVO)
        self.servo_start_calls.append(options)
        self.is_servoing = True
        self._token = token

    def servo_tick(self, target_angles, torque_ff):
        if (
            not self.is_servoing
            or not self.operation_owned_by_current_thread()
        ):
            raise _BusyError("another thread owns the active servo session")
        self.servo_tick_calls.append((target_angles, torque_ff))
        return types.SimpleNamespace(
            sent=True,
            message="",
            aborted=False,
        )

    def servo_end(self, finish_mode):
        if self.is_servoing:
            token = self._token
            self.is_servoing = False
            self.end_operation(token)
        if finish_mode is _FinishMode.BRAKE:
            self.state = _NativeRobotState.BRAKED
        elif finish_mode is _FinishMode.STOP:
            self.state = _NativeRobotState.DISABLED
        return self.servo_summary()

    def servo_summary(self):
        return types.SimpleNamespace(
            tick_count=len(self.servo_tick_calls),
            clamp_count=0,
            lag_count=0,
            elapsed_s=0.0,
            average_rate_hz=0.0,
            aborted_reason="",
        )


def _make_arm(
    *,
    state=None,
    core=None,
    driver=None,
    ages=None,
    responses=None,
):
    arm = controller.FafuRobotController.__new__(controller.FafuRobotController)
    public_state = state or controller.RobotState.IDLE
    native_state = getattr(_NativeRobotState, public_state.name)
    arm._core = core or _FakeRobotCore(state=native_state)
    arm._joint_motor_ids = [1, 2]
    arm._ht = driver or _FakeDriver(ages=ages, responses=responses)
    arm._cfg = _FakeRobotConfig()
    arm._config_lock = threading.RLock()
    arm._state_verbose = False
    arm._last_reported_state = public_state
    arm._servo_opts = None
    arm._dyn_torque_scale = np.ones(2)
    arm._dyn_motor_models = None
    arm._dyn_tau_limit = None
    arm._friction_params = None
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

    def test_torque_scale_rejects_busy_operation(self):
        arm = _make_arm(state=controller.RobotState.MOVING)
        before = arm._dyn_torque_scale.copy()

        with self.assertRaises(controller.RobotStateError):
            arm.set_torque_scale(2.0)
        np.testing.assert_array_equal(arm._dyn_torque_scale, before)

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
        core = arm._core
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
        self.assertIs(
            core.active_operation,
            _OperationKind.JOINT_MOTION,
        )
        self.assertTrue(core.operation_owned_by_current_thread())

        arm._exit_operation(owns)
        self.assertIs(core.active_operation, _OperationKind.NONE)
        self.assertIs(arm.state, controller.RobotState.IDLE)

    def test_operation_token_is_released_when_link_check_raises(self):
        arm = _make_arm()
        core = arm._core
        core.stream_link_ok = mock.Mock(
            side_effect=_StateError("link check failed")
        )

        with self.assertRaisesRegex(
            controller.RobotStateError, "link check failed"
        ):
            arm._enter_operation("move", controller.RobotState.MOVING)

        self.assertEqual(core._depth, 0)
        self.assertIs(core.active_operation, _OperationKind.NONE)
        self.assertIs(core.state, _NativeRobotState.IDLE)
        self.assertEqual(core.end_calls, [1])

    def test_disabled_state_is_never_promoted_from_cached_motor_mode(self):
        active = types.SimpleNamespace(mode=controller.MODE_POSITION)
        arm = _make_arm(
            state=controller.RobotState.DISABLED,
            responses={1: active, 2: active},
        )

        with self.assertRaisesRegex(
            controller.RobotStateError, "Call enable"
        ):
            arm._require_ready("move")

        self.assertIs(arm.state, controller.RobotState.DISABLED)
        self.assertIs(arm._core.state, _NativeRobotState.DISABLED)
        self.assertIs(arm.sync_state(), controller.RobotState.DISABLED)

    def test_sync_state_only_downgrades_idle_from_fresh_reads(self):
        active = types.SimpleNamespace(mode=controller.MODE_POSITION)
        stopped = types.SimpleNamespace(mode=controller.MODE_STOP)
        driver = _FakeDriver(responses={1: active, 2: active})
        driver.get_cached_state = mock.Mock(return_value=active)
        driver.read_motor_state = mock.Mock(return_value=stopped)
        arm = _make_arm(driver=driver)

        self.assertIs(arm.sync_state(), controller.RobotState.DISABLED)
        self.assertEqual(driver.read_motor_state.call_count, 1)
        driver.get_cached_state.assert_not_called()

    def test_sync_state_latches_dead_on_missing_feedback(self):
        core = _FakeRobotCore(
            alive=False,
            dead_reason="motor 1 did not respond",
        )
        arm = _make_arm(
            core=core,
            responses={1: None, 2: object()},
        )

        self.assertIs(arm.sync_state(), controller.RobotState.DEAD)
        self.assertIs(core.state, _NativeRobotState.DEAD)

    def test_gravity_brake_keeps_root_writer_until_braked(self):
        arm = _make_arm()
        core = arm._core
        token = core.begin_operation(_OperationKind.GRAVITY_COMP)

        arm._brake_joints()

        self.assertEqual(core._depth, 1)
        self.assertIs(core.active_operation, _OperationKind.GRAVITY_COMP)
        self.assertIs(core.state, _NativeRobotState.BRAKED)
        self.assertEqual(
            arm._ht.mode_calls,
            [(1, controller.MODE_BRAKE), (2, controller.MODE_BRAKE)],
        )
        core.end_operation(token)
        self.assertEqual(core._depth, 0)
        self.assertIs(core.active_operation, _OperationKind.NONE)
        self.assertIs(core.state, _NativeRobotState.BRAKED)

    def test_cross_thread_close_owns_gravity_cleanup(self):
        arm = _make_arm()
        core = arm._core
        token = core.begin_operation(_OperationKind.GRAVITY_COMP)
        errors = []

        def close_arm():
            try:
                arm.close_connection()
            except Exception as exc:
                errors.append(exc)

        closer = threading.Thread(target=close_arm)
        closer.start()
        deadline = time.monotonic() + 1.0
        while not core.closing and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(core.closing)

        # Native shutdown has cancellation and final release ownership. The
        # gravity owner must not emit a competing brake command.
        arm._brake_joints()
        self.assertEqual(arm._ht.mode_calls, [])
        core.end_operation(token)

        closer.join(timeout=2.0)
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(core.state, _NativeRobotState.DISCONNECTED)
        self.assertFalse(arm._ht.is_open())

    def test_same_thread_close_rejection_does_not_poison_core(self):
        arm = _make_arm()
        core = arm._core
        token = core.begin_operation(_OperationKind.GRAVITY_COMP)

        with self.assertRaisesRegex(
            controller.RobotStateError, "active operation owner"
        ):
            arm.close_connection()

        self.assertFalse(core.closing)
        self.assertFalse(core.cancel_requested)
        self.assertIs(core.state, _NativeRobotState.GRAVITY_COMP)
        self.assertTrue(arm._ht.is_open())
        core.end_operation(token)
        with mock.patch("builtins.print"):
            arm.close_connection()
        self.assertIs(core.state, _NativeRobotState.DISCONNECTED)

    def test_partial_feedback_loss_latches_dead(self):
        core = _FakeRobotCore(
            link_ok=False,
            dead_reason="stale motor feedback: M2=750ms",
        )
        arm = _make_arm(core=core)

        self.assertFalse(arm._stream_link_ok())
        self.assertIs(arm.state, controller.RobotState.DEAD)
        self.assertIn("M2=750ms", arm.dead_reason)

    def test_check_alive_requires_every_joint(self):
        core = _FakeRobotCore(
            alive=False,
            dead_reason="motors [2] did not respond",
        )
        arm = _make_arm(core=core)

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

    def test_move_mit_rejects_invalid_timeout_without_sending(self):
        arm = _make_arm()

        for bad in (np.nan, np.inf, -0.01):
            with self.subTest(timeout=bad):
                with self.assertRaisesRegex(ValueError, "timeout"):
                    arm.move_MIT(
                        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], timeout=bad)
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

    def test_move_j_delegates_radians_and_options_to_native_core(self):
        arm = _make_arm()

        arm.move_j(
            [np.pi, -np.pi / 2.0],
            speed=25,
            block=False,
            tolerance=0.02,
            settle_timeout=0.4,
        )

        self.assertEqual(len(arm._core.move_j_calls), 1)
        angles, options = arm._core.move_j_calls[0]
        np.testing.assert_allclose(angles, [np.pi, -np.pi / 2.0])
        self.assertFalse(options.block)
        self.assertAlmostEqual(
            options.max_velocity_rad_s, 0.25 * 0.5 * 2.0 * np.pi
        )
        self.assertEqual(options.control_rate_hz, 100.0)
        self.assertEqual(options.min_duration_s, 0.3)
        self.assertEqual(options.tolerance_rad, 0.02)
        self.assertEqual(options.settle_timeout_s, 0.4)
        self.assertEqual(arm._ht.mit_calls, [])
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertEqual(arm._ht.pos_vel_acc_calls, [])
        self.assertEqual(arm._core._depth, 0)
        self.assertIs(arm.state, controller.RobotState.IDLE)

    def test_move_j_rejects_non_radians_before_native_send(self):
        arm = _make_arm()

        with self.assertRaisesRegex(ValueError, "radians-only"):
            arm.move_j([0.0, 0.0], is_radians=False)

        self.assertEqual(arm._core.move_j_calls, [])
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])

    def test_blocking_move_j_requires_native_feedback_confirmation(self):
        arm = _make_arm()
        arm._core.move_j = mock.Mock(
            return_value=types.SimpleNamespace(sent=True, reached=False)
        )

        with self.assertRaisesRegex(RuntimeError, "confirmed target feedback"):
            arm.move_j([0.0, 0.0], block=True)

        arm._core.move_j.assert_called_once()
        self.assertIs(arm.state, controller.RobotState.IDLE)

    def test_servo_start_always_delegates_ownership_to_native_core(self):
        arm = _make_arm()
        core = mock.Mock()
        core.is_servoing = False
        arm._core = core
        arm._sync_state_from_core = mock.Mock()

        arm.servo_start(
            controller.ServoOpts(position_error_deadband_rad=0.0123))

        core.servo_start.assert_called_once()
        native = core.servo_start.call_args.args[0]
        self.assertIsInstance(native, _ServoOptions)
        self.assertAlmostEqual(native.position_error_deadband_rad, 0.0123)

    def test_repeated_servo_start_keeps_native_session_options(self):
        arm = _make_arm()
        first = controller.ServoOpts(is_radians=True)
        second = controller.ServoOpts(max_vel=0.25)

        arm.servo_start(first)
        arm.servo_start(second)

        self.assertEqual(len(arm._core.servo_start_calls), 1)
        self.assertTrue(arm._servo_opts.is_radians)
        self.assertEqual(arm._servo_opts.max_vel, first.max_vel)
        self.assertEqual(arm._core._depth, 1)
        arm.servo_end()
        self.assertEqual(arm._core._depth, 0)

    def test_servo_start_rejects_non_radians_even_when_already_active(self):
        arm = _make_arm()
        arm.servo_start()

        with self.assertRaisesRegex(ValueError, "radians-only"):
            arm.servo_start(controller.ServoOpts(is_radians=False))

        self.assertEqual(len(arm._core.servo_start_calls), 1)
        arm.servo_end()
    def test_servo_owner_may_send_nonblocking_gripper(self):
        core = _FakeRobotCore(state=_NativeRobotState.SERVOING)
        arm = _make_arm(
            state=controller.RobotState.SERVOING,
            core=core,
        )

        result = arm.gripper_control(0.0, block=False)

        self.assertIsNone(result)
        self.assertEqual(len(arm._ht.pos_vel_acc_calls), 1)
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertEqual(core.begin_calls, [])
        self.assertEqual(core.end_calls, [])
        self.assertIs(core.active_operation, _OperationKind.SERVO)
        self.assertIs(core.state, _NativeRobotState.SERVOING)
        self.assertIs(arm.state, controller.RobotState.SERVOING)

        self.assertTrue(arm.servo_j([0.0, 0.0]))
        self.assertEqual(len(core.servo_tick_calls), 1)
        self.assertIs(core.active_operation, _OperationKind.SERVO)

    def test_other_thread_cannot_borrow_servo_for_gripper(self):
        core = _FakeRobotCore(state=_NativeRobotState.SERVOING)
        arm = _make_arm(
            state=controller.RobotState.SERVOING,
            core=core,
        )
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
        core = _FakeRobotCore(state=_NativeRobotState.SERVOING)
        arm = _make_arm(
            state=controller.RobotState.SERVOING,
            core=core,
        )

        with self.assertRaises(controller.RobotStateError):
            arm.gripper_control(0.0, effort=123, block=True)
        with self.assertRaises(controller.RobotStateError):
            arm.grasp(force_threshold=123)

        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertIs(core.active_operation, _OperationKind.SERVO)
        self.assertIs(core.state, _NativeRobotState.SERVOING)

    def test_servo_rejects_new_joint_motion(self):
        core = _FakeRobotCore(state=_NativeRobotState.SERVOING)
        arm = _make_arm(
            state=controller.RobotState.SERVOING,
            core=core,
        )

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

    def test_mit_command_boundary_forwards_radians_unchanged(self):
        arm = _make_arm()
        pos = [np.pi, -np.pi / 2.0]
        vel = [2.0 * np.pi, -np.pi]

        arm.move_MIT(pos, vel, [0.0, 0.0], kp=0.0, kd=0.0)

        self.assertEqual(len(arm._ht.mit_calls), 1)
        args = arm._ht.mit_calls[0]
        np.testing.assert_allclose(args[1], pos)
        np.testing.assert_allclose(args[2], vel)
        self.assertEqual(args[6], 2)
        self.assertEqual(args[7], 0.0)

    def test_mit_and_joint_paths_reject_non_radians_before_work(self):
        arm = _make_arm()

        with self.assertRaisesRegex(ValueError, "radians-only"):
            arm.move_MIT(
                [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
                is_radians=False,
            )
        with self.assertRaisesRegex(ValueError, "radians-only"):
            arm.move_jntspace_path([[0.0, 0.0]], is_radians=False)
        with self.assertRaisesRegex(ValueError, "radians-only"):
            arm.move_jntspace_path_mit(
                [[0.0, 0.0]], is_radians=False
            )

        self.assertEqual(arm._ht.mit_calls, [])
        self.assertEqual(arm._core.move_j_calls, [])

    def test_high_level_controller_does_not_expose_raw_driver_or_turns_reader(self):
        arm = _make_arm()

        self.assertFalse(hasattr(arm, "driver"))
        self.assertFalse(hasattr(arm, "get_joint_values_raw"))
    def test_gripper_command_boundary_forwards_radians_unchanged(self):
        arm = _make_arm()

        arm.gripper_control(
            np.pi,
            vel=2.0 * np.pi,
            acc=3.0 * np.pi,
            block=False,
        )

        self.assertEqual(len(arm._ht.pos_vel_acc_calls), 1)
        args = arm._ht.pos_vel_acc_calls[0]
        self.assertEqual(args[0], 7)
        self.assertAlmostEqual(args[1], np.pi)
        self.assertAlmostEqual(args[2], 2.0 * np.pi)
        self.assertAlmostEqual(args[3], 3.0 * np.pi)

    def test_all_gripper_entry_points_reject_non_radians_without_sending(self):
        arm = _make_arm()
        calls = (
            lambda: arm.gripper_control(0.0, is_radians=False, block=False),
            lambda: arm.grasp(is_radians=False),
            lambda: arm.open_gripper(is_radians=False, block=False),
            lambda: arm.close_gripper(is_radians=False, block=False),
            lambda: arm.release(is_radians=False, block=False),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "radians-only"):
                    call()

        self.assertEqual(arm._ht.pos_vel_acc_calls, [])
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])

    def test_soft_limit_boundary_is_radians_only_and_validates_before_write(self):
        arm = _make_arm()

        arm.set_limit(1, -np.pi, np.pi)
        self.assertEqual(
            arm._ht.position_limit_calls,
            [(1, -np.pi, np.pi, _PosUnit.Radians)],
        )
        np.testing.assert_allclose(arm.get_limit(1), [-np.pi, np.pi])
        np.testing.assert_allclose(
            arm._cfg.limits[1], [-0.5, 0.5]
        )

        with self.assertRaisesRegex(ValueError, "radians-only"):
            arm.set_limit(1, -180.0, 180.0, is_radians=False)
        with self.assertRaisesRegex(ValueError, "radians-only"):
            arm.get_limit(1, is_radians=False)
        with self.assertRaisesRegex(ValueError, "finite"):
            arm.set_limit(1, np.nan, np.pi)

        self.assertEqual(len(arm._ht.position_limit_calls), 1)

        arm.disable_limit(1)
        self.assertEqual(arm._ht.disable_limit_calls, [1])
        self.assertNotIn(1, arm._cfg.limits)
        self.assertIsNone(arm.get_limit(1))

        arm.set_limit(1, -np.pi, np.pi)
        arm.clear_limits()
        self.assertEqual(arm._ht.clear_limit_calls, 1)
        self.assertEqual(arm._cfg.limits, {})
        self.assertIsNone(arm.get_limit(1))

    def test_soft_limit_mutations_validate_id_and_reject_busy(self):
        arm = _make_arm()

        for call in (
            lambda: arm.set_limit(99, -1.0, 1.0),
            lambda: arm.get_limit(99),
            lambda: arm.disable_limit(99),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "cfg.motor_ids"):
                    call()

        arm._core.state = _NativeRobotState.MOVING
        busy_calls = (
            lambda: arm.set_limit(1, -1.0, 1.0),
            lambda: arm.disable_limit(1),
            arm.clear_limits,
        )
        for call in busy_calls:
            with self.subTest(call=call):
                with self.assertRaises(controller.RobotStateError):
                    call()

        self.assertEqual(arm._ht.position_limit_calls, [])
        self.assertEqual(arm._ht.disable_limit_calls, [])
        self.assertEqual(arm._ht.clear_limit_calls, 0)

    def test_fault_recovery_waits_for_inflight_operation_exit(self):
        arm = _make_arm()
        core = arm._core
        owns = arm._enter_operation("move", controller.RobotState.MOVING)

        try:
            core.state = _NativeRobotState.ESTOP
            core.cancel_requested = True
            with self.assertRaisesRegex(
                controller.RobotStateError,
                "enable rejected",
            ):
                arm.enable()

            core.state = _NativeRobotState.DEAD
            core.dead_reason = "feedback lost"
            with self.assertRaisesRegex(
                controller.RobotStateError,
                "recover rejected",
            ):
                arm.recover(confirm=True)
        finally:
            arm._exit_operation(owns)

        self.assertIs(core.active_operation, _OperationKind.NONE)
        self.assertFalse(core.operation_owned_by_current_thread())
        self.assertIs(arm.state, controller.RobotState.DEAD)

    def test_invalid_close_policy_has_no_side_effects(self):
        arm = _make_arm()
        driver = mock.Mock()
        arm._ht = driver
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

    def test_move_j_native_state_error_is_translated_without_driver_send(self):
        arm = _make_arm()
        arm._core.move_j = mock.Mock(side_effect=_StateError("link lost"))

        with self.assertRaisesRegex(controller.RobotStateError, "link lost"):
            arm.move_j([0.0, 0.0])

        arm._core.move_j.assert_called_once()
        self.assertEqual(arm._ht.mit_calls, [])
        self.assertEqual(arm._ht.pos_vel_tqe_calls, [])
        self.assertEqual(arm._ht.pos_vel_acc_calls, [])
        self.assertIs(arm.state, controller.RobotState.IDLE)

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


    def test_gravity_start_enables_from_native_disabled_state(self):
        active = types.SimpleNamespace(mode=controller.MODE_POSITION)
        arm = _make_arm(
            state=controller.RobotState.DISABLED,
            responses={1: active, 2: active},
        )
        arm._dynamics = object()
        arm._ht.get_cached_state = mock.Mock(return_value=active)

        with mock.patch.object(
            arm, "enable", wraps=arm.enable
        ) as enable, mock.patch("builtins.print"):
            arm.start_gravity_compensation(duration=0.0)

        enable.assert_called_once_with()
        arm._ht.get_cached_state.assert_not_called()
        self.assertIs(arm.state, controller.RobotState.BRAKED)

    def test_gravity_begin_failure_does_not_leave_native_operation(self):
        arm = _make_arm()
        arm._dynamics = object()
        arm._core.begin_operation = mock.Mock(
            side_effect=_BusyError("native busy")
        )

        with mock.patch.object(
            arm, "_motors_in_position_mode", return_value=True
        ), self.assertRaisesRegex(controller.RobotStateError, "native busy"):
            arm.start_gravity_compensation(duration=0.0)

        self.assertFalse(arm.is_gravity_compensating)
        self.assertIs(
            arm._core.active_operation,
            _OperationKind.NONE,
        )


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
        core = arm._core
        old_dynamics = object()
        arm._dynamics = old_dynamics
        loaded = types.SimpleNamespace(
            model=types.SimpleNamespace(nq=2),
            eef_frame_name="tool_link",
            gravity_vector=np.array([0.0, 0.0, -9.81]),
        )
        operation_tokens = []

        def load_and_start_operation(*_args, **_kwargs):
            operation_tokens.append(
                core.begin_operation(_OperationKind.JOINT_MOTION)
            )
            return loaded

        try:
            with mock.patch.object(
                controller, "resolve_urdf_path", return_value="robot.urdf"
            ), mock.patch.object(
                controller.DynamicsModel,
                "load",
                side_effect=load_and_start_operation,
            ), self.assertRaises(controller.RobotStateError):
                arm.setup_dynamics()
        finally:
            if operation_tokens:
                core.end_operation(operation_tokens[0])

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
