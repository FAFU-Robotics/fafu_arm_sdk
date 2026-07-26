# -*- coding: utf-8 -*-
"""
Fafu Robot Controller Wrapper
=============================

This module provides a high-level Python wrapper around the
``fafu_motor`` (pybind11) module for the **Fafu robot arm**
(built on the Hightorque debug board hardware).
The goal is to present an interface similar in spirit to the
``PiperArmController`` shown in :mod:`piper.py` while hiding the
low-level details of motor IDs, modes, units and the
``HightorqueSerial`` driver.

Naming note
-----------
The underlying pybind11 binding module is ``fafu_motor`` (the C++
debug-board driver built from ``../fafu_robot_cpp/bindings.cpp``).
Everything user-facing — module names, class names, log prefixes,
docstrings — is exposed under the **Fafu** identity.

Conventions
-----------

* All hardware-command angles and angular rates exposed to the user are in
  **radians**; compatibility flags such as ``is_radians`` only accept
  ``True``.  The native C++ layer owns protocol-unit conversion.
* Velocities are expressed as a percentage in the ``speed`` argument
  (``0 - 100``); the native motion core maps this to a conservative
  angular-velocity limit.
* The Fafu arm is a chain of independent motors driven over a
  USB-CAN debug board. Cartesian motion uses the optional Pinocchio
  model configured through :meth:`setup_dynamics`; joint-space motion
  remains available without it.

Example
-------

>>> from fafu_robot_controller import FafuRobotController
>>> import numpy as np
>>>
>>> # cfg_path is required; gripper is optional (motor id 7 in the
>>> # default robot.cfg).
>>> arm = FafuRobotController(
...     cfg_path="robot.cfg",
...     has_gripper=True,
...     gripper_motor_id=7,
... )
>>>
>>> # current joint angles (rad)
>>> q = arm.get_joint_values()
>>>
>>> # move to a target configuration with S-curve and wait for finish
>>> arm.move_j([0, 0.2, 0.5, 0, 0, 0], speed=20, block=True)
>>>
>>> arm.open_gripper()
>>> arm.close_gripper()
>>>
>>> # Piper-style position+effort (firmware-side torque cap)
>>> arm.gripper_control(angle=math.radians(-90), effort=600)
>>>
>>> # Force-aware grasp (Python-side torque monitoring + early stop)
>>> result = arm.grasp(force_threshold=500)
>>> if result.grasped:
...     print(f"got it, peak torque {result.peak_torque_raw} raw, "
...           f"closed {result.closed_deg:.1f} deg in {result.duration_s:.2f}s")
>>>
>>> arm.disable()
>>> arm.close_connection()
"""

from __future__ import annotations

import functools
import math
import os
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
#  Import the native extension as a package, with a direct-module fallback for
#  legacy scripts that put this directory on sys.path.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if __package__:
    from . import fafu_motor as pm
    from ._api_types import (
        FrictionParams,
        GraspResult,
        MODE_BRAKE,
        MODE_MIT,
        MODE_POSITION,
        MODE_STOP,
        RobotState,
        RobotStateError,
        ServoOpts,
        _BUSY_STATES,
    )
    from ._dynamics import (
        DynamicsModel,
        friction_compensation,
        resolve_urdf_path,
    )
else:  # pragma: no cover - direct module import compatibility
    import fafu_motor as pm
    from _api_types import (
        FrictionParams,
        GraspResult,
        MODE_BRAKE,
        MODE_MIT,
        MODE_POSITION,
        MODE_STOP,
        RobotState,
        RobotStateError,
        ServoOpts,
        _BUSY_STATES,
    )
    from _dynamics import (
        DynamicsModel,
        friction_compensation,
        resolve_urdf_path,
    )

__all__ = [
    "FafuRobotController",
    "FrictionParams",
    "GraspResult",
    "MODE_BRAKE",
    "MODE_MIT",
    "MODE_POSITION",
    "MODE_STOP",
    "RobotState",
    "RobotStateError",
    "ServoOpts",
]


# Optional: TOPPRA-based time-optimal interpolation (matches piper.py).
try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp  # type: ignore

    _TOPPRA_EXIST = True
except Exception:  # pragma: no cover - optional dependency
    _TOPPRA_EXIST = False

# ============================================================================
#  Operation guard
# ============================================================================
_BORROWED_SERVO_WRITER = object()
_NATIVE_STATE_ERRORS = tuple(
    cls for cls in (getattr(pm, "StateError", None),)
    if isinstance(cls, type)
)


def _guard_operation(
    action: str,
    busy_state: "RobotState",
    *,
    allow_servo_nonblocking: bool = False,
):
    """Decorator: gate a call-scoped, blocking operation behind the
    controller state machine.

    On entry it verifies the controller is :attr:`RobotState.IDLE`
    (via :meth:`FafuRobotController._enter_operation`) and switches to
    ``busy_state``; on exit it restores ``IDLE``.  Re-entrant: a guarded
    method that internally calls another guarded method (e.g.
    ``go_home`` -> ``move_j``, ``move_l`` -> ``move_jntspace_path``)
    nests cleanly without the inner guard rejecting the outer one.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            borrow_servo_writer = (
                allow_servo_nonblocking
                and kwargs.get("block", True) is False
            )
            owns = self._enter_operation(
                action,
                busy_state,
                borrow_servo_writer=borrow_servo_writer,
            )
            try:
                return fn(self, *args, **kwargs)
            finally:
                self._exit_operation(owns)

        return wrapper

    return deco

# Native move_j speed and duration policy.
_VEL_AVG_MAX_TPS = 0.5
_DT_MIN_S = 0.3

# Native RobotCore owns feedback freshness and the latched DEAD transition.
_DEAD_RX_TIMEOUT_MS = 500.0

# 1 turn = 2*pi rad
_TWO_PI = 2.0 * math.pi

# ============================================================================
#  Controller
# ============================================================================
class FafuRobotController:
    """High-level controller for the Fafu robotic arm.

    Parameters
    ----------
    cfg_path : str
        Path to the ``robot.cfg`` file.  Relative paths are first
        resolved against the current working directory and then
        against the directory holding this file.  The configuration
        provides ``port``, ``baudrate``, ``motor_ids``, soft limits
        and the control rate.
    port : str, optional
        Override ``cfg.port``.  ``"auto"`` (or ``None``) triggers
        USB enumeration via :func:`fafu_motor.find_likely_debug_boards`.
    baudrate : int, optional
        Override ``cfg.baudrate``.  Defaults to ``cfg.baudrate``
        (typically 4 Mbps).
    has_gripper : bool, optional
        If ``True`` the motor with id ``gripper_motor_id`` is treated
        as the gripper rather than a manipulator joint.  Joint-space
        commands (``move_j``, ``get_joint_values`` ...) then ignore
        that motor.  Defaults to ``False``.
    gripper_motor_id : int, optional
        Motor id of the gripper.  Required when ``has_gripper`` is
        ``True``; must be present in ``cfg.motor_ids``.
    auto_enable : bool, optional
        If ``True`` (default) all motors are switched into
        position-control mode (``0x0A``) immediately after the serial
        port is opened.
    auto_polling : bool, optional
        If ``True`` (default) a 50 Hz background polling thread is
        started after enabling so that :meth:`get_joint_values` reads
        from a non-blocking cache.
    async_rx : bool, optional
        Override ``cfg.use_async_rx``.  When ``None`` the value from
        the configuration file is used.

    Notes
    -----
    The Fafu firmware requires :meth:`set_motor_mode` to be
    called *before* :meth:`enable_async_rx`, so this constructor
    enforces that order regardless of how the flags are combined.
    """

    # Re-exported as class attributes so users do not have to import
    # the module-level constants.
    MODE_POSITION = MODE_POSITION
    MODE_BRAKE    = MODE_BRAKE
    MODE_STOP     = MODE_STOP
    MODE_MIT      = MODE_MIT

    # High-level state machine (exposed for convenience so callers can do
    # ``if arm.state is arm.State.IDLE`` / ``except arm.StateError``).
    State      = RobotState
    StateError = RobotStateError

    # ------------------------------------------------------------------
    #  Construction / teardown
    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg_path: str,
        *,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        has_gripper: bool = False,
        gripper_motor_id: Optional[int] = None,
        auto_enable: bool = True,
        auto_polling: bool = True,
        async_rx: Optional[bool] = None,
    ) -> None:
        # Refuse an old native binary before loading configuration or opening
        # a serial port. P0 intentionally has no Python safety/state fallback.
        if getattr(pm, "CORE_ABI_VERSION", 0) != 4:
            raise RuntimeError(
                "fafu_motor native core ABI 4 is required; rebuild/install "
                "the C++ extension from this SDK version"
            )

        if not cfg_path:
            raise ValueError("cfg_path must be provided")

        cfg_path = self._resolve_cfg_path(cfg_path)
        try:
            cfg = pm.RobotConfig.load(cfg_path)
        except Exception as e:
            raise RuntimeError(f"failed to load config {cfg_path!r}: {e}") from e

        self._cfg_path = cfg_path
        self._cfg: pm.RobotConfig = cfg

        if has_gripper:
            if gripper_motor_id is None:
                raise ValueError("has_gripper=True requires gripper_motor_id")
            if gripper_motor_id not in cfg.motor_ids:
                raise ValueError(
                    f"gripper_motor_id {gripper_motor_id} not in cfg.motor_ids "
                    f"{list(cfg.motor_ids)}"
                )
        self._has_gripper = bool(has_gripper)
        self._gripper_motor_id = gripper_motor_id

        # Joint motors == all motor_ids minus the gripper id.
        if self._has_gripper:
            self._joint_motor_ids: List[int] = [
                m for m in cfg.motor_ids if m != gripper_motor_id
            ]
        else:
            self._joint_motor_ids = list(cfg.motor_ids)

        if not self._joint_motor_ids:
            raise ValueError("no joint motors after excluding the gripper")

        port_to_use = self._pick_serial_port(port if port else cfg.port)
        baud_to_use = int(baudrate or cfg.baudrate)

        try:
            self._ht = pm.HightorqueSerial(port_to_use, baud_to_use)
        except Exception as e:
            raise RuntimeError(
                f"failed to open serial port {port_to_use!r} @ {baud_to_use}: {e}"
            ) from e
        self._port = port_to_use
        self._baudrate = baud_to_use

        # Serializes Python configuration commits with native writer-lease
        # acquisition. RobotCore is the sole authority for lifecycle state,
        # operation ownership, cancellation and feedback-loss latching.
        self._config_lock = threading.RLock()
        self._state_verbose: bool = False
        # Diagnostic-only cache used for optional transition logging. Safety
        # decisions always read RobotCore directly.
        self._last_reported_state: Optional[RobotState] = None

        # Native P0 core is the sole authority for state, writer ownership,
        # recovery and Servo safety.
        core_cfg = pm.CoreConfig()
        core_cfg.all_motor_ids = list(cfg.motor_ids)
        core_cfg.joint_motor_ids = list(self._joint_motor_ids)
        core_cfg.max_torque_raw = int(cfg.max_torque_raw)
        core_cfg.stale_feedback_timeout_ms = _DEAD_RX_TIMEOUT_MS
        core_cfg.polling_rate_hz = max(
            10.0, float(cfg.control_rate_hz or 50.0))
        self._core = pm.RobotCore(self._ht, core_cfg)

        # Push soft limits configured in robot.cfg into the driver.
        try:
            cfg.apply_limits_to(self._ht)
        except Exception as e:
            print(f"[FafuRobot] warning: apply_limits_to failed: {e}")

        # Verify that every motor responds before doing anything risky.
        self._precheck_communication()

        # Order is significant: set_motor_mode MUST run before
        # enable_async_rx so that the SDK can verify the mode echo.
        #
        # enable() failure is non-fatal at construction so diagnostics and
        # recovery remain available. The native core stops every motor and
        # keeps motion disabled until a later enable() succeeds.
        if auto_enable:
            try:
                self.enable()
            except Exception as e:
                print(f"[FafuRobot] warning: enable() failed at startup: {e}")
                print("[FafuRobot] motion remains disabled; run diagnostics, "
                      "recover the fault, then call enable() again.")

        use_async = async_rx if async_rx is not None else bool(cfg.use_async_rx)
        polling_hz = max(10.0, float(cfg.control_rate_hz or 50.0))
        self._core.start_transport(
            bool(use_async), bool(auto_polling), polling_hz)

        # Python only keeps user options for optional dynamics feed-forward.
        # Servo lifecycle, counters and ownership live in the native core.
        self._servo_opts: Optional[ServoOpts] = None

        # ---- Optional numerical dynamics state ----
        # Hardware ownership and safety remain here. The helper owns only the
        # Pinocchio model and deterministic numerical operations.
        self._dynamics: Optional[DynamicsModel] = None
        self._dyn_motor_models: Optional[List[str]] = None
        self._dyn_tau_limit: Optional[np.ndarray] = None
        self._dyn_torque_scale: np.ndarray = np.ones(self.num_joints)
        self._friction_params: Optional[FrictionParams] = None

        # Feed-forward torque channel for gravity/friction compensation:
        #   True  -> group MIT (one 0x8093 frame, kp=kd=0), vendor-equivalent
        #            pos_vel_tqe_kp_kd. Validated on this firmware (one-to-many
        #            MIT is actuated; single-motor 0x15 is not).
        #   False -> legacy per-joint set_torque (0x0A), one frame per joint.
        # Auto-disabled when num_joints > 6 (one MIT frame holds <=6 motors).
        self._use_group_mit: bool = (
            self.num_joints <= 6 and max(self._joint_motor_ids) <= 6
        )

        print(
            f"[FafuRobot] connected on {self._port} @ {self._baudrate} "
            f"({len(self._joint_motor_ids)} joints"
            + (f" + gripper M{self._gripper_motor_id}" if self._has_gripper else "")
            + ")"
        )

    @staticmethod
    def _python_state_from_core(core_state) -> RobotState:
        return RobotState[core_state.name]

    def _sync_state_from_core(self) -> RobotState:
        state = self._python_state_from_core(self._core.state)
        previous = getattr(self, "_last_reported_state", None)
        if (
            self._state_verbose
            and previous is not None
            and previous is not state
        ):
            print(f"[FafuRobot] state: {previous} -> {state}")
        self._last_reported_state = state
        return state

    @staticmethod
    def _core_finish_mode(mode: str):
        return {
            "stop": pm.FinishMode.STOP,
            "brake": pm.FinishMode.BRAKE,
            "hold": pm.FinishMode.HOLD,
        }[mode]

    def _native_cancel_requested(self) -> bool:
        return bool(self._core.cancel_requested)

    def _command_guard(self):
        """Serialize one hardware write with ESTOP/DEAD for this instance."""
        return self._core.command_guard()

    def _combined_abort_check(
        self, abort_check: Optional[Callable[[], bool]]
    ) -> Callable[[], bool]:
        def cancelled() -> bool:
            if self._native_cancel_requested():
                return True
            if not self._core.stream_link_ok():
                self._sync_state_from_core()
                return True
            return bool(abort_check is not None and abort_check())
        return cancelled

    # ------------------------------------------------------------------
    #  Public properties
    # ------------------------------------------------------------------
    @property
    def cfg(self) -> pm.RobotConfig:
        """Underlying :class:`fafu_motor.RobotConfig`."""
        return self._cfg

    @property
    def port(self) -> str:
        return self._port

    @property
    def baudrate(self) -> int:
        return self._baudrate

    @property
    def joint_motor_ids(self) -> List[int]:
        """Motor ids that participate in joint-space commands (excludes gripper)."""
        return list(self._joint_motor_ids)

    @property
    def all_motor_ids(self) -> List[int]:
        return list(self._cfg.motor_ids)

    @property
    def num_joints(self) -> int:
        return len(self._joint_motor_ids)

    @property
    def has_gripper(self) -> bool:
        return self._has_gripper

    @property
    def gripper_motor_id(self) -> Optional[int]:
        return self._gripper_motor_id

    # ------------------------------------------------------------------
    #  State machine
    # ------------------------------------------------------------------
    @property
    def state(self) -> RobotState:
        """Current high-level state from the native concurrency core."""
        return self._sync_state_from_core()

    @property
    def state_verbose(self) -> bool:
        """Whether state transitions are printed to stdout (default off)."""
        return self._state_verbose

    @state_verbose.setter
    def state_verbose(self, value: bool) -> None:
        self._state_verbose = bool(value)

    def _require_ready(self, action: str, *, allow_disabled: bool = False) -> None:
        """Reject a command unless the native controller can accept it."""
        state = self.state
        if state is RobotState.IDLE:
            if not self._stream_link_ok():
                raise RobotStateError(
                    f"{action} rejected: feedback was lost (DEAD); "
                    "call recover(confirm=True) after power is restored"
                )
            return
        if state is RobotState.DEAD:
            detail = f": {self.dead_reason}" if self.dead_reason else ""
            raise RobotStateError(
                f"{action} rejected: feedback loss is latched (DEAD){detail}; "
                "call recover(confirm=True), then enable()"
            )
        if state in (RobotState.DISABLED, RobotState.BRAKED):
            if allow_disabled:
                return
            raise RobotStateError(
                f"{action} requires enabled motors; state={state.name}. "
                "Call enable() first"
            )
        if state is RobotState.DISCONNECTED:
            raise RobotStateError(f"{action} failed: connection is closed")
        if state is RobotState.ESTOP:
            raise RobotStateError(
                f"{action} rejected: emergency stop is latched; call resume()"
            )
        raise RobotStateError(
            f"{action} rejected: controller is busy (state={state})"
        )

    def _enter_operation(
        self,
        action: str,
        busy_state: RobotState,
        *,
        borrow_servo_writer: bool = False,
    ):
        """Acquire the native single-writer operation lease."""
        core = self._core
        if (
            action == "gripper_control"
            and borrow_servo_writer
            and core.active_operation == pm.OperationKind.SERVO
            and core.operation_owned_by_current_thread()
        ):
            if not core.stream_link_ok():
                self._sync_state_from_core()
                raise RobotStateError(
                    f"{action} rejected: {core.dead_reason}"
                )
            return _BORROWED_SERVO_WRITER

        if action == "reset_zero":
            kind = pm.OperationKind.RAW_STREAM
        elif busy_state is RobotState.MOVING:
            kind = pm.OperationKind.JOINT_MOTION
        elif busy_state is RobotState.GRASPING:
            kind = (
                pm.OperationKind.GRASP
                if action == "grasp"
                else pm.OperationKind.GRIPPER_MOTION
            )
        elif busy_state is RobotState.GRAVITY_COMP:
            kind = pm.OperationKind.GRAVITY_COMP
        else:
            kind = pm.OperationKind.RAW_STREAM

        token = None
        try:
            with self._config_lock:
                token = core.begin_operation(kind)
            if not core.stream_link_ok():
                core.end_operation(token)
                token = None
                self._sync_state_from_core()
                raise RobotStateError(
                    f"{action} rejected: {core.dead_reason}"
                )
            return int(token)
        except RobotStateError:
            raise
        except _NATIVE_STATE_ERRORS as exc:
            if token is not None:
                try:
                    core.end_operation(token)
                except Exception:
                    pass
            raise RobotStateError(f"{action} rejected: {exc}") from exc
        except BaseException:
            if token is not None:
                try:
                    core.end_operation(token)
                except Exception:
                    pass
            raise

    def _exit_operation(self, owns) -> None:
        if owns is _BORROWED_SERVO_WRITER:
            return
        self._core.end_operation(int(owns))
        self._sync_state_from_core()

    def _require_stream_command(
        self,
        action: str,
        *,
        allow_gravity_owner: bool = False,
    ) -> None:
        """Gate low-level stream writers using native ownership and state."""
        core = self._core
        owned = core.operation_owned_by_current_thread()
        kind = core.active_operation
        internal_move = owned and kind == pm.OperationKind.JOINT_MOTION
        internal_gravity = (
            allow_gravity_owner
            and owned
            and kind == pm.OperationKind.GRAVITY_COMP
        )
        if not (internal_move or internal_gravity):
            self._require_ready(action)
        if not core.stream_link_ok():
            self._sync_state_from_core()
            raise RobotStateError(
                f"{action} rejected: motor feedback is stale (DEAD)."
            )
        state = self.state
        if state in (
            RobotState.ESTOP,
            RobotState.DEAD,
            RobotState.DISCONNECTED,
        ):
            raise RobotStateError(f"{action} rejected: state={state}.")

    @property
    def dead_reason(self) -> Optional[str]:
        """Why the native controller latched DEAD, or ``None``."""
        return self._core.dead_reason or None

    def _stream_link_ok(self) -> bool:
        """Return native per-joint feedback freshness for streaming loops."""
        ok = bool(self._core.stream_link_ok())
        if not ok:
            self._sync_state_from_core()
        return ok

    def check_alive(self, *, fresh: bool = True, timeout: float = 0.1) -> bool:
        """Return whether every joint responds; failures latch native DEAD."""
        alive = bool(self._core.check_alive(bool(fresh), float(timeout)))
        self._sync_state_from_core()
        return alive

    def recover(self, *, confirm: bool = False) -> bool:
        """Verify a restored link and leave a DEAD controller safely BRAKED."""
        if self.state is RobotState.DEAD and not confirm:
            raise RuntimeError(
                "recover(confirm=True) required (safety): confirm the arm "
                "is powered and the workspace is clear."
            )
        try:
            recovered = bool(self._core.recover(bool(confirm), 0.2))
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"recover rejected: {exc}") from exc
        self._sync_state_from_core()
        return recovered

    def sync_state(self) -> RobotState:
        """Conservatively downgrade IDLE after fresh motor-mode reads."""
        state = self.state
        if state is not RobotState.IDLE:
            return state

        token = None
        try:
            token = self._core.begin_operation(pm.OperationKind.LIFECYCLE)
            if self.state is RobotState.IDLE:
                enabled = self._motors_in_position_mode(fresh=True)
                if enabled is None:
                    # Failed reads are link-health questions, not evidence that
                    # motors are safely disabled.
                    self._core.check_alive(True, 0.05)
                elif not enabled:
                    self._core.transition(pm.RobotState.DISABLED)
        except _NATIVE_STATE_ERRORS:
            pass
        finally:
            if token is not None:
                self._core.end_operation(token)
        return self._sync_state_from_core()

    def _motors_in_position_mode(self, *, fresh: bool) -> Optional[bool]:
        for motor_id in self._cfg.motor_ids:
            try:
                state = (
                    self._ht.read_motor_state(motor_id, 0.05)
                    if fresh
                    else self._ht.get_cached_state(motor_id)
                )
                if state is None and not fresh:
                    state = self._ht.read_motor_state(motor_id, 0.05)
            except Exception:
                return None
            if state is None:
                return None
            if getattr(state, "mode", None) != self.MODE_POSITION:
                return False
        return True

    @property
    def is_enabled(self) -> bool:
        """``True`` iff every motor is currently in position-control mode."""
        return bool(self._motors_in_position_mode(fresh=False))

    # ------------------------------------------------------------------
    #  Power management
    # ------------------------------------------------------------------
    def enable(self, *, allow_motor_reset: bool = True) -> None:
        """Enable position control, including the native recovery sequence."""
        self._enable_impl(allow_motor_reset=allow_motor_reset)

    def _enable_impl(self, *, allow_motor_reset: bool = True) -> None:
        options = pm.EnableOptions()
        options.allow_motor_reset = bool(allow_motor_reset)
        try:
            result = self._core.enable(options)
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"enable rejected: {exc}") from exc
        self._sync_state_from_core()
        if result.success:
            return

        failed = list(result.failed_motor_ids)
        suffix = f"; failed motors={failed}" if failed else ""
        raise RuntimeError(f"enable failed: {result.message}{suffix}")

    def disable(self) -> None:
        """Disable motor output and allow the arm to move freely."""
        try:
            self._core.disable()
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"disable rejected: {exc}") from exc
        self._sync_state_from_core()

    def brake(self) -> None:
        """Apply short-circuit braking to every configured motor."""
        try:
            self._core.brake()
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"brake rejected: {exc}") from exc
        self._sync_state_from_core()

    @_guard_operation("move_j", RobotState.MOVING)
    def move_j(
        self,
        joint_angles: Iterable[float],
        *,
        is_radians: bool = True,
        speed: int = 50,
        block: bool = True,
        tolerance: float = 0.01,
        settle_timeout: float = 1.0,
    ) -> None:
        """Move all arm joints through the native radians-only motion core.

        Speed is a percentage of a conservative pi rad/s ceiling. A blocking
        call returns only after fresh feedback confirms every joint is within
        tolerance. Timeout, link loss, cancellation and send failures are
        handled and braked by C++.
        """
        self._require_radians(is_radians, "move_j")
        angles_rad = self._validate_joint_angles(
            joint_angles, is_radians=True
        )
        speed = self._clamp_speed(speed)

        options = pm.MoveJOptions()
        options.block = bool(block)
        options.max_velocity_rad_s = (
            speed / 100.0
        ) * _VEL_AVG_MAX_TPS * _TWO_PI
        rate_hz = float(getattr(self._cfg, "control_rate_hz", 100.0))
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            rate_hz = 100.0
        options.control_rate_hz = min(1000.0, max(10.0, rate_hz))

        duration_s = float(getattr(self._cfg, "trajectory_dt_s", _DT_MIN_S))
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            duration_s = _DT_MIN_S
        options.min_duration_s = min(60.0, max(_DT_MIN_S, duration_s))

        options.tolerance_rad = float(tolerance)
        options.settle_timeout_s = float(settle_timeout)

        try:
            result = self._core.move_j(angles_rad, options)
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"move_j rejected: {exc}") from exc
        finally:
            self._sync_state_from_core()

        if block and not result.reached:
            raise RuntimeError(
                "move_j returned without confirmed target feedback"
            )

    def go_home(self, *, speed: int = 20, block: bool = True) -> None:
        """Move every manipulator joint back to 0 rad."""
        self.move_j(
            [0.0] * self.num_joints,
            is_radians=True,
            speed=speed,
            block=block,
        )

    @_guard_operation("move_jntspace_path", RobotState.MOVING)
    def move_jntspace_path(
        self,
        path,
        *,
        is_radians: bool = True,
        max_jntvel: Optional[List[float]] = None,
        max_jntacc: Optional[List[float]] = None,
        start_frame_id: int = 1,
        speed: int = 50,
        control_frequency: float = 0.05,
    ) -> None:
        """Follow a joint-space waypoint path via ``move_j`` streaming.

        TOPPRA time-parametrises ``path`` into a dense, uniformly-spaced
        (``control_frequency``) waypoint stream; each frame is then sent
        with :meth:`move_j` (``block=False``) on the firmware position
        channel (``pos_vel_MAXtqe`` / ``0x8090``).

        Parameters
        ----------
        path : array_like, shape (N, num_joints)
            Sequence of joint configurations to traverse in order.
        is_radians : bool, optional
            Compatibility guard; only ``True`` is accepted.
        max_jntvel, max_jntacc : list of float, optional
            Per-joint rad/s and rad/s^2 limits (passed straight
            to TOPPRA).
        start_frame_id : int, optional
            Skip the first ``start_frame_id`` interpolated frames
            (typically used to skip the robot's current configuration).
        speed : int, optional
            Speed percentage forwarded to each :meth:`move_j` call.
        control_frequency : float, optional
            ``ctrl_freq`` passed to TOPPRA (seconds); also the per-frame
            stream period (``sleep`` between ``move_j`` frames).

        Raises
        ------
        NotImplementedError
            When the optional ``wrs`` dependency is not available.
        """
        self._require_radians(is_radians, "move_jntspace_path")
        if not _TOPPRA_EXIST:
            raise NotImplementedError(
                "TOPPRA-based interpolation requires "
                "`wrs.motion.trajectory.piecewisepoly_toppra`; "
                "install it or use a custom interpolator."
            )
        if path is None:
            raise ValueError("path must not be None")

        path_arr = np.asarray(path, dtype=float)
        if (
            path_arr.ndim != 2
            or path_arr.shape[0] == 0
            or path_arr.shape[1] != self.num_joints
            or not np.all(np.isfinite(path_arr))
        ):
            raise ValueError(
                f"path must be a non-empty finite (N, {self.num_joints}) array; "
                f"got shape {path_arr.shape}"
            )

        tpply = pwp.PiecewisePolyTOPPRA()
        interpolated = tpply.interpolate_by_max_spdacc(
            path=path_arr,
            ctrl_freq=control_frequency,
            max_vels=max_jntvel,
            max_accs=max_jntacc,
            toggle_debug=False,
        )
        interpolated = interpolated[start_frame_id:]
        for jnt_values in interpolated:
            self.move_j(
                joint_angles=jnt_values,
                is_radians=True,
                speed=speed,
                block=False,
            )
            time.sleep(max(0.005, control_frequency))

    @_guard_operation("move_jntspace_path_mit", RobotState.MOVING)
    def move_jntspace_path_mit(
        self,
        path,
        *,
        is_radians: bool = True,
        max_jntvel: Optional[List[float]] = None,
        max_jntacc: Optional[List[float]] = None,
        start_frame_id: int = 1,
        speed: int = 50,
        control_frequency: float = 0.05,
        kp: "float | Iterable[float] | None" = None,
        kd: "float | Iterable[float] | None" = None,
        gravity_ff: bool = True,
    ) -> None:
        """Follow a joint-space waypoint path via **group MIT** streaming.

        Same TOPPRA time-parametrisation as :meth:`move_jntspace_path`, but
        each interpolated frame is sent on the one-to-many MIT channel
        (:meth:`move_MIT`, CAN ID ``0x8093``) with per-joint **target
        position + finite-difference velocity + gravity feed-forward +
        kp/kd**.  Prefer this when you want MIT tracking / gravity FF;
        use :meth:`move_jntspace_path` for firmware position (``move_j``).

        Parameters
        ----------
        path : array_like, shape (N, num_joints)
            Sequence of joint configurations to traverse in order.
        is_radians : bool, optional
            Compatibility guard; only ``True`` is accepted.
        max_jntvel, max_jntacc : list of float, optional
            Per-joint rad/s and rad/s^2 limits (passed to TOPPRA).
        start_frame_id : int, optional
            Skip the first ``start_frame_id`` interpolated frames.
        speed : int, optional
            Kept for signature compatibility with
            :meth:`move_jntspace_path`; **ignored** here (timing comes
            from TOPPRA + ``control_frequency``).
        control_frequency : float, optional
            ``ctrl_freq`` passed to TOPPRA (seconds); also the per-frame
            stream period.
        kp, kd : float or iterable, optional
            Per-joint MIT PD gains in **physical vendor units** (see
            :meth:`move_MIT`).  ``None`` uses vendor replay defaults for a
            6-DoF arm (``kp=[30,40,55,15,7,5]``,
            ``kd=[3,4,5.5,1.5,0.7,0.5]``); scalar otherwise.
        gravity_ff : bool, optional
            Add gravity feed-forward torque per frame (needs
            :meth:`setup_dynamics`).  Default ``True``; falls back to
            zero feed-forward when no model is loaded.

        Raises
        ------
        NotImplementedError
            When the optional ``wrs`` dependency is not available.
        RuntimeError
            When ``num_joints > 6`` (one MIT frame holds at most 6 motors).
        """
        self._require_radians(is_radians, "move_jntspace_path_mit")
        if not _TOPPRA_EXIST:
            raise NotImplementedError(
                "TOPPRA-based interpolation requires "
                "`wrs.motion.trajectory.piecewisepoly_toppra`; "
                "install it or use a custom interpolator."
            )
        if path is None:
            raise ValueError("path must not be None")

        path_arr = np.asarray(path, dtype=float)
        if (
            path_arr.ndim != 2
            or path_arr.shape[0] == 0
            or path_arr.shape[1] != self.num_joints
            or not np.all(np.isfinite(path_arr))
        ):
            raise ValueError(
                f"path must be a non-empty finite (N, {self.num_joints}) array; "
                f"got shape {path_arr.shape}"
            )

        if self.num_joints > 6 or max(self._joint_motor_ids) > 6:
            raise RuntimeError(
                "move_jntspace_path_mit requires at most six joint motors "
                "with IDs in 1..6; configure joint_motor_ids accordingly"
            )

        tpply = pwp.PiecewisePolyTOPPRA()
        interpolated = tpply.interpolate_by_max_spdacc(
            path=path_arr,
            ctrl_freq=control_frequency,
            max_vels=max_jntvel,
            max_accs=max_jntacc,
            toggle_debug=False,
        )
        interpolated = interpolated[start_frame_id:]
        if len(interpolated) == 0:
            return

        n = self.num_joints
        if kp is None:
            kp = ([30.0, 40.0, 55.0, 15.0, 7.0, 5.0][:n] if n == 6 else 20.0)
        if kd is None:
            kd = ([3.0, 4.0, 5.5, 1.5, 0.7, 0.5][:n] if n == 6 else 2.0)
        if gravity_ff and not self.has_dynamics:
            print("[FafuRobot] move_jntspace_path_mit: gravity_ff requested but "
                  "no dynamics model; streaming kp/kd only (setup_dynamics to "
                  "add gravity feed-forward).")

        dt = max(0.005, control_frequency)
        prev = None
        for jnt_values in interpolated:
            jv = np.asarray(jnt_values, dtype=float)
            vel = np.zeros(n) if prev is None else (jv - prev) / dt
            prev = jv
            if gravity_ff and self.has_dynamics:
                tau = self.compute_compensation_torque(
                    jv, vel, friction=False)
            else:
                tau = np.zeros(n)
            self.move_MIT(
                jv, vel, tau, kp=kp, kd=kd,
                is_radians=True, apply_torque_scale=True, timeout=0.0)
            time.sleep(dt)
        # Re-assert the final pose briefly so the arm settles on target.
        for _ in range(3):
            jv = np.asarray(interpolated[-1], dtype=float)
            if gravity_ff and self.has_dynamics:
                tau = self.compute_compensation_torque(
                    jv, np.zeros(n), friction=False)
            else:
                tau = np.zeros(n)
            self.move_MIT(jv, np.zeros(n), tau, kp=kp, kd=kd,
                          is_radians=True, apply_torque_scale=True,
                          timeout=0.0)
            time.sleep(dt)

    # ------------------------------------------------------------------
    #  Servo (online streaming) control
    # ------------------------------------------------------------------
    # Timing-sensitive safety and protocol work lives in RobotCore. Python
    # only maps public options and computes optional dynamics feed-forward.
    def servo_start(self, opts: Optional[ServoOpts] = None) -> None:
        """Start a caller-driven Servo session in the native core."""
        core = self._core
        opts = ServoOpts(**vars(opts)) if opts is not None else ServoOpts()
        self._require_radians(opts.is_radians, "servo_start")

        if core.is_servoing:
            if core.operation_owned_by_current_thread():
                return
            raise RobotStateError(
                "servo_start rejected: another thread owns the Servo session"
            )

        native = pm.ServoOptions()
        native.watchdog_ms = int(opts.watchdog_ms)
        native.max_velocity_rad_s = float(opts.max_vel)
        native.max_step_rad = float(opts.max_step_rad)
        native.max_lag_rad = float(opts.max_lag_rad)
        native.nominal_rate_hz = float(opts.rate_hz)
        native.input_is_radians = True
        native.feedforward_velocity = bool(opts.feedforward_vel)
        native.position_error_deadband_rad = float(
            opts.position_error_deadband_rad
        )
        native.lookahead_time_s = float(opts.lookahead_time)
        native.lag_abort_consecutive = int(opts.lag_abort_consecutive)
        native.channel = (
            pm.ServoChannel.MIT
            if opts.use_mit
            else pm.ServoChannel.POSITION
        )
        if opts.mit_kp is not None:
            native.mit_kp = (
                [float(opts.mit_kp)]
                if np.isscalar(opts.mit_kp)
                else [float(x) for x in opts.mit_kp]
            )
        if opts.mit_kd is not None:
            native.mit_kd = (
                [float(opts.mit_kd)]
                if np.isscalar(opts.mit_kd)
                else [float(x) for x in opts.mit_kd]
            )

        with self._config_lock:
            models = (
                list(opts.motor_models)
                if opts.motor_models
                else (
                    list(self._dyn_motor_models)
                    if self._dyn_motor_models is not None
                    else None
                )
            )
            if opts.use_mit and models is None:
                raise ValueError(
                    "MIT servo requires exact per-joint motor_models; pass "
                    "ServoOpts(motor_models=[...]) or call set_motor_models()"
                )
            if models is not None:
                self.set_motor_models(models)
            try:
                core.servo_start(native)
            except _NATIVE_STATE_ERRORS as exc:
                raise RobotStateError(f"servo_start rejected: {exc}") from exc
            self._servo_opts = opts
        self._sync_state_from_core()

    def servo_j(self, target_angles: Iterable[float]) -> bool:
        """Send one non-blocking joint target through the native Servo core."""
        values = [float(x) for x in target_angles]
        torque_ff: List[float] = []
        opts = self._servo_opts
        if (
            opts is not None
            and opts.use_mit
            and opts.mit_gravity_ff
            and self.has_dynamics
            and len(values) == self.num_joints
        ):
            q_rad = np.asarray(values, dtype=float)
            tau = self.compute_compensation_torque(
                q_rad, np.zeros(self.num_joints), friction=False
            )
            torque_ff = [
                float(x) for x in tau * self._dyn_torque_scale
            ]

        try:
            result = self._core.servo_tick(values, torque_ff)
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"servo_j rejected: {exc}") from exc
        self._sync_state_from_core()
        if not result.sent and result.message:
            print(f"[FafuRobot] servo_j: {result.message}")
        return bool(result.sent and not result.aborted)

    def servo_end(self, finish_mode: str = "hold") -> None:
        """End the Servo session with hold, brake or stop."""
        if finish_mode not in {"stop", "brake", "hold"}:
            raise ValueError(
                "finish_mode must be one of ['brake', 'hold', 'stop']"
            )
        try:
            self._core.servo_end(self._core_finish_mode(finish_mode))
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"servo_end rejected: {exc}") from exc
        self._sync_state_from_core()

    @property
    def is_servoing(self) -> bool:
        """Whether a Servo session is currently active."""
        return bool(self._core.is_servoing)

    @property
    def servo_lag_count(self) -> int:
        """Lag-tripped ticks in the current or most recent Servo session."""
        return int(self._core.servo_summary().lag_count)

    @property
    def servo_clamp_count(self) -> int:
        """Step-clamped ticks in the current or most recent Servo session."""
        return int(self._core.servo_summary().clamp_count)

    @property
    def servo_aborted_reason(self) -> Optional[str]:
        """Reason for the most recent automatic Servo abort, if any."""
        return self._core.servo_summary().aborted_reason or None
    # ------------------------------------------------------------------
    #  Dynamics: gravity + friction compensation ("float" / teach mode)
    # ------------------------------------------------------------------
    #
    #  Gravity + friction feed-forward compensation.  The control law is::
    #
    #      tau = clip( G(q) + [ fc*sign(v) + fv*v ],  ±tau_limit )
    #
    #  and is streamed to every joint motor in MIT mode with
    #  pos=vel=kp=kd=0 so only the feed-forward torque acts (pure
    #  open-loop torque == "weightless / float" behaviour you can push
    #  around by hand).  Coriolis / inertia terms are available too but
    #  are *not* part of the default compensation (they need q-dot-dot
    #  which we do not estimate online).
    #
    #  Prerequisites (all checked at call time with actionable errors):
    #    1) ``pinocchio`` installed (gravity / Coriolis / mass need it).
    #    2) :meth:`setup_dynamics` called once with a URDF that has
    #       <inertial> tags (the vendored follower URDF does).
    #    3) Per-joint motor models configured so Nm-to-raw conversion is
    #       physically correct; otherwise compensation will be inaccurate.
    # ------------------------------------------------------------------
    def setup_dynamics(
        self,
        urdf_path: Optional[str] = None,
        *,
        gravity_vec: Iterable[float] = (0.0, 0.0, -9.81),
        motor_models: Optional[List[str]] = None,
        tau_limit: Optional[Iterable[float]] = None,
        torque_scale: Optional[Iterable[float]] = None,
        friction: Optional[FrictionParams] = None,
        eef_frame: Optional[str] = None,
    ) -> None:
        """Load the rigid-body model used for gravity / dynamics terms.

        Call once before :meth:`get_gravity`,
        :meth:`compute_compensation_torque` or
        :meth:`start_gravity_compensation`.

        Parameters
        ----------
        urdf_path : str, optional
            Path to a URDF with ``<inertial>`` data for every link.  When
            ``None`` the controller searches, in order:

            1. ``<package_dir>/fafu_robot_description/*.urdf``
               (the vendored follower URDF shipped with this package
               for a fully self-contained deployment).

            The URDF joint order is assumed to match
            :attr:`joint_motor_ids` (true for the 6-DoF Fafu
            arm).  Both ``model.nq`` and ``model.nv`` must equal
            :attr:`num_joints`.
        gravity_vec : iterable of 3 float, optional
            Gravity direction/magnitude in the URDF base frame.  Default
            ``(0, 0, -9.81)`` (base ``z`` points up).  Flip / rotate this
            if the arm is wall- or ceiling-mounted.
        motor_models : list of str, optional
            One motor-model key **per joint** (in :attr:`joint_motor_ids`
            order) used by ``set_pos_vel_tqe_kp_kd`` to convert the
            commanded torque from Nm to the raw int16 the firmware wants.
            Valid keys are defined by the native calibration table
            (e.g. ``"M7256_35"``, ``"M60BM_35"``, ``"M4438_32"``,
            ``"M3536_32"`` ...). When ``None``, dynamics calculations
            remain available but non-zero torque/MIT output is rejected until
            exact models are configured.
        tau_limit : iterable of float, optional
            Per-joint torque clip (Nm).  Default
            ``[15, 30, 30, 15, 5, 5]`` (the reference script's values,
            conservative vs the motors' ``[21, 36, 36, 21, 10, 10]`` Nm
            ceiling).  Length must equal :attr:`num_joints`.
        torque_scale : float or iterable of float, optional
            Empirical gain multiplied into the torque right before it is
            sent (``tau_sent = tau * torque_scale``).  Use this to
            calibrate on real hardware when the Nm->raw coefficient is
            uncertain: start at ``1.0``, run with ``dry_run`` to read the
            raw int16 that would be sent, then raise it until the arm just
            floats.  Scalar applies to every joint; a list sets each joint.
            Default ``1.0`` (no extra gain).
        friction : FrictionParams, optional
            Default friction model used when
            :meth:`get_friction_compensation` is called without explicit
            params.  Defaults to :meth:`FrictionParams.reference_6dof`
            for six joints and zero friction for other joint counts.
        eef_frame : str, optional
            Name of the URDF frame treated as the end effector for
            :meth:`forward_kinematics` / :meth:`inverse_kinematics` /
            :meth:`move_p` / :meth:`move_l`.  Defaults to ``"tool_link"``
            (the follower tool frame); if that frame
            is absent the controller falls back to the last joint's child
            frame.

        Raises
        ------
        RuntimeError
            ``pinocchio`` not installed, URDF not found, or the model's
            DoF count does not match :attr:`num_joints`.
        """
        if self.state in _BUSY_STATES:
            raise RobotStateError(
                "dynamics configuration cannot change during an operation"
            )

        resolved = resolve_urdf_path(_HERE, urdf_path)
        if resolved is None:
            requested = repr(urdf_path) if urdf_path else "the vendored package"
            raise RuntimeError(f"could not find a URDF in {requested}")
        dynamics = DynamicsModel.load(
            resolved,
            num_joints=self.num_joints,
            gravity_vector=gravity_vec,
            eef_frame=eef_frame,
        )

        models = None
        if motor_models is not None:
            models = [str(model) for model in motor_models]
            if (
                len(models) != self.num_joints
                or any(not model for model in models)
            ):
                raise ValueError(
                    f"motor_models must contain {self.num_joints} "
                    "non-empty names"
                )

        if tau_limit is not None:
            limit = np.asarray(list(tau_limit), dtype=float)
            if limit.shape != (self.num_joints,):
                raise ValueError(
                    f"tau_limit must have {self.num_joints} elements"
                )
            if not np.all(np.isfinite(limit)):
                raise ValueError("tau_limit must contain only finite values")
            limit = np.abs(limit)
        elif self.num_joints == 6:
            limit = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
        else:
            limit = np.full(self.num_joints, 5.0)

        scale = np.asarray(
            torque_scale if torque_scale is not None else 1.0,
            dtype=float,
        )
        if scale.ndim == 0:
            scale = np.full(self.num_joints, float(scale))
        if scale.shape != (self.num_joints,):
            raise ValueError(
                f"torque_scale must be a scalar or "
                f"{self.num_joints} values"
            )
        if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
            raise ValueError(
                "torque_scale values must be finite and non-negative"
            )
        scale = scale.copy()

        if friction is None:
            if self.num_joints == 6:
                friction = FrictionParams.reference_6dof()
            else:
                friction = FrictionParams(
                    fc=np.zeros(self.num_joints),
                    fv=np.zeros(self.num_joints),
                )
        fc = np.asarray(friction.fc, dtype=float)
        fv = np.asarray(friction.fv, dtype=float)
        threshold = float(friction.vel_threshold)
        if fc.shape != (self.num_joints,) or fv.shape != (self.num_joints,):
            raise ValueError(
                f"friction fc/fv must each have {self.num_joints} elements"
            )
        if (
            not np.all(np.isfinite(fc))
            or not np.all(np.isfinite(fv))
            or not np.isfinite(threshold)
            or threshold < 0.0
        ):
            raise ValueError(
                "friction coefficients and threshold must be finite; "
                "threshold must be non-negative"
            )
        friction_model = FrictionParams(
            fc=fc.copy(),
            fv=fv.copy(),
            vel_threshold=threshold,
        )

        with self._config_lock:
            if self.state in _BUSY_STATES:
                raise RobotStateError(
                    "dynamics configuration cannot change during an operation"
                )
            try:
                self._core.set_joint_motor_models(
                    models if models is not None else [""] * self.num_joints
                )
            except _NATIVE_STATE_ERRORS as exc:
                raise RobotStateError(
                    f"dynamics configuration rejected: {exc}"
                ) from exc

            # Publish one coherent configuration only after native commit.
            self._dyn_motor_models = models
            self._dyn_tau_limit = limit
            self._dyn_torque_scale = scale
            self._friction_params = friction_model
            self._dynamics = dynamics

        if models is None:
            print(
                "[FafuRobot] setup_dynamics: no motor_models given; "
                "non-zero torque/MIT output remains disabled until exact "
                "models are configured."
            )
        if eef_frame and dynamics.eef_frame_name != eef_frame:
            print(
                f"[FafuRobot] setup_dynamics: requested eef_frame "
                f"{eef_frame!r} not found; using "
                f"{dynamics.eef_frame_name!r}."
            )
        print(
            f"[FafuRobot] dynamics ready: URDF={os.path.basename(resolved)}, "
            f"dof={dynamics.model.nq}, "
            f"eef_frame={dynamics.eef_frame_name!r}, "
            f"gravity={dynamics.gravity_vector.tolist()}, "
            f"tau_limit={limit.tolist()}, "
            f"torque_scale={scale.tolist()}"
        )

    def set_motor_models(self, motor_models: Iterable[str]) -> None:
        """Configure exact per-joint models for torque and MIT conversion."""
        models = [str(model) for model in motor_models]
        if len(models) != self.num_joints or any(not model for model in models):
            raise ValueError(
                f"motor_models must contain {self.num_joints} non-empty names")
        with self._config_lock:
            if self.state in _BUSY_STATES:
                raise RobotStateError(
                    "motor models cannot change during a control operation")
            try:
                self._core.set_joint_motor_models(models)
            except _NATIVE_STATE_ERRORS as exc:
                raise RobotStateError(
                    f"motor model configuration rejected: {exc}"
                ) from exc
            self._dyn_motor_models = models

    def set_torque_scale(self, scale: "float | Iterable[float]") -> None:
        """Set the empirical per-joint torque gain (see ``torque_scale`` in
        :meth:`setup_dynamics`).  Accepts a scalar or a per-joint list.

        Can be called between calibration runs without reloading the dynamics
        model. Active control operations reject changes.
        """
        arr = np.asarray(scale, dtype=float).copy()
        if arr.ndim == 0:
            arr = np.full(self.num_joints, float(arr))
        if arr.shape != (self.num_joints,):
            raise ValueError(
                f"torque_scale must be a scalar or {self.num_joints} values")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(
                "torque_scale values must be finite and non-negative")
        with self._config_lock:
            if self.state in _BUSY_STATES:
                raise RobotStateError(
                    "torque scale cannot change during a control operation"
                )
            self._dyn_torque_scale = arr

    def tau_to_raw(self, tau: Iterable[float]) -> np.ndarray:
        """Convert per-joint torque in Nm with the native calibration table."""
        values = [float(x) for x in tau]
        if len(values) != self.num_joints:
            raise ValueError(f"tau must have {self.num_joints} elements")
        with self._config_lock:
            models = list(self._dyn_motor_models or [""] * self.num_joints)
        return np.asarray(
            pm.torques_to_raw(values, models, 1.0), dtype=np.int64)

    @property
    def has_dynamics(self) -> bool:
        """True once setup_dynamics has loaded a model."""
        return getattr(self, "_dynamics", None) is not None

    def _require_dynamics(self) -> DynamicsModel:
        dynamics = getattr(self, "_dynamics", None)
        if dynamics is None:
            raise RuntimeError(
                "dynamics model not loaded; call setup_dynamics() first"
            )
        return dynamics

    def _require_kinematics(self) -> DynamicsModel:
        return self._require_dynamics()

    def _q_in(
        self,
        q: Optional[Iterable[float]],
        is_radians: bool,
    ) -> np.ndarray:
        """Normalize a joint vector to radians."""
        if q is None:
            return self.get_joint_values()
        value = np.asarray(list(q), dtype=float)
        if value.shape != (self.num_joints,):
            raise ValueError(
                f"expected {self.num_joints} joint values, got {value.size}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("joint values must be finite")
        return value if is_radians else np.deg2rad(value)

    def _joint_limits_rad(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return configured joint limits in radians, or None."""
        lower = np.empty(self.num_joints)
        upper = np.empty(self.num_joints)
        for index, motor_id in enumerate(self._joint_motor_ids):
            try:
                limit = self.get_limit(motor_id, is_radians=True)
            except Exception:
                limit = None
            if limit is None:
                return None
            lower[index], upper[index] = limit
        return lower, upper

    def forward_kinematics(
        self,
        q: Optional[Iterable[float]] = None,
        *,
        is_radians: bool = True,
    ) -> Dict[str, object]:
        """Forward kinematics of the end-effector frame.

        Parameters
        ----------
        q : iterable of float, optional
            Joint angles for the manipulator joints (in
            :attr:`joint_motor_ids` order).  Defaults to the live pose.
        is_radians : bool, optional
            Interpret ``q`` in radians (default) or degrees.

        Returns
        -------
        dict with keys
            ``position`` (3,) end-effector position (m),
            ``rotation`` (3, 3) rotation matrix,
            ``rpy`` (3,) roll/pitch/yaw (rad),
            ``transform`` (4, 4) homogeneous transform,
            ``q`` (num_joints,) the joint configuration used (rad).
        """
        dynamics = self._require_kinematics()
        return dynamics.forward_kinematics(self._q_in(q, is_radians))

    def inverse_kinematics(
        self,
        target_position: Iterable[float],
        target_rotation=None,
        *,
        is_euler: bool = False,
        is_radians: bool = True,
        init_q: Optional[Iterable[float]] = None,
        max_iter: int = 1000,
        eps: float = 1e-3,
        damping: float = 1e-2,
        adaptive_damping: bool = True,
        multi_init: bool = True,
        num_attempts: int = 8,
        clamp_limits: bool = True,
    ) -> Optional[np.ndarray]:
        """Damped least-squares inverse kinematics for the end effector.

        Parameters
        ----------
        target_position : iterable of 3 float
            Desired end-effector position ``[x, y, z]`` (m).
        target_rotation : array_like, optional
            Desired orientation: a 3x3 rotation matrix, or an Euler/RPY
            triple when ``is_euler=True``.  ``None`` -> identity.
        is_euler, is_radians : bool, optional
            Treat ``target_rotation`` as RPY (radians unless
            ``is_radians=False``).  ``is_radians`` also sets the unit of
            the returned joint vector.
        init_q : iterable of float, optional
            Seed configuration for the single-init solve.  Defaults to
            the live pose.  Ignored when ``multi_init=True``.
        max_iter, eps, damping, adaptive_damping : numeric, optional
            Solver tuning (damped least-squares IK).
        multi_init : bool, optional
            Try several seeds (current pose, zero, limit mid-point, random
            within limits) and keep the best -- far more robust.  Default
            ``True``.
        num_attempts : int, optional
            Number of seeds for ``multi_init``.
        clamp_limits : bool, optional
            Abort an iterate that leaves the joint soft limits (matches the
            vendor behaviour) when limits are configured.

        Returns
        -------
        np.ndarray or None
            ``num_joints`` joint angles (rad by default, deg if
            ``is_radians=False``) on success, else ``None``.
        """
        dynamics = self._require_kinematics()
        initial = None if init_q is None else self._q_in(init_q, is_radians)
        if multi_init:
            try:
                current_q = self.get_joint_values()
            except Exception:
                current_q = np.zeros(self.num_joints)
        elif initial is not None:
            current_q = initial
        else:
            current_q = self.get_joint_values()
        result = dynamics.inverse_kinematics(
            target_position,
            target_rotation,
            current_q=current_q,
            init_q=initial,
            is_euler=is_euler,
            rotation_is_radians=is_radians,
            max_iter=max_iter,
            eps=eps,
            damping=damping,
            adaptive_damping=adaptive_damping,
            multi_init=multi_init,
            num_attempts=num_attempts,
            limits=self._joint_limits_rad() if clamp_limits else None,
        )
        if result is None:
            return None
        return result if is_radians else np.rad2deg(result)

    def get_gravity(
        self,
        q: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Generalized gravity torque ``G(q)`` (Nm), one entry per joint.

        Parameters
        ----------
        q : iterable of float, optional
            Joint angles (rad).  Defaults to the live measured pose.
        """
        dynamics = self._require_dynamics()
        value = self.get_joint_values() if q is None else list(q)
        return dynamics.gravity(value)

    def get_mass_matrix(
        self,
        q: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Return the joint-space inertia matrix."""
        dynamics = self._require_dynamics()
        value = self.get_joint_values() if q is None else list(q)
        return dynamics.mass_matrix(value)

    def get_coriolis(
        self,
        q: Optional[Iterable[float]] = None,
        v: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Return the Coriolis/centrifugal matrix."""
        dynamics = self._require_dynamics()
        q_value = self.get_joint_values() if q is None else list(q)
        v_value = self.get_joint_velocities() if v is None else list(v)
        return dynamics.coriolis(q_value, v_value)

    def get_dynamics(
        self,
        q: Optional[Iterable[float]] = None,
        v: Optional[Iterable[float]] = None,
        a: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Return full inverse dynamics torque in Nm."""
        dynamics = self._require_dynamics()
        q_value = self.get_joint_values() if q is None else list(q)
        v_value = self.get_joint_velocities() if v is None else list(v)
        a_value = np.zeros(self.num_joints) if a is None else list(a)
        return dynamics.inverse_dynamics(q_value, v_value, a_value)

    def get_friction_compensation(
        self,
        vel: Optional[Iterable[float]] = None,
        *,
        params: Optional[FrictionParams] = None,
    ) -> np.ndarray:
        """Coulomb + viscous friction torque (Nm), per joint.

        ``tau = fc*sign(v) + fv*v`` with the low-speed dead-band described
        in :class:`FrictionParams`.  Pure numpy — does **not** need
        pinocchio, so it works even on a Windows box without dynamics.

        Parameters
        ----------
        vel : iterable of float, optional
            Joint velocities (rad/s).  Defaults to the live measured
            velocities.
        params : FrictionParams, optional
            Override the model.  Defaults to the one given to
            :meth:`setup_dynamics`, else :meth:`FrictionParams.reference_6dof`.
        """
        velocity = self.get_joint_velocities() if vel is None else vel
        model = (
            params
            or self._friction_params
            or FrictionParams.reference_6dof()
        )
        return friction_compensation(velocity, model)

    def compute_compensation_torque(
        self,
        q: Optional[Iterable[float]] = None,
        v: Optional[Iterable[float]] = None,
        *,
        friction: bool = True,
    ) -> np.ndarray:
        """Return clipped gravity and optional friction feed-forward torque."""
        torque = self.get_gravity(q)
        if friction:
            torque = torque + self.get_friction_compensation(v)
        if self._dyn_tau_limit is not None:
            torque = np.clip(
                torque,
                -self._dyn_tau_limit,
                self._dyn_tau_limit,
            )
        return torque

    @_guard_operation(
        "apply_compensation_torque", RobotState.GRAVITY_COMP)
    def apply_compensation_torque(
        self,
        tau: Iterable[float],
        *,
        damping_kd: float = 0.0,
    ) -> None:
        """Stream a feed-forward torque vector (Nm) to every joint motor.

        Channel (see :attr:`_use_group_mit`):

        * **Group MIT** (default when ``num_joints <= 6``): one one-to-many
          ``set_many_mit`` frame (CAN ID ``0x8093``) with ``kp=kd=0``, i.e.
          pure torque feed-forward.  This is the vendor-equivalent of
          ``pos_vel_tqe_kp_kd(q, 0, tau, 0, 0)`` and is **validated on this
          firmware** (the one-to-many MIT frame is actuated; the single-motor
          ``0x15`` frame is not).
        * **Legacy per-joint** (``num_joints > 6`` or ``_use_group_mit=False``):
          one ``set_torque`` frame (mode ``0x0A``) per joint.

        Both send the identical feed-forward torque; only the CAN framing
        differs.  ``damping_kd`` is kept for API compatibility but has no
        firmware effect on either channel; add velocity damping via the
        software impedance net (``b_soft`` in
        :meth:`start_gravity_compensation`) instead.
        """
        self._require_stream_command(
            "apply_compensation_torque", allow_gravity_owner=True)
        tau = np.asarray(list(tau), dtype=float)
        if tau.shape != (self.num_joints,) or not np.all(np.isfinite(tau)):
            raise ValueError(
                f"tau must have {self.num_joints} finite elements, got {tau}")
        # Empirical calibration gain (1.0 by default).
        tau = tau * self._dyn_torque_scale

        with self._command_guard():
            if (
                self._use_group_mit
                and self.num_joints <= 6
                and max(self._joint_motor_ids) <= 6
            ):
                # One 0x8093 frame, kp=kd=0 => pure torque feed-forward.
                raw = self.tau_to_raw(tau)
                zeros = [0.0] * self.num_joints
                self._ht.set_many_mit_rad(
                    list(self._joint_motor_ids),
                    zeros, zeros,
                    [int(x) for x in raw],
                    [0] * self.num_joints,
                    [0] * self.num_joints,
                    max(self._joint_motor_ids),
                    0.0,
                )
                return

            for i, mid in enumerate(self._joint_motor_ids):
                model = (self._dyn_motor_models[i]
                         if self._dyn_motor_models is not None else "")
                self._ht.set_torque(mid, float(tau[i]), model)

    @_guard_operation("move_MIT", RobotState.MOVING)
    def move_MIT(
        self,
        pos: Iterable[float],
        vel: Iterable[float],
        tau: Iterable[float],
        kp: "float | Iterable[float]" = 0.0,
        kd: "float | Iterable[float]" = 0.0,
        *,
        is_radians: bool = True,
        apply_torque_scale: bool = True,
        kp_kd_raw: bool = False,
        timeout: float = 0.0,
    ) -> Dict[int, "pm.MotorState"]:
        """Stream one **group MIT** frame (CAN ID ``0x8093``) to J1..J6.

        A single one-to-many MIT/PD broadcast carrying, per joint::

            tau_out = kp*(pos - q) + kd*(vel - qd) + tau_ff

        Unlike the single-motor MIT channel (mode ``0x15``, silently ignored by
        this arm's firmware), the *one-to-many* MIT frame (``0x8093``) **is**
        actuated -- validated on hardware via ``diag_torque_ramp.py
        --path mit-many`` (J1 spins up with rising raw torque).  So this is the
        preferred channel for gravity comp / drag-teaching / replay.

        Parameters
        ----------
        pos : iterable of float
            Per-joint target position in radians. Feeds the MIT
            position term; only matters when ``kp != 0``. Soft limits applied.
        vel : iterable of float
            Per-joint target velocity in rad/s. Only matters when ``kd != 0``.
        tau : iterable of float
            Per-joint feed-forward torque in **Nm** (e.g. gravity + friction).
            Converted to raw int16 with the per-joint motor coeff, exactly like
            :meth:`apply_compensation_torque`.
        kp, kd : float or iterable of float
            Per-joint PD gains in **physical units**,
            e.g. ``kp=[30,40,55,15,7,5]``,
            ``kd=[3,4,5.5,1.5,0.7,0.5]``.  Converted
            to the firmware ``rkp``/``rkd`` int16 with the vendor formula
            (``kp_float2int``, radian convention)::

                raw = int16( (kp / coeff) * 10 * 2*pi )

            where ``coeff`` is the joint's motor torque coefficient.  Scalar
            broadcasts to every joint.  ``0`` (default) => pure torque
            feed-forward (gravity comp on the group-MIT channel).  Pass
            ``kp_kd_raw=True`` to skip the conversion and send raw int16 (used
            by low-level diagnostics / :meth:`apply_compensation_torque`).
        is_radians : bool
            Compatibility guard; only ``True`` is accepted.
        apply_torque_scale : bool
            Multiply ``tau`` by the per-joint ``torque_scale`` calibration gain
            (default ``True``, matches :meth:`apply_compensation_torque`).
        kp_kd_raw : bool
            When ``True`` treat ``kp``/``kd`` as already-raw int16 (skip the
            vendor physical->raw conversion).  Default ``False``.
        timeout : float
            Reply-wait seconds. ``0`` (default) = fire-and-forget (fastest, for
            high-rate loops); >0 blocks for state readback and returns it.

        Returns
        -------
        dict[int, MotorState]
            Motor states keyed by id when ``timeout > 0``; empty dict otherwise.

        Notes
        -----
        One MIT frame holds at most **6 motors** (10 bytes each, CAN-FD 64 B
        cap). This method sends the manipulator joints only; drive the gripper
        separately (``gripper_control`` / group with <=6 total).
        """
        self._require_radians(is_radians, "move_MIT")
        self._require_stream_command("move_MIT")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout must be finite and non-negative")
        n = self.num_joints
        if n > 6 or max(self._joint_motor_ids) > 6:
            raise RuntimeError(
                "move_MIT requires at most six joint motors with IDs in 1..6; "
                "configure joint_motor_ids accordingly"
            )
        pos = np.asarray(list(pos), dtype=float)
        vel = np.asarray(list(vel), dtype=float)
        tau = np.asarray(list(tau), dtype=float)
        for name, arr in (("pos", pos), ("vel", vel), ("tau", tau)):
            if arr.shape != (n,) or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} must have {n} finite elements, got {arr}")

        def _as_vec(g) -> np.ndarray:
            if np.isscalar(g):
                g = np.full(n, float(g))
            else:
                g = np.asarray(list(g), dtype=float)
            if g.shape != (n,) or not np.all(np.isfinite(g)):
                raise ValueError(
                    f"kp/kd must be scalar or {n} finite elements")
            return g

        kp_v = _as_vec(kp)
        kd_v = _as_vec(kd)

        models = self._dyn_motor_models or [""] * n
        if kp_kd_raw:
            kp_out = [int(np.clip(round(x), -32768, 32767)) for x in kp_v]
            kd_out = [int(np.clip(round(x), -32768, 32767)) for x in kd_v]
        else:
            kp_out = [
                int(pm.gain_to_raw(float(kp_v[i]), models[i]))
                for i in range(n)
            ]
            kd_out = [
                int(pm.gain_to_raw(float(kd_v[i]), models[i]))
                for i in range(n)
            ]

        if apply_torque_scale:
            tau = tau * self._dyn_torque_scale
        tau_raw = self.tau_to_raw(tau)   # per-joint coeff * 0.01 -> int16

        motor_ids = list(self._joint_motor_ids)
        with self._command_guard():
            return self._ht.set_many_mit_rad(
                motor_ids,
                [float(x) for x in pos],
                [float(x) for x in vel],
                [int(x) for x in tau_raw],
                kp_out,
                kd_out,
                max(motor_ids),
                timeout,
            )

    def gravity_compensation_step(
        self,
        *,
        friction: bool = True,
        damping_kd: float = 0.0,
        dry_run: bool = False,
    ) -> np.ndarray:
        """One tick of gravity(+friction) compensation; returns tau (Nm).

        Reads the live pose/velocity, computes the clipped feed-forward
        torque and (unless ``dry_run``) sends it.  Call this yourself in a
        custom loop, or use :meth:`start_gravity_compensation` for a
        ready-made blocking loop.
        """
        self._require_dynamics()
        q = self.get_joint_values()
        v = self.get_joint_velocities()
        tau = self.compute_compensation_torque(q, v, friction=friction)
        if not dry_run:
            self.apply_compensation_torque(tau, damping_kd=damping_kd)
        return tau

    # ---- tuned gravity-comp defaults (6-DOF follower arm) --------------
    # Empirically calibrated lead-through / float-mode impedance net:
    #   * heavy gravity joints J2/J3/J4 get a stiff spring K + integral Ki
    #     (the integral removes the static-friction droop without the
    #     high-K divergence the latency-limited loop suffers);
    #   * the light base/wrist joints J1/J5/J6 get a soft K, light damping
    #     and NO integral (so they don't hunt/drift in their friction
    #     deadband).
    # Only auto-applied when the arm actually has 6 joints; otherwise the
    # caller must pass their own arrays.  Requires torque_scale ~= 90 from
    # setup_dynamics() to feel right.
    _GRAVITY_COMP_DOF = 6
    _GRAVITY_COMP_TORQUE_SCALE = 90.0
    _GRAVITY_COMP_K_DEFAULT = (1.5, 8.0, 10.0, 10.0, 1.5, 1.5)
    _GRAVITY_COMP_B_DEFAULT = (0.4, 0.4, 0.6, 0.8, 0.2, 0.2)
    # Ki raised ~1.7x on J2/J3/J4 vs the first tune: with a small held error
    # (e.g. 0.7 deg of stiction deadband) the integral ramp rate is Ki*err, so a
    # low Ki took >10 s to build enough torque to break static friction and
    # close the gap ("have to hold it for ages before it locks").  The higher Ki
    # ramps to the SAME i_clamp ceiling far quicker (settles in ~2-3 s) without
    # raising the max authority.  If a joint starts to hunt/overshoot at these
    # values, dial it back per-joint via --i-soft.
    _GRAVITY_COMP_I_DEFAULT = (0.0, 2.5, 3.5, 6.0, 0.0, 0.0)

    def start_gravity_compensation(
        self,
        *,
        friction: bool = True,
        rate_hz: float = 200.0,
        duration: Optional[float] = None,
        damping_kd: float = 0.0,
        dry_run: bool = False,
        verbose: bool = False,
        abort_check: Optional[Callable[[], bool]] = None,
        k_soft: Optional[Iterable[float]] = None,
        b_soft: Optional[Iterable[float]] = None,
        q_des: Optional[Iterable[float]] = None,
        tau_lpf_alpha: float = 0.4,
        tau_slew_per_s: Optional[float] = 40.0,
        i_soft: Optional[Iterable[float]] = None,
        i_clamp: float = 3.0,
        vel_abort_rad_s: float = 4.0,
        vel_lpf_alpha: float = 0.3,
        hold_on_release: bool = True,
        move_vel_thresh_rad_s: float = 0.15,
        home_on_exit: bool = False,
        home_speed: int = 15,
        home_brake_pause: float = 0.0,
    ) -> None:
        """Run a blocking gravity(+friction) compensation loop ("float mode").

        The arm becomes weightless and can be guided by hand. The loop adds
        rate control, a duration cap, dry-run preview and safe cancellation.

        Safety:
          * Validate torque output with dry_run=True first.
          * On exit the joint release policy is applied; support heavy links.
          * The gripper is never touched.

        Parameters
        ----------
        friction : bool, optional
            Add the friction feed-forward term.  Default ``True``.
        rate_hz : float, optional
            Loop rate.  Default ``200`` Hz (matches the reference's ~5 ms
            sleep).  Each tick sends ``num_joints`` CAN frames.
        duration : float, optional
            Stop after this many seconds.  ``None`` (default) runs until
            Ctrl+C or ``abort_check``.
        damping_kd : float, optional
            Extra firmware velocity damping (>= 0).  ``0`` = reference.
        dry_run : bool, optional
            Compute and (if ``verbose``) print torque but do **not** send
            anything.  Default ``False``.
        verbose : bool, optional
            Print q / v / tau every ~0.5 s.  Default ``False``.
        abort_check : callable, optional
            Called every tick; return ``True`` to stop early.
        k_soft, b_soft : iterable of float, optional
            Software **impedance / PD safety net** (Nm/rad, Nm·s/rad), one
            per joint.  ``tau += K*(q_des - q) + B*(-v_filt)`` is added to the
            gravity feed-forward (firmware kp=kd=0).  Keeps the arm pulled
            toward ``q_des`` so it **cannot run away**.  On a 6-DOF arm
            ``None`` (default) auto-applies the tuned lead-through net
            (``_GRAVITY_COMP_K_DEFAULT`` / ``_GRAVITY_COMP_B_DEFAULT``);
            pass an explicit all-zeros array to opt out (pure float mode).
        q_des : iterable of float, optional
            Hold target (rad) for the PD net.  ``None`` (default) captures the
            pose at loop start, i.e. "hold where it is now".
        tau_lpf_alpha : float, optional
            First-order low-pass on the commanded torque (0..1, 1 = no
            filter).  Lower = smoother.  Default ``0.4``.
        tau_slew_per_s : float, optional
            Max torque change rate (Nm/s).  Caps single-tick jumps to damp
            limit-cycle oscillation.  ``None`` / <=0 disables.  Default ``40``.
        i_soft : iterable of float, optional
            Software **integral** gain (Nm per rad·s), one per joint.  Use
            this — NOT a bigger ``k_soft`` — to remove the static-friction
            "droop" (steady-state offset): a large K diverges on this
            high-latency loop, but the integral only acts at low frequency so
            it eliminates the deadband without oscillating.  Only integrates
            while the joint is (nearly) at rest, so the arm stays compliant
            when you push it.  On a 6-DOF arm ``None`` (default) auto-applies
            ``_GRAVITY_COMP_I_DEFAULT`` (integral only on the gravity joints
            J2/J3/J4); pass all-zeros to disable.
        i_clamp : float, optional
            Per-joint cap on the integral torque contribution (Nm,
            anti-windup).  Default ``3.0``.  This bounds how much static droop
            the integral can fight; too small and a joint with a large
            model/friction deficit keeps slowly sagging because the integral
            saturates before it generates enough holding torque.
        vel_abort_rad_s : float, optional
            **Runaway guard** (safety): if any joint's |velocity| exceeds this
            (rad/s) the loop aborts and re-holds — catches divergence before
            the arm flings itself.  ``0`` disables.  Default ``4.0``.
        vel_lpf_alpha : float, optional
            Low-pass on the velocity used by the ``b_soft`` damping term
            (0..1, lower = smoother).  The raw firmware velocity is noisy; a
            light filter lets ``b_soft`` actually damp the pure-P limit-cycle
            (e.g. J1 hunting) instead of chattering.  Default ``0.3``.
        hold_on_release : bool, optional
            **Lead-through teach mode.**  When ``True`` (default) the hold
            target ``q_des`` continuously follows the live pose for any joint
            whose speed exceeds ``move_vel_thresh_rad_s`` (so while you drag it the
            spring force is ~0 and the arm is weightless), and freezes the
            instant the joint stops — the spring + integral then lock it
            exactly where you let go.  Drag again to a new pose and it holds
            there.  Set ``False`` for the classic fixed-``q_des`` behaviour
            (springs back to the start pose when pushed away).
        move_vel_thresh_rad_s : float, optional
            Speed (rad/s) above which a joint is considered "being dragged"
            for ``hold_on_release``.  Default ``0.15``.  Raise it if the arm
            slowly creeps/drifts when untouched; lower it if dragging feels
            stiff before it starts following.
        home_on_exit : bool, optional
            When ``True``, on loop exit (normal, Ctrl+C, or ``abort_check``)
            the teardown sequence is: **brake every joint** (arrest motion
            gently) → **pause ``home_brake_pause`` s** → re-enable position
            mode and **slowly drive every joint back to 0 rad**
            (via :meth:`go_home`) → brake again at home.  Default ``False``
            (just brake in place).  Ignored in ``dry_run``.
        home_speed : int, optional
            Speed percentage (0..100] for the ``home_on_exit`` return move.
            Default ``15`` (slow / gentle).
        home_brake_pause : float, optional
            Seconds to stay braked after Ctrl+C before starting the homing
            move (``home_on_exit`` only).  Default ``0.0`` (home immediately
            after braking).
        """
        self._require_dynamics()

        # Validate every pure option before a live run can enable a motor.
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")

        def _finite(name: str, value: float) -> float:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            return value

        rate_hz = _finite("rate_hz", rate_hz)
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if duration is not None:
            duration = _finite("duration", duration)
            if duration < 0.0:
                raise ValueError("duration cannot be negative")
        damping_kd = _finite("damping_kd", damping_kd)
        if damping_kd < 0.0:
            raise ValueError("damping_kd cannot be negative")
        tau_lpf_alpha = _finite("tau_lpf_alpha", tau_lpf_alpha)
        vel_lpf_alpha = _finite("vel_lpf_alpha", vel_lpf_alpha)
        if not 0.0 <= tau_lpf_alpha <= 1.0:
            raise ValueError("tau_lpf_alpha must be in [0, 1]")
        if not 0.0 <= vel_lpf_alpha <= 1.0:
            raise ValueError("vel_lpf_alpha must be in [0, 1]")
        if tau_slew_per_s is not None:
            tau_slew_per_s = _finite("tau_slew_per_s", tau_slew_per_s)
        i_clamp = _finite("i_clamp", i_clamp)
        vel_abort_rad_s = _finite("vel_abort_rad_s", vel_abort_rad_s)
        move_vel_thresh_rad_s = _finite(
            "move_vel_thresh_rad_s", move_vel_thresh_rad_s
        )
        home_brake_pause = _finite("home_brake_pause", home_brake_pause)
        if i_clamp < 0.0:
            raise ValueError("i_clamp cannot be negative")
        if vel_abort_rad_s < 0.0:
            raise ValueError("vel_abort_rad_s cannot be negative")
        if move_vel_thresh_rad_s < 0.0:
            raise ValueError("move_vel_thresh_rad_s cannot be negative")
        if home_brake_pause < 0.0:
            raise ValueError("home_brake_pause cannot be negative")
        home_speed = int(home_speed)
        if not 1 <= home_speed <= 100:
            raise ValueError("home_speed must be in [1, 100]")

        # Auto-apply the tuned 6-DOF lead-through net when the caller did not
        # override it.  Pass an explicit array (e.g. all-zeros) to opt out.
        if self.num_joints == self._GRAVITY_COMP_DOF:
            if k_soft is None:
                k_soft = self._GRAVITY_COMP_K_DEFAULT
            if b_soft is None:
                b_soft = self._GRAVITY_COMP_B_DEFAULT
            if i_soft is None:
                i_soft = self._GRAVITY_COMP_I_DEFAULT
            # The tuned net is calibrated for torque_scale ~= 90; warn if the
            # arm is still on the uncalibrated default (would feel "no force").
            if float(np.min(self._dyn_torque_scale)) < 10.0:
                print("[FafuRobot] WARN: torque_scale looks uncalibrated "
                      f"({self._dyn_torque_scale.tolist()}); the tuned "
                      f"gravity-comp net expects ~{self._GRAVITY_COMP_TORQUE_SCALE}. "
                      "Call setup_dynamics(..., torque_scale=90) or "
                      "set_torque_scale(90) first, or the arm will feel weak.")

        def _broadcast(val, name):
            if val is None:
                return None
            if np.isscalar(val):
                arr = np.full(self.num_joints, float(val), dtype=float)
            else:
                arr = np.asarray(list(val), dtype=float)
                if arr.shape == (1,):
                    arr = np.full(self.num_joints, arr[0], dtype=float)
            if arr.shape != (self.num_joints,):
                raise ValueError(
                    f"{name} must be a scalar or have {self.num_joints} "
                    f"elements, got {arr.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values")
            if np.any(arr < 0.0):
                raise ValueError(f"{name} cannot contain negative gains")
            return arr

        k_soft_np = _broadcast(k_soft, "k_soft")
        b_soft_np = _broadcast(b_soft, "b_soft")
        i_soft_np = _broadcast(i_soft, "i_soft")

        q_des_value: Optional[np.ndarray] = None
        if q_des is not None:
            q_des_value = np.asarray(list(q_des), dtype=float)
            if (
                q_des_value.shape != (self.num_joints,)
                or not np.all(np.isfinite(q_des_value))
            ):
                raise ValueError(
                    f"q_des must have {self.num_joints} finite elements"
                )

        q_des_np: Optional[np.ndarray] = None
        if k_soft_np is not None or i_soft_np is not None:
            q_des_np = (
                q_des_value
                if q_des_value is not None
                else self.get_joint_values().copy()
            )
            if not np.all(np.isfinite(q_des_np)):
                raise RuntimeError("joint feedback for q_des is not finite")
            print(
                "[FafuRobot] impedance net ON: "
                f"K={(k_soft_np.tolist() if k_soft_np is not None else None)} "
                f"B={(b_soft_np.tolist() if b_soft_np is not None else None)} "
                f"Ki={(i_soft_np.tolist() if i_soft_np is not None else None)} "
                f"q_des(deg)={np.degrees(q_des_np).round(1).tolist()}"
            )

        # Only after all validation may a live run energize the motors.
        if not dry_run:
            self._require_ready(
                "start_gravity_compensation", allow_disabled=True
            )
            if (
                self.state is not RobotState.IDLE
                or self._motors_in_position_mode(fresh=True) is not True
            ):
                print(
                    "[FafuRobot] start_gravity_compensation: "
                    "enabling motors ..."
                )
                self.enable()

        if dry_run:
            print("[FafuRobot] gravity-comp DRY-RUN: computing torque, "
                  "NOT sending to motors.")
        else:
            print("[FafuRobot] gravity-comp LIVE: arm will go weightless. "
                  "Keep a hand on it / the E-stop. Ctrl+C to stop.")

        period = 1.0 / rate_hz
        # Anti-convulsion: limit how fast the commanded torque may change
        # between ticks (slew) and low-pass it.  With high torque_scale the
        # K/B impedance + friction sign terms can otherwise flip the command
        # by hundreds of raw counts in one tick, which excites a limit-cycle
        # oscillation ("convulsion") through the USB-CAN latency.
        max_dtau = (
            tau_slew_per_s * period
            if tau_slew_per_s is not None and tau_slew_per_s > 0.0
            else None
        )
        alpha = tau_lpf_alpha
        tau_prev: Optional[np.ndarray] = None
        # Integral state (kills the static-friction "droop" deadband without
        # the high-frequency gain that makes a large K diverge on this
        # high-latency loop).  Only integrates while (nearly) at rest so it
        # stays compliant while you push, and never winds up.
        integ = np.zeros(self.num_joints, dtype=float)
        rest_thresh_rad_s = 0.05
        # Per-joint lead-through state machine: a joint enters "dragging" only
        # after its speed stays above `move_vel_thresh_rad_s` continuously for
        # `enter_time` seconds, and LOCKS again once its speed has stayed below
        # that threshold continuously for `settle_time` seconds.
        #
        # The enter DEBOUNCE is the key fix for "gradually droops, never
        # stabilises": under gravity a joint creeps down in stick-slip jerks,
        # and a single slip spike easily exceeds `move_vel_thresh_rad_s` for one
        # tick.  With an instantaneous enter-test that spike would flip the
        # joint into "dragging", snap q_des down to the (sagged) live pose and
        # zero the integral -- so every micro-slip ratchets the hold point
        # lower and the joint walks down forever.  Requiring the fast motion to
        # PERSIST for `enter_time` rejects those momentary slips (they stop
        # almost immediately) while a real hand-drag (sustained) still engages.
        dragging = np.zeros(self.num_joints, dtype=bool)
        slow_time = np.zeros(self.num_joints, dtype=float)
        fast_time = np.zeros(self.num_joints, dtype=float)
        enter_time = 0.08
        settle_time = 0.25
        # Filtered velocity for the damping (B) term: the raw firmware
        # velocity is quantized/noisy, and feeding it straight into B*(-v)
        # either does nothing or chatters.  A light LPF gives B clean phase to
        # actually damp the P limit-cycle.
        v_alpha = vel_lpf_alpha
        v_filt = np.zeros(self.num_joints, dtype=float)
        cancelled = self._combined_abort_check(abort_check)
        t0 = time.monotonic()
        last_t = t0
        last_log = t0
        gravity_token = None
        if not dry_run:
            try:
                with self._config_lock:
                    gravity_token = int(
                        self._core.begin_operation(
                            pm.OperationKind.GRAVITY_COMP
                        )
                    )
            except _NATIVE_STATE_ERRORS as exc:
                raise RobotStateError(
                    f"start_gravity_compensation rejected: {exc}"
                ) from exc
        try:
            while True:
                tick_start = time.monotonic()
                if cancelled():
                    print("[FafuRobot] gravity-comp: cancellation -> stop")
                    break
                if duration is not None and (tick_start - t0) >= duration:
                    break

                q = self.get_joint_values()
                v = self.get_joint_velocities()
                # ---- runaway / divergence guard (safety) ----
                if vel_abort_rad_s > 0.0:
                    vmax = float(np.max(np.abs(v)))
                    if vmax > vel_abort_rad_s:
                        print(
                            "[FafuRobot] gravity-comp: RUNAWAY guard "
                            f"(|v|={vmax:.2f} > {vel_abort_rad_s} rad/s) "
                            "-> abort & re-hold"
                        )
                        break
                dt = tick_start - last_t
                last_t = tick_start
                v_filt = v_alpha * v + (1.0 - v_alpha) * v_filt
                absv = np.abs(v)
                # ---- lead-through teach (debounced): a joint enters "dragging"
                # only after SUSTAINED fast motion (`enter_time`), so momentary
                # gravity stick-slip spikes can't ratchet the hold point down;
                # while dragging its q_des follows the live pose (spring -> 0,
                # weightless to move); once it stays slow for `settle_time` it
                # locks q_des where you let go and the spring + integral hold it
                # there.  A slow gravity sag never sustains above the threshold,
                # so it is treated as "held" and the integral pulls it back out.
                if hold_on_release and q_des_np is not None:
                    fast = absv > move_vel_thresh_rad_s
                    fast_time = np.where(fast, fast_time + dt, 0.0)
                    slow_time = np.where(fast, 0.0, slow_time + dt)
                    # sustained fast -> enter drag; sustained slow -> lock
                    dragging = np.where(fast_time >= enter_time, True, dragging)
                    dragging = np.where(slow_time >= settle_time, False, dragging)
                    q_des_np = np.where(dragging, q, q_des_np)
                    # ZERO the integral while dragging.  The integral is a
                    # *position-error* term, NOT a transferable gravity-model
                    # bias: carrying it from one pose to a very different one
                    # (e.g. dragging from q3=140 deg back to q3~0) discharges a
                    # huge wound-up value as a violent forward "spring kick".
                    # Resetting on drag keeps it pose-local and safe; the
                    # slightly slower re-lock is handled by a higher Ki, not by
                    # carrying stale integral across the workspace.
                    integ = np.where(dragging, 0.0, integ)
                    hold_mask = ~dragging
                else:
                    hold_mask = absv < rest_thresh_rad_s
                tau = self.compute_compensation_torque(q, v, friction=friction)
                if k_soft_np is not None:
                    tau = tau + k_soft_np * (q_des_np - q)
                    if b_soft_np is not None:
                        tau = tau + b_soft_np * (-v_filt)
                if i_soft_np is not None and dt > 0.0:
                    err = q_des_np - q
                    # integrate while "held" (locked, incl. slow gravity sag)
                    # so the droop is actively removed; bleed off otherwise so
                    # it never winds up.
                    integ = np.where(hold_mask, integ + err * dt, integ * 0.95)
                    # anti-windup: clamp so |Ki*integ| <= i_clamp (per joint)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cap = np.where(i_soft_np > 0.0,
                                       i_clamp / np.maximum(i_soft_np, 1e-9),
                                       0.0)
                    integ = np.clip(integ, -cap, cap)
                    tau = tau + i_soft_np * integ
                if self._dyn_tau_limit is not None:
                    tau = np.clip(tau, -self._dyn_tau_limit, self._dyn_tau_limit)
                # smooth + slew-limit before sending
                if tau_prev is not None:
                    if alpha < 1.0:
                        tau = alpha * tau + (1.0 - alpha) * tau_prev
                    if max_dtau is not None:
                        tau = np.clip(tau, tau_prev - max_dtau, tau_prev + max_dtau)
                tau_prev = tau.copy()
                if not dry_run:
                    self.apply_compensation_torque(tau, damping_kd=damping_kd)

                if verbose and (tick_start - last_log) >= 0.5:
                    last_log = tick_start
                    raw = self.tau_to_raw(tau * self._dyn_torque_scale)
                    print(f"[grav] q(deg)={np.degrees(q).round(1).tolist()} "
                          f"tau(Nm)={tau.round(3).tolist()} "
                          f"x{self._dyn_torque_scale.tolist()} "
                          f"-> raw={raw.tolist()}")

                # rate control
                sleep_s = period - (time.monotonic() - tick_start)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\n[FafuRobot] gravity-comp interrupted by user")
        finally:
            skip_cleanup = False
            try:
                if not dry_run:
                    health = self._core.health()
                    current = self._python_state_from_core(health.state)
                    skip_cleanup = bool(
                        health.closing
                        or health.cancel_requested
                        or current in (
                            RobotState.ESTOP,
                            RobotState.DEAD,
                            RobotState.DISCONNECTED,
                        )
                    )
                    if not skip_cleanup:
                        # Keep the GRAVITY_COMP lease until the brake write and
                        # state transition are complete. This leaves no IDLE
                        # window in which another writer can steal ownership.
                        self._brake_joints()
            finally:
                if gravity_token is not None:
                    self._core.end_operation(gravity_token)

            health = self._core.health()
            current = self._python_state_from_core(health.state)
            skip_cleanup = bool(
                skip_cleanup
                or health.closing
                or health.cancel_requested
                or current in (
                    RobotState.ESTOP,
                    RobotState.DEAD,
                    RobotState.DISCONNECTED,
                )
            )
            if dry_run:
                pass
            elif skip_cleanup:
                print(
                    f"[FafuRobot] gravity-comp stopped ({current.name}); "
                    "native safety/shutdown owns hardware cleanup."
                )
            elif home_on_exit:
                pause = max(0.0, float(home_brake_pause))
                if pause > 0.0:
                    print(
                        f"[FafuRobot] gravity-comp: braked; pausing "
                        f"{pause:.1f}s before homing ..."
                    )
                    time.sleep(pause)
                else:
                    print("[FafuRobot] gravity-comp: braked.")
                print(
                    "[FafuRobot] gravity-comp: switching to position & "
                    f"returning home (0 rad) @ speed={home_speed} ... "
                    "(do NOT press Ctrl+C again)"
                )
                try:
                    self.enable()
                    self.go_home(speed=home_speed, block=True)
                    print("[FafuRobot] gravity-comp homed to 0 rad.")
                except KeyboardInterrupt:
                    print("\n[FafuRobot] homing interrupted.")
                except Exception as exc:  # noqa: BLE001
                    print(f"[FafuRobot] homing failed ({exc}).")
                self._brake_joints()
                print("[FafuRobot] gravity-comp stopped (joints braked).")
            else:
                print("[FafuRobot] gravity-comp stopped (joints braked).")

    @property
    def is_gravity_compensating(self) -> bool:
        """Whether native gravity-compensation owns the writer lease."""
        return (
            self._core.active_operation
            == pm.OperationKind.GRAVITY_COMP
        )

    def _brake_joints(self) -> None:
        """Brake manipulator joints without changing the gripper."""
        core = self._core
        borrowed_gravity_writer = (
            core.active_operation == pm.OperationKind.GRAVITY_COMP
            and core.operation_owned_by_current_thread()
        )
        token = None
        try:
            if not borrowed_gravity_writer:
                token = core.begin_operation(pm.OperationKind.LIFECYCLE)
            with self._command_guard():
                for motor_id in self._joint_motor_ids:
                    try:
                        self._ht.set_motor_mode(motor_id, self.MODE_BRAKE)
                    except Exception:
                        try:
                            self._ht.stop(motor_id)
                        except Exception:
                            pass
            core.transition(pm.RobotState.BRAKED)
        except Exception:
            health = core.health()
            current = self._python_state_from_core(health.state)
            if (
                not health.closing
                and current not in (
                    RobotState.ESTOP,
                    RobotState.DEAD,
                    RobotState.DISCONNECTED,
                )
            ):
                raise
        finally:
            if token is not None:
                core.end_operation(token)
            self._sync_state_from_core()

    # ------------------------------------------------------------------
    #  Cartesian motion (placeholders)
    # ------------------------------------------------------------------
    @_guard_operation("move_p", RobotState.MOVING)
    def move_p(
        self,
        pos: Iterable[float],
        rot=None,
        *,
        is_euler: bool = False,
        is_radians: bool = True,
        speed: int = 50,
        block: bool = True,
        init_q: Optional[Iterable[float]] = None,
        **ik_kwargs,
    ) -> np.ndarray:
        """Move the end effector to a Cartesian pose (IK + joint move).

        Parameters
        ----------
        pos : iterable of 3 float
            Target end-effector position ``[x, y, z]`` (m).
        rot : array_like, optional
            Target orientation: a 3x3 rotation matrix, or an Euler/RPY
            triple when ``is_euler=True``.  ``None`` keeps the identity
            orientation.
        is_euler, is_radians : bool, optional
            Treat ``rot`` as RPY (radians unless ``is_radians=False``).
        speed : int, optional
            Speed percentage forwarded to :meth:`move_j`.
        block : bool, optional
            Forwarded to :meth:`move_j`.
        init_q : iterable of float, optional
            IK seed; defaults to the live pose.
        **ik_kwargs
            Extra keyword args forwarded to :meth:`inverse_kinematics`
            (e.g. ``multi_init``, ``eps``, ``damping``).

        Returns
        -------
        np.ndarray
            The joint solution (rad) that was commanded.

        Raises
        ------
        RuntimeError
            IK failed to converge (target likely outside the workspace).
        """
        self._require_kinematics()
        q = self.inverse_kinematics(
            pos,
            rot,
            is_euler=is_euler,
            is_radians=is_radians,
            init_q=init_q,
            **ik_kwargs,
        )
        if q is None:
            raise RuntimeError(
                "move_p: IK failed to converge; target pose is likely "
                "outside the reachable workspace or near a singularity."
            )
        q_rad = q if is_radians else np.deg2rad(q)
        self.move_j(q_rad, is_radians=True, speed=speed, block=block)
        return q_rad

    @_guard_operation("move_l", RobotState.MOVING)
    def move_l(
        self,
        pos: Iterable[float],
        rot=None,
        *,
        is_euler: bool = False,
        is_radians: bool = True,
        speed: int = 50,
        steps: int = 20,
        **ik_kwargs,
    ) -> np.ndarray:
        """Move the end effector along a straight Cartesian line.

        Samples the geodesic from the current pose to ``(pos, rot)`` in
        ``steps`` waypoints, solves IK for each (seeded from the previous
        waypoint for continuity) and runs the resulting joint path.

        Parameters
        ----------
        pos, rot, is_euler, is_radians, speed
            See :meth:`move_p`.
        steps : int, optional
            Number of Cartesian waypoints (>= 1).  More steps == straighter
            line but more IK solves.  Default 20.
        **ik_kwargs
            Forwarded to :meth:`inverse_kinematics` for each waypoint.

        Returns
        -------
        np.ndarray
            The joint-space path actually commanded, shape ``(steps, num_joints)``.

        Raises
        ------
        RuntimeError
            IK failed at some waypoint (path leaves the workspace).
        """
        dynamics = self._require_kinematics()
        steps = max(1, int(steps))
        q_now = self.get_joint_values()
        waypoints = dynamics.cartesian_waypoints(
            q_now,
            pos,
            rot,
            is_euler=is_euler,
            rotation_is_radians=is_radians,
            steps=steps,
        )

        # multi_init off for waypoints: we want continuity from the seed.
        ik_kwargs.setdefault("multi_init", False)

        path: List[np.ndarray] = []
        prev_q = q_now
        for k, (translation, rotation) in enumerate(waypoints, start=1):
            q = self.inverse_kinematics(
                translation,
                rotation,
                is_radians=True,
                init_q=prev_q,
                **ik_kwargs,
            )
            if q is None:
                raise RuntimeError(
                    f"move_l: IK failed at waypoint {k}/{steps}; the "
                    "straight-line path leaves the reachable workspace.")
            path.append(q)
            prev_q = q

        path_arr = np.asarray(path, dtype=float)
        try:
            self.move_jntspace_path(path_arr, is_radians=True, speed=speed)
        except NotImplementedError:
            # No TOPPRA/wrs: fall back to sequential blocking joint moves.
            print("[FafuRobot] move_l: TOPPRA/wrs unavailable; falling back "
                  "to sequential move_j per waypoint.")
            for q in path_arr:
                self.move_j(q, is_radians=True, speed=speed, block=True)
        return path_arr

    # ------------------------------------------------------------------
    #  Feedback
    # ------------------------------------------------------------------
    def get_joint_values(self, *, prefer_cache: bool = True) -> np.ndarray:
        """Return current manipulator joint angles, in radians."""
        states = self._read_states(self._joint_motor_ids, prefer_cache=prefer_cache)
        out = np.zeros(self.num_joints, dtype=float)
        for i, mid in enumerate(self._joint_motor_ids):
            s = states.get(mid)
            if s is None:
                raise RuntimeError(f"no feedback from motor {mid}")
            out[i] = self._turns_to_rad(s.position)
        return out

    def get_joint_velocities(self, *, prefer_cache: bool = True) -> np.ndarray:
        """Return current manipulator joint velocities, in rad/s."""
        states = self._read_states(self._joint_motor_ids, prefer_cache=prefer_cache)
        out = np.zeros(self.num_joints, dtype=float)
        for i, mid in enumerate(self._joint_motor_ids):
            s = states.get(mid)
            if s is None:
                raise RuntimeError(f"no feedback from motor {mid}")
            # velocity is reported in turns/s -> rad/s
            out[i] = self._turns_to_rad(s.velocity)
        return out

    def get_motor_states(self, *, prefer_cache: bool = True) -> Dict[int, "pm.MotorState"]:
        """Return raw :class:`MotorState` objects keyed by motor id."""
        return self._read_states(self._cfg.motor_ids, prefer_cache=prefer_cache)

    def get_pose(self, *, prefer_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return the live end-effector pose as ``(position, rotation)``.

        ``position`` is a length-3 vector in metres and ``rotation`` a
        3x3 matrix, both in the URDF base frame.  Requires
        :meth:`setup_dynamics` (pinocchio + URDF).
        """
        self._require_kinematics()
        q = self.get_joint_values(prefer_cache=prefer_cache)
        fk = self.forward_kinematics(q)
        return fk["position"], fk["rotation"]

    # ------------------------------------------------------------------
    #  Gripper
    # ------------------------------------------------------------------
    # Public gripper motion units are rad, rad/s and rad/s^2.
    _GRIPPER_VEL_DEFAULT = 0.3 * _TWO_PI
    _GRIPPER_ACC_DEFAULT = 0.5 * _TWO_PI
    _GRIPPER_TOLERANCE_RAD = 0.005 * _TWO_PI  # ~ 1.8 deg
    _GRIPPER_STALL_VEL_RAD_S = 0.005 * _TWO_PI
    _GRIPPER_STALL_PATIENCE_S = 0.3           # treat as done if stalled this long

    @_guard_operation(
        "gripper_control",
        RobotState.GRASPING,
        allow_servo_nonblocking=True,
    )
    def gripper_control(
        self,
        angle: float,
        effort: Optional[int] = None,
        *,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        block: bool = True,
        timeout: float = 8.0,
        tolerance_rad: float = math.radians(1.5),
        effort_threshold: Optional[int] = None,
    ) -> Optional[GraspResult]:
        """Drive the gripper joint to an explicit ``angle`` (Piper-style).

        Unlike the Piper gripper (linear width in metres), the
        Fafu gripper is just another rotational motor, so
        ``angle`` is interpreted as a **joint angle** (radians by
        default).  No "open / close" semantics are applied here -
        the value is sent as-is and clamped by the soft limit.

        On the stock Fafu cfg the convention is::

            limits.7 = -114.98 deg (lower)  ─ closer to fully closed
                        -1.83  deg (upper)  ─ open

        ...so a more *open* gripper is a *less negative* angle.  Use
        :meth:`open_gripper` / :meth:`close_gripper` if you do not
        want to think about which direction is which.

        This signature intentionally mirrors :meth:`piper.PiperArmController.gripper_control`
        (``angle``, ``effort``) so client code can be ported across
        the two arms with minimal changes.  The semantics are slightly
        different though: Piper's firmware does the force-limited
        closure internally, whereas on Fafu the wrapper either
        passes ``effort`` to ``set_pos_vel_tqe`` as a torque cap, or
        polls ``MotorState.torque`` in Python to implement the early
        exit (``effort_threshold``).

        Parameters
        ----------
        angle : float
            Target joint angle.  Required.
        effort : int, optional
            Maximum torque (**raw int16**, as reported by
            :attr:`MotorState.torque`) the firmware is allowed to use
            while following ``angle``.  When ``None`` (default) the
            command goes through ``set_pos_vel_acc`` and inherits the
            motor's internal current limit; when given, the command
            goes through ``set_pos_vel_tqe``.  Typical safe values
            are ``cfg.max_torque_raw`` (full effort) down to a few
            hundred for soft contact.
        is_radians : bool, optional
            Compatibility guard; only ``True`` is accepted.
        vel : float, optional
            Non-negative velocity limit in rad/s.
        acc : float, optional
            Non-negative acceleration limit in rad/s^2. Ignored when
            ``effort`` is provided.
        block : bool, optional
            * ``True`` (default): poll the gripper position until it
              reaches the target (within ``tolerance_rad``), stalls,
              ``effort_threshold`` is exceeded, or ``timeout`` elapses.
            * ``False``: just send the command and return immediately.
              The Servo owner thread may use this form during a Servo session;
              it borrows the existing writer lease without changing the
              active operation. No other gripper form gets this exception.
        timeout : float, optional
            Maximum seconds to wait when ``block`` is True.
        tolerance_rad : float, optional
            Non-negative "reached" tolerance in radians.
        effort_threshold : int, optional
            Raw int16 ``|torque|`` value that triggers force detection.
            When provided, it also limits the firmware command (or lowers
            ``effort`` when that is smaller) so returning early cannot
            leave an over-force command active. The method then returns a
            :class:`GraspResult`; otherwise it returns ``None`` for
            backwards compatibility.

        Returns
        -------
        GraspResult or None
            A :class:`GraspResult` only when ``block=True`` and
            ``effort_threshold`` is provided; otherwise ``None`` for
            backwards compatibility.
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        self._require_radians(is_radians, "gripper_control")
        angle = float(angle)
        vel = float(vel)
        acc = float(acc)
        timeout = float(timeout)
        tolerance_rad = float(tolerance_rad)
        if not all(
            math.isfinite(value)
            for value in (angle, vel, acc, timeout, tolerance_rad)
        ):
            raise ValueError("gripper motion parameters must be finite")
        if vel < 0.0 or acc < 0.0:
            raise ValueError("gripper velocity and acceleration cannot be negative")
        if timeout <= 0.0:
            raise ValueError("timeout must be positive")
        if tolerance_rad < 0.0:
            raise ValueError("tolerance_rad cannot be negative")
        effort_limit = None if effort is None else int(effort)
        if effort_limit is not None and not 1 <= effort_limit <= 32767:
            raise ValueError("effort must be in [1, 32767]")
        threshold = (
            None if effort_threshold is None else int(effort_threshold)
        )
        if threshold is not None and not 1 <= threshold <= 32767:
            raise ValueError("effort_threshold must be in [1, 32767]")
        command_effort = effort_limit
        if threshold is not None:
            command_effort = (
                threshold if command_effort is None
                else min(command_effort, threshold)
            )

        with self._command_guard():
            if command_effort is None:
                self._ht.set_pos_vel_acc_rad(
                    self._gripper_motor_id,
                    angle,
                    vel,
                    acc,
                )
            else:
                self._ht.set_pos_vel_tqe_rad(
                    self._gripper_motor_id,
                    angle,
                    vel,
                    command_effort,
                )

        if not block:
            return None

        # When the caller did not ask for force-aware blocking we keep
        # the legacy return-None behaviour so existing call sites
        # don't suddenly start receiving objects.
        result = self._wait_until_gripper_done(
            angle,
            timeout=timeout,
            tolerance_rad=tolerance_rad,
            effort_threshold=threshold,
        )
        if threshold is None:
            return None
        return result

    def _gripper_limit_rad(self) -> Tuple[Optional[float], Optional[float]]:
        """Return the gripper soft limit in radians."""
        try:
            lim = self._ht.get_position_limit_turns(self._gripper_motor_id)
        except Exception:
            lim = None
        if lim is None:
            return (None, None)
        return (
            self._turns_to_rad(float(lim[0])),
            self._turns_to_rad(float(lim[1])),
        )

    def open_gripper(
        self,
        angle: Optional[float] = None,
        effort: Optional[int] = None,
        *,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        block: bool = True,
        timeout: float = 8.0,
    ) -> None:
        """Open the gripper.

        On Fafu a more *open* gripper corresponds to the
        **upper** soft limit (a less negative angle).  When ``angle``
        is ``None`` (default) the gripper is driven to that upper
        limit, which gives the largest opening allowed by the
        configuration.

        Parameters
        ----------
        angle : float, optional
            Explicit target angle.  When omitted, the upper soft
            limit is used (or +pi/2 rad if no limit is
            configured).
        effort : int, optional
            Max torque (raw int16) the firmware may use during the
            move; forwarded to :meth:`gripper_control`.  ``None`` keeps
            the legacy ``set_pos_vel_acc`` behaviour.
        is_radians : bool, optional
            Compatibility guard; only ``True`` is accepted.
        vel : float, optional
            Velocity limit in rad/s.
        acc : float, optional
            Acceleration limit in rad/s^2.  Ignored when ``effort``
            is provided.
        block : bool, optional
            Block until the gripper reaches the target / stalls /
            ``timeout`` elapses.  Defaults to ``True``.
        timeout : float, optional
            Max seconds to wait when ``block`` is True.
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        self._require_radians(is_radians, "open_gripper")
        if angle is None:
            _, hi_rad = self._gripper_limit_rad()
            target_rad = hi_rad if hi_rad is not None else math.pi / 2.0
            self.gripper_control(
                target_rad,
                effort,
                is_radians=True,
                vel=vel, acc=acc,
                block=block, timeout=timeout,
            )
        else:
            self.gripper_control(
                angle, effort,
                is_radians=is_radians, vel=vel, acc=acc,
                block=block, timeout=timeout,
            )

    def close_gripper(
        self,
        angle: Optional[float] = None,
        effort: Optional[int] = None,
        *,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        block: bool = True,
        timeout: float = 8.0,
    ) -> None:
        """Close the gripper.

        On Fafu a more *closed* gripper corresponds to the
        **lower** soft limit (a more negative angle).  When ``angle``
        is ``None`` (default) the gripper is driven to that lower
        limit, which gives the tightest grip allowed by the
        configuration.

        Parameters
        ----------
        angle : float, optional
            Explicit target angle.  When omitted, the lower soft
            limit is used (or -pi/2 rad if no limit is
            configured).
        effort : int, optional
            Max torque (raw int16) the firmware may use during the
            move; forwarded to :meth:`gripper_control`.  ``None`` keeps
            the legacy ``set_pos_vel_acc`` behaviour.  For **force-aware
            grasping with an early stop on contact**, use
            :meth:`grasp` instead.
        is_radians : bool, optional
            Compatibility guard; only ``True`` is accepted.
        vel : float, optional
            Velocity limit in rad/s.
        acc : float, optional
            Acceleration limit in rad/s^2.  Ignored when ``effort``
            is provided.
        block : bool, optional
            Block until the gripper reaches the target / stalls /
            ``timeout`` elapses.  Defaults to ``True`` so that a
            grasp action does not get cut short by a subsequent
            ``close_connection()``.
        timeout : float, optional
            Max seconds to wait when ``block`` is True.
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        self._require_radians(is_radians, "close_gripper")
        if angle is None:
            lo_rad, _ = self._gripper_limit_rad()
            target_rad = lo_rad if lo_rad is not None else -math.pi / 2.0
            self.gripper_control(
                target_rad,
                effort,
                is_radians=True,
                vel=vel, acc=acc,
                block=block, timeout=timeout,
            )
        else:
            self.gripper_control(
                angle, effort,
                is_radians=is_radians, vel=vel, acc=acc,
                block=block, timeout=timeout,
            )

    # Default grasp tuning.  Slower than open/close because we want
    # to feel contact, not slam through it.
    _GRASP_VEL_DEFAULT = 0.15 * _TWO_PI
    _GRASP_FORCE_THRESHOLD_DEFAULT = 500 # raw int16, conservative

    @_guard_operation("grasp", RobotState.GRASPING)
    def grasp(
        self,
        *,
        target_angle: Optional[float] = None,
        is_radians: bool = True,
        force_threshold: int = _GRASP_FORCE_THRESHOLD_DEFAULT,
        effort: Optional[int] = None,
        vel: float = _GRASP_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        timeout: float = 5.0,
        min_close_rad: float = math.radians(3.0),
    ) -> GraspResult:
        """Close the gripper until contact is detected (Piper-style force grasp).

        This is the Fafu-side equivalent of Piper's
        ``close_gripper(effort=...)``: it commands the gripper toward
        the closing direction and returns as soon as the **measured**
        torque exceeds ``force_threshold`` (or the gripper plateaus
        at finite stiffness against an object).

        Contact detection runs in Python by polling ``MotorState.torque``
        from the background poller. The firmware command always includes a
        torque cap, so polling latency cannot exceed that configured limit.
        This is useful for catching objects but is **not** a substitute for
        a real wrist F/T sensor.

        Parameters
        ----------
        target_angle : float, optional
            Maximum closure angle.  When ``None`` (default) the lower
            soft limit from ``cfg.limits[gripper_motor_id]`` is used,
            which is the tightest grip the configuration allows.
        is_radians : bool, optional
            Compatibility guard; only ``True`` is accepted.
        force_threshold : int, optional
            Raw int16 ``|torque|`` value that, when exceeded, ends the
            grasp early and marks it as successful
            (``reason='detected_object_force'``).  Calibrate by
            running an empty close and noting the steady-state torque.
        effort : int, optional
            Torque cap passed to the **firmware** via
            ``set_pos_vel_tqe`` (raw int16). When ``None``,
            ``force_threshold`` is also used as the hard cap.
        vel : float, optional
            Non-negative closing velocity in rad/s.
        acc : float, optional
            Retained for API compatibility; grasp uses the torque-capped
            ``set_pos_vel_tqe`` command, which has no acceleration field.
        timeout : float, optional
            Maximum wall-clock seconds to wait.
        min_close_rad : float, optional
            Non-negative minimum closure in radians before a stall counts as
            "object grasped".

        Returns
        -------
        GraspResult
            Always returned; inspect ``.grasped`` and ``.reason``
            to decide what happened.

        Raises
        ------
        RuntimeError
            If this controller was built without a gripper.

        Examples
        --------
        Empty close (will report no object)::

            r = arm.grasp(force_threshold=500)
            print(r.grasped, r.reason, r.peak_torque_raw)

        Grasp with a hard hardware torque cap as well::

            r = arm.grasp(force_threshold=600, effort=800, vel=0.1)
            if not r.grasped:
                arm.open_gripper()
                raise RuntimeError(f"grasp failed: {r.reason}")
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        self._require_radians(is_radians, "grasp")
        vel = float(vel)
        acc = float(acc)
        timeout = float(timeout)
        min_close_rad = float(min_close_rad)
        values = (vel, acc, timeout, min_close_rad)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("grasp motion parameters must be finite")
        if vel < 0.0 or acc < 0.0 or min_close_rad < 0.0:
            raise ValueError("grasp velocity, acceleration and closure cannot be negative")
        if timeout <= 0.0:
            raise ValueError("timeout must be positive")

        threshold = int(force_threshold)
        if not 1 <= threshold <= 32767:
            raise ValueError("force_threshold must be in [1, 32767]")
        effort_limit = threshold if effort is None else int(effort)
        if not 1 <= effort_limit <= 32767:
            raise ValueError("effort must be in [1, 32767]")

        if target_angle is None:
            lo_rad, _ = self._gripper_limit_rad()
            target_rad = lo_rad if lo_rad is not None else -math.pi / 2.0
        else:
            target_angle = float(target_angle)
            if not math.isfinite(target_angle):
                raise ValueError("target_angle must be finite")
            target_rad = target_angle

        with self._command_guard():
            self._ht.set_pos_vel_tqe_rad(
                self._gripper_motor_id,
                target_rad,
                vel,
                effort_limit,
            )

        return self._wait_until_gripper_done(
            target_rad,
            timeout=timeout,
            effort_threshold=threshold,
            min_progress_rad=min_close_rad,
        )

    def release(
        self,
        *,
        target_angle: Optional[float] = None,
        is_radians: bool = True,
        vel: float = _GRIPPER_VEL_DEFAULT,
        acc: float = _GRIPPER_ACC_DEFAULT,
        timeout: float = 5.0,
        block: bool = True,
    ) -> None:
        """Counterpart to :meth:`grasp`: drive the gripper open to drop the object.

        Thin alias of :meth:`open_gripper` — provided so that grasp /
        release form a symmetric pair in user code.
        """
        self._require_radians(is_radians, "release")
        self.open_gripper(
            angle=target_angle,
            is_radians=is_radians,
            vel=vel,
            acc=acc,
            block=block,
            timeout=timeout,
        )

    def get_gripper_state(self) -> "pm.MotorState":
        """Return the latest :class:`MotorState` of the gripper motor."""
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        s = self._ht.get_cached_state(self._gripper_motor_id)
        if s is None:
            s = self._ht.read_motor_state(self._gripper_motor_id, 0.1)
        if s is None:
            raise RuntimeError("no feedback from gripper motor")
        return s

    # ------------------------------------------------------------------
    #  Soft limits
    # ------------------------------------------------------------------
    def set_limit(
        self,
        motor_id: int,
        lo: float,
        hi: float,
        *,
        is_radians: bool = True,
    ) -> None:
        """Enable a soft position limit for ``motor_id``.

        ``lo`` / ``hi`` are radians; ``is_radians`` must be ``True``.
        """
        self._require_radians(is_radians, "set_limit")
        lo = float(lo)
        hi = float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("lo and hi must be finite radians")

        if lo > hi:
            raise ValueError(f"lo ({lo}) > hi ({hi})")
        with self._config_lock:
            if motor_id not in self._cfg.motor_ids:
                raise ValueError(f"motor {motor_id} is not in cfg.motor_ids")
            if self.state in _BUSY_STATES:
                raise RobotStateError(
                    "soft limits cannot change during a control operation"
                )
            self._ht.enable_position_limit(
                motor_id, lo, hi, pm.PosUnit.Radians
            )
            # RobotConfig stores protocol limits, but this conversion is
            # never exposed through the public command boundary.
            limits = dict(self._cfg.limits)
            limits[motor_id] = (lo / _TWO_PI, hi / _TWO_PI)
            self._cfg.limits = limits

    def get_limit(
        self,
        motor_id: int,
        *,
        is_radians: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Return the radians ``(lo, hi)`` limit, or ``None`` if unset."""
        self._require_radians(is_radians, "get_limit")
        with self._config_lock:
            if motor_id not in self._cfg.motor_ids:
                raise ValueError(f"motor {motor_id} is not in cfg.motor_ids")
            limit = self._ht.get_position_limit_turns(motor_id)
            if limit is None:
                return None
            lo_t, hi_t = limit
            return (self._turns_to_rad(lo_t), self._turns_to_rad(hi_t))

    def disable_limit(self, motor_id: int) -> None:
        """Disable the soft limit for ``motor_id``."""
        with self._config_lock:
            if motor_id not in self._cfg.motor_ids:
                raise ValueError(f"motor {motor_id} is not in cfg.motor_ids")
            if self.state in _BUSY_STATES:
                raise RobotStateError(
                    "soft limits cannot change during a control operation"
                )
            self._ht.disable_position_limit(motor_id)
            limits = dict(self._cfg.limits)
            limits.pop(motor_id, None)
            self._cfg.limits = limits

    def clear_limits(self) -> None:
        """Disable every soft position limit."""
        with self._config_lock:
            if self.state in _BUSY_STATES:
                raise RobotStateError(
                    "soft limits cannot change during a control operation"
                )
            self._ht.clear_all_position_limits()
            self._cfg.limits = {}

    # ------------------------------------------------------------------
    #  Safety
    # ------------------------------------------------------------------
    def emergency_stop(self) -> None:
        """Immediately stop every motor and latch native ESTOP."""
        self._core.emergency_stop()
        self._sync_state_from_core()
        print("[FafuRobot] EMERGENCY STOP issued (all motors stopped).")

    def resume(self) -> None:
        """Clear native ESTOP and re-enable position control."""
        if self._core.state == pm.RobotState.DEAD:
            print(
                "[FafuRobot] resume: state is DEAD; use "
                "recover(confirm=True), then enable()."
            )
            return
        try:
            resumed = bool(self._core.resume())
        except _NATIVE_STATE_ERRORS as exc:
            raise RobotStateError(f"resume rejected: {exc}") from exc
        if not resumed:
            raise RuntimeError("resume failed to enable all motors")
        self._sync_state_from_core()

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    def get_status(self):
        """Return the latest :class:`Stats` object from the driver."""
        return self._ht.get_stats()

    def get_can_status(self):
        """Return the latest :class:`CanStatus` (live read)."""
        return self._ht.read_can_status()

    @_guard_operation("reset_zero", RobotState.MOVING)
    def reset_zero(self, motor_id: int, *, confirm: bool = False) -> None:
        """Set the current position of ``motor_id`` as the new zero.

        ``confirm=True`` is required because this is destructive.
        """
        if not confirm:
            raise RuntimeError("reset_zero is destructive; pass confirm=True")
        if motor_id not in self._cfg.motor_ids:
            raise ValueError(f"motor {motor_id} not in cfg.motor_ids")
        with self._command_guard():
            self._ht.reset_zero(motor_id)

    # ------------------------------------------------------------------
    #  Connection lifecycle
    # ------------------------------------------------------------------
    def close_connection(
        self,
        *,
        joint_release: str = "brake",
        gripper_release: str = "brake",
    ) -> None:
        """Cancel active work, release motors and close the serial port.

        Both joint and gripper release default to short-circuit brake."""
        valid = {"stop", "brake", "hold"}
        if joint_release not in valid:
            raise ValueError(f"joint_release must be one of {valid}")
        if gripper_release not in valid:
            raise ValueError(f"gripper_release must be one of {valid}")

        core = self._core
        needs_shutdown = core.state != pm.RobotState.DISCONNECTED
        port_open = self._ht.is_open()
        if not needs_shutdown and not port_open:
            return

        if needs_shutdown:
            try:
                core.shutdown(
                    self._core_finish_mode(joint_release),
                    self._core_finish_mode(gripper_release),
                    5.0,
                )
            except _NATIVE_STATE_ERRORS as exc:
                raise RobotStateError(
                    f"close_connection rejected: {exc}") from exc
        if self._ht.is_open():
            self._ht.close()
        self._sync_state_from_core()
        print(f"[FafuRobot] connection closed "
              f"(joints={joint_release}, gripper={gripper_release}).")

    def __enter__(self) -> "FafuRobotController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_connection()

    # ==================================================================
    #  Internals
    # ==================================================================
    @staticmethod
    def _require_radians(is_radians: bool, action: str) -> None:
        if is_radians is not True:
            raise ValueError(
                f"{action} is radians-only; convert with np.deg2rad first"
            )

    @staticmethod
    def _turns_to_rad(turns: float) -> float:
        return float(turns) * _TWO_PI

    @staticmethod
    def _clamp_speed(speed: int) -> int:
        s = int(speed)
        if s < 1:
            s = 1
        if s > 100:
            s = 100
        return s

    @staticmethod
    def _resolve_cfg_path(cfg_path: str) -> str:
        if os.path.isabs(cfg_path) or os.path.exists(cfg_path):
            return cfg_path
        candidate = os.path.join(_HERE, cfg_path)
        if os.path.exists(candidate):
            return candidate
        return cfg_path

    def _pick_serial_port(self, preferred: Optional[str]) -> str:
        """Resolve a serial port without silently selecting the wrong arm."""
        pref = (preferred or "").strip()
        if pref and pref.lower() != "auto":
            return pref

        try:
            ports = list(pm.find_likely_debug_boards())
        except Exception as exc:
            raise RuntimeError(
                f"failed to enumerate USB debug boards: {exc}") from exc
        if not ports:
            raise RuntimeError(
                "no USB debug board detected; specify an explicit port or "
                "check the cable")
        if len(ports) != 1:
            names = ", ".join(str(p.port) for p in ports)
            raise RuntimeError(
                "multiple USB debug boards detected "
                f"({names}); specify one explicit port per SDK instance")
        return str(ports[0].port)

    def _precheck_communication(self) -> None:
        bad: List[int] = []
        for mid in self._cfg.motor_ids:
            if self._ht.read_motor_state(mid, 0.5) is None:
                bad.append(mid)
        if bad:
            raise RuntimeError(
                f"motors {bad} did not respond within 500ms; "
                f"check power, wiring and motor IDs"
            )

    def _validate_joint_angles(
        self,
        joint_angles: Iterable[float],
        is_radians: bool,
    ) -> List[float]:
        """Validate a radians-only joint vector."""
        self._require_radians(is_radians, "joint motion")
        arr = np.asarray(list(joint_angles), dtype=float)
        if arr.shape != (self.num_joints,):
            raise ValueError(
                f"expected a 1-D vector of {self.num_joints} joint values; "
                f"got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("joint values must all be finite (no NaN/Inf)")
        return [float(value) for value in arr]

    def _read_states(
        self,
        motor_ids: Iterable[int],
        *,
        prefer_cache: bool,
    ) -> Dict[int, "pm.MotorState"]:
        out: Dict[int, "pm.MotorState"] = {}
        ids = list(motor_ids)
        if prefer_cache:
            for mid in ids:
                s = self._ht.get_cached_state(mid)
                if s is not None:
                    out[mid] = s
            missing = [m for m in ids if m not in out]
            if not missing:
                return out
        else:
            missing = ids
        # Fall back to a synchronous read for whatever is missing.
        for mid in missing:
            s = self._ht.read_motor_state(mid, 0.1)
            if s is not None:
                out[mid] = s
        return out

    # Minimum closure before a stall counts as "grasped object"
    # rather than "no movement / command never took effect".
    _GRIPPER_MIN_PROGRESS_RAD = 0.008 * _TWO_PI  # ~ 2.9 deg

    def _wait_until_gripper_done(
        self,
        target_rad: float,
        *,
        timeout: float = 8.0,
        tolerance_rad: Optional[float] = None,
        effort_threshold: Optional[int] = None,
        min_progress_rad: Optional[float] = None,
    ) -> GraspResult:
        """Block until the gripper reaches ``target_rad``, stalls,
        ``|torque| >= effort_threshold``, or ``timeout`` elapses.

        Returns a :class:`GraspResult` regardless of why we stopped;
        callers that don't care can simply ignore the return value.

        We treat the move as "done" if any of:

        * ``|position - target| <= tolerance_rad`` (reached target).
        * ``|torque| >= effort_threshold`` (force trip, if given).
        * ``|velocity| < stall threshold`` for
          ``_GRIPPER_STALL_PATIENCE_S`` and the gripper has moved at
          least ``min_progress_rad`` (grasped something).
        * Stall without enough movement → ``'no_movement'``.
        * Wall-clock ``timeout`` exceeded → ``'timeout'``.
        """
        if tolerance_rad is None:
            tolerance_rad = self._GRIPPER_TOLERANCE_RAD
        if min_progress_rad is None:
            min_progress_rad = self._GRIPPER_MIN_PROGRESS_RAD

        t0 = time.monotonic()
        deadline = t0 + max(0.05, float(timeout))

        # Capture the starting position so we can report closed_deg
        # and classify "no movement" vs "real grasp".
        start_state = self._ht.get_cached_state(self._gripper_motor_id)
        if start_state is None:
            start_state = self._ht.read_motor_state(self._gripper_motor_id, 0.05)
        start_pos = (
            self._turns_to_rad(start_state.position)
            if start_state is not None
            else float("nan")
        )
        last_pos = start_pos

        stall_since: Optional[float] = None
        peak_torque = 0

        while True:
            now = time.monotonic()
            if self._native_cancel_requested() or not self._stream_link_ok():
                return self._make_grasp_result(
                    reason="cancelled", grasped=False,
                    last_pos=last_pos, start_pos=start_pos,
                    peak_torque=peak_torque, duration=now - t0,
                )
            if now >= deadline:
                return self._make_grasp_result(
                    reason="timeout", grasped=False,
                    last_pos=last_pos, start_pos=start_pos,
                    peak_torque=peak_torque, duration=now - t0,
                )

            s = self._ht.get_cached_state(self._gripper_motor_id)
            if s is None:
                s = self._ht.read_motor_state(self._gripper_motor_id, 0.05)

            if s is not None:
                last_pos = self._turns_to_rad(s.position)
                velocity_rad_s = self._turns_to_rad(s.velocity)
                t_raw = int(abs(s.torque))
                if t_raw > peak_torque:
                    peak_torque = t_raw

                if effort_threshold is not None and t_raw >= int(effort_threshold):
                    return self._make_grasp_result(
                        reason="detected_object_force", grasped=True,
                        last_pos=last_pos, start_pos=start_pos,
                        peak_torque=peak_torque, duration=now - t0,
                    )

                if abs(last_pos - target_rad) <= tolerance_rad:
                    return self._make_grasp_result(
                        reason="reached_target", grasped=False,
                        last_pos=last_pos, start_pos=start_pos,
                        peak_torque=peak_torque, duration=now - t0,
                    )

                if abs(velocity_rad_s) < self._GRIPPER_STALL_VEL_RAD_S:
                    if stall_since is None:
                        stall_since = now
                    elif now - stall_since >= self._GRIPPER_STALL_PATIENCE_S:
                        progress = abs(last_pos - start_pos)
                        if progress >= min_progress_rad:
                            return self._make_grasp_result(
                                reason="detected_object_stall", grasped=True,
                                last_pos=last_pos, start_pos=start_pos,
                                peak_torque=peak_torque, duration=now - t0,
                            )
                        return self._make_grasp_result(
                            reason="no_movement", grasped=False,
                            last_pos=last_pos, start_pos=start_pos,
                            peak_torque=peak_torque, duration=now - t0,
                        )
                else:
                    stall_since = None

            time.sleep(0.02)

    def _make_grasp_result(
        self,
        *,
        reason: str,
        grasped: bool,
        last_pos: float,
        start_pos: float,
        peak_torque: int,
        duration: float,
    ) -> GraspResult:
        if math.isnan(last_pos) or math.isnan(start_pos):
            closed_deg = 0.0
            angle_rad = float("nan") if math.isnan(last_pos) else last_pos
        else:
            closed_deg = math.degrees(abs(last_pos - start_pos))
            angle_rad = last_pos
        return GraspResult(
            grasped=grasped,
            reason=reason,
            angle_rad=angle_rad,
            closed_deg=closed_deg,
            peak_torque_raw=int(peak_torque),
            duration_s=float(duration),
        )
