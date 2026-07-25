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

* All joint angles exposed to the user are in **radians** by default
  (``is_radians=True``).  Internally the wrapper converts radians to
  *turns* (the protocol native unit; ``1 turn = 2*pi rad``).
* Velocities are expressed as a percentage in the ``speed`` argument
  (``0 - 100``); this is mapped to a peak average velocity in
  turns/second.
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
from contextlib import nullcontext
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
#  Import the native extension as a package, with a direct-module fallback for
#  legacy scripts that put this directory on sys.path.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if __package__:
    from . import fafu_motor as pm
else:  # pragma: no cover - direct module import compatibility
    import fafu_motor as pm

# Optional: TOPPRA-based time-optimal interpolation (matches piper.py).
try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp  # type: ignore

    _TOPPRA_EXIST = True
except Exception:  # pragma: no cover - optional dependency
    _TOPPRA_EXIST = False

# Optional: wrs robot_math, only needed if move_p / move_l are wired to IK.
try:
    import wrs.basis.robot_math as rm  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    rm = None  # type: ignore

# Optional: pinocchio rigid-body dynamics, only needed for the
# gravity / Coriolis / mass-matrix terms behind :meth:`setup_dynamics`
# and :meth:`start_gravity_compensation`.  It is notoriously hard to
# install on Windows (conda-forge / Linux are the smooth paths), so the
# whole dynamics feature degrades gracefully: when ``pinocchio`` is
# missing every dynamics method raises a clear, actionable error and the
# rest of the controller (move_j / servo_j / gripper) keeps working.
try:
    import pinocchio as pin  # type: ignore

    _PIN_EXIST = True
except Exception:  # pragma: no cover - optional dependency
    pin = None  # type: ignore
    _PIN_EXIST = False


# ============================================================================
#  Constants
# ============================================================================

# Operating modes (matches motor_example_debug / arm_multi_joint_example).
MODE_POSITION = 0x0A   # release / position-control hold
MODE_BRAKE    = 0x0F   # short-circuit braking (no torque to spin)
MODE_STOP     = 0x00   # PWM off, free to move by hand
MODE_MIT      = 0x0B   # MIT residual mode left by set_many_mit (0x8093);
                       # on this firmware only motor_reset can leave it


# ============================================================================
#  High-level controller state machine
# ============================================================================
class RobotState(Enum):
    """High-level lifecycle state of :class:`FafuRobotController`.

    This is a *software* flow-control state (distinct from the per-motor
    hardware ``mode`` byte, see ``MODE_*``).  It gates which high-level
    API calls are legal at any moment so that mistakes like "``move_j``
    while not enabled" or "issue motion after an emergency stop" fail
    loudly with a clear message instead of silently doing the wrong
    thing on the hardware.

    Transition map (arrows are the *only* legal transitions)::

        DISCONNECTED --open--> DISABLED
        DISABLED  --enable()-->  IDLE
        IDLE      --disable()-->  DISABLED
        IDLE      --brake()---->  BRAKED
        BRAKED/DISABLED --enable()--> IDLE
        IDLE      --move_j/move_p/...-->  MOVING     --(done)--> IDLE
        IDLE      --servo_start()------>  SERVOING   --servo_end("hold")--> IDLE
                                                     --servo_end("brake")--> BRAKED
                                                     --servo_end("stop")--> DISABLED
        IDLE      --grasp()/gripper---->  GRASPING   --(done)--> IDLE
        IDLE      --start_gravity_comp->  GRAVITY_COMP --(exit)--> BRAKED
        <any connected> --emergency_stop()--> ESTOP  --resume()--> IDLE
        <any connected> --power/CAN lost---> DEAD     --recover()--> DISABLED
        <any>     --close_connection()-->  DISCONNECTED
    """

    DISCONNECTED  = "disconnected"   # serial port closed (initial / after close_connection)
    DISABLED      = "disabled"       # connected, motors free-spin (MODE_STOP 0x00), hand-movable
    BRAKED        = "braked"         # connected, short-circuit brake (MODE_BRAKE 0x0F), resists motion
    IDLE          = "idle"           # connected + enabled (MODE_POSITION), ready & not busy
    MOVING        = "moving"         # blocking joint / cartesian / gripper motion in progress
    SERVOING      = "servoing"       # servo_start .. servo_end streaming session open
    GRASPING      = "grasping"       # force-aware grasp() in progress
    GRAVITY_COMP  = "gravity_comp"   # start_gravity_compensation() float/drag-teach loop running
    ESTOP         = "estop"          # emergency_stop() latched — sticky until resume()
    DEAD          = "dead"           # motor power / CAN link lost — latched; recover() after restore

    def __str__(self) -> str:  # nicer prints
        return self.value


class RobotStateError(RuntimeError):
    """Raised when a high-level API call is illegal in the current
    :class:`RobotState` (e.g. ``move_j`` before ``enable``, or any
    motion while the controller is latched in ``ESTOP``)."""


# States in which the arm is actively executing an operation and must
# not accept a *new* one.  emergency_stop() is the only exception.
_BUSY_STATES = frozenset(
    {
        RobotState.MOVING,
        RobotState.SERVOING,
        RobotState.GRASPING,
        RobotState.GRAVITY_COMP,
    }
)


def _guard_operation(action: str, busy_state: "RobotState"):
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
            owns = self._enter_operation(action, busy_state)
            try:
                return fn(self, *args, **kwargs)
            finally:
                self._exit_operation(owns)

        return wrapper

    return deco

# S-curve trajectory tuning (matches arm_multi_joint_example.py).
_VEL_AVG_MAX_TPS = 0.5    # absolute cap for average velocity (turns/s)
_DT_MIN_S        = 0.3    # shortest segment time (s)
_SETTLE_MS       = 300    # extra hold time after the trajectory ends
# If the measured pose is within this of the last *commanded* pose we
# treat the joint as "still held by us" and start the next trajectory
# from the command (continuity).  Beyond it, something moved the joint
# externally, so we start from the measured value instead.
# 0.05 turns = 18 deg, comfortably larger than any steady-state error
# but small enough to detect a real hand-drag.
_CMD_CONTINUITY_TOL_T = 0.05

# 1 turn = 2*pi rad
_TWO_PI = 2.0 * math.pi

# ============================================================================
#  Public dataclasses
# ============================================================================
@dataclass
class GraspResult:
    """Outcome of a :meth:`FafuRobotController.grasp` call.

    Attributes
    ----------
    grasped : bool
        ``True`` iff the wrapper concluded that an object was caught
        (torque threshold reached, or the gripper stalled after having
        clearly moved towards the target).  ``False`` for "reached the
        target with no resistance" / "did not move at all" / "timeout".
    reason : str
        One of:

        * ``'detected_object_force'`` — ``|torque| >= effort_threshold``
        * ``'detected_object_stall'`` — speed plateaued after closing
          at least ``min_close_deg``
        * ``'reached_target'``      — got to the commanded angle, nothing in the way
        * ``'no_movement'``         — stalled but barely moved (command
          may not have taken effect, or jaws were already shut)
        * ``'timeout'``             — neither condition met within ``timeout``
        * ``'cancelled'``           — safety stop, close, or link loss
    angle_rad : float
        Final gripper angle (radians).
    closed_deg : float
        Absolute change in the gripper angle since the call started
        (degrees).  Always non-negative.
    peak_torque_raw : int
        Maximum ``|MotorState.torque|`` observed during the move
        (raw int16 from the motor).
    duration_s : float
        Wall-clock time the call took (seconds).
    """
    grasped: bool
    reason: str
    angle_rad: float
    closed_deg: float
    peak_torque_raw: int
    duration_s: float


@dataclass
class ServoOpts:
    """Tunables for an online ``servo_j`` streaming session.

    See the four safety lines documented on
    :meth:`FafuRobotController.servo_start`.

    Attributes
    ----------
    watchdog_ms : int, optional
        Firmware-side watchdog in milliseconds.  If the motor receives
        no new command for this long it automatically brakes.  Default
        100. Set to 0 to disable (not recommended).
    max_vel : float, optional
        Per-joint velocity cap in **rad/s** written into every frame
        (``set_many_pos_vel_tqe`` ``vel`` field).  Default ``1.0``
        (~57 deg/s).  When ``feedforward_vel=True`` this acts as an
        upper safety bound on the computed feedforward velocity.
    max_step_rad : float, optional
        Per-step jump limit in **rad**.  ``|target - last_target|``
        larger than this is clamped to ``±max_step_rad`` and a warning
        is logged.  Default ``0.05`` (~2.9 deg).
    max_lag_rad : float, optional
        Tracking-error guard in **rad**.  If any motor's measured
        position deviates from its last target by more than this,
        the tick is flagged as a "lag-trip" and the running counter
        :attr:`FafuRobotController.servo_lag_count` is incremented
        (and ``servo_end`` prints the total).  The frame is **still
        sent** so the firmware watchdog cannot brake the rest of the
        arm; pass ``lag_abort_consecutive`` if you need automatic
        protective stop on persistent lag.  Default ``0.2``
        (~11.5 deg).  Set to ``0`` / negative to disable the
        counter entirely.
    is_radians : bool, optional
        Interpret arguments to :meth:`servo_j` in radians (default) or
        degrees.  Matches :meth:`move_j`.
    rate_hz : float, optional
        Nominal upper-layer call rate **only used to compute dt for
        feedforward and lookahead**.  Default ``100.0``.  The actual
        call rate is still set by however fast the caller invokes
        :meth:`servo_j`; this number does not throttle anything.
    feedforward_vel : bool, optional
        Default ``True``.  When enabled, the per-frame ``vel`` field
        written to each motor is the **true required velocity**
        ``(target[k] - target[k-1]) * rate_hz`` (clamped to
        ``±max_vel``), exactly like UR servoj's internal velocity
        feedforward. When ``False``, position-channel frames use
        ``max_vel`` as a positive velocity limit, while MIT frames use zero
        desired velocity (MIT velocity is signed, not a limit).
    lookahead_time : float, optional
        Default ``0.0`` (no smoothing).  When ``> 0``, an exponential
        moving average (first-order low-pass) with time constant
        ``lookahead_time`` is applied to every target before it is
        sent.  Recommended ``0.03 - 0.10`` s for noisy upper layers
        (joystick teleop, VR, vision servoing); leave at ``0`` when the
        upper layer already produces a smooth trajectory (planned
        path).  This is the same knob UR servoj exposes as
        ``lookahead_time``.
    lag_abort_consecutive : int, optional
        Default ``0`` (off).  When ``> 0``, the servo session is
        automatically terminated with :meth:`servo_end` (``"brake"``)
        after this many *consecutive* lag-tripped ticks.  This is the
        old "fail-stop" behaviour, just bounded so a single bad tick
        does not kill the loop.  Recommended in production: ``5`` -
        ``10`` (50 - 100 ms of persistent lag).  Leave at ``0`` for
        diagnostic scripts.
    """

    watchdog_ms: int = 100
    max_vel: float = 1.0
    max_step_rad: float = 0.05
    max_lag_rad: float = 0.2
    is_radians: bool = True
    # ---- noise / lag tuning (UR-servoj style) ----
    rate_hz: float = 100.0
    feedforward_vel: bool = True
    lookahead_time: float = 0.0
    # ---- lag-trip policy ----
    # Old behaviour was "lag > max_lag_rad ⇒ servo_j returns False ⇒ frame
    # NOT sent". That turned a single lagging joint into a cascade: the
    # firmware watchdog on the *good* joints tripped after watchdog_ms of
    # no frames and braked them too. The new default is "keep sending +
    # accumulate lag_count"; the lag count is reported by servo_end and
    # can be polled live via :attr:`FafuRobotController.servo_lag_count`.
    # Set ``lag_abort_consecutive > 0`` to opt back into auto-abort (handy
    # for production safety, but not for diagnostic scripts).
    lag_abort_consecutive: int = 0
    # ---- control channel ----
    # False (default): position channel; no motor calibration is needed.
    # True: group-MIT impedance channel. Non-zero MIT gains/torque require
    # exact per-joint motor_models; unknown coefficients are rejected.
    use_mit: bool = False
    # Per-joint MIT PD gains (physical vendor units, same as move_MIT). None =>
    # vendor replay defaults for 6-DoF (kp=[30,40,55,15,7,5], kd=[3,4,5.5,1.5,
    # 0.7,0.5]); scalar otherwise. Only used when use_mit=True.
    mit_kp: "float | Iterable[float] | None" = None
    mit_kd: "float | Iterable[float] | None" = None
    motor_models: "Optional[List[str]]" = None
    # Add gravity feed-forward each MIT tick (needs setup_dynamics; silently
    # falls back to kp/kd-only when no model). Only used when use_mit=True.
    mit_gravity_ff: bool = True


@dataclass
class FrictionParams:
    """Coulomb + viscous joint-friction model for gravity-comp / float mode.

    Mirrors ``2_gravity_friction_compensation_control.py``::

        tau_friction = fc * sign(v) + fv * v

    with a low-speed dead-band: when ``|v| < vel_threshold`` only the
    viscous term ``fv * v`` is used, so the Coulomb ``sign(v)`` term does
    not chatter around zero velocity.

    Attributes
    ----------
    fc : np.ndarray
        Per-joint Coulomb friction (Nm) — constant magnitude, opposes the
        direction of motion.  Identify by driving each joint at a very low
        constant speed and reading the minimum steady torque needed.
    fv : np.ndarray
        Per-joint viscous friction (Nm*s/rad) — proportional to speed.
        Identify from the slope of the torque-vs-speed curve.  Usually an
        order of magnitude smaller than ``fc``.
    vel_threshold : float
        Speed (rad/s) below which the Coulomb term is suppressed.
        ``0.01 - 0.05`` is reasonable; default ``0.02``.
    """

    fc: np.ndarray
    fv: np.ndarray
    vel_threshold: float = 0.02

    @staticmethod
    def reference_6dof() -> "FrictionParams":
        """Starting values from the reference friction model.

        These are starting values only. Friction is arm- and wear-specific
        and must be re-identified on the actual hardware.
        """
        return FrictionParams(
            fc=np.array([0.20, 0.15, 0.15, 0.15, 0.04, 0.04]),
            fv=np.array([0.06, 0.06, 0.06, 0.03, 0.02, 0.02]),
            vel_threshold=0.02,
        )


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

        # ---- High-level state machine (see RobotState) ----
        # Port is open but motors are not yet in position control, so we
        # start in DISABLED.  auto_enable below promotes us to IDLE.
        # ``_op_depth`` tracks nested guarded operations for re-entrancy
        # (e.g. go_home -> move_j); ``_state_verbose`` toggles transition
        # logging (off by default to avoid spamming per-move_j).
        self._state_lock = threading.RLock()
        self._state: RobotState = RobotState.DISABLED
        self._op_depth: int = 0
        self._op_owner_thread_id: Optional[int] = None
        self._state_verbose: bool = False
        self._gravity_comp_owner_thread_id: Optional[int] = None
        self._gravity_core_token: Optional[int] = None
        # Python-side timestamps support the hardware-free policy tests and
        # diagnostics; the native core is authoritative in real instances.
        self._motor_last_seen_monotonic: Dict[int, float] = {}
        # DEAD (motor power / CAN loss) detection. During a live async/polling
        # session, every configured joint must have feedback newer than this
        # threshold. A stale joint latches DEAD and stops all streaming, so
        # power-return cannot silently resume motion.
        self._dead_rx_timeout_ms: float = 500.0
        self._dead_reason: Optional[str] = None

        # Native P0 core is the single authority for state, writer ownership,
        # recovery and Servo safety. Refuse an old binary instead of silently
        # running a second, divergent implementation in Python.
        if getattr(pm, "CORE_ABI_VERSION", 0) != 2:
            raise RuntimeError(
                "fafu_motor native core ABI 2 is required; rebuild/install "
                "the C++ extension from this SDK version")
        core_cfg = pm.CoreConfig()
        core_cfg.all_motor_ids = list(cfg.motor_ids)
        core_cfg.joint_motor_ids = list(self._joint_motor_ids)
        core_cfg.max_torque_raw = int(cfg.max_torque_raw)
        core_cfg.stale_feedback_timeout_ms = self._dead_rx_timeout_ms
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

        # Last *commanded* joint positions (turns) from a blocking
        # move (S-curve).  Used to start the next move's trajectory from
        # where the motor was last *told* to go, not from the measured
        # position.  The two differ by the steady-state error (gravity /
        # backlash on heavy joints like the shoulder); starting the new
        # trajectory at the measured value re-commands that error as a
        # step and makes the joint jerk in the wrong direction for one
        # blink before the S-curve takes over.  Cleared whenever the
        # commanded value becomes unknown (disable / manual drag).
        self._last_cmd_turns: Optional[Dict[int, float]] = None

        # Python only keeps user options for optional dynamics feed-forward.
        # Servo lifecycle, counters and ownership live in the native core.
        self._servo_active: bool = False
        self._servo_opts: Optional[ServoOpts] = None

        # ---- Dynamics (gravity / friction compensation) state ----
        # All None until setup_dynamics() succeeds. Kept on the instance
        # so the per-tick compensation loop does not re-load the URDF.
        self._pin_model = None
        self._pin_data = None
        # End-effector frame used by FK/IK (move_p / move_l). Resolved in
        # setup_dynamics from the URDF ("tool_link" for the follower arm).
        self._eef_frame_id: Optional[int] = None
        self._eef_frame_name: Optional[str] = None
        self._dyn_gravity_vec: np.ndarray = np.array([0.0, 0.0, -9.81])
        # Per-joint motor models used for physical torque/MIT conversion.
        # Unknown models are rejected before any non-zero command is sent.
        self._dyn_motor_models: Optional[List[str]] = None
        # Per-joint torque clip (Nm). Defaults applied in setup_dynamics.
        self._dyn_tau_limit: Optional[np.ndarray] = None
        # Per-joint empirical torque gain applied right before sending. Used
        # to calibrate gravity-comp on real hardware when the Nm->raw coeff
        # is uncertain: bump it up until the arm just floats. Defaults to 1.0.
        self._dyn_torque_scale: np.ndarray = np.ones(self.num_joints)
        self._friction_params: Optional[FrictionParams] = None
        self._gravity_comp_active: bool = False
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
        core = getattr(self, "_core", None)
        if core is None:
            return self._state
        state = self._python_state_from_core(core.state)
        with self._state_lock:
            self._state = state
            self._dead_reason = core.dead_reason or None
        return state

    @staticmethod
    def _core_finish_mode(mode: str):
        return {
            "stop": pm.FinishMode.STOP,
            "brake": pm.FinishMode.BRAKE,
            "hold": pm.FinishMode.HOLD,
        }[mode]

    def _native_cancel_requested(self) -> bool:
        core = getattr(self, "_core", None)
        return bool(core is not None and core.cancel_requested)

    def _command_guard(self):
        """Serialize one hardware write with ESTOP/DEAD for this instance."""
        core = getattr(self, "_core", None)
        return core.command_guard() if core is not None else nullcontext()

    def _combined_abort_check(
        self, abort_check: Optional[Callable[[], bool]]
    ) -> Callable[[], bool]:
        def cancelled() -> bool:
            core = getattr(self, "_core", None)
            if self._native_cancel_requested():
                return True
            if core is not None and not core.stream_link_ok():
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
        
    @property
    def driver(self) -> pm.HightorqueSerial:
        """Escape hatch to the underlying ``HightorqueSerial`` instance."""
        return self._ht

    # ------------------------------------------------------------------
    #  State machine
    # ------------------------------------------------------------------
    @property
    def state(self) -> RobotState:
        """Current high-level state from the native concurrency core."""
        core = getattr(self, "_core", None)
        if core is not None:
            return self._sync_state_from_core()
        with self._state_lock:
            return self._state

    @property
    def state_verbose(self) -> bool:
        """Whether state transitions are printed to stdout (default off)."""
        return self._state_verbose

    @state_verbose.setter
    def state_verbose(self, value: bool) -> None:
        self._state_verbose = bool(value)

    def _set_state(self, new: RobotState) -> None:
        core = getattr(self, "_core", None)
        if core is not None and new not in (
            RobotState.DEAD, RobotState.ESTOP, RobotState.DISCONNECTED
        ):
            target = getattr(pm.RobotState, new.name)
            try:
                if core.state != target:
                    core.transition(target)
            except Exception as exc:
                raise RobotStateError(
                    f"invalid native state transition to {new.name}: {exc}"
                ) from exc
        with self._state_lock:
            old = self._state
            if old is new:
                return
            self._state = new
            if self._state_verbose:
                print(f"[FafuRobot] state: {old} -> {new}")

    def _require_ready(self, action: str, *, allow_disabled: bool = False) -> None:
        """Reject a command unless the controller can accept it."""
        st = self._state
        if st is RobotState.IDLE:
            if not self._stream_link_ok():
                raise RobotStateError(
                    f"{action} rejected: feedback was lost (DEAD); "
                    "call recover(confirm=True) after power is restored")
            return
        if st is RobotState.DEAD:
            detail = f": {self._dead_reason}" if self._dead_reason else ""
            raise RobotStateError(
                f"{action} rejected: feedback loss is latched (DEAD){detail}; "
                "call recover(confirm=True), then enable()")
        if st in (RobotState.DISABLED, RobotState.BRAKED):
            if allow_disabled:
                return
            try:
                if self.is_enabled:
                    self._set_state(RobotState.IDLE)
                    return
            except Exception:
                pass
            raise RobotStateError(
                f"{action} requires enabled motors; state={st.name}. "
                "Call enable() first")
        if st is RobotState.DISCONNECTED:
            raise RobotStateError(
                f"{action} failed: connection is closed")
        if st is RobotState.ESTOP:
            raise RobotStateError(
                f"{action} rejected: emergency stop is latched; call resume()")
        raise RobotStateError(
            f"{action} rejected: controller is busy (state={st})")

    def _enter_operation(self, action: str, busy_state: RobotState):
        """Acquire the single per-controller writer lease.

        Reads remain concurrent. A second writer fails immediately instead of
        interleaving frames; same-thread nested joint calls remain supported.
        """
        core = getattr(self, "_core", None)
        if core is not None:
            if action == "reset_zero":
                kind = pm.OperationKind.RAW_STREAM
            elif busy_state is RobotState.MOVING:
                kind = pm.OperationKind.JOINT_MOTION
            elif busy_state is RobotState.GRASPING:
                kind = (pm.OperationKind.GRASP
                        if action == "grasp"
                        else pm.OperationKind.GRIPPER_MOTION)
            elif busy_state is RobotState.GRAVITY_COMP:
                kind = pm.OperationKind.GRAVITY_COMP
            else:
                kind = pm.OperationKind.RAW_STREAM
            try:
                token = core.begin_operation(kind)
                if not core.stream_link_ok():
                    core.end_operation(token)
                    self._sync_state_from_core()
                    raise RobotStateError(
                        f"{action} rejected: {core.dead_reason}")
            except RobotStateError:
                raise
            except Exception as exc:
                raise RobotStateError(f"{action} rejected: {exc}") from exc

            owner = threading.get_ident()
            with self._state_lock:
                if self._op_depth == 0:
                    self._op_owner_thread_id = owner
                self._op_depth += 1
                self._state = busy_state
            return int(token)

        owner = threading.get_ident()
        with self._state_lock:
            if self._op_depth > 0:
                if self._op_owner_thread_id != owner:
                    raise RobotStateError(
                        f"{action} rejected: another thread owns the active "
                        f"control operation (state={self._state})."
                    )
                self._op_depth += 1
                return False
            self._require_ready(action)
            self._op_depth = 1
            self._op_owner_thread_id = owner
            self._set_state(busy_state)
            return True

    def _exit_operation(self, owns) -> None:
        core = getattr(self, "_core", None)
        if core is not None and not isinstance(owns, bool):
            try:
                core.end_operation(int(owns))
            finally:
                with self._state_lock:
                    if self._op_depth > 0:
                        self._op_depth -= 1
                    if self._op_depth == 0:
                        self._op_owner_thread_id = None
                    self._state = self._python_state_from_core(core.state)
            return

        owner = threading.get_ident()
        with self._state_lock:
            if self._op_depth <= 0:
                return
            if self._op_owner_thread_id != owner:
                raise RuntimeError("operation guard exited by a non-owner thread")
            self._op_depth -= 1
            if owns and self._op_depth == 0:
                self._op_owner_thread_id = None
                if self._state in (RobotState.MOVING, RobotState.GRASPING):
                    self._set_state(RobotState.IDLE)

    def _require_stream_command(
        self,
        action: str,
        *,
        allow_gravity_owner: bool = False,
    ) -> None:
        """Gate low-level streaming APIs without breaking internal loops.

        Direct callers must be in ``IDLE``.  A guarded composite move may call
        a low-level sender only from the same thread that owns the operation;
        the gravity-compensation loop gets the same narrowly-scoped exception.
        ESTOP/DEAD/DISCONNECTED are never bypassed.
        """
        owner = threading.get_ident()
        with self._state_lock:
            internal_move = (
                self._op_depth > 0
                and self._op_owner_thread_id == owner
                and self._state in (RobotState.MOVING, RobotState.GRASPING)
            )
            internal_gravity = (
                allow_gravity_owner
                and self._state is RobotState.GRAVITY_COMP
                and (
                    (self._gravity_comp_active
                     and self._gravity_comp_owner_thread_id == owner)
                    or (self._op_depth > 0
                        and self._op_owner_thread_id == owner)
                )
            )
        if not (internal_move or internal_gravity):
            self._require_ready(action)
            return
        if not self._stream_link_ok():
            raise RobotStateError(
                f"{action} rejected: motor feedback is stale (DEAD).")
        with self._state_lock:
            if self._state in (
                RobotState.ESTOP,
                RobotState.DEAD,
                RobotState.DISCONNECTED,
            ):
                raise RobotStateError(
                    f"{action} rejected: state={self._state}.")

    def _note_motor_seen(self, motor_id: int) -> None:
        self._motor_last_seen_monotonic[int(motor_id)] = time.monotonic()

    def _on_motor_states_updated(self, motor_ids: Iterable[int]) -> None:
        now = time.monotonic()
        for motor_id in motor_ids:
            self._motor_last_seen_monotonic[int(motor_id)] = now

    # ------------------------------------------------------------------
    #  Power / link loss -> DEAD (latched, survives power-return)
    # ------------------------------------------------------------------
    @property
    def dead_reason(self) -> Optional[str]:
        """Why the controller latched DEAD, or None."""
        core = getattr(self, "_core", None)
        if core is not None:
            return core.dead_reason or None
        return self._dead_reason

    def _enter_dead(self, reason: str) -> None:
        """Latch ``DEAD``: kill any streaming session so no further frames go
        out, remember the reason, and transition.  Sticky (like ESTOP): only
        :meth:`recover` leaves it.  This is the safety guarantee that a power
        blip cannot silently resume motion when power returns — the moment we
        notice the link is gone we stop commanding and refuse to restart until
        the user explicitly recovers.
        """
        if self._state in (RobotState.DEAD, RobotState.DISCONNECTED):
            return
        # Kill any streaming session so its loop stops sending immediately.
        self._servo_active = False
        self._last_cmd_turns = None
        self._dead_reason = reason
        self._set_state(RobotState.DEAD)
        print(f"[FafuRobot] DEAD latched: {reason}\n"
              "           Motion is disabled and all streaming stopped. The arm "
              "will NOT move on power-return; call recover(confirm=True) after "
              "power / CAN is restored.")

    def _stream_link_ok(self) -> bool:
        """Cheap per-tick liveness check for high-rate loops (servo_j /
        move_MIT / gravity comp).

        Every commanded joint must have fresh feedback; a healthy motor must
        never hide stale feedback from another commanded joint.
        """
        core = getattr(self, "_core", None)
        if core is not None:
            ok = bool(core.stream_link_ok())
            if not ok:
                self._servo_active = False
                self._sync_state_from_core()
            return ok

        try:
            async_rx = bool(self._ht.is_async_rx())
            polling = bool(self._ht.is_polling())
            if not async_rx and not polling:
                return True
        except Exception:
            return True

        now = time.monotonic()
        age_fn = getattr(self._ht, "get_state_age_ms", None)
        stale: List[Tuple[int, float]] = []
        for mid in self._joint_motor_ids:
            age_ms: Optional[float] = None
            if age_fn is not None:
                try:
                    driver_age = float(age_fn(mid))
                    if math.isfinite(driver_age):
                        age_ms = driver_age
                except Exception:
                    pass
            if age_ms is None:
                seen_at = self._motor_last_seen_monotonic.get(mid)
                if seen_at is not None:
                    age_ms = max(0.0, (now - seen_at) * 1000.0)
            if age_ms is None or age_ms > self._dead_rx_timeout_ms:
                stale.append((mid, float("inf") if age_ms is None else age_ms))

        if stale:
            detail = ", ".join(
                f"M{mid}=never" if not math.isfinite(age)
                else f"M{mid}={age:.0f}ms"
                for mid, age in stale
            )
            self._enter_dead(
                f"stale motor feedback: {detail} "
                f"(limit={self._dead_rx_timeout_ms:.0f}ms)"
            )
            return False
        return True

    def check_alive(self, *, fresh: bool = True, timeout: float = 0.1) -> bool:
        """Probe the joint motors for a live power / CAN link.

        Returns ``True`` only if every joint motor answers a state read.  If
        **any** fail to answer — indicating partial/full power, wiring, or CAN
        failure — the controller latches :attr:`RobotState.DEAD` (sticky) so no
        motion can be issued, and returns ``False``.  Call :meth:`recover`
        after restoring power.  No-op (returns ``False``) if already
        disconnected.
        """
        core = getattr(self, "_core", None)
        if core is not None:
            alive = bool(core.check_alive(bool(fresh), float(timeout)))
            self._sync_state_from_core()
            return alive

        if self._state is RobotState.DISCONNECTED:
            return False
        missing: List[int] = []
        for mid in self._joint_motor_ids:
            try:
                s = (self._ht.read_motor_state(mid, timeout) if fresh
                     else self._ht.get_cached_state(mid))
            except Exception:
                s = None
            if s is None:
                missing.append(mid)
            else:
                self._note_motor_seen(mid)
        if missing:
            self._enter_dead(
                f"motors {missing} did not respond to a state read "
                "(partial power/wiring/CAN failure?)"
            )
            return False
        return True

    def recover(self, *, confirm: bool = False) -> bool:
        """Leave ``DEAD`` after motor power / CAN has been restored.

        ★ SAFETY ★ This never moves the arm.  It verifies the motors respond
        again, restarts background RX / polling if needed, forces every motor
        to the safe non-energised ``STOP`` (``0x00``) mode, and transitions to
        ``DISABLED`` — **not** ``IDLE``.  You must then call :meth:`enable`
        explicitly to re-arm; ``enable`` holds the *current* measured pose (no
        jump), so the arm cannot lurch toward a stale pre-blackout target.

        ``confirm=True`` is required because this re-touches the bus after a
        fault; make sure the workspace is clear first.

        Returns ``True`` if the link is back and we moved to ``DISABLED``;
        ``False`` if the motors are still unresponsive (stays ``DEAD``).
        """
        core = getattr(self, "_core", None)
        if core is not None:
            if self._state is RobotState.DEAD and not confirm:
                raise RuntimeError(
                    "recover(confirm=True) required (safety): confirm the arm "
                    "is powered and the workspace is clear.")
            recovered = bool(core.recover(bool(confirm), 0.2))
            self._last_cmd_turns = None
            self._sync_state_from_core()
            return recovered

        with self._state_lock:
            if self._op_depth > 0:
                raise RobotStateError(
                    "recover rejected: wait for the in-flight operation to exit")

        if self._state is not RobotState.DEAD:
            print(f"[FafuRobot] recover: not in DEAD (state={self._state}); ignored.")
            return self._state is not RobotState.DEAD
        if not confirm:
            raise RuntimeError(
                "recover(confirm=True) required (safety): confirm the arm is "
                "powered and the workspace is clear before re-touching the bus.")
        # Restart background tasks if they dropped.
        try:
            if not self._ht.is_async_rx():
                self._ht.enable_async_rx()
                time.sleep(0.1)
        except Exception as e:
            print(f"[FafuRobot] recover: enable_async_rx failed: {e}")
        try:
            if not self._ht.is_polling():
                hz = float(self._cfg.control_rate_hz) if self._cfg.control_rate_hz else 50.0
                self._ht.start_state_polling(
                    list(self._cfg.motor_ids),
                    max(10.0, hz),
                    self._on_motor_states_updated,
                )
        except Exception as e:
            print(f"[FafuRobot] recover: start_state_polling failed: {e}")
        # Probe every configured motor; partial recovery is not safe.
        missing: List[int] = []
        for mid in self._cfg.motor_ids:
            try:
                if self._ht.read_motor_state(mid, 0.2) is not None:
                    self._note_motor_seen(mid)
                else:
                    missing.append(mid)
            except Exception:
                missing.append(mid)
        if missing:
            print(f"[FafuRobot] recover: motors {missing} still unresponsive; "
                  "check power/wiring/USB/CAN and retry. (state stays DEAD)")
            return False
        # Link is back. Force a known-safe non-energised state; never move.
        for mid in self._cfg.motor_ids:
            try:
                self._ht.stop(mid)
            except Exception:
                pass
        self._last_cmd_turns = None
        self._dead_reason = None
        self._set_state(RobotState.DISABLED)
        print("[FafuRobot] recover: link restored; motors left in STOP (0x00, "
              "free). Call enable() to re-arm (holds current pose, no jump) "
              "before commanding motion.")
        return True

    def sync_state(self) -> RobotState:
        """Reconcile the software state with live motor hardware and
        return the (possibly updated) :class:`RobotState`.

        Only reconciles the quiescent states (``IDLE`` <-> ``DISABLED``)
        based on :meth:`is_enabled`; the busy / latched states
        (``MOVING`` / ``SERVOING`` / ``GRASPING`` / ``GRAVITY_COMP`` /
        ``ESTOP`` / ``DISCONNECTED``) are left untouched.  Useful after a
        firmware watchdog trip or a manual driver poke.
        """
        if self._state in (RobotState.IDLE, RobotState.DISABLED):
            try:
                self._set_state(
                    RobotState.IDLE if self.is_enabled else RobotState.DISABLED
                )
            except Exception:
                pass
        return self._state

    @property
    def is_enabled(self) -> bool:
        """``True`` iff every motor is currently in position-control mode."""
        for mid in self._cfg.motor_ids:
            s = self._ht.get_cached_state(mid)
            if s is None:
                s = self._ht.read_motor_state(mid, 0.05)
            if s is None or s.mode != self.MODE_POSITION:
                return False
        return True

    # ------------------------------------------------------------------
    #  Power management
    # ------------------------------------------------------------------
    def enable(self, *, allow_motor_reset: bool = True) -> None:
        """Enable position control, including the native recovery sequence."""
        with self._state_lock:
            if self._op_depth > 0:
                raise RobotStateError(
                    "enable rejected: wait for the in-flight operation to exit")

        if self._state is RobotState.DISCONNECTED:
            raise RobotStateError("enable failed: connection is closed")
        if self._state is RobotState.DEAD:
            raise RobotStateError(
                "enable rejected: feedback loss is latched (DEAD); "
                "call recover(confirm=True)")
        if self._state in _BUSY_STATES:
            raise RobotStateError(
                f"enable rejected: controller is busy (state={self._state})")

        self._enable_impl(allow_motor_reset=allow_motor_reset)
        self._set_state(RobotState.IDLE)

    def _enable_impl(self, *, allow_motor_reset: bool = True) -> None:
        """Run the native position-mode recovery sequence."""
        core = getattr(self, "_core", None)
        if core is None:
            raise RuntimeError("native RobotCore is not initialized")

        options = pm.EnableOptions()
        options.allow_motor_reset = bool(allow_motor_reset)
        result = core.enable(options)
        self._sync_state_from_core()
        if result.success:
            return

        failed = list(result.failed_motor_ids)
        suffix = f"; failed motors={failed}" if failed else ""
        raise RuntimeError(f"enable failed: {result.message}{suffix}")

    def disable(self) -> None:
        """Disable motor output and allow the arm to move freely."""
        core = getattr(self, "_core", None)
        if core is None:
            raise RuntimeError("native RobotCore is not initialized")
        core.disable()
        self._last_cmd_turns = None
        self._sync_state_from_core()

    def brake(self) -> None:
        """Apply short-circuit braking to every configured motor."""
        core = getattr(self, "_core", None)
        if core is None:
            raise RuntimeError("native RobotCore is not initialized")
        core.brake()
        self._last_cmd_turns = None
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
        style: str = "scurve",
        duration: Optional[float] = None,
        timeout: float = 10.0,
    ) -> None:
        """Move every manipulator joint to a target configuration.

        Parameters
        ----------
        joint_angles : iterable of float
            Sequence of ``num_joints`` joint angles for the manipulator
            joints (in the order of :attr:`joint_motor_ids`).  The
            gripper, if any, is held at its current position.
        is_radians : bool, optional
            Interpret ``joint_angles`` in radians (default) or degrees.
        speed : int, optional
            Speed percentage in ``(0, 100]``.  Mapped linearly to a
            target average velocity ``(speed / 100) * 0.5`` turns/s.
            Defaults to 50 (~ 90 deg/s average).
        block : bool, optional
            * ``True`` (default): generate an S-curve trajectory and
              run it through :meth:`HightorqueSerial.run_control_loop`
              at ``cfg.control_rate_hz``; returns only after the
              trajectory finishes (plus a short settle window).
            * ``False``: send a single ``set_many_pos_vel_tqe`` frame
              and return immediately.
        tolerance : float, optional
            Joint tolerance for the *fast* one-shot blocking fallback
            used when TOPPRA / S-curve cannot run.  In radians (or
            degrees, matching ``is_radians``).  Defaults to ``0.01``.
        style : {"scurve", "linear", "acc"}, optional
            Trajectory style for ``block=True``:

            * ``"scurve"`` (default): host-streamed cosine ease-in/out
              profile via :meth:`_move_scurve` (smooth start/stop, uses
              ``set_many_pos_vel_tqe`` == ``pos_vel_MAXtqe``, no
              integral -> gravity steady-state error).
            * "linear": synchronized-arrival mode. It computes one velocity
              per joint, sends one group frame, then waits for settling.
            * "acc": per-joint set_pos_vel_acc (firmware
              trapezoidal *internal* position loop, MODE_POS_VEL_ACC).
              This is
              the channel the firmware drives with its own profile +
              (likely) integral action, so it is the one path that may
              reduce the gravity steady-state error for free.  Single
              shot per joint, then poll until settled
              (:meth:`_move_acc_sync`).
        duration : float, optional
            Only used when ``style="linear"``.  Explicit move duration
            (seconds).  When ``None`` it is derived from ``speed``.
        timeout : float, optional
            Only used when ``style="linear"`` and ``block=True``.  Max
            seconds to wait for the joints to settle before giving up.
        """
        angles = self._validate_joint_angles(joint_angles, is_radians)
        targets_turns: Dict[int, float] = {
            mid: angles[i] for i, mid in enumerate(self._joint_motor_ids)
        }
        speed = self._clamp_speed(speed)

        style = (style or "scurve").strip().lower()
        valid_styles = {"scurve", "linear", "acc"}
        if style not in valid_styles:
            raise ValueError(
                f"style must be one of {sorted(valid_styles)}, got {style!r}"
            )

        if style == "linear":
            tol_turns = abs(tolerance) / (_TWO_PI if is_radians else 360.0)
            self._move_linear_sync(
                targets_turns,
                speed_pct=speed,
                duration=duration,
                tolerance_turns=tol_turns,
                timeout_s=timeout,
                block=block,
            )
            return

        if style == "acc":
            tol_turns = abs(tolerance) / (_TWO_PI if is_radians else 360.0)
            self._move_acc_sync(
                targets_turns,
                speed_pct=speed,
                tolerance_turns=tol_turns,
                timeout_s=timeout,
                block=block,
            )
            return

        if block:
            self._move_scurve(targets_turns, speed_pct=speed)
            return

        # block=False: single shot, no waiting.
        v_avg = (speed / 100.0) * _VEL_AVG_MAX_TPS
        cmds = self._build_many_cmds_holding_others(targets_turns, vel_rps=v_avg)
        with self._command_guard():
            self._ht.set_many_pos_vel_tqe(
                cmds,
                pm.PosUnit.Turns,
                max(self._cfg.motor_ids),
                0.05,
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
            Interpret ``path`` in radians (default) or degrees.
        max_jntvel, max_jntacc : list of float, optional
            Per-joint velocity and acceleration limits (passed straight
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
                is_radians=is_radians,
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
            Interpret ``path`` in radians (default) or degrees.
        max_jntvel, max_jntacc : list of float, optional
            Per-joint velocity and acceleration limits (passed to TOPPRA).
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
        _to_rad = 1.0 if is_radians else (math.pi / 180.0)
        prev = None
        for jnt_values in interpolated:
            jv = np.asarray(jnt_values, dtype=float)
            vel = np.zeros(n) if prev is None else (jv - prev) / dt
            prev = jv
            if gravity_ff and self.has_dynamics:
                tau = self.compute_compensation_torque(
                    jv * _to_rad, vel * _to_rad, friction=False)
            else:
                tau = np.zeros(n)
            self.move_MIT(
                jv, vel, tau, kp=kp, kd=kd,
                is_radians=is_radians, apply_torque_scale=True, timeout=0.0)
            time.sleep(dt)
        # Re-assert the final pose briefly so the arm settles on target.
        for _ in range(3):
            jv = np.asarray(interpolated[-1], dtype=float)
            if gravity_ff and self.has_dynamics:
                tau = self.compute_compensation_torque(
                    jv * _to_rad, np.zeros(n), friction=False)
            else:
                tau = np.zeros(n)
            self.move_MIT(jv, np.zeros(n), tau, kp=kp, kd=kd,
                          is_radians=is_radians, apply_torque_scale=True,
                          timeout=0.0)
            time.sleep(dt)

    # ------------------------------------------------------------------
    #  Servo (online streaming) control
    # ------------------------------------------------------------------
    # Timing-sensitive safety and protocol work lives in RobotCore. Python
    # only maps public options and computes optional dynamics feed-forward.
    def servo_start(self, opts: Optional[ServoOpts] = None) -> None:
        """Start a caller-driven Servo session.

        The native core owns the watchdog, single-writer lease, step clamp,
        lag monitor, command channel and session cleanup.
        """
        core = getattr(self, "_core", None)
        if core is None:
            raise RuntimeError("native RobotCore is not initialized")
        opts = ServoOpts(**vars(opts)) if opts is not None else ServoOpts()
        native = pm.ServoOptions()
        native.watchdog_ms = int(opts.watchdog_ms)
        native.max_velocity_rad_s = float(opts.max_vel)
        native.max_step_rad = float(opts.max_step_rad)
        native.max_lag_rad = float(opts.max_lag_rad)
        native.nominal_rate_hz = float(opts.rate_hz)
        native.input_is_radians = bool(opts.is_radians)
        native.feedforward_velocity = bool(opts.feedforward_vel)
        native.lookahead_time_s = float(opts.lookahead_time)
        native.lag_abort_consecutive = int(opts.lag_abort_consecutive)
        native.channel = (
            pm.ServoChannel.MIT if opts.use_mit
            else pm.ServoChannel.POSITION)
        if opts.mit_kp is not None:
            native.mit_kp = (
                [float(opts.mit_kp)] if np.isscalar(opts.mit_kp)
                else [float(x) for x in opts.mit_kp])
        if opts.mit_kd is not None:
            native.mit_kd = (
                [float(opts.mit_kd)] if np.isscalar(opts.mit_kd)
                else [float(x) for x in opts.mit_kd])

        models = opts.motor_models or self._dyn_motor_models
        if opts.use_mit and models is None:
            raise ValueError(
                "MIT servo requires exact per-joint motor_models; pass "
                "ServoOpts(motor_models=[...]) or call set_motor_models()")
        if models is not None:
            self.set_motor_models(models)
        core.servo_start(native)
        self._last_cmd_turns = None
        self._servo_opts = opts
        self._servo_active = True
        self._sync_state_from_core()

    def servo_j(self, target_angles: Iterable[float]) -> bool:
        """Send one non-blocking joint target through the native Servo core."""
        core = getattr(self, "_core", None)
        if core is None:
            raise RuntimeError("native RobotCore is not initialized")

        values = [float(x) for x in target_angles]
        torque_ff: List[float] = []
        opts = self._servo_opts
        if (opts is not None and opts.use_mit and opts.mit_gravity_ff
                and self.has_dynamics and len(values) == self.num_joints):
            q_rad = np.asarray(values, dtype=float)
            if not opts.is_radians:
                q_rad = np.radians(q_rad)
            tau = self.compute_compensation_torque(
                q_rad, np.zeros(self.num_joints), friction=False)
            torque_ff = [
                float(x) for x in tau * self._dyn_torque_scale
            ]

        result = core.servo_tick(values, torque_ff)
        self._servo_active = bool(core.is_servoing)
        self._sync_state_from_core()
        if not result.sent and result.message:
            print(f"[FafuRobot] servo_j: {result.message}")
        return bool(result.sent and not result.aborted)

    def servo_end(self, finish_mode: str = "hold") -> None:
        """End the Servo session with hold, brake or stop."""
        if finish_mode not in {"stop", "brake", "hold"}:
            raise ValueError(
                "finish_mode must be one of ['brake', 'hold', 'stop']")
        core = getattr(self, "_core", None)
        if core is None:
            raise RuntimeError("native RobotCore is not initialized")

        core.servo_end(self._core_finish_mode(finish_mode))
        self._servo_active = bool(core.is_servoing)
        self._sync_state_from_core()

    @property
    def is_servoing(self) -> bool:
        """Whether a Servo session is currently active."""
        core = getattr(self, "_core", None)
        return bool(core.is_servoing) if core is not None else self._servo_active

    @property
    def servo_lag_count(self) -> int:
        """Lag-tripped ticks in the current or most recent Servo session."""
        core = getattr(self, "_core", None)
        return int(core.servo_summary().lag_count) if core is not None else 0

    @property
    def servo_clamp_count(self) -> int:
        """Step-clamped ticks in the current or most recent Servo session."""
        core = getattr(self, "_core", None)
        return int(core.servo_summary().clamp_count) if core is not None else 0

    @property
    def servo_aborted_reason(self) -> Optional[str]:
        """Reason for the most recent automatic Servo abort, if any."""
        core = getattr(self, "_core", None)
        if core is None:
            return None
        return core.servo_summary().aborted_reason or None

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
            arm).  ``model.nq`` must equal :attr:`num_joints`.
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
            params.  Defaults to :meth:`FrictionParams.reference_6dof`.
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
        if not _PIN_EXIST:
            raise RuntimeError(
                "gravity/dynamics need the 'pinocchio' package, which is "
                "not installed.\n"
                "  - conda:  conda install -c conda-forge pinocchio\n"
                "  - linux:  pip install pin\n"
                "  (Windows pip wheels are unreliable; conda-forge or WSL "
                "is the smooth path.)\n"
                "Friction-only compensation does NOT need pinocchio."
            )

        resolved = self._resolve_urdf_path(urdf_path)
        if resolved is None:
            raise RuntimeError(
                "setup_dynamics: could not find a URDF. Pass urdf_path "
                "explicitly, or drop one under "
                "'<package>/fafu_robot_description/'."
            )

        try:
            model = pin.buildModelFromUrdf(resolved)
        except Exception as e:
            raise RuntimeError(f"setup_dynamics: failed to load URDF "
                               f"{resolved!r}: {e}") from e

        if model.nq != self.num_joints:
            raise RuntimeError(
                f"setup_dynamics: URDF DoF ({model.nq}) != num_joints "
                f"({self.num_joints}). The URDF must describe exactly the "
                f"{self.num_joints} manipulator joints (gripper excluded), "
                f"as a simple revolute chain."
            )

        self._pin_model = model
        self._pin_data = model.createData()
        self._resolve_eef_frame(model, eef_frame)
        self._dyn_gravity_vec = np.asarray(list(gravity_vec), dtype=float)
        if self._dyn_gravity_vec.shape != (3,):
            raise ValueError("gravity_vec must have exactly 3 elements")

        if motor_models is not None:
            if len(motor_models) != self.num_joints:
                raise ValueError(
                    f"motor_models must have {self.num_joints} entries "
                    f"(one per joint), got {len(motor_models)}")
            self.set_motor_models(motor_models)
        else:
            self._dyn_motor_models = None
            if self._core is not None:
                self._core.set_joint_motor_models([""] * self.num_joints)
            print("[FafuRobot] setup_dynamics: no motor_models given; "
                  "non-zero torque/MIT output remains disabled until exact "
                  "models are configured.")

        if tau_limit is not None:
            tl = np.asarray(list(tau_limit), dtype=float)
            if tl.shape != (self.num_joints,):
                raise ValueError(
                    f"tau_limit must have {self.num_joints} elements")
            self._dyn_tau_limit = np.abs(tl)
        else:
            ref = np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])
            if self.num_joints == 6:
                self._dyn_tau_limit = ref
            else:
                # Unknown geometry: pick a conservative blanket cap.
                self._dyn_tau_limit = np.full(self.num_joints, 5.0)

        self.set_torque_scale(torque_scale if torque_scale is not None else 1.0)

        self._friction_params = friction or FrictionParams.reference_6dof()

        print(f"[FafuRobot] dynamics ready: URDF={os.path.basename(resolved)}, "
              f"dof={model.nq}, eef_frame={self._eef_frame_name!r}, "
              f"gravity={self._dyn_gravity_vec.tolist()}, "
              f"tau_limit={self._dyn_tau_limit.tolist()}, "
              f"torque_scale={self._dyn_torque_scale.tolist()}")

    def _resolve_eef_frame(self, model, eef_frame: Optional[str]) -> None:
        """Pick and cache the end-effector frame used by FK/IK.

        Preference: explicit argument, tool_link, then the child frame of the
        last actuated joint.
        """
        candidates: List[str] = []
        if eef_frame:
            candidates.append(eef_frame)
        candidates.append("tool_link")
        # Last joint name in the chain (joint1..jointN for this arm).
        try:
            last_joint = model.names[model.njoints - 1]
            candidates.append(last_joint)
        except Exception:
            pass

        for name in candidates:
            if model.existFrame(name):
                self._eef_frame_name = name
                self._eef_frame_id = model.getFrameId(name)
                if eef_frame and name != eef_frame:
                    print(f"[FafuRobot] setup_dynamics: requested eef_frame "
                          f"{eef_frame!r} not found; using {name!r}.")
                return

        # Last resort: the very last frame in the model.
        self._eef_frame_id = model.nframes - 1
        self._eef_frame_name = model.frames[self._eef_frame_id].name
        print(f"[FafuRobot] setup_dynamics: no tool frame found; using last "
              f"frame {self._eef_frame_name!r} as end effector.")

    def set_motor_models(self, motor_models: Iterable[str]) -> None:
        """Configure exact per-joint models for torque and MIT conversion."""
        if self.state in _BUSY_STATES:
            raise RobotStateError(
                "motor models cannot change during a control operation")
        models = [str(model) for model in motor_models]
        if len(models) != self.num_joints or any(not model for model in models):
            raise ValueError(
                f"motor_models must contain {self.num_joints} non-empty names")
        core = getattr(self, "_core", None)
        if core is not None:
            core.set_joint_motor_models(models)
        self._dyn_motor_models = models

    def set_torque_scale(self, scale: "float | Iterable[float]") -> None:
        """Set the empirical per-joint torque gain (see ``torque_scale`` in
        :meth:`setup_dynamics`).  Accepts a scalar or a per-joint list.

        Can be called live (e.g. between calibration runs) without
        reloading the dynamics model.
        """
        arr = np.asarray(scale, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self.num_joints, float(arr))
        if arr.shape != (self.num_joints,):
            raise ValueError(
                f"torque_scale must be a scalar or {self.num_joints} values")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(
                "torque_scale values must be finite and non-negative")
        self._dyn_torque_scale = arr

    def tau_to_raw(self, tau: Iterable[float]) -> np.ndarray:
        """Convert per-joint torque in Nm with the native calibration table."""
        values = [float(x) for x in tau]
        if len(values) != self.num_joints:
            raise ValueError(f"tau must have {self.num_joints} elements")
        models = self._dyn_motor_models or [""] * self.num_joints
        return np.asarray(
            pm.torques_to_raw(values, models, 1.0), dtype=np.int64)

    @property
    def has_dynamics(self) -> bool:
        """``True`` once :meth:`setup_dynamics` has loaded a model."""
        return self._pin_model is not None

    def _require_dynamics(self) -> None:
        if self._pin_model is None:
            raise RuntimeError(
                "dynamics model not loaded; call setup_dynamics() first")

    @staticmethod
    def _resolve_urdf_path(urdf_path: Optional[str]) -> Optional[str]:
        """Find a URDF: explicit arg → vendored copy under the package."""
        if urdf_path:
            return urdf_path if os.path.exists(urdf_path) else None

        candidates: List[str] = []
        # vendored URDF, for a self-contained deployment.
        desc = os.path.join(_HERE, "fafu_robot_description")
        if os.path.isdir(desc):
            for fn in sorted(os.listdir(desc)):
                if fn.endswith(".urdf"):
                    candidates.append(os.path.join(desc, fn))
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    # ------------------------------------------------------------------
    #  Kinematics (FK / IK) -- needs setup_dynamics() (pinocchio + URDF)
    # ------------------------------------------------------------------
    def _require_kinematics(self) -> None:
        self._require_dynamics()
        if self._eef_frame_id is None:
            raise RuntimeError(
                "end-effector frame not resolved; call setup_dynamics() "
                "(optionally with eef_frame=...) first")

    def _q_in(self, q: Optional[Iterable[float]], is_radians: bool) -> np.ndarray:
        """Normalize a joint vector arg to a radians ndarray of length
        ``num_joints`` (defaults to the live measured pose)."""
        if q is None:
            return self.get_joint_values()
        arr = np.asarray(list(q), dtype=float)
        if arr.size != self.num_joints:
            raise ValueError(
                f"expected {self.num_joints} joint values, got {arr.size}")
        return arr if is_radians else np.deg2rad(arr)

    @staticmethod
    def _rot_from_arg(rot, is_euler: bool, is_radians: bool) -> np.ndarray:
        """Accept either a 3x3 rotation matrix or an Euler/RPY triple and
        return a 3x3 rotation matrix."""
        if rot is None:
            return np.eye(3)
        arr = np.asarray(rot, dtype=float)
        if is_euler or arr.shape == (3,):
            rpy = arr.reshape(3)
            if not is_radians:
                rpy = np.deg2rad(rpy)
            return np.asarray(pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2]),
                              dtype=float)
        if arr.shape != (3, 3):
            raise ValueError(
                "rot must be a 3x3 rotation matrix or a length-3 RPY triple")
        return arr

    def _fk_se3(self, q_rad: np.ndarray):
        """Internal: return the end-effector ``pin.SE3`` for ``q`` (rad)."""
        pin.forwardKinematics(self._pin_model, self._pin_data, q_rad)
        pin.updateFramePlacements(self._pin_model, self._pin_data)
        return self._pin_data.oMf[self._eef_frame_id]

    def _joint_limits_rad(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Per-joint (lower, upper) soft limits in radians, or ``None`` if
        any joint has no configured limit."""
        lo = np.empty(self.num_joints)
        hi = np.empty(self.num_joints)
        for i, mid in enumerate(self._joint_motor_ids):
            try:
                lim = self.get_limit(mid, is_radians=True)
            except Exception:
                lim = None
            if lim is None:
                return None
            lo[i], hi[i] = lim
        return lo, hi

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
        self._require_kinematics()
        qv = self._q_in(q, is_radians)
        oMf = self._fk_se3(qv)
        pos = np.asarray(oMf.translation, dtype=float).copy()
        rot = np.asarray(oMf.rotation, dtype=float).copy()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return {
            "position": pos,
            "rotation": rot,
            "rpy": np.asarray(pin.rpy.matrixToRpy(rot), dtype=float),
            "transform": T,
            "q": qv,
        }

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
        self._require_kinematics()
        R = self._rot_from_arg(target_rotation, is_euler, is_radians)
        p = np.asarray(list(target_position), dtype=float).reshape(3)
        oMdes = pin.SE3(R, p)

        limits = self._joint_limits_rad() if clamp_limits else None

        if multi_init:
            q_sol = self._ik_multi_init(oMdes, num_attempts, max_iter, eps,
                                        damping, adaptive_damping, limits)
        else:
            seed = (self.get_joint_values() if init_q is None
                    else self._q_in(init_q, is_radians))
            q_sol = self._ik_single(oMdes, seed, max_iter, eps, damping,
                                    adaptive_damping, limits)

        if q_sol is None:
            return None
        return q_sol if is_radians else np.rad2deg(q_sol)

    def _ik_single(self, oMdes, seed, max_iter, eps, damping,
                   adaptive_damping, limits) -> Optional[np.ndarray]:
        """Single-seed damped least-squares loop (q in/out are rad)."""
        q = np.asarray(seed, dtype=float).copy()
        fid = self._eef_frame_id
        dt = 1e-1
        err_norm = float("inf")
        for _ in range(max_iter):
            pin.forwardKinematics(self._pin_model, self._pin_data, q)
            pin.updateFramePlacements(self._pin_model, self._pin_data)
            iMd = self._pin_data.oMf[fid].actInv(oMdes)
            err = pin.log(iMd).vector
            err_norm = float(np.linalg.norm(err))
            if err_norm < eps:
                return q
            J = pin.computeFrameJacobian(self._pin_model, self._pin_data, q,
                                         fid, pin.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            lam = (damping * (1.0 + 1.0 / (err_norm + 0.1))
                   if adaptive_damping else damping)
            JJT = J.dot(J.T) + (lam ** 2) * np.eye(6)
            try:
                alpha = np.linalg.solve(JJT, err)
            except np.linalg.LinAlgError:
                return None
            v = -J.T.dot(alpha)
            v_norm = np.linalg.norm(v)
            if v_norm > 10.0:
                v *= 10.0 / v_norm
            q = pin.integrate(self._pin_model, q, v * dt)
            if limits is not None:
                lo, hi = limits
                if np.any(q < lo) or np.any(q > hi):
                    return None
        return None

    def _ik_multi_init(self, oMdes, num_attempts, max_iter, eps, damping,
                       adaptive_damping, limits) -> Optional[np.ndarray]:
        """Try several seeds; return the configuration with the smallest
        Cartesian error (early-out once within ``eps``)."""
        seeds: List[np.ndarray] = []
        try:
            seeds.append(self.get_joint_values())
        except Exception:
            pass
        seeds.append(np.zeros(self.num_joints))
        if limits is not None:
            lo, hi = limits
            seeds.append((lo + hi) / 2.0)
            rng = np.random.default_rng()
            while len(seeds) < num_attempts:
                seeds.append(rng.uniform(lo, hi))
        else:
            while len(seeds) < num_attempts:
                seeds.append(np.random.uniform(
                    -np.pi / 4, np.pi / 4, self.num_joints))

        best_q = None
        best_err = float("inf")
        p_des = np.asarray(oMdes.translation, dtype=float)
        for seed in seeds[:num_attempts]:
            q = self._ik_single(oMdes, seed, max_iter, eps, damping,
                                 adaptive_damping, limits)
            if q is None:
                continue
            actual = np.asarray(self._fk_se3(q).translation, dtype=float)
            err = float(np.linalg.norm(actual - p_des))
            if err < best_err:
                best_err, best_q = err, q
            if err < eps:
                return q
        return best_q

    def get_gravity(self, q: Optional[Iterable[float]] = None) -> np.ndarray:
        """Generalized gravity torque ``G(q)`` (Nm), one entry per joint.

        Parameters
        ----------
        q : iterable of float, optional
            Joint angles (rad).  Defaults to the live measured pose.
        """
        self._require_dynamics()
        qv = (self.get_joint_values() if q is None
              else np.asarray(list(q), dtype=float))
        original_linear = self._pin_model.gravity.linear.copy()
        self._pin_model.gravity.linear = self._dyn_gravity_vec
        try:
            g = pin.computeGeneralizedGravity(self._pin_model, self._pin_data, qv)
        finally:
            self._pin_model.gravity.linear = original_linear
        return np.asarray(g, dtype=float)

    def get_mass_matrix(self, q: Optional[Iterable[float]] = None) -> np.ndarray:
        """Joint-space inertia matrix ``M(q)`` (CRBA)."""
        self._require_dynamics()
        qv = (self.get_joint_values() if q is None
              else np.asarray(list(q), dtype=float))
        M = pin.crba(self._pin_model, self._pin_data, qv)
        n = len(qv)
        return np.asarray(M[:n, :n], dtype=float)

    def get_coriolis(
        self,
        q: Optional[Iterable[float]] = None,
        v: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Coriolis/centrifugal matrix ``C(q, v)``."""
        self._require_dynamics()
        qv = (self.get_joint_values() if q is None
              else np.asarray(list(q), dtype=float))
        vv = (self.get_joint_velocities() if v is None
              else np.asarray(list(v), dtype=float))
        C = pin.computeCoriolisMatrix(self._pin_model, self._pin_data, qv, vv)
        return np.asarray(C, dtype=float)

    def get_dynamics(
        self,
        q: Optional[Iterable[float]] = None,
        v: Optional[Iterable[float]] = None,
        a: Optional[Iterable[float]] = None,
    ) -> np.ndarray:
        """Full inverse dynamics ``tau = M(q)a + C(q,v)v + G(q)`` (RNEA)."""
        self._require_dynamics()
        qv = (self.get_joint_values() if q is None
              else np.asarray(list(q), dtype=float))
        vv = (self.get_joint_velocities() if v is None
              else np.asarray(list(v), dtype=float))
        av = (np.zeros(self.num_joints) if a is None
              else np.asarray(list(a), dtype=float))
        original_linear = self._pin_model.gravity.linear.copy()
        self._pin_model.gravity.linear = self._dyn_gravity_vec
        try:
            tau = pin.rnea(self._pin_model, self._pin_data, qv, vv, av)
        finally:
            self._pin_model.gravity.linear = original_linear
        return np.asarray(tau, dtype=float)

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
        if vel is None:
            vel = self.get_joint_velocities()
        vel = np.asarray(list(vel), dtype=float)

        p = params or self._friction_params or FrictionParams.reference_6dof()
        fc = np.asarray(p.fc, dtype=float)
        fv = np.asarray(p.fv, dtype=float)
        if fc.shape != vel.shape or fv.shape != vel.shape:
            raise ValueError(
                f"friction fc/fv length {fc.shape}/{fv.shape} != velocity "
                f"length {vel.shape}")

        full = fc * np.sign(vel) + fv * vel
        low = fv * vel
        return np.where(np.abs(vel) < p.vel_threshold, low, full)

    def compute_compensation_torque(
        self,
        q: Optional[Iterable[float]] = None,
        v: Optional[Iterable[float]] = None,
        *,
        friction: bool = True,
    ) -> np.ndarray:
        """Gravity (+ optional friction) feed-forward torque, clipped (Nm).

        ``tau = clip( G(q) + [friction(v)], ±tau_limit )``.
        """
        self._require_dynamics()
        tau = self.get_gravity(q)
        if friction:
            tau = tau + self.get_friction_compensation(v)
        if self._dyn_tau_limit is not None:
            tau = np.clip(tau, -self._dyn_tau_limit, self._dyn_tau_limit)
        return tau

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
                self._ht.set_many_mit(
                    list(self._joint_motor_ids),
                    zeros, zeros,
                    [int(x) for x in raw],
                    [0] * self.num_joints,
                    [0] * self.num_joints,
                    pm.PosUnit.Radians,
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
            Per-joint target position (radians by default). Feeds the MIT
            position term; only matters when ``kp != 0``. Soft limits applied.
        vel : iterable of float
            Per-joint target velocity (rad/s by default). Only matters when
            ``kd != 0``. Converted to turns/s internally.
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
            Interpret ``pos``/``vel`` as radians / rad/s (default) or deg / deg/s.
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
        self._require_stream_command("move_MIT")
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

        # pos: keep in the caller's unit and let the driver convert; vel: the
        # driver wants turns/s, so convert from rad/s (or deg/s).
        if is_radians:
            unit = pm.PosUnit.Radians
            vel_tps = vel / _TWO_PI
        else:
            unit = pm.PosUnit.Degrees
            vel_tps = vel / 360.0

        motor_ids = list(self._joint_motor_ids)
        with self._command_guard():
            return self._ht.set_many_mit(
                motor_ids,
                [float(x) for x in pos],
                [float(x) for x in vel_tps],
                [int(x) for x in tau_raw],
                kp_out,
                kd_out,
                unit,
                max(motor_ids),
                float(timeout),
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
        vel_abort_rps: float = 4.0,
        vel_lpf_alpha: float = 0.3,
        hold_on_release: bool = True,
        move_vel_thresh: float = 0.15,
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
        vel_abort_rps : float, optional
            **Runaway guard** (safety): if any joint's |velocity| exceeds this
            (rev/s) the loop aborts and re-holds — catches divergence before
            the arm flings itself.  ``0`` disables.  Default ``4.0``.
        vel_lpf_alpha : float, optional
            Low-pass on the velocity used by the ``b_soft`` damping term
            (0..1, lower = smoother).  The raw firmware velocity is noisy; a
            light filter lets ``b_soft`` actually damp the pure-P limit-cycle
            (e.g. J1 hunting) instead of chattering.  Default ``0.3``.
        hold_on_release : bool, optional
            **Lead-through teach mode.**  When ``True`` (default) the hold
            target ``q_des`` continuously follows the live pose for any joint
            whose speed exceeds ``move_vel_thresh`` (so while you drag it the
            spring force is ~0 and the arm is weightless), and freezes the
            instant the joint stops — the spring + integral then lock it
            exactly where you let go.  Drag again to a new pose and it holds
            there.  Set ``False`` for the classic fixed-``q_des`` behaviour
            (springs back to the start pose when pushed away).
        move_vel_thresh : float, optional
            Speed (rev/s) above which a joint is considered "being dragged"
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
        # State-machine guard. A dry-run only computes/prints torque and
        # never touches the motors, so it is allowed from any live state;
        # a live run enables the motors itself (allow_disabled) but must
        # not start while disconnected / estopped / busy.
        if not dry_run:
            self._require_ready("start_gravity_compensation", allow_disabled=True)
        if not dry_run and not self.is_enabled:
            print("[FafuRobot] start_gravity_compensation: enabling motors ...")
            self.enable()

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
            arr = np.atleast_1d(np.asarray(list(val), dtype=float))
            if arr.shape == (1,):                       # scalar -> all joints
                arr = np.full(self.num_joints, arr[0], dtype=float)
            if arr.shape != (self.num_joints,):
                raise ValueError(
                    f"{name} must be a scalar or have {self.num_joints} "
                    f"elements, got {arr.shape}")
            return arr

        k_soft_np = _broadcast(k_soft, "k_soft")
        b_soft_np = _broadcast(b_soft, "b_soft")
        i_soft_np = _broadcast(i_soft, "i_soft")
        q_des_np: Optional[np.ndarray] = None
        if k_soft_np is not None or i_soft_np is not None:
            q_des_np = (np.asarray(list(q_des), dtype=float)
                        if q_des is not None else self.get_joint_values().copy())
            print(f"[FafuRobot] impedance net ON: "
                  f"K={(k_soft_np.tolist() if k_soft_np is not None else None)} "
                  f"B={(b_soft_np.tolist() if b_soft_np is not None else None)} "
                  f"Ki={(i_soft_np.tolist() if i_soft_np is not None else None)} "
                  f"q_des(deg)={np.degrees(q_des_np).round(1).tolist()}")

        if dry_run:
            print("[FafuRobot] gravity-comp DRY-RUN: computing torque, "
                  "NOT sending to motors.")
        else:
            print("[FafuRobot] gravity-comp LIVE: arm will go weightless. "
                  "Keep a hand on it / the E-stop. Ctrl+C to stop.")

        period = 1.0 / max(1.0, float(rate_hz))
        # Anti-convulsion: limit how fast the commanded torque may change
        # between ticks (slew) and low-pass it.  With high torque_scale the
        # K/B impedance + friction sign terms can otherwise flip the command
        # by hundreds of raw counts in one tick, which excites a limit-cycle
        # oscillation ("convulsion") through the USB-CAN latency.
        max_dtau = (tau_slew_per_s * period
                    if tau_slew_per_s and tau_slew_per_s > 0 else None)
        alpha = float(np.clip(tau_lpf_alpha, 0.0, 1.0))
        tau_prev: Optional[np.ndarray] = None
        # Integral state (kills the static-friction "droop" deadband without
        # the high-frequency gain that makes a large K diverge on this
        # high-latency loop).  Only integrates while (nearly) at rest so it
        # stays compliant while you push, and never winds up.
        integ = np.zeros(self.num_joints, dtype=float)
        rest_thresh = 0.05            # rev/s: below this = "at rest", integrate
        # Per-joint lead-through state machine: a joint enters "dragging" only
        # after its speed has stayed ABOVE `move_vel_thresh` continuously for
        # `enter_time` seconds, and LOCKS again once its speed has stayed below
        # that threshold continuously for `settle_time` seconds.
        #
        # The enter DEBOUNCE is the key fix for "gradually droops, never
        # stabilises": under gravity a joint creeps down in stick-slip jerks,
        # and a single slip spike easily exceeds `move_vel_thresh` for one
        # tick.  With an instantaneous enter-test that spike would flip the
        # joint into "dragging", snap q_des down to the (sagged) live pose and
        # zero the integral -- so every micro-slip ratchets the hold point
        # lower and the joint walks down forever.  Requiring the fast motion to
        # PERSIST for `enter_time` rejects those momentary slips (they stop
        # almost immediately) while a real hand-drag (sustained) still engages.
        dragging = np.zeros(self.num_joints, dtype=bool)
        slow_time = np.zeros(self.num_joints, dtype=float)
        fast_time = np.zeros(self.num_joints, dtype=float)
        enter_time = 0.08            # s above move_vel_thresh -> start drag
        settle_time = 0.25           # s below move_vel_thresh -> lock & hold
        # Filtered velocity for the damping (B) term: the raw firmware
        # velocity is quantized/noisy, and feeding it straight into B*(-v)
        # either does nothing or chatters.  A light LPF gives B clean phase to
        # actually damp the P limit-cycle.
        v_alpha = float(np.clip(vel_lpf_alpha, 0.01, 1.0))
        v_filt = np.zeros(self.num_joints, dtype=float)
        t0 = time.monotonic()
        last_t = t0
        last_log = t0
        self._gravity_comp_active = True
        self._gravity_comp_owner_thread_id = threading.get_ident()
        if not dry_run:
            if self._core is not None:
                self._gravity_core_token = int(
                    self._core.begin_operation(
                        pm.OperationKind.GRAVITY_COMP))
                with self._state_lock:
                    self._state = RobotState.GRAVITY_COMP
            else:
                self._set_state(RobotState.GRAVITY_COMP)
        try:
            while True:
                tick_start = time.monotonic()
                if self._combined_abort_check(abort_check)():
                    print("[FafuRobot] gravity-comp: cancellation -> stop")
                    break
                # Power/CAN-loss guard: stop feeding torque the instant the
                # link drops so power-return cannot resume the float loop.
                if not self._stream_link_ok():
                    break
                if duration is not None and (tick_start - t0) >= duration:
                    break

                q = self.get_joint_values()
                v = self.get_joint_velocities()
                # ---- runaway / divergence guard (safety) ----
                if vel_abort_rps and vel_abort_rps > 0:
                    vmax = float(np.max(np.abs(v)))
                    if vmax > vel_abort_rps:
                        print(f"[FafuRobot] gravity-comp: RUNAWAY guard "
                              f"(|v|={vmax:.2f} > {vel_abort_rps} rev/s) "
                              f"-> abort & re-hold")
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
                    fast = absv > move_vel_thresh
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
                    hold_mask = absv < rest_thresh
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
            self._gravity_comp_active = False
            self._gravity_comp_owner_thread_id = None
            token = self._gravity_core_token
            self._gravity_core_token = None
            if token is not None and self._core is not None:
                self._core.end_operation(token)
                current = self._sync_state_from_core()
            else:
                current = self._state
            safety_latched = current in (
                RobotState.ESTOP,
                RobotState.DEAD,
                RobotState.DISCONNECTED,
            )
            # Once safety is latched, RobotCore has already serialized STOP.
            # Do not emit a later brake/hold/home command from this cleanup.
            if dry_run:
                pass
            elif safety_latched:
                print(f"[FafuRobot] gravity-comp stopped ({current.name}); "
                      "hardware cleanup skipped after safety STOP.")
            elif home_on_exit:
                # Sequence: brake (arrest motion gently) -> pause -> re-enable
                # position mode -> slow S-curve home to 0 -> brake at home.
                # NOTE: brake is mode 0x0F; position frames only actuate in
                # 0x0A, so we MUST re-enable before go_home or it won't move.
                self._brake_joints()
                pause = max(0.0, float(home_brake_pause))
                if pause > 0.0:
                    print(f"[FafuRobot] gravity-comp: braked; pausing "
                          f"{pause:.1f}s before homing ...")
                    time.sleep(pause)
                else:
                    print("[FafuRobot] gravity-comp: braked.")
                print("[FafuRobot] gravity-comp: switching to position & "
                      f"returning home (0 rad) @ speed={home_speed} ... "
                      "(do NOT press Ctrl+C again)")
                try:
                    self.enable()
                    self.go_home(speed=home_speed, block=True)
                    print("[FafuRobot] gravity-comp homed to 0 rad.")
                except KeyboardInterrupt:
                    print("\n[FafuRobot] homing interrupted.")
                except Exception as exc:  # noqa: BLE001
                    print(f"[FafuRobot] homing failed ({exc}).")
                # brake at the final pose
                self._brake_joints()
                print("[FafuRobot] gravity-comp stopped (joints braked).")
            else:
                # Kill the float torque and engage short-circuit brake on
                # every joint (freeze-in-place, no stiff grab jolt).
                self._brake_joints()
                print("[FafuRobot] gravity-comp stopped (joints braked).")
            # Final resting state: joints are braked (or homed+braked) via
            # _brake_joints() (mode 0x0F) -> BRAKED. The transient enable()/
            # go_home() inside home_on_exit may have flipped us to IDLE, so
            # correct it here. Do not clobber a latched ESTOP / DEAD.
            if not dry_run and not safety_latched:
                self._sync_state_from_core()
                if self._state not in (
                    RobotState.ESTOP,
                    RobotState.DEAD,
                    RobotState.DISCONNECTED,
                ):
                    self._set_state(RobotState.BRAKED)

    @property
    def is_gravity_compensating(self) -> bool:
        """``True`` while :meth:`start_gravity_compensation` loop is running."""
        return self._gravity_comp_active

    def _brake_joints(self) -> None:
        """Put every manipulator joint into short-circuit brake mode (0x0F),
        falling back to ``stop`` per motor.  The gripper is left untouched.

        Note: brake is velocity-damping, not a position lock — heavy joints
        under sustained gravity can still creep slowly; it never goes fully
        limp, applies no holding current, and engages without the stiff
        "grab" jolt of a position-hold."""
        core = getattr(self, "_core", None)
        token = None
        try:
            if core is not None:
                token = core.begin_operation(pm.OperationKind.LIFECYCLE)
            with self._command_guard():
                for mid in self._joint_motor_ids:
                    try:
                        self._ht.set_motor_mode(mid, self.MODE_BRAKE)
                    except Exception:
                        try:
                            self._ht.stop(mid)
                        except Exception:
                            pass
            if core is not None:
                core.transition(pm.RobotState.BRAKED)
        except Exception:
            if core is None:
                raise
            current = self._sync_state_from_core()
            if current not in (
                RobotState.ESTOP,
                RobotState.DEAD,
                RobotState.DISCONNECTED,
            ):
                raise
        finally:
            if token is not None:
                core.end_operation(token)
            if core is not None:
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
            pos, rot, is_euler=is_euler, is_radians=True,
            init_q=init_q, **ik_kwargs,
        )
        if q is None:
            raise RuntimeError(
                "move_p: IK failed to converge; target pose is likely "
                "outside the reachable workspace or near a singularity.")
        self.move_j(q, is_radians=True, speed=speed, block=block)
        return q

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
        self._require_kinematics()
        steps = max(1, int(steps))
        R = self._rot_from_arg(rot, is_euler, is_radians)
        p = np.asarray(list(pos), dtype=float).reshape(3)
        goal = pin.SE3(R, p)

        q_now = self.get_joint_values()
        start = self._fk_se3(q_now)
        # Geodesic twist from start to goal in the start frame.
        rel = pin.log6(start.actInv(goal))

        # multi_init off for waypoints: we want continuity from the seed.
        ik_kwargs.setdefault("multi_init", False)

        path: List[np.ndarray] = []
        prev_q = q_now
        for k in range(1, steps + 1):
            u = k / steps
            Tk = start * pin.exp6(rel * u)
            q = self.inverse_kinematics(
                Tk.translation, Tk.rotation, is_radians=True,
                init_q=prev_q, **ik_kwargs,
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

    def get_joint_values_raw(self, *, prefer_cache: bool = True) -> List[float]:
        """Return current manipulator joint positions in *turns*."""
        states = self._read_states(self._joint_motor_ids, prefer_cache=prefer_cache)
        return [
            (states[mid].position if states.get(mid) is not None else float("nan"))
            for mid in self._joint_motor_ids
        ]

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
    # Default gripper speed: 0.3 turns/s = 108 deg/s.  Full Fafu
    # gripper range (~113 deg) finishes in roughly 1 second.
    _GRIPPER_VEL_DEFAULT = 0.3
    _GRIPPER_ACC_DEFAULT = 0.5
    _GRIPPER_TOLERANCE_TURNS = 0.005          # ~ 1.8 deg
    _GRIPPER_STALL_VEL_TPS = 0.005            # < 1.8 deg/s ⇒ "not moving"
    _GRIPPER_STALL_PATIENCE_S = 0.3           # treat as done if stalled this long

    @_guard_operation("gripper_control", RobotState.GRASPING)
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
        tolerance_deg: float = 1.5,
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
            Interpret ``angle`` as radians (default) or degrees.
        vel : float, optional
            Velocity limit in turns/s.  Defaults to ``0.3`` turns/s
            (~ 108 deg/s).
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
            is provided (``set_pos_vel_tqe`` has no acceleration arg).
        block : bool, optional
            * ``True`` (default): poll the gripper position until it
              reaches the target (within ``tolerance_deg``), stalls,
              ``effort_threshold`` is exceeded, or ``timeout`` elapses.
            * ``False``: just send the command and return immediately.
        timeout : float, optional
            Maximum seconds to wait when ``block`` is True.
        tolerance_deg : float, optional
            "Reached" tolerance in degrees.  Defaults to 1.5 deg.
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

        pos_turns = self._rad_to_turns(angle) if is_radians else angle / 360.0

        with self._command_guard():
            if command_effort is None:
                self._ht.set_pos_vel_acc(
                    self._gripper_motor_id,
                    pos_turns,
                    float(vel),
                    float(acc),
                    pm.PosUnit.Turns,
                )
            else:
                self._ht.set_pos_vel_tqe(
                    self._gripper_motor_id,
                    pos_turns,
                    float(vel),
                    command_effort,
                    pm.PosUnit.Turns,
                )

        if not block:
            return None

        # When the caller did not ask for force-aware blocking we keep
        # the legacy return-None behaviour so existing call sites
        # don't suddenly start receiving objects.
        result = self._wait_until_gripper_done(
            pos_turns,
            timeout=float(timeout),
            tolerance_turns=tolerance_deg / 360.0,
            effort_threshold=threshold,
        )
        if threshold is None:
            return None
        return result

    def _gripper_limit_turns(self) -> Tuple[Optional[float], Optional[float]]:
        """Helper: return the gripper's (lo, hi) soft limit in turns."""
        try:
            lim = self._ht.get_position_limit_turns(self._gripper_motor_id)
        except Exception:
            lim = None
        if lim is None:
            return (None, None)
        return (float(lim[0]), float(lim[1]))

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
            limit is used (or +0.25 turns ~ 90 deg if no limit is
            configured).
        effort : int, optional
            Max torque (raw int16) the firmware may use during the
            move; forwarded to :meth:`gripper_control`.  ``None`` keeps
            the legacy ``set_pos_vel_acc`` behaviour.
        is_radians : bool, optional
            Interpret ``angle`` as radians (default) or degrees.
        vel : float, optional
            Velocity limit in turns/s.
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
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
        if angle is None:
            _, hi_t = self._gripper_limit_turns()
            target_turns = hi_t if hi_t is not None else 0.25
            # Convert turns -> radians so we can reuse gripper_control's
            # full plumbing (effort, blocking, GraspResult-free void path).
            self.gripper_control(
                self._turns_to_rad(target_turns),
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
            limit is used (or -0.25 turns ~ -90 deg if no limit is
            configured).
        effort : int, optional
            Max torque (raw int16) the firmware may use during the
            move; forwarded to :meth:`gripper_control`.  ``None`` keeps
            the legacy ``set_pos_vel_acc`` behaviour.  For **force-aware
            grasping with an early stop on contact**, use
            :meth:`grasp` instead.
        is_radians : bool, optional
            Interpret ``angle`` as radians (default) or degrees.
        vel : float, optional
            Velocity limit in turns/s.
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
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
        if angle is None:
            lo_t, _ = self._gripper_limit_turns()
            target_turns = lo_t if lo_t is not None else -0.25
            self.gripper_control(
                self._turns_to_rad(target_turns),
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
    _GRASP_VEL_DEFAULT = 0.15            # turns/s (~ 54 deg/s)
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
        min_close_deg: float = 3.0,
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
            Interpret ``target_angle`` as radians (default) or degrees.
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
            Closing velocity in turns/s.  Defaults to 0.15 turns/s
            (deliberately slower than ``open_gripper`` so that
            contact is gentle).
        acc : float, optional
            Retained for API compatibility; grasp uses the torque-capped
            ``set_pos_vel_tqe`` command, which has no acceleration field.
        timeout : float, optional
            Maximum wall-clock seconds to wait.
        min_close_deg : float, optional
            Minimum closure (in degrees) before a stall counts as
            "object grasped".  Anything below this is reported as
            ``'no_movement'`` instead, to catch cases where the
            command never reached the motor or the jaws were already
            closed.  Defaults to 3 deg.

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

        threshold = int(force_threshold)
        if not 1 <= threshold <= 32767:
            raise ValueError("force_threshold must be in [1, 32767]")
        effort_limit = threshold if effort is None else int(effort)
        if not 1 <= effort_limit <= 32767:
            raise ValueError("effort must be in [1, 32767]")

        if target_angle is None:
            lo_t, _ = self._gripper_limit_turns()
            target_turns = lo_t if lo_t is not None else -0.25
        else:
            target_turns = (
                self._rad_to_turns(target_angle) if is_radians
                else target_angle / 360.0
            )

        with self._command_guard():
            self._ht.set_pos_vel_tqe(
                self._gripper_motor_id,
                target_turns,
                float(vel),
                effort_limit,
                pm.PosUnit.Turns,
            )

        return self._wait_until_gripper_done(
            target_turns,
            timeout=float(timeout),
            effort_threshold=threshold,
            min_progress_turns=max(0.0, float(min_close_deg)) / 360.0,
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

        ``lo`` / ``hi`` are interpreted as radians (default) or degrees.
        """
        if motor_id not in self._cfg.motor_ids:
            raise ValueError(f"motor {motor_id} is not in cfg.motor_ids")
        if lo > hi:
            raise ValueError(f"lo ({lo}) > hi ({hi})")
        unit = pm.PosUnit.Radians if is_radians else pm.PosUnit.Degrees
        self._ht.enable_position_limit(motor_id, float(lo), float(hi), unit)
        # Mirror into cfg.limits (kept in turns) so subsequent saves work.
        lo_t = pm.to_turns(float(lo), unit)
        hi_t = pm.to_turns(float(hi), unit)
        try:
            self._cfg.limits[motor_id] = (lo_t, hi_t)
        except Exception:
            pass

    def get_limit(
        self,
        motor_id: int,
        *,
        is_radians: bool = True,
    ) -> Optional[Tuple[float, float]]:
        """Return ``(lo, hi)`` for the given motor or ``None`` if unset."""
        r = self._ht.get_position_limit_turns(motor_id)
        if r is None:
            return None
        lo_t, hi_t = r
        if is_radians:
            return (self._turns_to_rad(lo_t), self._turns_to_rad(hi_t))
        return (lo_t * 360.0, hi_t * 360.0)

    def disable_limit(self, motor_id: int) -> None:
        """Disable the soft limit for ``motor_id``."""
        self._ht.disable_position_limit(motor_id)
        try:
            if motor_id in self._cfg.limits:
                del self._cfg.limits[motor_id]
        except Exception:
            pass

    def clear_limits(self) -> None:
        """Disable every soft position limit."""
        self._ht.clear_all_position_limits()
        try:
            self._cfg.limits.clear()
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Safety
    # ------------------------------------------------------------------
    def emergency_stop(self) -> None:
        """Immediately stop every motor (free-spin mode) and latch ESTOP.

        Legal from **any** connected state (this is the one call allowed
        to interrupt a blocking operation, e.g. from a signal handler).
        Latches :attr:`RobotState.ESTOP`; all subsequent motion calls are
        rejected until :meth:`resume` is called.  The latch also survives
        an in-flight guarded operation finishing (its ``finally`` will
        not overwrite ESTOP).
        """
        core = getattr(self, "_core", None)
        if core is not None:
            core.emergency_stop()
            self._last_cmd_turns = None
            self._servo_active = False
            self._sync_state_from_core()
            print("[FafuRobot] EMERGENCY STOP issued (all motors stopped).")
            return

        for mid in self._cfg.motor_ids:
            try:
                self._ht.stop(mid)
            except Exception:
                pass
        self._last_cmd_turns = None
        self._servo_active = False
        # Preserve a latched DEAD (needs recover(), not resume()); otherwise latch ESTOP.
        if self._state is not RobotState.DEAD:
            self._set_state(RobotState.ESTOP)
        print("[FafuRobot] EMERGENCY STOP issued (all motors → mode 0x00).")

    def resume(self) -> None:
        """Clear an emergency stop and re-enable position control.

        Routes through :meth:`enable`, which transitions the state back
        to ``IDLE`` on success.
        """
        core = getattr(self, "_core", None)
        if core is not None:
            if core.state == pm.RobotState.DEAD:
                print("[FafuRobot] resume: state is DEAD; use "
                      "recover(confirm=True), then enable().")
                return
            if not core.resume():
                raise RuntimeError("resume failed to enable all motors")
            self._sync_state_from_core()
            return

        if self._state is RobotState.DEAD:
            print("[FafuRobot] resume: state is DEAD (power/CAN lost); use "
                  "recover(confirm=True) after restoring power, not resume().")
            return
        if self._state is not RobotState.ESTOP:
            print(f"[FafuRobot] resume: not in ESTOP (state={self._state}); "
                  "re-enabling anyway.")
        self.enable()

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
        joint_release: str = "stop",
        gripper_release: str = "brake",
    ) -> None:
        """Cancel active work, release motors and close the serial port."""
        valid = {"stop", "brake", "hold"}
        if joint_release not in valid:
            raise ValueError(f"joint_release must be one of {valid}")
        if gripper_release not in valid:
            raise ValueError(f"gripper_release must be one of {valid}")

        core = getattr(self, "_core", None)
        if core is None:
            raise RuntimeError("native RobotCore is not initialized")

        needs_shutdown = core.state != pm.RobotState.DISCONNECTED
        port_open = self._ht.is_open()
        if not needs_shutdown and not port_open:
            return

        if needs_shutdown:
            core.shutdown(
                self._core_finish_mode(joint_release),
                self._core_finish_mode(gripper_release),
                5.0,
            )
        if self._ht.is_open():
            self._ht.close()
        self._servo_active = False
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
    def _rad_to_turns(rad: float) -> float:
        return float(rad) / _TWO_PI

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
            s = self._ht.read_motor_state(mid, 0.5)
            if s is None:
                bad.append(mid)
            else:
                self._note_motor_seen(mid)
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
        """Validate length and convert input angles to **turns**."""
        arr = np.asarray(list(joint_angles), dtype=float)
        if arr.shape != (self.num_joints,):
            raise ValueError(
                f"expected a 1-D vector of {self.num_joints} joint values; "
                f"got shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("joint values must all be finite (no NaN/Inf)")
        if not is_radians:
            return [v / 360.0 for v in arr]  # degrees -> turns
        return [self._rad_to_turns(v) for v in arr]

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
                self._note_motor_seen(mid)
        return out

    # Minimum closure (in turns) before a stall counts as "grasped object"
    # rather than "no movement / command never took effect".
    _GRIPPER_MIN_PROGRESS_TURNS = 0.008   # ~ 2.9 deg

    def _wait_until_gripper_done(
        self,
        target_turns: float,
        *,
        timeout: float = 8.0,
        tolerance_turns: Optional[float] = None,
        effort_threshold: Optional[int] = None,
        min_progress_turns: Optional[float] = None,
    ) -> GraspResult:
        """Block until the gripper reaches ``target_turns``, stalls,
        ``|torque| >= effort_threshold``, or ``timeout`` elapses.

        Returns a :class:`GraspResult` regardless of why we stopped;
        callers that don't care can simply ignore the return value.

        We treat the move as "done" if any of:

        * ``|position - target| <= tolerance_turns`` (reached target).
        * ``|torque| >= effort_threshold`` (force trip, if given).
        * ``|velocity| < stall threshold`` for
          ``_GRIPPER_STALL_PATIENCE_S`` and the gripper has moved at
          least ``min_progress_turns`` (grasped something).
        * Stall without enough movement → ``'no_movement'``.
        * Wall-clock ``timeout`` exceeded → ``'timeout'``.
        """
        if tolerance_turns is None:
            tolerance_turns = self._GRIPPER_TOLERANCE_TURNS
        if min_progress_turns is None:
            min_progress_turns = self._GRIPPER_MIN_PROGRESS_TURNS

        t0 = time.monotonic()
        deadline = t0 + max(0.05, float(timeout))

        # Capture the starting position so we can report closed_deg
        # and classify "no movement" vs "real grasp".
        start_state = self._ht.get_cached_state(self._gripper_motor_id)
        if start_state is None:
            start_state = self._ht.read_motor_state(self._gripper_motor_id, 0.05)
        start_pos = start_state.position if start_state is not None else float("nan")
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
                last_pos = s.position
                t_raw = int(abs(s.torque))
                if t_raw > peak_torque:
                    peak_torque = t_raw

                if effort_threshold is not None and t_raw >= int(effort_threshold):
                    return self._make_grasp_result(
                        reason="detected_object_force", grasped=True,
                        last_pos=last_pos, start_pos=start_pos,
                        peak_torque=peak_torque, duration=now - t0,
                    )

                if abs(s.position - target_turns) <= tolerance_turns:
                    return self._make_grasp_result(
                        reason="reached_target", grasped=False,
                        last_pos=last_pos, start_pos=start_pos,
                        peak_torque=peak_torque, duration=now - t0,
                    )

                if abs(s.velocity) < self._GRIPPER_STALL_VEL_TPS:
                    if stall_since is None:
                        stall_since = now
                    elif now - stall_since >= self._GRIPPER_STALL_PATIENCE_S:
                        progress = abs(s.position - start_pos)
                        if progress >= min_progress_turns:
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
            angle_rad = float("nan") if math.isnan(last_pos) else self._turns_to_rad(last_pos)
        else:
            closed_deg = abs(last_pos - start_pos) * 360.0
            angle_rad = self._turns_to_rad(last_pos)
        return GraspResult(
            grasped=grasped,
            reason=reason,
            angle_rad=angle_rad,
            closed_deg=closed_deg,
            peak_torque_raw=int(peak_torque),
            duration_s=float(duration),
        )

    def _build_many_cmds_holding_others(
        self,
        targets_turns: Dict[int, float],
        *,
        vel_rps: float,
    ) -> List["pm.ManyMotorCmd"]:
        """Build a ``set_many_pos_vel_tqe`` payload for all motors.

        Motors *not* listed in ``targets_turns`` (e.g. the gripper
        when issuing a ``move_j``) are commanded to hold their
        current position with zero velocity.
        """
        cmds: List[pm.ManyMotorCmd] = []
        max_torque = int(self._cfg.max_torque_raw)
        for mid in self._cfg.motor_ids:
            if mid in targets_turns:
                cmds.append(
                    pm.ManyMotorCmd(mid, float(targets_turns[mid]),
                                    float(vel_rps), max_torque)
                )
            else:
                s = self._ht.get_cached_state(mid)
                if s is None:
                    s = self._ht.read_motor_state(mid, 0.1)
                hold_pos = s.position if s is not None else 0.0
                cmds.append(pm.ManyMotorCmd(mid, hold_pos, 0.0, max_torque))
        return cmds

    def _move_scurve(
        self,
        targets_turns: Dict[int, float],
        *,
        speed_pct: int,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """S-curve trajectory + ``run_control_loop``, mirrors the
        reference :func:`arm_multi_joint_example.move_scurve`."""
        rate_hz = max(10.0, float(self._cfg.control_rate_hz) or 100.0)
        v_avg_target = (speed_pct / 100.0) * _VEL_AVG_MAX_TPS

        # 1) capture starting positions for *every* motor (so we can
        #    hold non-target ones, e.g. the gripper).  Always query
        #    fresh: after teach-record enable / hand-drag the cache
        #    can be stale and the first S-curve tick will jerk.
        meas_pos: Dict[int, float] = {}
        for mid in self._cfg.motor_ids:
            s = self._ht.read_motor_state(mid, 0.1)
            if s is None:
                raise RuntimeError(
                    f"could not read motor {mid} starting position; "
                    f"aborting move_j for safety"
                )
            meas_pos[mid] = s.position

        # Prefer the *commanded* position from the previous blocking move
        # as the trajectory start, so consecutive moves are continuous and
        # we do not re-command the steady-state error as a step (which made
        # heavy joints like J2 jerk the wrong way for one blink).  Fall back
        # to the measured value when there is no trustworthy last command
        # (first move, or measured drifted far from it -> hand-dragged /
        # external disturbance).
        start_pos: Dict[int, float] = {}
        for mid in self._cfg.motor_ids:
            cmd = (self._last_cmd_turns or {}).get(mid)
            if cmd is not None and abs(cmd - meas_pos[mid]) <= _CMD_CONTINUITY_TOL_T:
                start_pos[mid] = cmd
            else:
                start_pos[mid] = meas_pos[mid]

        # 2) adaptive segment time based on the largest delta.
        max_abs_dpos = 0.0
        for mid, tgt in targets_turns.items():
            max_abs_dpos = max(max_abs_dpos, abs(tgt - start_pos[mid]))

        dt_s = max(_DT_MIN_S, float(self._cfg.trajectory_dt_s) or 1.0)
        if max_abs_dpos > 1e-5:
            dt_target = max_abs_dpos / max(v_avg_target, 1e-3)
            dt_s = max(_DT_MIN_S, dt_target)

        # Plan: per-motor (delta, signed peak velocity).
        plans: Dict[int, Tuple[float, float]] = {}
        for mid, tgt in targets_turns.items():
            dpos = tgt - start_pos[mid]
            if abs(dpos) < 1e-5:
                plans[mid] = (0.0, 0.0)
                continue
            v_avg = abs(dpos) / dt_s
            v_peak = min(_VEL_AVG_MAX_TPS, v_avg) * (math.pi / 2.0)
            plans[mid] = (dpos, math.copysign(v_peak, dpos))

        total_ticks  = max(1, int(dt_s * rate_hz))
        settle_ticks = max(1, int(_SETTLE_MS * rate_hz / 1000.0))
        last_tick    = total_ticks + settle_ticks
        max_mid      = max(self._cfg.motor_ids)
        max_torque   = int(self._cfg.max_torque_raw)

        # Cache once so we do not allocate Python lists every tick.
        all_ids = list(self._cfg.motor_ids)

        def on_tick(tick: int, _dt_ms: float) -> bool:
            if tick >= last_tick:
                return False

            alpha = min(1.0, tick / total_ticks)
            smooth = 0.5 * (1.0 - math.cos(math.pi * alpha))
            vel_factor = math.sin(math.pi * alpha)

            cmds: List[pm.ManyMotorCmd] = []
            for mid in all_ids:
                if mid in plans and plans[mid][1] != 0.0:
                    dpos, v_peak_signed = plans[mid]
                    desired = start_pos[mid] + smooth * dpos
                    v_inst = v_peak_signed * vel_factor
                    cmds.append(pm.ManyMotorCmd(mid, desired, v_inst, max_torque))
                else:
                    # Motors with no target (or zero delta) hold their start.
                    cmds.append(
                        pm.ManyMotorCmd(mid, start_pos[mid], 0.0, max_torque)
                    )

            with self._command_guard():
                self._ht.set_many_pos_vel_tqe(
                    cmds, pm.PosUnit.Turns, max_mid, 0.002,
                )
            return True

        # The control loop runs on the C++ side with GIL released.
        rc = self._ht.run_control_loop(
            rate_hz,
            list(self._cfg.motor_ids),
            on_tick,
            abort_check=self._combined_abort_check(abort_check),
            on_exception=lambda msg: print(f"[FafuRobot] ctrl-loop exception: {msg}"),
            stop_on_finish=False,
            stop_on_abort=True,
        )
        if rc == 1:
            # Aborted mid-trajectory: the commanded position is no longer
            # the planned target, so we cannot trust it for continuity.
            self._last_cmd_turns = None
            raise RuntimeError("move_j aborted (abort_check returned True)")
        if rc == 2:
            self._last_cmd_turns = None
            raise RuntimeError("move_j control loop exited after a send error")

        # Remember what we last *commanded* every motor to, so the next
        # blocking move can start continuously from here (see start_pos
        # selection above).
        last_cmd: Dict[int, float] = {}
        for mid in self._cfg.motor_ids:
            if mid in targets_turns:
                last_cmd[mid] = targets_turns[mid]
            else:
                last_cmd[mid] = start_pos[mid]
        self._last_cmd_turns = last_cmd

    def _move_linear_sync(
        self,
        targets_turns: Dict[int, float],
        *,
        speed_pct: int,
        duration: Optional[float] = None,
        tolerance_turns: float = 0.01,
        timeout_s: float = 10.0,
        block: bool = True,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Official-style synchronized *linear* move.

        Synchronized-arrival move: set per-joint velocity for a common
        arrival time, broadcast pos+vel+max-torque, then wait for arrival:

        1. read the current position of every motor,
        2. pick ``v_i = (target_i - current_i) / duration`` so all joints
           finish together,
        3. broadcast the ``set_many_pos_vel_tqe`` frame (the motors'
           on-board loop drives toward the target with that velocity as
           feed-forward / target speed),
        4. when ``block`` is ``True``, **keep re-sending that same frame**
           while polling the measured positions until every targeted joint
           is within ``tolerance_turns`` or ``timeout_s`` elapses.

        IMPORTANT: this uses the same ``set_many_pos_vel_tqe`` (CAN ID
        ``0x8090``, firmware ``MODE_POS_VEL_TQE``) path as the S-curve
        ``move_j``.  On this FDCAN bridge that channel is a **streaming**
        channel: a single one-shot frame does *not* sustain motion (and a
        watchdog left over from a prior MIT/servo session would kill it
        outright), so we must refresh the frame continuously — exactly like
        :meth:`_move_scurve`'s ``run_control_loop`` and :meth:`servo_j`.
        The difference vs S-curve is only the profile: linear holds a
        constant target + velocity cap (harder start/stop, official
        ``jointsSyncArrival`` semantics) instead of a cosine ease-in/out.
        """
        # 1) fresh current positions for every motor (so held motors can
        #    keep their place, and so velocities are computed correctly).
        meas_pos: Dict[int, float] = {}
        for mid in self._cfg.motor_ids:
            s = self._ht.read_motor_state(mid, 0.1)
            if s is None:
                raise RuntimeError(
                    f"could not read motor {mid} starting position; "
                    f"aborting move_j(style=linear) for safety"
                )
            meas_pos[mid] = s.position

        # 2) duration: explicit, or derived from speed so the joint with
        #    the largest delta travels at <= v_avg_target turns/s.
        v_avg_target = (speed_pct / 100.0) * _VEL_AVG_MAX_TPS
        max_abs_dpos = 0.0
        for mid, tgt in targets_turns.items():
            max_abs_dpos = max(max_abs_dpos, abs(float(tgt) - meas_pos[mid]))
        if duration is None:
            if max_abs_dpos < 1e-5:
                dur = _DT_MIN_S
            else:
                dur = max(_DT_MIN_S, max_abs_dpos / max(v_avg_target, 1e-3))
        else:
            dur = max(_DT_MIN_S, float(duration))

        # 3) v_i = (target - current) / duration; held motors stay put.
        max_torque = int(self._cfg.max_torque_raw)
        cmds: List[pm.ManyMotorCmd] = []
        for mid in self._cfg.motor_ids:
            if mid in targets_turns:
                tgt = float(targets_turns[mid])
                vel = (tgt - meas_pos[mid]) / dur
                cmds.append(pm.ManyMotorCmd(mid, tgt, vel, max_torque))
            else:
                cmds.append(pm.ManyMotorCmd(mid, meas_pos[mid], 0.0, max_torque))

        max_mid = max(self._cfg.motor_ids)
        # first broadcast == posVelMaxTorque() + motor_send_cmd()
        with self._command_guard():
            self._ht.set_many_pos_vel_tqe(
                cmds, pm.PosUnit.Turns, max_mid, 0.05)

        last_cmd: Dict[int, float] = {}
        for mid in self._cfg.motor_ids:
            last_cmd[mid] = (
                float(targets_turns[mid]) if mid in targets_turns else meas_pos[mid]
            )

        if not block:
            self._last_cmd_turns = last_cmd
            return

        # 4) waitForPosition: keep re-sending the (constant) frame to sustain
        #    motion on the streaming 0x8090 channel while polling until every
        #    targeted joint settles.  A single one-shot frame does NOT move
        #    the joint here (see docstring), so the resend is mandatory.
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        reached = False
        while time.monotonic() < deadline:
            if self._combined_abort_check(abort_check)():
                self._last_cmd_turns = None
                raise RuntimeError(
                    "move_j(style=linear) aborted (abort_check returned True)"
                )
            # keep-alive resend (fire-and-forget, no reply wait) — refreshes
            # the position loop / watchdog exactly like the S-curve loop.
            with self._command_guard():
                self._ht.set_many_pos_vel_tqe(
                    cmds, pm.PosUnit.Turns, max_mid, 0.0)
            ok = True
            for mid, tgt in targets_turns.items():
                s = self._ht.read_motor_state(mid, 0.1)
                if s is None or abs(s.position - float(tgt)) > tolerance_turns:
                    ok = False
                    break
            if ok:
                reached = True
                break
            time.sleep(0.02)

        if not reached:
            print(
                f"[FafuRobot] move_j(style=linear): not settled within "
                f"{timeout_s:.1f}s. If the residual is small (~1 deg) it is "
                f"the pos_vel_MAXtqe gravity steady-state error; if the joint "
                f"barely moved, raise speed/max_torque or check for a leftover "
                f"watchdog from a prior MIT/servo session."
            )
        self._last_cmd_turns = last_cmd

    def _move_acc_sync(
        self,
        targets_turns: Dict[int, float],
        *,
        speed_pct: int,
        acc_rpss: Optional[float] = None,
        tolerance_turns: float = 0.01,
        timeout_s: float = 10.0,
        block: bool = True,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Per-joint ``set_pos_vel_acc`` move (firmware trapezoidal loop).

        Unlike the ``pos_vel_MAXtqe`` paths (``move_j`` S-curve / linear),
        this commands each joint's **on-board** position profile
        (MODE_POS_VEL_ACC: target pos + max velocity + acceleration).  The
        firmware
        runs its own profile generator and position loop, which on many
        actuators includes an integral term — so this is the one path that
        may shrink the gravity steady-state error *without* any torque
        feed-forward / calibration.  Provided as an A/B test against the
        ``pos_vel_MAXtqe`` paths.

        One CAN frame per joint (no one-to-many ``set_pos_vel_acc`` exists),
        all fired back-to-back; the firmware profiles handle synchronization
        loosely (they finish near the same time because we share velocity /
        acceleration caps).  Held / non-target motors are left untouched.
        """
        v_max = (speed_pct / 100.0) * _VEL_AVG_MAX_TPS
        v_max = max(0.02, v_max)
        # Acceleration cap: reach v_max in ~0.3 s by default (gentle ramp),
        # so the start is not a hard step like style="linear".
        acc = float(acc_rpss) if acc_rpss is not None else max(0.05, v_max / 0.3)

        with self._command_guard():
            for mid, tgt in targets_turns.items():
                self._ht.set_pos_vel_acc(
                    int(mid), float(tgt), float(v_max), float(acc),
                    pm.PosUnit.Turns,
                )

        last_cmd: Dict[int, float] = {}
        cur = self._last_cmd_turns or {}
        for mid in self._cfg.motor_ids:
            if mid in targets_turns:
                last_cmd[mid] = float(targets_turns[mid])
            elif mid in cur:
                last_cmd[mid] = cur[mid]

        if not block:
            self._last_cmd_turns = last_cmd or None
            return

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        reached = False
        while time.monotonic() < deadline:
            if self._combined_abort_check(abort_check)():
                self._last_cmd_turns = None
                raise RuntimeError(
                    "move_j(style=acc) aborted (cancellation requested)"
                )
            ok = True
            for mid, tgt in targets_turns.items():
                s = self._ht.read_motor_state(mid, 0.1)
                if s is None or abs(s.position - float(tgt)) > tolerance_turns:
                    ok = False
                    break
            if ok:
                reached = True
                break
            time.sleep(0.02)

        if not reached:
            print(
                f"[FafuRobot] move_j(style=acc): not settled within "
                f"{timeout_s:.1f}s (if the residual is still ~1 deg, the "
                f"firmware acc loop has no integral either)."
            )
        self._last_cmd_turns = last_cmd or None


# ============================================================================
#  Demo
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="FafuRobotController demo (Piper-style high-level API)"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(_HERE, "robot.cfg"),
        help="path to robot.cfg",
    )
    parser.add_argument(
        "--gripper-id",
        type=int,
        default=None,
        help="motor id of the gripper (omit if no gripper)",
    )
    parser.add_argument("--speed", type=int, default=15,
                        help="speed percent for the demo (default 15)")
    parser.add_argument("--no-move", action="store_true",
                        help="only read state, do not issue any motion")
    args = parser.parse_args()

    has_gripper = args.gripper_id is not None
    arm = FafuRobotController(
        cfg_path=args.config,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id,
    )

    def _print(*a, **kw):
        # The 50Hz polling thread can interleave its log output with
        # ours. Sleep briefly so its line is fully flushed first,
        # then print our own line with explicit flush.
        time.sleep(0.02)
        kw.setdefault("flush", True)
        print(*a, **kw)

    try:
        _print("\n--- Initial state ---")
        q = arm.get_joint_values()
        _print(f"  joint angles (deg): {np.degrees(q).round(2).tolist()}")
        _print(f"  is_enabled        : {arm.is_enabled}")
        _print(f"  stats             : {arm.get_status().to_string()}")

        if args.no_move:
            _print("\n[--no-move] skipping motion demo.")
        else:
            _print("\n--- Going home (joint zero) ---")
            arm.go_home(speed=args.speed, block=True)

            _print("\n--- Tiny sinusoidal demo (block=False) ---")
            base = arm.get_joint_values()
            q0 = base.copy()
            amp = math.radians(10.0)      # swing amplitude
            margin = math.radians(2.0)    # keep this far inside soft limits

            # Pick a joint whose [q0-amp, q0+amp] swing stays inside its
            # soft limits.  This avoids the old bug of driving the last
            # joint to an *absolute* +-10 deg around 0, which on this arm
            # slams J7 into its soft limit (upper bound -1.836 deg).
            lims = arm._joint_limits_rad()
            demo_idx = None
            # Prefer the wrist/last joints first, then fall back inward.
            for idx in range(arm.num_joints - 1, -1, -1):
                if lims is None:
                    demo_idx = idx          # no limits known: trust the swing
                    break
                lo, hi = lims[0][idx], lims[1][idx]
                if (q0[idx] - amp) >= (lo + margin) and \
                   (q0[idx] + amp) <= (hi - margin):
                    demo_idx = idx
                    break

            if demo_idx is None:
                _print("  no joint has +-10 deg of room inside its soft "
                       "limits at this pose; skipping the sinusoidal demo.")
            else:
                _print(f"  swinging J{demo_idx + 1} around its current angle "
                       f"({math.degrees(q0[demo_idx]):+.1f} deg) by +-10 deg")
                for i in range(40):
                    # Oscillate *around the current angle*, not around 0.
                    base[demo_idx] = q0[demo_idx] + amp * math.sin(i * math.pi / 20.0)
                    arm.move_j(base, speed=args.speed, block=False)
                    time.sleep(0.05)
                # Return that joint to where it started.
                base[demo_idx] = q0[demo_idx]
                arm.move_j(base, speed=args.speed, block=False)

            if has_gripper:
                _print("\n--- Gripper demo (open -> close, using soft limits) ---")
                arm.open_gripper()        # blocking, default vel 0.3 turns/s
                gs = arm.get_gripper_state()
                _print(f"  after open : gripper "
                       f"{math.degrees(arm._turns_to_rad(gs.position)):+.2f} deg")
                time.sleep(0.3)
                arm.close_gripper()
                gs = arm.get_gripper_state()
                _print(f"  after close: gripper "
                       f"{math.degrees(arm._turns_to_rad(gs.position)):+.2f} deg")

        _print("\n--- Final state ---")
        q = arm.get_joint_values()
        _print(f"  joint angles (deg): {np.degrees(q).round(2).tolist()}")
        if has_gripper:
            gs = arm.get_gripper_state()
            _print(f"  gripper           : "
                   f"{math.degrees(arm._turns_to_rad(gs.position)):+.2f} deg")
    finally:
        arm.close_connection()
