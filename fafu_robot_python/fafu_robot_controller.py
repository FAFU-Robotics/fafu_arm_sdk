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
  USB-CAN debug board; there is no built-in inverse kinematics.
  ``move_p`` / ``move_l`` therefore raise :class:`NotImplementedError`
  unless an external IK solver is plugged in.

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

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------------
#  Make sure fafu_motor.pyd next to this file is importable.
# ----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fafu_motor as pm  # noqa: E402

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

# Per-motor torque coefficient (Nm per raw-int16 LSB), mirrors the C++
# TORQUE_COEFF table in hightorque_serial.cpp.  The firmware torque command
# is a raw int16; the driver converts a desired Nm to raw via
# ``raw = round(tau_nm / coeff)``.  An unknown / empty model maps to coeff
# 1.0 (raw == Nm, i.e. essentially unscaled -- see setup_dynamics warning).
TORQUE_COEFF: Dict[str, float] = {
    # ---- Fafu arm actual motors (measured, authoritative) ----
    "M5036_02": 0.67,     # J1, J4
    "M6036_02": 0.677,    # J2, J3
    "M4438_30": 0.5256,   # J5, J6, J7  (vendor motor_tqe_adj value)
    # ---- legacy / other Hightorque models ----
    "M3536_32": 0.458105,
    "M4438_32": 0.485565,
    "M4538_19": 0.493835,
    "M5043_20": 0.966,
    "M5046_20": 0.533654,
    "M5047_09": 0.547474,
    "M5047_36": 0.803,
    "M6056_36": 0.677,
    "M7256_35": 0.676524,
    "M60SG_35": 0.7942,
    "M60BM_35": 0.7942,
}


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
        ``100``.  Set to ``0`` to disable (★ not recommended ★).
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
        feedforward.  When ``False`` every frame uses the constant
        ``max_vel`` as ``vel``, which makes the motor "always sprint"
        toward each target and is the dominant source of high-frequency
        whine in joystick / teleop scenarios.
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
    # True (default): send on the group-MIT channel (0x8093, kp/kd + gravity
    #   feed-forward, like move_MIT) -- impedance tracking, softer feel, matches
    #   the vendor pos_vel_tqe_kp_kd streaming. Uses kp/kd (mit_kp/mit_kd) and,
    #   for heavy joints, a dynamics model for gravity feed-forward.
    # False: send on the position channel (0x8090 set_many_pos_vel_tqe_partial)
    #   -- firmware position loop tracks; robust, no gain tuning (fallback).
    use_mit: bool = True
    # Per-joint MIT PD gains (physical vendor units, same as move_MIT). None =>
    # vendor replay defaults for 6-DoF (kp=[30,40,55,15,7,5], kd=[3,4,5.5,1.5,
    # 0.7,0.5]); scalar otherwise. Only used when use_mit=True.
    mit_kp: "float | Iterable[float] | None" = None
    mit_kd: "float | Iterable[float] | None" = None
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

        ★ These are a *starting point only* — friction is arm- and
        wear-specific and MUST be re-identified on your hardware. ★
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
        # enable() failure is NON-FATAL at construction: if a few motors
        # are stuck (e.g. MIT 0x0B that only a hard power-cycle clears),
        # crashing here locks the user out of the menu entirely — they
        # can't even run diagnostics, retry, or operate the other joints.
        # So warn loudly and continue; the offending joints simply won't
        # move until recovered ([y] soft-reboot / power-cycle).
        if auto_enable:
            try:
                self.enable()
            except Exception as e:
                print(f"[FafuRobot] warning: enable() failed at startup: {e}")
                print("[FafuRobot] continuing anyway so you can run diagnostics "
                      "/ operate other joints; stuck joints won't move until "
                      "recovered.")

        use_async = async_rx if async_rx is not None else bool(cfg.use_async_rx)
        if use_async:
            try:
                self._ht.enable_async_rx()
                time.sleep(0.1)
                # Prime the RX cache so get_cached_state returns immediately.
                for mid in cfg.motor_ids:
                    self._ht.read_motor_state(mid, 0.1)
            except Exception as e:
                print(f"[FafuRobot] warning: enable_async_rx failed: {e}")

        if auto_polling:
            try:
                hz = float(cfg.control_rate_hz) if cfg.control_rate_hz else 50.0
                hz = max(10.0, hz)
                self._ht.start_state_polling(list(cfg.motor_ids), hz)
            except Exception as e:
                print(f"[FafuRobot] warning: start_state_polling failed: {e}")

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

        # Servo (online streaming) session state. None when not servoing.
        self._servo_active: bool = False
        self._servo_opts: Optional[ServoOpts] = None
        self._servo_last_target_turns: List[float] = []
        # filtered_target_turns mirrors last_target when lookahead_time == 0;
        # when smoothing is enabled it lags the raw target by ~lookahead_time.
        self._servo_filtered_target_turns: List[float] = []
        # Pre-allocated numpy buffers for set_many_pos_vel_tqe_partial.
        # active_ids / hold_ids are set once in servo_start and never
        # re-allocated; active_pos / active_vel are overwritten in place
        # every tick to avoid per-call numpy allocation.
        self._servo_active_ids_np: Optional[np.ndarray] = None
        self._servo_hold_ids_np: Optional[np.ndarray] = None
        self._servo_active_pos_np: Optional[np.ndarray] = None
        self._servo_active_vel_np: Optional[np.ndarray] = None
        self._servo_max_motor_id: int = 0
        self._servo_max_torque: int = 0
        self._servo_started_at: float = 0.0
        self._servo_tick_count: int = 0
        # warning counters for batched logging at servo_end
        self._servo_clamp_count: int = 0
        self._servo_lag_count: int = 0
        # Track *consecutive* lag-tripped ticks to support
        # lag_abort_consecutive. Reset to 0 every time a tick comes
        # in clean (lag <= max_lag_rad).
        self._servo_lag_streak: int = 0
        # One-shot warning so the user sees a live notice on the first
        # lag-trip without flooding the console at 100Hz.
        self._servo_lag_warned: bool = False
        # Sticky flag — once auto-abort fires we refuse further servo_j
        # frames so callers can detect the condition. Cleared by
        # servo_start.
        self._servo_aborted_reason: Optional[str] = None

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
        # Per-joint motor model strings used to convert Nm -> raw int16
        # inside set_pos_vel_tqe_kp_kd. None => "" (coeff 1.0, see
        # setup_dynamics docstring for why that is unsafe-but-quiet).
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
        self._use_group_mit: bool = (self.num_joints <= 6)

        print(
            f"[FafuRobot] connected on {self._port} @ {self._baudrate} "
            f"({len(self._joint_motor_ids)} joints"
            + (f" + gripper M{self._gripper_motor_id}" if self._has_gripper else "")
            + ")"
        )

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
        """Switch every motor to position control (mode ``0x0A``).

        Resolution path (cheapest to most aggressive):

        0. **Fast path.** Read every motor's live state; if all are
           already in MODE_POSITION (left over from a previous
           ``servo_end("hold")`` or ``close_connection(release="hold")``)
           return without sending any mode-change frame.  This both
           saves ~30 ms and avoids a known false-failure mode of
           ``set_motor_mode(0x0A)`` whose internal "switch + read-back"
           sequence sometimes reports ``None`` on a freshly-reopened
           port even when the mode change actually succeeded.
        1. **Normal path.** Call :meth:`_switch_mode_all` with up to
           3 retries; it now does a fresh ``read_motor_state`` to
           verify failures, filtering out the false-failure case
           above.
        2. **Soft-reboot path.** If the normal path still fails, print
           a diagnostic for every motor (mode / fault / position read
           live), then issue a firmware-level ``motor_reset`` to every
           failed motor (fire-and-forget reset of the on-motor
           controller — does **not** clear zero / config), wait 1 s
           for them to come back up, and try the normal path one more
           time.

        Most cases of "refused mode 0x0A" after a previous
        ``servo_end`` / ``close_connection`` are resolved at step 2
        without a hardware power-cycle.

        Parameters
        ----------
        allow_motor_reset : bool, optional
            Default ``True``.  Set to ``False`` to keep the legacy
            behaviour (raise immediately if step 1 fails).  Useful
            when something else owns the bus and you do not want a
            stealth ``motor_reset``.

        Raises
        ------
        RuntimeError
            Either step 1 (when ``allow_motor_reset=False``) or both
            stages exhausted.  In the latter case the printed
            diagnostic above tells you whether the motors are
            unreachable (no state read), faulted (non-zero ``fault``
            register), or mechanically jammed (mode = 0x0A but the
            commanded position cannot be tracked).
        """
        # Stage-0: fast path. read once, skip the whole switch if every
        # motor is already in MODE_POSITION. This handles the common
        # case of "previous program left motors holding" cleanly and
        # cheaply, and dodges the sync-read false-failure issue.
        already_active = True
        seen_mit_mode = False   # 0x0B = MIT (set_many_mit / 0x8093) 残留模式
        for mid in self._cfg.motor_ids:
            s = self._ht.read_motor_state(mid, 0.2)
            if s is None or int(s.mode) != self.MODE_POSITION:
                already_active = False
                if s is not None and int(s.mode) == self.MODE_MIT:
                    seen_mit_mode = True
                break
        if already_active:
            print("[FafuRobot] all motors already in position control hold; "
                  "enable() is a no-op.")
            return

        # MIT 残留 (mode 0x0B): 实测本固件下 set_motor_mode(0x0A) 切不动 (哪怕用
        # per-motor 位置帧隐式切也失败, 见 set_motor_mode 注释), 必须 motor_reset
        # 软重启才能回到位置模式. 常规 _switch_mode_all 的 3 次重试注定失败, 只会
        # 白白拉长"电机掉电 -> 下垂"的窗口. 检测到 0x0B 就直接跳到 motor_reset.
        if seen_mit_mode and allow_motor_reset:
            print("[FafuRobot] enable: MIT residual mode (0x0B) detected; "
                  "motor_reset directly (skip doomed mode-switch retries) ...")
            for mid in self._cfg.motor_ids:
                try:
                    self._ht.motor_reset(mid)
                except Exception as e:
                    print(f"  motor {mid}: motor_reset failed: {e}")
            # 与 stage-2 一致用 1.0s: 上次日志 J2/J3/J4 在 0.6s 后仍卡在 0x0B.
            time.sleep(1.0)
            if self._switch_mode_all(self.MODE_POSITION,
                                     label="position (post-MIT-reset)", max_retry=3):
                time.sleep(0.05)
                print("[FafuRobot] all motors enabled (recovered via motor_reset).")
                return

            # 部分电机 reset 后仍卡在 0x0B -> 对这些电机做几轮更"用力"的重试:
            # 每轮对每个卡住电机 stop()(踢一下) + 连发 3 次 motor_reset (文本
            # 后门命令 fire-and-forget, 可能丢帧, 多发几次更稳), 再等 1.2s.
            for rnd in range(1, 4):
                still_mit: List[int] = []
                for mid in self._cfg.motor_ids:
                    try:
                        s = self._ht.read_motor_state(mid, 0.2)
                        if s is not None and int(s.mode) == self.MODE_MIT:
                            still_mit.append(mid)
                    except Exception:
                        still_mit.append(mid)
                if not still_mit:
                    break
                print(f"[FafuRobot] enable: motors {still_mit} still in MIT 0x0B; "
                      f"aggressive motor_reset round {rnd} (stop + reset x3) ...")
                for mid in still_mit:
                    try:
                        self._ht.stop(mid)
                    except Exception:
                        pass
                    for _ in range(3):
                        try:
                            self._ht.motor_reset(mid)
                        except Exception as e:
                            print(f"  motor {mid}: motor_reset failed: {e}")
                        time.sleep(0.05)
                time.sleep(1.2)
                if self._switch_mode_all(self.MODE_POSITION,
                                         label=f"position (post-MIT-reset-{rnd + 1})",
                                         max_retry=3):
                    time.sleep(0.05)
                    print(f"[FafuRobot] all motors enabled "
                          f"(recovered via motor_reset round {rnd + 1}).")
                    return

            # 反复软重启仍有电机卡在 0x0B -> 本固件把 MIT 模式锁死, 纯软件救不回,
            # 直接给出断电指引 (落到 stage-2 也只是再白等一遍).
            stuck = []
            for mid in self._cfg.motor_ids:
                try:
                    s = self._ht.read_motor_state(mid, 0.2)
                    if s is not None and int(s.mode) == self.MODE_MIT:
                        stuck.append(mid)
                except Exception:
                    pass
            if stuck:
                self._print_motor_diagnostic(
                    prefix="enable: MIT 0x0B stuck after repeated motor_reset; ")
                raise RuntimeError(
                    f"enable failed: motors {stuck} stuck in MIT mode (0x0B) and "
                    f"will not respond to motor_reset. This firmware locks the MIT/"
                    f"运控 mode so tightly that only a HARD POWER-CYCLE of the arm "
                    f"clears it. Please power the arm off and on, then rerun. "
                    f"(To avoid this: after using MIT/持续保持, press [y] 软重启 "
                    f"before quitting so motors are left in position mode 0x0A.)"
                )

            # (理论上到不了这里) 快速路径没完全恢复 -> 落到 stage-2 全量 reset.
            print("[FafuRobot] enable: MIT fast-path reset incomplete; "
                  "falling through to full motor_reset recovery ...")

        if self._switch_mode_all(self.MODE_POSITION, label="position", max_retry=3):
            time.sleep(0.05)
            print("[FafuRobot] all motors enabled (position control hold).")
            return

        # Stage-1 failed. Print a diagnostic so the next message is
        # actionable instead of a generic "refused".
        self._print_motor_diagnostic(prefix="enable: stage-1 failed; ")

        if not allow_motor_reset:
            raise RuntimeError(
                "enable failed: at least one motor refused mode 0x0A "
                "(motor_reset recovery disabled by allow_motor_reset=False)"
            )

        # Stage-2: motor_reset is a firmware-level soft reboot of the
        # on-motor controller. It is the equivalent of pulling power on
        # *only that motor* for a few hundred ms, without losing zero
        # calibration / soft limits / etc. Use as last resort before
        # asking the user to power-cycle.
        print("[FafuRobot] enable: attempting motor_reset on every motor (soft reboot) ...")
        for mid in self._cfg.motor_ids:
            try:
                self._ht.motor_reset(mid)
            except Exception as e:
                print(f"  motor {mid}: motor_reset failed: {e}")
        # Firmware needs time to come back. Empirically ~0.6 s is
        # enough on the boards we have; 1.0 s gives margin.
        time.sleep(1.0)

        if not self._switch_mode_all(self.MODE_POSITION, label="position (post-reset)", max_retry=3):
            self._print_motor_diagnostic(prefix="enable: stage-2 (post-reset) failed; ")
            raise RuntimeError(
                "enable failed even after motor_reset; check the "
                "diagnostic above. Likely causes: (a) motor controller "
                "in latched FAULT state -> hard power-cycle; "
                "(b) USB-CAN bus disconnected / wrong COM port; "
                "(c) mechanical jam holding the joint outside soft limits."
            )

        time.sleep(0.05)
        print("[FafuRobot] all motors enabled (recovered via motor_reset).")

    def _print_motor_diagnostic(self, *, prefix: str = "") -> None:
        """Read every motor's live state and print a compact summary.

        Used by :meth:`enable` to turn an opaque "refused mode 0x0A"
        into something a user can act on.  Reads are synchronous with
        a generous timeout (300 ms) so the failure mode "motor does
        not respond" is also obvious.
        """
        print(f"[FafuRobot] {prefix}motor states:")
        for mid in self._cfg.motor_ids:
            try:
                s = self._ht.read_motor_state(mid, 0.3)
            except Exception as e:
                print(f"  motor {mid}: READ EXCEPTION {e}")
                continue
            if s is None:
                print(f"  motor {mid}: NO RESPONSE (read timeout) — bus or motor power off?")
                continue
            fault_val = getattr(s, "fault", 0)
            fault_tag = "fault=OK" if not fault_val else f"fault=0x{int(fault_val):02X} ★"
            print(f"  motor {mid}: mode=0x{int(s.mode):02X}  {fault_tag}  "
                  f"pos={s.position:+.3f}t  vel={s.velocity:+.3f}t/s")

    def disable(self) -> None:
        """Switch every motor to free-spin mode (mode ``0x00``)."""
        # Commanded position is now meaningless (the user may hand-drag
        # the arm); force the next move to start from the measured pose.
        self._last_cmd_turns = None
        ok = self._switch_mode_all(self.MODE_STOP, label="stop", max_retry=2)
        if ok:
            print("[FafuRobot] all motors disabled (free spin).")

    def brake(self) -> None:
        """Engage short-circuit braking on every motor (mode ``0x0F``)."""
        ok = self._switch_mode_all(self.MODE_BRAKE, label="brake", max_retry=2)
        if ok:
            print("[FafuRobot] all motors braked.")

    # ------------------------------------------------------------------
    #  Joint-space motion
    # ------------------------------------------------------------------
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
            * ``"linear"``: synchronized-arrival mode —
              compute ``v_i = (target_i - current_i)/duration``
              for synchronized arrival, broadcast **one**
              ``set_many_pos_vel_tqe`` frame, then poll until settled
              (:meth:`_move_linear_sync`).  Same ``pos_vel_MAXtqe`` path
              (no integral) so the gravity steady-state error is
              **identical** to S-curve; harder start/stop.  A/B only.
            * ``"acc"``: per-joint ``set_pos_vel_acc`` (firmware
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
        if path_arr.ndim != 2 or path_arr.shape[1] != self.num_joints:
            raise ValueError(
                f"path must have shape (N, {self.num_joints}); got {path_arr.shape}"
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
        if path_arr.ndim != 2 or path_arr.shape[1] != self.num_joints:
            raise ValueError(
                f"path must have shape (N, {self.num_joints}); got {path_arr.shape}"
            )

        if self.num_joints > 6:
            raise RuntimeError(
                f"move_jntspace_path_mit needs <=6 joints, but "
                f"num_joints={self.num_joints}. Exclude the gripper "
                f"(gripper_motor_id) so only manipulator joints stream.")

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

        # Active mode so 0x8093 MIT frames are actuated.
        self.enable()

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
    #
    #  Unlike :meth:`move_j` (offline S-curve, blocking), ``servo_j`` is
    #  designed for upper-layer code that streams a fresh joint target
    #  every ~10 ms (e.g. teleop, VR, visual servoing, in-the-loop IK).
    #
    #  Lifecycle::
    #
    #      arm.servo_start(ServoOpts(...))     # once: arm watchdog, async RX
    #      while not quit:
    #          arm.servo_j(target_angles)      # ~100 Hz, non-blocking
    #          time.sleep_until(next_tick)
    #      arm.servo_end("brake")              # clear watchdog + brake
    #
    #  Four safety lines (cannot be skipped — they all live inside
    #  ``servo_j`` and ``servo_start``):
    #
    #    1) Firmware-side watchdog ( :func:`HightorqueSerial.set_timeout` )
    #       — if the motor sees no new command for ``watchdog_ms`` it
    #       brakes itself. Survives host crash, Ctrl+C, USB unplug.
    #    2) Per-step jump clamp (``max_step_rad``) — protects against
    #       upper-layer planner step bugs.
    #    3) Tracking-error monitor (``max_lag_rad``) — if measured
    #       position trails ``last_target`` by more than this, ``servo_j``
    #       returns ``False`` (stuck joint, motor too weak, planner too
    #       fast, etc.).
    #    4) Soft limits — reuses :meth:`enable_position_limit`; the
    #       underlying ``set_many_pos_vel_tqe`` clamps automatically.
    # ------------------------------------------------------------------
    def servo_start(self, opts: Optional[ServoOpts] = None) -> None:
        """Enter an online-streaming joint-servo session.

        Idempotent on the second call (warns and returns).  After this
        every joint motor has a firmware watchdog armed (1) and the
        wrapper has cached the current joint configuration as the
        initial ``last_target`` (so the first :meth:`servo_j` cannot
        trip the step-clamp warning).  The gripper is **not** touched
        and is not subject to the servo watchdog.

        Parameters
        ----------
        opts : ServoOpts, optional
            Tunables.  Defaults to :class:`ServoOpts` (watchdog 100 ms,
            max_vel 1.0 rad/s, max_step 0.05 rad, max_lag 0.2 rad).

        Raises
        ------
        RuntimeError
            If the driver fails to enable async RX (servo cannot run on
            the blocking sync-RX path) or any joint motor fails to
            report a starting position.
        """
        if self._servo_active:
            print("[FafuRobot] servo_start: already servoing; call servo_end first")
            return

        opts = ServoOpts(**vars(opts)) if opts is not None else ServoOpts()
        if opts.watchdog_ms < 0:
            opts.watchdog_ms = 0
        if opts.watchdog_ms == 0:
            print("[FafuRobot] servo_start: WARNING watchdog disabled; "
                  "host crash will not stop motors. Use only for offline tests.")
        elif opts.watchdog_ms < 30:
            print(f"[FafuRobot] servo_start: WARNING watchdog_ms={opts.watchdog_ms} "
                  "< 30 may trip spuriously at 100Hz; recommend >= 50 ms.")
        if opts.max_vel <= 0.0:
            opts.max_vel = 1.0
        if opts.max_step_rad <= 0.0:
            opts.max_step_rad = 0.05
        if opts.rate_hz <= 0.0:
            opts.rate_hz = 100.0
        if opts.lookahead_time < 0.0:
            opts.lookahead_time = 0.0

        if not self.is_enabled:
            print("[FafuRobot] servo_start: motors not enabled, calling enable() ...")
            self.enable()

        # Servo cannot tolerate the 5-15 ms sync RX wait inside
        # send_can_and_recv_, so force async RX on if it is not already.
        if not self._ht.is_async_rx():
            try:
                self._ht.enable_async_rx()
                time.sleep(0.05)
            except Exception as e:
                raise RuntimeError(f"servo_start: enable_async_rx failed: {e}") from e

        # A servo session moves motors outside the move_j S-curve, so any
        # previously-cached commanded pose is no longer valid; force the
        # next move_j to start from the measured pose.
        self._last_cmd_turns = None

        # Capture current pos as last_target so the first servo_j is a
        # zero-step move and never trips the step-clamp warning.
        last_target: List[float] = []
        for mid in self._joint_motor_ids:
            s = self._ht.get_cached_state(mid)
            if s is None:
                s = self._ht.read_motor_state(mid, 0.1)
            if s is None:
                raise RuntimeError(
                    f"servo_start: cannot read motor {mid} starting position")
            last_target.append(s.position)

        # Pre-build the numpy arrays consumed by
        # ``set_many_pos_vel_tqe_partial`` so the per-tick allocation
        # cost is zero (we only memcpy target_pos / target_vel into
        # the pre-allocated buffers below; active_ids / hold_ids never
        # change for a session).
        active_ids_np = np.asarray(self._joint_motor_ids, dtype=np.int32)
        # Non-joint motors are normally "held" at cached pos each tick.
        # The gripper must be excluded: otherwise every servo_j overwrites
        # M7 back to the position captured at servo_start, canceling
        # open_gripper/close_gripper (non-blocking) commands.
        hold_motor_ids = [
            mid for mid in self._cfg.motor_ids if mid not in self._joint_motor_ids
        ]
        if self._has_gripper and self._gripper_motor_id is not None:
            hold_motor_ids = [
                mid for mid in hold_motor_ids if mid != self._gripper_motor_id
            ]
        hold_ids_np = np.asarray(hold_motor_ids, dtype=np.int32)
        # Pre-allocated output buffers reused every tick.
        N = len(self._joint_motor_ids)
        active_pos_np = np.zeros(N, dtype=np.float64)
        active_vel_np = np.zeros(N, dtype=np.float64)

        # Arm the firmware watchdog on every JOINT motor (gripper excluded
        # — gripper is not part of the servo loop, watching it would brake
        # the jaws while the loop is still in start-up).
        if opts.watchdog_ms > 0:
            for mid in self._joint_motor_ids:
                try:
                    self._ht.set_timeout(mid, int(opts.watchdog_ms))
                except Exception as e:
                    print(f"[FafuRobot] servo_start: set_timeout({mid}) failed: {e}")

        # If MIT channel requested, resolve + convert kp/kd to raw ONCE (per
        # vendor kp_float2int, radian_2pi) so the per-tick hot path only does a
        # cheap gravity-ff call + one set_many_mit. Physical gains default to
        # the vendor replay values (6-DoF) or a scalar.
        self._servo_mit_kp_raw: List[int] = []
        self._servo_mit_kd_raw: List[int] = []
        # MIT frame slot count = max JOINT motor id (gripper excluded), so a
        # 6-joint arm builds a 6-slot 62B frame, not a 7-slot 72B (>64) one.
        self._servo_mit_max_motor_id = int(max(self._joint_motor_ids))
        if opts.use_mit:
            if self.num_joints > 6:
                raise RuntimeError(
                    f"servo use_mit needs <=6 joints, but num_joints="
                    f"{self.num_joints}; exclude the gripper.")
            kp = opts.mit_kp
            kd = opts.mit_kd
            nj = self.num_joints
            if kp is None:
                kp = ([30.0, 40.0, 55.0, 15.0, 7.0, 5.0][:nj] if nj == 6 else 20.0)
            if kd is None:
                kd = ([3.0, 4.0, 5.5, 1.5, 0.7, 0.5][:nj] if nj == 6 else 2.0)
            kp_v = ([float(kp)] * nj if np.isscalar(kp)
                    else [float(x) for x in kp])
            kd_v = ([float(kd)] * nj if np.isscalar(kd)
                    else [float(x) for x in kd])
            for i in range(nj):
                model = (self._dyn_motor_models[i]
                         if self._dyn_motor_models is not None else "")
                coeff = TORQUE_COEFF.get(model, 1.0) or 1.0
                self._servo_mit_kp_raw.append(
                    int(np.clip(round((kp_v[i] / coeff) * 10.0 * _TWO_PI),
                                -32768, 32767)))
                self._servo_mit_kd_raw.append(
                    int(np.clip(round((kd_v[i] / coeff) * 10.0 * _TWO_PI),
                                -32768, 32767)))
            if opts.mit_gravity_ff and not self.has_dynamics:
                print("[FafuRobot] servo_start: use_mit gravity_ff requested but "
                      "no dynamics model; MIT will run kp/kd-only.")

        self._servo_opts = opts
        self._servo_last_target_turns = list(last_target)
        self._servo_filtered_target_turns = list(last_target)
        self._servo_active_ids_np = active_ids_np
        self._servo_hold_ids_np = hold_ids_np
        self._servo_active_pos_np = active_pos_np
        self._servo_active_vel_np = active_vel_np
        self._servo_max_motor_id = int(max(self._cfg.motor_ids))
        self._servo_max_torque = int(self._cfg.max_torque_raw)
        self._servo_active = True
        self._servo_tick_count = 0
        self._servo_clamp_count = 0
        self._servo_lag_count = 0
        self._servo_lag_streak = 0
        self._servo_lag_warned = False
        self._servo_aborted_reason = None
        self._servo_started_at = time.monotonic()

        ff_tag = "on" if opts.feedforward_vel else "off"
        la_tag = (f"{opts.lookahead_time * 1000:.0f}ms"
                  if opts.lookahead_time > 0 else "off")
        chan = ("MIT 0x8093 kp={} kd={} (raw)".format(
                    self._servo_mit_kp_raw, self._servo_mit_kd_raw)
                if opts.use_mit else "pos+vel+maxtqe 0x8090")
        print(
            f"[FafuRobot] servo_start: watchdog={opts.watchdog_ms}ms, "
            f"max_vel={opts.max_vel}rad/s, max_step={opts.max_step_rad}rad, "
            f"max_lag={opts.max_lag_rad}rad, "
            f"feedforward={ff_tag}, lookahead={la_tag}, rate={opts.rate_hz:.0f}Hz, "
            f"channel={chan}"
        )

    def servo_j(self, target_angles: Iterable[float]) -> bool:
        """Stream one joint-space target (non-blocking).

        Call repeatedly at a fixed cadence (recommended 100-200 Hz).
        Caller is responsible for the sleep between calls; we do not
        block to enforce a rate.

        Parameters
        ----------
        target_angles : iterable of float
            ``num_joints`` joint angles in the unit declared by
            :attr:`ServoOpts.is_radians` (radians by default).

        Returns
        -------
        bool
            * ``True``  — frame was sent.  ``last_target`` advanced.
              (Step- or limit-clamping does **not** count as failure;
              it just bumps :attr:`servo_clamp_count`.  Lag-trip does
              not count as failure either: the frame is still sent
              and :attr:`servo_lag_count` is bumped.)
            * ``False`` — payload was rejected and **no frame was
              sent**.  Reasons: wrong length, NaN/Inf, send exception,
              :meth:`servo_start` was never called, or an earlier
              ``lag_abort_consecutive`` event already terminated the
              session.  ``last_target`` stays untouched so the next
              ``servo_j`` will still see the same step delta.
        """
        if self._servo_aborted_reason is not None:
            # Session was auto-terminated earlier; refuse silently.
            # (The abort site already printed a banner.) Caller can
            # check :attr:`servo_aborted_reason` to know why.
            return False
        if not self._servo_active or self._servo_opts is None:
            print("[FafuRobot] servo_j: not in a servo session; call servo_start first")
            return False

        opts = self._servo_opts
        arr = np.asarray(list(target_angles), dtype=float)
        if arr.size != self.num_joints:
            print(f"[FafuRobot] servo_j: expected {self.num_joints} angles, "
                  f"got {arr.size}")
            return False
        if not np.all(np.isfinite(arr)):
            print("[FafuRobot] servo_j: target contains NaN/Inf, refused")
            return False

        N = self.num_joints
        dt = 1.0 / opts.rate_hz
        max_vel_tps = opts.max_vel / _TWO_PI    # rad/s -> turns/s

        # (a) Convert user-units -> turns (protocol native)
        if opts.is_radians:
            target_turns = [float(v) / _TWO_PI for v in arr]
        else:
            target_turns = [float(v) / 360.0 for v in arr]

        # (b) Defense 2 — per-step clamp on the *raw* upper-layer target,
        # measured against last successfully-sent filtered target. Without
        # this an upper-layer bug could push the EMA filter far in one
        # step (lookahead does NOT bound a single step, only smoothes it).
        max_step_turns = abs(opts.max_step_rad) / _TWO_PI
        was_clamped = False
        for i in range(N):
            delta = target_turns[i] - self._servo_last_target_turns[i]
            if abs(delta) > max_step_turns:
                target_turns[i] = (self._servo_last_target_turns[i]
                                   + math.copysign(max_step_turns, delta))
                was_clamped = True
        if was_clamped:
            self._servo_clamp_count += 1
            # No per-tick print — would dominate jitter on 100Hz loops.
            # The aggregated count is reported in servo_end.

        # (c) Lookahead smoothing — exponential moving average with time
        # constant ``lookahead_time``. Matches UR servoj's lookahead.
        # alpha = dt / (lookahead + dt); alpha=1 when lookahead=0 (no smoothing).
        if opts.lookahead_time > 0.0:
            alpha = dt / (opts.lookahead_time + dt)
            target_to_send = [
                self._servo_filtered_target_turns[i]
                + alpha * (target_turns[i] - self._servo_filtered_target_turns[i])
                for i in range(N)
            ]
        else:
            target_to_send = target_turns

        # (d) Per-joint velocity, written directly into the pre-allocated
        # buffer (no per-tick allocation):
        #   - feedforward_vel=True (default, UR-style):
        #         vel[i] = (target_to_send[i] - last_sent[i]) / dt, clamped to ±max_vel
        #     Eliminates the dominant "motor sprints to every micro-step"
        #     noise source. Tiny target changes get tiny vel commands.
        #   - feedforward_vel=False (legacy):
        #         vel[i] = max_vel_tps for every joint
        active_pos_buf = self._servo_active_pos_np
        active_vel_buf = self._servo_active_vel_np
        if opts.feedforward_vel:
            for i in range(N):
                active_pos_buf[i] = target_to_send[i]
                v = (target_to_send[i] - self._servo_filtered_target_turns[i]) / dt
                if v > max_vel_tps:    v = max_vel_tps
                elif v < -max_vel_tps: v = -max_vel_tps
                active_vel_buf[i] = v
        else:
            for i in range(N):
                active_pos_buf[i] = target_to_send[i]
            active_vel_buf.fill(max_vel_tps)

        # (e) Defense 3 — tracking-error monitor.
        #
        # ★ 2026-05 policy change ★
        # Old behaviour was "lag > max_lag_rad ⇒ return False ⇒ frame NOT
        # sent". On hardware that turned a single lagging joint into a
        # cascade: the firmware watchdog on the *other* joints fired
        # after watchdog_ms of silence and braked them too. Confirmed
        # with --no-safety-nets (lag check off): J4 still drooped but
        # J2/J3 actually completed their motion, so the lag check was
        # making things worse, not better.
        #
        # New behaviour: ALWAYS send the frame; just count lag events
        # for diagnostics. ``lag_abort_consecutive > 0`` opts back into
        # protective stop with a sensible debounce.
        lag_tripped_this_tick = False
        if opts.max_lag_rad > 0.0:
            max_lag_turns = opts.max_lag_rad / _TWO_PI
            for i, mid in enumerate(self._joint_motor_ids):
                s = self._ht.get_cached_state(mid)
                if s is None:
                    continue
                lag = abs(s.position - target_to_send[i])
                if lag > max_lag_turns:
                    lag_tripped_this_tick = True
                    break   # one bad joint is enough for the counter
        if lag_tripped_this_tick:
            self._servo_lag_count += 1
            self._servo_lag_streak += 1
            if not self._servo_lag_warned:
                self._servo_lag_warned = True
                print(f"[FafuRobot] servo_j: first lag-trip "
                      f"(|lag| > {opts.max_lag_rad:.3f}rad); "
                      f"frame is still being sent. "
                      f"Counter at servo_end will show the total.")
        else:
            self._servo_lag_streak = 0

        # (f) Defense 4 — soft limits handled inside set_many_pos_vel_tqe_partial.

        # (g) Send one frame on the configured channel.
        #   - default (use_mit=False): 0x8090 position+vel+maxtqe partial.
        #     Firmware position loop tracks; hold motors held at cached pos.
        #     Hot path is entirely C++ (numpy memcpy), ~5us per tick.
        #   - use_mit=True: 0x8093 group MIT (kp/kd + gravity feed-forward).
        #     pos/vel here are already in turns / turns-per-sec, exactly what
        #     set_many_mit wants (PosUnit.Turns); kp/kd raw precomputed in
        #     servo_start; gravity tau computed from the target pose each tick.
        try:
            if opts.use_mit:
                if opts.mit_gravity_ff and self.has_dynamics:
                    q_rad = np.asarray(target_to_send, dtype=float) * _TWO_PI
                    v_rad = active_vel_buf * _TWO_PI     # turns/s -> rad/s
                    tau = self.compute_compensation_torque(
                        q_rad, v_rad, friction=False) * self._dyn_torque_scale
                    tau_raw = [int(x) for x in self.tau_to_raw(tau)]
                else:
                    tau_raw = [0] * self.num_joints
                self._ht.set_many_mit(
                    list(self._joint_motor_ids),
                    [float(x) for x in active_pos_buf],
                    [float(x) for x in active_vel_buf],
                    tau_raw,
                    self._servo_mit_kp_raw,
                    self._servo_mit_kd_raw,
                    pm.PosUnit.Turns,
                    # MIT frame holds joints only (gripper excluded); slot count
                    # must be max JOINT id, NOT max of all motor_ids (which
                    # includes M7 gripper -> 7 slots -> 72B > 64B frame error).
                    self._servo_mit_max_motor_id,
                    0.0,                # async_rx is on, don't wait for replies
                )
            else:
                self._ht.set_many_pos_vel_tqe_partial(
                    self._servo_active_ids_np,
                    active_pos_buf,
                    active_vel_buf,
                    self._servo_max_torque,
                    self._servo_hold_ids_np,
                    pm.PosUnit.Turns,
                    self._servo_max_motor_id,
                    0.0,                # async_rx is on, don't wait for replies
                )
        except Exception as e:
            print(f"[FafuRobot] servo_j: send failed: {e}")
            return False

        # Advance state only after a successful send so the next tick's
        # clamp / feedforward see a consistent "last sent" reference.
        self._servo_last_target_turns = list(target_turns)
        self._servo_filtered_target_turns = list(target_to_send)
        self._servo_tick_count += 1

        # Optional protective-stop: if lag persisted for N ticks in a
        # row, brake and refuse further frames.  Default opts.value 0
        # means never auto-abort (preferred for diagnostic scripts).
        if (opts.lag_abort_consecutive > 0
                and self._servo_lag_streak >= opts.lag_abort_consecutive):
            reason = (f"lag_abort_consecutive={opts.lag_abort_consecutive} "
                      f"exceeded (streak={self._servo_lag_streak})")
            print(f"[FafuRobot] servo_j: PROTECTIVE STOP — {reason}. "
                  f"Calling servo_end('brake').")
            try:
                self.servo_end(finish_mode="brake")
            except Exception as e:
                print(f"[FafuRobot] servo_j: auto servo_end failed: {e}")
            self._servo_aborted_reason = reason
            return False

        return True

    def servo_end(self, finish_mode: str = "hold") -> None:
        """End the servo session and place every joint motor in
        ``finish_mode``.

        Parameters
        ----------
        finish_mode : {"hold", "brake", "stop"}, optional
            * ``"hold"`` (default): keep the motor in MODE_POSITION
              (``0x0A``) holding its last commanded frame.  Identical
              behaviour to the state every motor is in after a
              successful :meth:`move_j`, so you can immediately chain
              another ``move_j`` / ``servo_start`` without calling
              :meth:`enable` again.  Motors stay energised.
            * ``"brake"``: switch to short-circuit braking
              (``0x0F``).  No torque output, no current draw, but the
              joint resists motion (it may still drift slowly under
              load).  Use when you want to release the loop but keep
              the arm roughly in place without burning power.
            * ``"stop"``: PWM off (``0x00``).  Joints free to be moved
              by hand.  Use only when a human will reposition the arm
              right after, otherwise the arm will sag.

        Always clears the firmware watchdog first so that the mode
        change is not interrupted by the watchdog firing in the
        middle.  The gripper is left alone (we never touched it).
        """
        if not self._servo_active:
            print("[FafuRobot] servo_end: not in a servo session, ignored")
            return

        valid = {"stop", "brake", "hold"}
        if finish_mode not in valid:
            raise ValueError(
                f"finish_mode must be one of {sorted(valid)}, got {finish_mode!r}")

        opts = self._servo_opts
        # 1) clear watchdog FIRST so the mode switch is not pre-empted
        if opts is not None and opts.watchdog_ms > 0:
            for mid in self._joint_motor_ids:
                try:
                    self._ht.set_timeout(mid, 0)
                except Exception:
                    pass

        # 2) place motors in finish_mode
        used_mit = bool(opts is not None and getattr(opts, "use_mit", False))
        for mid in self._joint_motor_ids:
            try:
                if finish_mode == "stop":
                    self._ht.stop(mid)
                elif finish_mode == "brake":
                    self._ht.set_motor_mode(mid, self.MODE_BRAKE)
                else:  # "hold"
                    # A MIT session (0x8093) leaves the motor in a MIT control
                    # mode. Two problems if we just "leave it": (a) it holds
                    # only the last PD frame (no integral) and SAGS under
                    # gravity once we stop refreshing, and (b) the firmware
                    # then ignores subsequent 0x8090 position frames (move_j /
                    # go_home / pos-mode servo appear dead) because the
                    # broadcast 0x8090 frame cannot switch mode. So actively
                    # switch back to MODE_POSITION (0x0A): the firmware
                    # position loop (with integral) holds it, and 0x8090 works
                    # again. Non-MIT (0x8090) sessions already end in 0x0A, so
                    # only do this for MIT.
                    if used_mit:
                        self._ht.set_motor_mode(mid, self.MODE_POSITION)
                    # else: leave in MODE_POSITION with last frame intact
            except Exception:
                pass

        elapsed = max(1e-3, time.monotonic() - self._servo_started_at)
        rate = self._servo_tick_count / elapsed
        warn_tag = ""
        if self._servo_clamp_count or self._servo_lag_count:
            warn_tag = (f"   [warn: step-clamp={self._servo_clamp_count}, "
                        f"lag-trip={self._servo_lag_count}]")
        if self._servo_aborted_reason is not None:
            warn_tag += f"   [auto-aborted: {self._servo_aborted_reason}]"
        print(f"[FafuRobot] servo_end ({finish_mode}): "
              f"{self._servo_tick_count} ticks in {elapsed:.2f}s "
              f"(~{rate:.1f} Hz){warn_tag}")

        self._servo_active = False
        # NOTE: tick_count / clamp_count / lag_count / aborted_reason
        # are intentionally NOT zeroed here so user code can read them
        # via :attr:`servo_lag_count` etc. *after* the loop ends.
        # They are reset at the start of the next servo_start.
        self._servo_lag_streak = 0
        self._servo_last_target_turns = []
        self._servo_filtered_target_turns = []
        self._servo_active_ids_np = None
        self._servo_hold_ids_np = None
        self._servo_active_pos_np = None
        self._servo_active_vel_np = None
        self._servo_max_motor_id = 0
        self._servo_max_torque = 0

    @property
    def is_servoing(self) -> bool:
        """``True`` while a :meth:`servo_start` / :meth:`servo_end`
        window is open."""
        return self._servo_active

    @property
    def servo_lag_count(self) -> int:
        """Number of lag-tripped ticks observed in the current (or
        most recently ended) servo session.

        A "lag-trip" is a tick where some joint's measured position
        differs from the last sent target by more than
        :attr:`ServoOpts.max_lag_rad`.  The frame is **still** sent on
        a lag-trip (see :meth:`servo_j` docstring); this counter is
        purely informational unless ``lag_abort_consecutive`` is in
        use.
        """
        return int(self._servo_lag_count)

    @property
    def servo_clamp_count(self) -> int:
        """Number of step-clamped ticks in the current / last session
        (caller asked for a jump > ``max_step_rad`` and we limited it)."""
        return int(self._servo_clamp_count)

    @property
    def servo_aborted_reason(self) -> Optional[str]:
        """Non-``None`` iff the current session was auto-terminated by
        ``lag_abort_consecutive``.  Cleared by the next :meth:`servo_start`.
        While set, :meth:`servo_j` refuses every frame and returns ``False``.
        """
        return self._servo_aborted_reason

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
    #    3) Per-joint ``motor_models`` configured so Nm -> raw int16 is
    #       physically correct (otherwise the arm under-drives and sags —
    #       which is *safe* but not useful).
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
            Valid keys are the ones in the driver's ``TORQUE_COEFF`` table
            (e.g. ``"M7256_35"``, ``"M60BM_35"``, ``"M4438_32"``,
            ``"M3536_32"`` ...).  When ``None`` every joint uses ``""``
            (coefficient ``1.0``): the loop still runs but the torque is
            *not* physically scaled, so the arm will merely sag — safe,
            but you must fill these in for real compensation.
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
            self._dyn_motor_models = [str(m) for m in motor_models]
        else:
            self._dyn_motor_models = None
            print("[FafuRobot] setup_dynamics: WARNING no motor_models given; "
                  "torque will NOT be physically scaled (coeff=1.0). The arm "
                  "will under-drive / sag. Pass motor_models for real "
                  "compensation.")

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
        """Pick the end-effector frame used by FK/IK.

        Preference order: explicit ``eef_frame`` arg → ``"tool_link"`` →
        the child frame of the last actuated joint.  Stored on the
        instance as ``_eef_frame_id`` / ``_eef_frame_name``.
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
        self._dyn_torque_scale = arr

    def tau_to_raw(self, tau: Iterable[float]) -> np.ndarray:
        """Convert a per-joint torque (Nm) to the firmware raw int16 that
        :meth:`apply_compensation_torque` would actually send.

        ``raw = round(tau / (coeff * 0.01))`` per joint, where ``coeff``
        comes from the joint's motor model (1.0 if no model configured).
        The ``*0.01`` matches the firmware torque LSB (vendor motor.cpp
        ``tqe_float2int``); it MUST mirror the C++ ``set_torque`` path.
        Handy for the ``dry_run`` calibration preview.  Does **not** apply
        the torque-scale gain (pass already-scaled torque if you want that).
        """
        tau = np.asarray(list(tau), dtype=float)
        raw = np.zeros(self.num_joints, dtype=np.int64)
        for i in range(self.num_joints):
            model = (self._dyn_motor_models[i]
                     if self._dyn_motor_models is not None else "")
            coeff = TORQUE_COEFF.get(model, 1.0)
            v = tau[i] / (coeff * 0.01) if coeff != 0.0 else 0.0
            raw[i] = int(np.clip(round(v), -32768, 32767))
        return raw

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
        tau = np.asarray(list(tau), dtype=float)
        if tau.shape != (self.num_joints,):
            raise ValueError(
                f"tau must have {self.num_joints} elements, got {tau.shape}")
        # Empirical calibration gain (1.0 by default).
        tau = tau * self._dyn_torque_scale

        if self._use_group_mit and self.num_joints <= 6:
            # One 0x8093 frame, kp=kd=0 => pure torque feed-forward. pos/vel
            # are multiplied by 0 in-firmware so their value is irrelevant; send
            # zeros. Pre-scaled tau -> raw here, so pass raw ints directly.
            raw = self.tau_to_raw(tau)   # tau already * torque_scale above
            zeros = [0.0] * self.num_joints
            try:
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
            except Exception as e:
                print(f"[FafuRobot] apply_compensation_torque (group MIT) "
                      f"failed: {e}")
            return

        for i, mid in enumerate(self._joint_motor_ids):
            model = (self._dyn_motor_models[i]
                     if self._dyn_motor_models is not None else "")
            try:
                self._ht.set_torque(mid, float(tau[i]), model)
            except Exception as e:
                print(f"[FafuRobot] apply_compensation_torque: motor {mid} "
                      f"failed: {e}")

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
        n = self.num_joints
        if n > 6:
            raise RuntimeError(
                f"move_MIT: group MIT frame holds <=6 motors, but "
                f"num_joints={n}. Use --gripper-id so the gripper is excluded, "
                f"or drive extra joints on a separate frame.")
        pos = np.asarray(list(pos), dtype=float)
        vel = np.asarray(list(vel), dtype=float)
        tau = np.asarray(list(tau), dtype=float)
        for name, arr in (("pos", pos), ("vel", vel), ("tau", tau)):
            if arr.shape != (n,):
                raise ValueError(
                    f"{name} must have {n} elements, got {arr.shape}")

        def _as_vec(g) -> np.ndarray:
            if np.isscalar(g):
                return np.full(n, float(g))
            g = np.asarray(list(g), dtype=float)
            if g.shape != (n,):
                raise ValueError(f"kp/kd must be scalar or {n} elements")
            return g

        kp_v = _as_vec(kp)
        kd_v = _as_vec(kd)

        if not kp_kd_raw:
            # Vendor kp_float2int / kd_float2int (radian_2pi convention):
            #   raw = int16( (gain / coeff) * 10 * 2*pi )
            # coeff = per-joint torque coefficient (same table as tau_to_raw).
            kp_raw = np.zeros(n)
            kd_raw = np.zeros(n)
            for i in range(n):
                model = (self._dyn_motor_models[i]
                         if self._dyn_motor_models is not None else "")
                coeff = TORQUE_COEFF.get(model, 1.0) or 1.0
                kp_raw[i] = (kp_v[i] / coeff) * 10.0 * _TWO_PI
                kd_raw[i] = (kd_v[i] / coeff) * 10.0 * _TWO_PI
            kp_v, kd_v = kp_raw, kd_raw

        kp_out = [int(np.clip(round(x), -32768, 32767)) for x in kp_v]
        kd_out = [int(np.clip(round(x), -32768, 32767)) for x in kd_v]

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

        The arm becomes weightless and can be guided by hand — the classic
        teaching / lead-through mode.  Mirrors the reference
        ``while True: ...`` loop but adds rate control, a duration cap, a
        ``dry_run`` preview and Ctrl+C-safe teardown.

        ★ SAFETY ★
          * Always test with ``dry_run=True`` first and read the printed
            torques: if any column is wildly larger than the joint's
            ``tau_limit`` your ``motor_models`` / URDF are wrong.
          * On exit (normal, Ctrl+C, or exception) every joint is put in
            ``stop`` (free) mode and the firmware will hold nothing — keep a
            hand on the arm / E-stop, heavy links will drop.
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
        if not self.is_enabled:
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
        try:
            while True:
                tick_start = time.monotonic()
                if abort_check is not None and abort_check():
                    print("[FafuRobot] gravity-comp: abort_check -> stop")
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
            # SAFER teardown: instead of dropping to free-spin (limp -> heavy
            # links fall), re-engage *position hold* at the current pose via
            # the verified position channel (set_pos_vel_acc, NOT a torque
            # frame).  This stops any leftover feed-forward torque and keeps
            # the arm where it is.
            if dry_run:
                pass
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
                    # brake (0x0F) -> position (0x0A).  Use _switch_mode_all
                    # DIRECTLY rather than enable(): enable()'s Stage-0 pre-reads
                    # every motor (7 x 0.2s ~= 1.4s of "nothing happening")
                    # which is the dead window that tempts a second Ctrl+C and
                    # aborts the homing.  Only fall back to the heavier enable()
                    # (motor_reset recovery) if the quick switch genuinely fails.
                    if not self._switch_mode_all(self.MODE_POSITION,
                                                 label="position", max_retry=3):
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
        for mid in self._joint_motor_ids:
            try:
                self._ht.set_motor_mode(mid, self.MODE_BRAKE)
            except Exception:
                try:
                    self._ht.stop(mid)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    #  Cartesian motion (placeholders)
    # ------------------------------------------------------------------
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
            Raw int16 ``|torque|`` value that, when exceeded while
            blocking, causes an early return.  This is the lightweight
            "did I grab something?" check.  When ``None`` (default)
            only position / stall / timeout are used.  When given,
            this method returns a :class:`GraspResult` describing the
            outcome; otherwise it returns ``None`` for backwards
            compatibility with the Piper-style void signature.

        Returns
        -------
        GraspResult or None
            A :class:`GraspResult` iff ``block=True`` AND a stop
            reason was recorded (i.e. ``effort_threshold`` was given,
            or the stall / target conditions fired). Returns ``None``
            otherwise.
        """
        if not self._has_gripper:
            raise RuntimeError(
                "FafuRobotController was constructed without a gripper"
            )
        pos_turns = self._rad_to_turns(angle) if is_radians else angle / 360.0

        if effort is None:
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
                int(effort),
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
            effort_threshold=effort_threshold,
        )
        if effort_threshold is None:
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
        the closing direction and stops as soon as the **measured**
        torque exceeds ``force_threshold`` (or the gripper plateaus
        at finite stiffness against an object).

        Because Fafu's firmware does not implement the position
        + effort loop itself, the detection logic runs in Python by
        polling ``MotorState.torque`` from the background poller
        (``start_state_polling``).  This is good enough for catching
        objects but is **not** a substitute for a real wrist F/T
        sensor.

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
            ``set_pos_vel_tqe`` (raw int16).  When ``None`` the
            firmware uses its default current limit.  Pass this if
            you also want a hard hardware-level torque ceiling, not
            just the soft Python-side trip wire.
        vel : float, optional
            Closing velocity in turns/s.  Defaults to 0.15 turns/s
            (deliberately slower than ``open_gripper`` so that
            contact is gentle).
        acc : float, optional
            Acceleration limit in turns/s^2.  Ignored when ``effort``
            is provided (``set_pos_vel_tqe`` has no acceleration arg).
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

        if target_angle is None:
            lo_t, _ = self._gripper_limit_turns()
            target_turns = lo_t if lo_t is not None else -0.25
        else:
            target_turns = (
                self._rad_to_turns(target_angle) if is_radians
                else target_angle / 360.0
            )

        if effort is None:
            self._ht.set_pos_vel_acc(
                self._gripper_motor_id,
                target_turns,
                float(vel),
                float(acc),
                pm.PosUnit.Turns,
            )
        else:
            self._ht.set_pos_vel_tqe(
                self._gripper_motor_id,
                target_turns,
                float(vel),
                int(effort),
                pm.PosUnit.Turns,
            )

        return self._wait_until_gripper_done(
            target_turns,
            timeout=float(timeout),
            effort_threshold=int(force_threshold),
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
        """Immediately stop every motor (free-spin mode)."""
        for mid in self._cfg.motor_ids:
            try:
                self._ht.stop(mid)
            except Exception:
                pass
        print("[FafuRobot] EMERGENCY STOP issued (all motors → mode 0x00).")

    def resume(self) -> None:
        """Re-enable position control after an emergency stop."""
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

    def reset_zero(self, motor_id: int, *, confirm: bool = False) -> None:
        """Set the current position of ``motor_id`` as the new zero.

        ``confirm=True`` is required because this is destructive.
        """
        if not confirm:
            raise RuntimeError("reset_zero is destructive; pass confirm=True")
        if motor_id not in self._cfg.motor_ids:
            raise ValueError(f"motor {motor_id} not in cfg.motor_ids")
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
        """Stop background tasks and close the serial port.

        Parameters
        ----------
        joint_release : {"stop", "brake", "hold"}, optional
            What to do with the manipulator joints before closing.

            * ``"stop"`` (default): mode 0x00, PWM off, joints free
              to be moved by hand. Safest if a human will reposition
              the arm. Heavy joints may sag slightly under gravity.
            * ``"brake"``: mode 0x0F, short-circuit damping. No
              current required, but resists motion. Use this if you
              want the pose to roughly hold without keeping the
              motors energised.
            * ``"hold"``: keep mode 0x0A, motors actively hold their
              last commanded position. Uses current; do not leave
              like this for long.
        gripper_release : {"stop", "brake", "hold"}, optional
            Same options for the gripper.  Defaults to ``"brake"`` so
            that the gripper does **not** drift open the moment the
            program disconnects (the previous default of ``"stop"``
            caused exactly that issue).
        """
        # If a servo session is still open, end it gracefully BEFORE we
        # stop polling / disable async RX. servo_end clears the firmware
        # watchdog (which would otherwise fire as soon as we stop
        # sending frames) and parks the joints in ``joint_release`` mode.
        if self._servo_active:
            try:
                self.servo_end(finish_mode=joint_release)
            except Exception as e:
                print(f"[FafuRobot] close_connection: servo_end fallback failed: {e}")

        try:
            if self._ht.is_polling():
                self._ht.stop_state_polling()
        except Exception:
            pass
        try:
            if self._ht.is_async_rx():
                self._ht.disable_async_rx()
        except Exception:
            pass

        valid = {"stop", "brake", "hold"}
        if joint_release not in valid:
            raise ValueError(f"joint_release must be one of {valid}")
        if gripper_release not in valid:
            raise ValueError(f"gripper_release must be one of {valid}")

        mode_map = {
            "stop":  self.MODE_STOP,
            "brake": self.MODE_BRAKE,
            "hold":  self.MODE_POSITION,
        }
        for mid in self._cfg.motor_ids:
            policy = (gripper_release if (self._has_gripper
                                          and mid == self._gripper_motor_id)
                      else joint_release)
            try:
                if policy == "stop":
                    self._ht.stop(mid)
                else:
                    # brake (0x0F) or hold (0x0A) — both are mode changes,
                    # not the dedicated stop()/brake() RPCs.
                    self._ht.set_motor_mode(mid, mode_map[policy])
            except Exception:
                pass
        try:
            self._ht.close()
        except Exception:
            pass
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
        pref = (preferred or "").strip()
        is_auto = (not pref) or pref.lower() == "auto"
        try:
            usb_ports = pm.find_likely_debug_boards()
        except Exception as e:
            print(f"[FafuRobot] failed to enumerate ports: {e}")
            return preferred or ""
        if not is_auto:
            for p in usb_ports:
                if p.port == pref:
                    return pref
            print(
                f"[FafuRobot] preferred port {pref!r} not found; "
                f"falling back to auto"
            )
        if not usb_ports:
            raise RuntimeError(
                "no USB debug board detected; check the cable and try again"
            )
        return usb_ports[0].port

    def _precheck_communication(self) -> None:
        bad: List[int] = []
        for mid in self._cfg.motor_ids:
            s = self._ht.read_motor_state(mid, 0.5)
            if s is None:
                bad.append(mid)
        if bad:
            raise RuntimeError(
                f"motors {bad} did not respond within 500ms; "
                f"check power, wiring and motor IDs"
            )

    def _switch_mode_all(
        self,
        mode: int,
        *,
        label: str,
        max_retry: int = 3,
    ) -> bool:
        """Drive every motor to ``mode``, verifying the actual state.

        Returns True as soon as every motor's *measured* state is at
        ``mode``.  This intentionally trusts the post-switch read
        rather than ``set_motor_mode``'s return value, because that
        return value can legitimately be ``None`` on a busy /
        freshly-reopened port even when the mode change succeeded
        (the binding's internal "switch -> sleep -> read confirm"
        sequence is timing-sensitive).  We back-stop by issuing a
        fresh read on every motor that the binding flagged as
        failing, and only treat it as a real failure if the read
        confirms a wrong mode.
        """
        for attempt in range(1, max_retry + 1):
            failed: List[int] = []
            for mid in self._cfg.motor_ids:
                s = self._ht.set_motor_mode(mid, mode)
                if s is None or int(s.mode) != mode:
                    failed.append(mid)

            # ★ Post-failure verify: do a fresh read on every "failed"
            # motor to filter out binding false negatives. If the read
            # confirms the target mode is already there, the motor is
            # actually fine — common case after a sync-read timeout
            # inside set_motor_mode(0x0A).
            if failed:
                time.sleep(0.03)
                really_failed: List[int] = []
                for mid in failed:
                    s = self._ht.read_motor_state(mid, 0.3)
                    if s is None or int(s.mode) != mode:
                        really_failed.append(mid)
                failed = really_failed

            if not failed:
                return True
            if attempt < max_retry:
                print(
                    f"[FafuRobot] switch to {label}: retry {attempt + 1}, "
                    f"failed motors = {failed}"
                )
                time.sleep(0.1)
        return False

    def _validate_joint_angles(
        self,
        joint_angles: Iterable[float],
        is_radians: bool,
    ) -> List[float]:
        """Validate length and convert input angles to **turns**."""
        arr = np.asarray(list(joint_angles), dtype=float)
        if arr.size != self.num_joints:
            raise ValueError(
                f"expected {self.num_joints} joint values, got {arr.size}"
            )
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

            self._ht.set_many_pos_vel_tqe(
                cmds, pm.PosUnit.Turns, max_mid, 0.002,
            )
            return True

        # The control loop runs on the C++ side with GIL released.
        rc = self._ht.run_control_loop(
            rate_hz,
            list(self._cfg.motor_ids),
            on_tick,
            abort_check=abort_check,
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
            print("[FafuRobot] warning: control loop exited abnormally")

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
        self._ht.set_many_pos_vel_tqe(cmds, pm.PosUnit.Turns, max_mid, 0.05)

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
            if abort_check is not None and abort_check():
                self._last_cmd_turns = None
                raise RuntimeError(
                    "move_j(style=linear) aborted (abort_check returned True)"
                )
            # keep-alive resend (fire-and-forget, no reply wait) — refreshes
            # the position loop / watchdog exactly like the S-curve loop.
            self._ht.set_many_pos_vel_tqe(cmds, pm.PosUnit.Turns, max_mid, 0.0)
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

        for mid, tgt in targets_turns.items():
            try:
                self._ht.set_pos_vel_acc(
                    int(mid), float(tgt), float(v_max), float(acc),
                    pm.PosUnit.Turns,
                )
            except Exception as e:
                print(f"[FafuRobot] move_j(style=acc): motor {mid} failed: {e}")

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
            if abort_check is not None and abort_check():
                self._last_cmd_turns = None
                raise RuntimeError(
                    "move_j(style=acc) aborted (abort_check returned True)"
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
