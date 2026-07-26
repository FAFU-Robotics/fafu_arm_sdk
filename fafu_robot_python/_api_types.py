"""Public value types shared by the high-level Python SDK.

This module has no dependency on the controller or native extension, avoiding
circular imports and keeping passive types usable in hardware-free tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional

import numpy as np


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
        <any connected> --power/CAN lost---> DEAD     --recover()--> BRAKED
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
        (~57 deg/s). Position frames use it as a non-negative velocity
        limit; MIT frames use it as the absolute bound on signed desired
        velocity.
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
        Compatibility guard; only ``True`` is accepted.
    rate_hz : float, optional
        Nominal upper-layer call rate **only used to compute dt for
        feedforward and lookahead**.  Default ``100.0``.  The actual
        call rate is still set by however fast the caller invokes
        :meth:`servo_j`; this number does not throttle anything.
    feedforward_vel : bool, optional
        Default ``True``. For the Position channel, the per-frame ``vel``
        field is a **non-negative limit**: the larger of target path speed
        and measured-error catch-up speed, clamped to ``max_vel``. A
        stationary target therefore keeps converging until it enters
        ``position_error_deadband_rad``. When ``False``, Position uses a
        fixed positive ``max_vel``. For MIT, ``True`` retains signed
        finite-difference desired velocity and ``False`` sends zero.
    position_error_deadband_rad : float, optional
        Position tracking-error deadband in radians for the position
        channel.  A stationary target inside this band is allowed to settle
        with zero velocity limit, reducing tiny corrective motions.  Default
        ``0.001``.
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
    # Appended to preserve the positional order of every pre-existing field.
    # Prefer keyword arguments for all ServoOpts construction.
    position_error_deadband_rad: float = 0.001


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

__all__ = [
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
