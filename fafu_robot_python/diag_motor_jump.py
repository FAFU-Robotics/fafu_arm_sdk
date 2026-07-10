"""
诊断脚本：检查 J5 / J6 (或其他关节) 在 servo_j 命令之外，是否会出现位置反馈瞬变。

包含两个独立测试：

  --mode static
      只 enable 电机、不下发任何运动命令，纯读取 get_joint_values()。
      如果电机静止时仍然出现 >5° 的相邻两帧瞬变，说明：
        (1) cached_state / async_rx 数据不同步，或
        (2) 多圈编码器有 wrap-around，或
        (3) 电机驱动板返回数据本身就跳。
      ★ 这种情况是真机/驱动层问题，跟 servo_j 完全无关。

  --mode movej
      让 6 个关节做一段小幅度的 move_j 运动，再回到原位。
      move_j 走的是 S 曲线轨迹（不是 servo_j 高频流），如果这种模式下
      J5/J6 也跳，说明跳变来源不是 servo_j。

  --mode servo-hold
      启动 servo_j 模式，命令永远等于"当前位置"，运行 N 秒。
      电机理应纹丝不动。如果在此模式下 J5/J6 仍然出现跳变，
      说明问题是 servo_j 内部状态或 async_rx 在高频 TX 时的竞态。

  --mode servo-sin
      启动 servo_j 模式，命令是"当前位置 + 1° 正弦扰动"，运行 N 秒。
      电机会做幅度 ±1° 的缓慢摆动。如果只有此模式跳变（而 servo-hold
      不跳），说明问题是命令变化才触发的 partial set 路径。

  --mode servo-ramp
      启动 servo_j 模式，让指定的关节（默认 J2 + J3）以恒定速度
      同步缓慢上升，到达 --ramp-target-deg 后停住。模拟 teleop
      抬升过程，但排除 IK 影响。如果某个角度处突然出现跳变，
      可精准定位到这个角度（即"高度"）的问题。

输出：
  - 每秒打印一次最近 1 秒的：max(meas_step) / 平均 meas_step / 当前位姿
  - 任意一帧 meas_step > --jump-threshold 立刻打印 [JUMP] 详情
  - 退出时打印每个关节的总跳变事件数

用法（在仓库根目录）：
  python fafu_robot_sdk/fafu_robot_python/diag_motor_jump.py --mode static --duration 30
  python fafu_robot_sdk/fafu_robot_python/diag_motor_jump.py --mode movej --duration 30

按 Ctrl-C 可随时退出，电机会保留在 hold 状态。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Windows high-resolution timer
# ---------------------------------------------------------------------------
# Default Windows system timer resolution is 15.6 ms; that means
# ``time.sleep(0.01)`` actually sleeps ~15 ms, so a "100Hz" servo loop
# silently degrades to ~60-65Hz with massive jitter, AND it makes every
# diagnostic measurement (jump events / track_err) impossible to compare
# across runs.
#
# ``winmm.timeBeginPeriod(1)`` puts the whole process in 1ms timer mode
# until ``timeEndPeriod(1)`` is called. We RAII it through atexit so a
# Ctrl+C / unexpected return still restores the system default.
# ---------------------------------------------------------------------------
def _enable_high_res_timer() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import atexit
        winmm = ctypes.WinDLL("winmm")
        if winmm.timeBeginPeriod(1) == 0:   # 0 == TIMERR_NOERROR
            atexit.register(lambda: winmm.timeEndPeriod(1))
            print("[INIT] winmm.timeBeginPeriod(1) enabled — 1ms timer.")
        else:
            print("[INIT][WARN] timeBeginPeriod(1) refused; "
                  "loop may show 15ms jitter on Windows.")
    except Exception as exc:
        print(f"[INIT][WARN] could not enable 1ms timer: {exc}")


def _resolve_default_cfg() -> str:
    """Locate robot.cfg next to this script, or in a sibling fafu_robot_python/.

    The script lives inside fafu_robot_python/ in the current layout (so the
    cfg is its sibling); we also keep a fallback for the legacy layout where
    the script was at the parent directory and the cfg was inside a
    fafu_robot_python/ subdir.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "robot.cfg", here / "fafu_robot_python" / "robot.cfg"):
        if cand.is_file():
            return str(cand)
    # Nothing matched — return the most likely path so the [ERROR] message
    # points the user at where we expected the file.
    return str(here / "robot.cfg")


def _import_controller():
    """Import FafuRobotController, adding the package dir to sys.path if needed."""
    here = Path(__file__).resolve().parent
    pkg_dir = here / "fafu_robot_python"
    if str(pkg_dir) not in sys.path:
        sys.path.insert(0, str(pkg_dir))
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        from fafu_robot_python.fafu_robot_controller import FafuRobotController  # type: ignore
    except Exception:
        from fafu_robot_controller import FafuRobotController  # type: ignore
    return FafuRobotController


def run_static(arm, duration_s: float, jump_threshold_deg: float) -> None:
    """Mode A: never send any motion command, only poll position feedback."""
    print(f"[STATIC] duration={duration_s:.1f}s, jump_threshold={jump_threshold_deg:.2f}deg")
    print("[STATIC] motors are in position-hold from enable(); no servo_j / move_j is issued.")
    print("[STATIC] starting feedback poll loop ...")

    num_joints = arm.num_joints
    jump_count = np.zeros(num_joints, dtype=np.int64)
    max_jump = np.zeros(num_joints, dtype=float)

    prev = arm.get_joint_values()
    start_t = time.time()
    last_print_t = start_t
    window_max = np.zeros(num_joints, dtype=float)
    window_sum = np.zeros(num_joints, dtype=float)
    window_count = 0

    print(f"[STATIC] init q (deg) = {np.rad2deg(prev)}")
    try:
        while time.time() - start_t < duration_s:
            cur = arm.get_joint_values()
            dq_deg = np.rad2deg(np.abs(cur - prev))
            for i in range(num_joints):
                if dq_deg[i] > max_jump[i]:
                    max_jump[i] = dq_deg[i]
                if dq_deg[i] > jump_threshold_deg:
                    jump_count[i] += 1
                    print(
                        f"[STATIC][JUMP] t={time.time()-start_t:6.2f}s J{i+1}: "
                        f"dq={dq_deg[i]:.2f}deg, prev={math.degrees(prev[i]):+.2f}, "
                        f"cur={math.degrees(cur[i]):+.2f}"
                    )
            window_max = np.maximum(window_max, dq_deg)
            window_sum += dq_deg
            window_count += 1

            now_t = time.time()
            if now_t - last_print_t >= 1.0:
                avg = window_sum / max(window_count, 1)
                print(
                    f"[STATIC] t={now_t-start_t:5.1f}s "
                    f"max_step(deg)={np.array2string(window_max, precision=3, suppress_small=True)} "
                    f"avg_step(deg)={np.array2string(avg, precision=4, suppress_small=True)} "
                    f"q(deg)={np.array2string(np.rad2deg(cur), precision=2, suppress_small=True)}"
                )
                window_max[:] = 0.0
                window_sum[:] = 0.0
                window_count = 0
                last_print_t = now_t

            prev = cur
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[STATIC] interrupted by user")

    print("\n[STATIC] ===== summary =====")
    for i in range(num_joints):
        print(
            f"  J{i+1}: jump_events(>{jump_threshold_deg:.1f}deg)={jump_count[i]}, "
            f"max_observed_step={max_jump[i]:.3f}deg"
        )


def _build_servo_opts(arm, *, watchdog_ms: int = 350, max_lag_deg: float = 35.0):
    """Construct a ServoOpts instance matching the teleop bridge defaults.

    ``watchdog_ms`` and ``max_lag_deg`` are exposed so the diagnostic harness
    can turn both safety nets off and check whether observed "J4 droops mid
    ramp" cascades originate from our own defenses misfiring (servo_j returns
    False -> no frames sent -> firmware watchdog after 350ms -> motors lose
    torque -> gravity pulls J4 down). When either argument is <=0 a loud
    runtime warning is printed.
    """
    mod = sys.modules.get(arm.__class__.__module__)
    if mod is None or not hasattr(mod, "ServoOpts"):
        raise RuntimeError("Cannot locate ServoOpts class in fafu_robot_controller module")
    ServoOpts = getattr(mod, "ServoOpts")
    kwargs = dict(
        watchdog_ms=int(watchdog_ms),
        max_vel=3.0,
        max_step_rad=math.radians(8.0),
        max_lag_rad=math.radians(max_lag_deg),
        is_radians=True,
    )
    ann = getattr(ServoOpts, "__annotations__", {})
    if "rate_hz" in ann:
        kwargs["rate_hz"] = 100.0
    if "feedforward_vel" in ann:
        kwargs["feedforward_vel"] = True
    if "lookahead_time" in ann:
        kwargs["lookahead_time"] = 0.05

    if watchdog_ms <= 0 or max_lag_deg <= 0:
        print("=" * 70)
        print("!! SAFETY NETS DISABLED — DIAGNOSTIC MODE !!")
        print(f"   watchdog_ms = {watchdog_ms}  "
              "(0 = firmware will NOT brake if host hangs)")
        print(f"   max_lag_deg = {max_lag_deg}  "
              "(0 = servo_j will NOT reject bad-lag frames)")
        print("   Keep your hand on the E-stop.")
        print("=" * 70)

    return ServoOpts(**kwargs)


def _run_servo_loop(
    arm,
    duration_s: float,
    jump_threshold_deg: float,
    *,
    sin_amplitude_deg: float = 0.0,
    sin_period_s: float = 4.0,
    tag: str = "SERVO",
    watchdog_ms: int = 350,
    max_lag_deg: float = 35.0,
) -> None:
    """Common servo_j loop: command = current_pose + optional sin perturbation."""
    num_joints = arm.num_joints
    jump_count = np.zeros(num_joints, dtype=np.int64)
    max_jump = np.zeros(num_joints, dtype=float)
    reject_count = 0

    opts = _build_servo_opts(arm, watchdog_ms=watchdog_ms, max_lag_deg=max_lag_deg)
    arm.servo_start(opts)
    print(f"[{tag}] servo_start done. duration={duration_s:.1f}s, "
          f"jump_threshold={jump_threshold_deg:.2f}deg, "
          f"sin_amp={sin_amplitude_deg:.2f}deg, sin_period={sin_period_s:.1f}s")

    q_base = arm.get_joint_values().copy()
    print(f"[{tag}] q_base (deg) = {np.rad2deg(q_base)}")
    prev = q_base.copy()
    start_t = time.time()
    last_print_t = start_t
    window_max = np.zeros(num_joints, dtype=float)
    window_sum = np.zeros(num_joints, dtype=float)
    window_count = 0

    amp_rad = math.radians(sin_amplitude_deg)
    omega = 2.0 * math.pi / max(sin_period_s, 1e-3)

    try:
        while time.time() - start_t < duration_s:
            t = time.time() - start_t
            # Same sin offset broadcast to all joints; small enough not to
            # trip servo_j step clamp (default 8 deg / cycle).
            offset = amp_rad * math.sin(omega * t) if amp_rad > 0 else 0.0
            q_target = q_base + offset
            ok = arm.servo_j(q_target)
            if not ok:
                reject_count += 1

            cur = arm.get_joint_values()
            dq_deg = np.rad2deg(np.abs(cur - prev))
            for i in range(num_joints):
                if dq_deg[i] > max_jump[i]:
                    max_jump[i] = dq_deg[i]
                if dq_deg[i] > jump_threshold_deg:
                    jump_count[i] += 1
                    track_err = np.rad2deg(np.abs(cur - q_target))
                    print(
                        f"[{tag}][JUMP] t={t:6.2f}s J{i+1}: "
                        f"dq={dq_deg[i]:.2f}deg, prev={math.degrees(prev[i]):+.2f}, "
                        f"cur={math.degrees(cur[i]):+.2f}, "
                        f"target={math.degrees(q_target[i]):+.2f}, "
                        f"track_err={track_err[i]:.2f}deg"
                    )
            window_max = np.maximum(window_max, dq_deg)
            window_sum += dq_deg
            window_count += 1

            now_t = time.time()
            if now_t - last_print_t >= 1.0:
                avg = window_sum / max(window_count, 1)
                track_err_deg = np.rad2deg(np.abs(cur - q_target))
                print(
                    f"[{tag}] t={t:5.1f}s "
                    f"max_step(deg)={np.array2string(window_max, precision=3, suppress_small=True)} "
                    f"avg_step(deg)={np.array2string(avg, precision=4, suppress_small=True)} "
                    f"track_err(deg)={np.array2string(track_err_deg, precision=2, suppress_small=True)} "
                    f"reject={reject_count}"
                )
                window_max[:] = 0.0
                window_sum[:] = 0.0
                window_count = 0
                last_print_t = now_t

            prev = cur
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"[{tag}] interrupted by user")
    finally:
        try:
            arm.servo_end(finish_mode="hold")
        except Exception as exc:
            print(f"[{tag}][WARN] servo_end failed: {exc}")

    print(f"\n[{tag}] ===== summary =====")
    print(f"  total servo_j rejects (hard, no-frame-sent): {reject_count}")
    # Lag-trip count: tick where some joint lag > max_lag_rad. Since 2026-05
    # the frame is STILL sent on a lag-trip (counter only); old behaviour was
    # "drop frame", which caused the firmware watchdog cascade.
    try:
        lag_count = int(getattr(arm, "servo_lag_count", 0))
        clamp_count = int(getattr(arm, "servo_clamp_count", 0))
        print(f"  total lag-trips (frame still sent): {lag_count}")
        print(f"  total step-clamps                  : {clamp_count}")
    except Exception:
        pass
    for i in range(num_joints):
        print(
            f"  J{i+1}: jump_events(>{jump_threshold_deg:.1f}deg)={jump_count[i]}, "
            f"max_observed_step={max_jump[i]:.3f}deg"
        )


def run_servo_hold(arm, duration_s: float, jump_threshold_deg: float,
                   *, watchdog_ms: int = 350, max_lag_deg: float = 35.0) -> None:
    """servo_j with constant target = initial pose."""
    _run_servo_loop(arm, duration_s, jump_threshold_deg,
                    sin_amplitude_deg=0.0, tag="SERVO-HOLD",
                    watchdog_ms=watchdog_ms, max_lag_deg=max_lag_deg)


def run_servo_sin(arm, duration_s: float, jump_threshold_deg: float,
                  amp_deg: float, period_s: float,
                  *, watchdog_ms: int = 350, max_lag_deg: float = 35.0) -> None:
    """servo_j with target = initial + sin perturbation."""
    _run_servo_loop(arm, duration_s, jump_threshold_deg,
                    sin_amplitude_deg=amp_deg, sin_period_s=period_s,
                    tag="SERVO-SIN",
                    watchdog_ms=watchdog_ms, max_lag_deg=max_lag_deg)


def run_servo_ramp(
    arm,
    duration_s: float,
    jump_threshold_deg: float,
    *,
    ramp_joints: list,
    ramp_target_deg: float,
    ramp_speed_deg_s: float,
    hold_at_top_s: float = 2.0,
    return_to_home: bool = True,
    watchdog_ms: int = 350,
    max_lag_deg: float = 35.0,
) -> None:
    """servo_j with target ramping selected joints from current pose toward
    a target offset. Simulates "slow lift" trajectory without IK.

    Parameters
    ----------
    ramp_joints : list[int]
        Joint indices (1-based, like J2 -> 2) that should be ramped up.
        Other joints stay at their initial value.
    ramp_target_deg : float
        Total offset (deg) to apply to ramped joints.
    ramp_speed_deg_s : float
        Ramp speed (deg/s). E.g. 5.0 means each ramped joint rises 5deg
        per second; teleop default upper-layer max_joint_step is 1deg/cycle
        @ 100Hz = 100 deg/s, but with --max-lin-speed 0.15 the effective
        rate observed in logs was ~5-15 deg/s, so 5.0 is realistic.
    hold_at_top_s : float
        After reaching the target, hold there for this duration so we
        can see if jumps occur at the top.
    return_to_home : bool
        Whether to ramp back down at the end.
    """
    tag = "SERVO-RAMP"
    num_joints = arm.num_joints
    jump_count = np.zeros(num_joints, dtype=np.int64)
    max_jump = np.zeros(num_joints, dtype=float)
    reject_count = 0

    opts = _build_servo_opts(arm, watchdog_ms=watchdog_ms, max_lag_deg=max_lag_deg)
    arm.servo_start(opts)
    print(f"[{tag}] servo_start done.")
    print(f"[{tag}] ramp_joints(1-based)={ramp_joints} "
          f"target_offset={ramp_target_deg:.2f}deg "
          f"speed={ramp_speed_deg_s:.2f}deg/s "
          f"hold_at_top={hold_at_top_s:.1f}s")

    q_base = arm.get_joint_values().copy()
    print(f"[{tag}] q_base (deg) = {np.rad2deg(q_base)}")
    prev = q_base.copy()
    start_t = time.time()
    last_print_t = start_t
    last_height_t = start_t  # for per-second pose summary

    target_offset_rad = math.radians(ramp_target_deg)
    speed_rad_s = math.radians(ramp_speed_deg_s)
    rise_duration_s = abs(target_offset_rad) / max(speed_rad_s, 1e-6)
    print(f"[{tag}] phase plan: rise={rise_duration_s:.1f}s, hold={hold_at_top_s:.1f}s, "
          f"fall={rise_duration_s if return_to_home else 0.0:.1f}s")
    ramp_idxs_0based = [int(j) - 1 for j in ramp_joints if 1 <= int(j) <= num_joints]

    window_max = np.zeros(num_joints, dtype=float)
    window_sum = np.zeros(num_joints, dtype=float)
    window_count = 0

    def _offset_at(t_local: float) -> float:
        """Compute current ramp offset (rad) at time t since start."""
        if t_local <= rise_duration_s:
            # rising linearly
            return min(target_offset_rad, math.copysign(speed_rad_s * t_local, target_offset_rad))
        if t_local <= rise_duration_s + hold_at_top_s:
            return target_offset_rad
        if return_to_home:
            t_fall = t_local - rise_duration_s - hold_at_top_s
            if t_fall <= rise_duration_s:
                remain = target_offset_rad - math.copysign(speed_rad_s * t_fall, target_offset_rad)
                # clamp to 0
                if (target_offset_rad >= 0 and remain < 0) or (target_offset_rad < 0 and remain > 0):
                    return 0.0
                return remain
        return 0.0

    try:
        while time.time() - start_t < duration_s:
            t_local = time.time() - start_t
            offset = _offset_at(t_local)
            q_target = q_base.copy()
            for idx in ramp_idxs_0based:
                q_target[idx] = q_base[idx] + offset
            ok = arm.servo_j(q_target)
            if not ok:
                reject_count += 1

            cur = arm.get_joint_values()
            dq_deg = np.rad2deg(np.abs(cur - prev))
            for i in range(num_joints):
                if dq_deg[i] > max_jump[i]:
                    max_jump[i] = dq_deg[i]
                if dq_deg[i] > jump_threshold_deg:
                    jump_count[i] += 1
                    track_err = np.rad2deg(np.abs(cur - q_target))
                    print(
                        f"[{tag}][JUMP] t={t_local:6.2f}s J{i+1}: "
                        f"dq={dq_deg[i]:.2f}deg, prev={math.degrees(prev[i]):+.2f}, "
                        f"cur={math.degrees(cur[i]):+.2f}, "
                        f"target={math.degrees(q_target[i]):+.2f}, "
                        f"track_err={track_err[i]:.2f}deg | "
                        f"offset_now={math.degrees(offset):+.2f}deg"
                    )
            window_max = np.maximum(window_max, dq_deg)
            window_sum += dq_deg
            window_count += 1

            now_t = time.time()
            if now_t - last_print_t >= 1.0:
                avg = window_sum / max(window_count, 1)
                track_err_deg = np.rad2deg(np.abs(cur - q_target))
                phase = (
                    "RISE" if t_local <= rise_duration_s
                    else "HOLD" if t_local <= rise_duration_s + hold_at_top_s
                    else "FALL" if return_to_home and t_local <= 2 * rise_duration_s + hold_at_top_s
                    else "DONE"
                )
                lag_now = int(getattr(arm, "servo_lag_count", 0))
                print(
                    f"[{tag}] t={t_local:5.1f}s [{phase}] offset={math.degrees(offset):+5.2f}deg "
                    f"q_now(deg)={np.array2string(np.rad2deg(cur), precision=2, suppress_small=True)} "
                    f"max_step={np.array2string(window_max, precision=3, suppress_small=True)} "
                    f"track_err={np.array2string(track_err_deg, precision=2, suppress_small=True)} "
                    f"reject={reject_count} lag-trips={lag_now}"
                )
                window_max[:] = 0.0
                window_sum[:] = 0.0
                window_count = 0
                last_print_t = now_t

            prev = cur
            time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"[{tag}] interrupted by user")
    finally:
        try:
            arm.servo_end(finish_mode="hold")
        except Exception as exc:
            print(f"[{tag}][WARN] servo_end failed: {exc}")

    print(f"\n[{tag}] ===== summary =====")
    print(f"  total servo_j rejects (hard, no-frame-sent): {reject_count}")
    try:
        lag_count = int(getattr(arm, "servo_lag_count", 0))
        clamp_count = int(getattr(arm, "servo_clamp_count", 0))
        print(f"  total lag-trips (frame still sent): {lag_count}")
        print(f"  total step-clamps                  : {clamp_count}")
    except Exception:
        pass
    for i in range(num_joints):
        marker = "  <-- ramped" if (i + 1) in ramp_joints else ""
        print(
            f"  J{i+1}: jump_events(>{jump_threshold_deg:.1f}deg)={jump_count[i]}, "
            f"max_observed_step={max_jump[i]:.3f}deg{marker}"
        )


def run_movej(arm, duration_s: float, jump_threshold_deg: float, speed_pct: int) -> None:
    """Mode B: use move_j (S-curve trajectory) instead of servo_j."""
    print(f"[MOVEJ] duration={duration_s:.1f}s, jump_threshold={jump_threshold_deg:.2f}deg, "
          f"speed={speed_pct}%")
    num_joints = arm.num_joints
    jump_count = np.zeros(num_joints, dtype=np.int64)
    max_jump = np.zeros(num_joints, dtype=float)

    q_home = arm.get_joint_values().copy()
    print(f"[MOVEJ] q_home (deg) = {np.rad2deg(q_home)}")

    # 小幅度 +/- 5° 抖动每个关节，看能否复现 servo_j 下看到的瞬变
    amp_deg = 5.0
    amp_rad = math.radians(amp_deg)

    # 监控线程不方便，这里采用串行：每个 move_j 之间快速 poll
    def _move_and_monitor(q_target: np.ndarray) -> None:
        """非阻塞 move_j，期间持续 poll 实测位置，记录 jump 事件。"""
        prev = arm.get_joint_values()
        arm.move_j(joint_angles=q_target, is_radians=True, speed=speed_pct, block=False)
        # 等待到位 / 超时
        t0 = time.time()
        while time.time() - t0 < 3.0:
            cur = arm.get_joint_values()
            dq_deg = np.rad2deg(np.abs(cur - prev))
            for i in range(num_joints):
                if dq_deg[i] > max_jump[i]:
                    max_jump[i] = dq_deg[i]
                if dq_deg[i] > jump_threshold_deg:
                    jump_count[i] += 1
                    print(
                        f"[MOVEJ][JUMP] t={time.time()-t0:5.2f}s J{i+1}: "
                        f"dq={dq_deg[i]:.2f}deg, prev={math.degrees(prev[i]):+.2f}, "
                        f"cur={math.degrees(cur[i]):+.2f}"
                    )
            # 到位检测
            if np.max(np.abs(cur - q_target)) < math.radians(0.5):
                break
            prev = cur
            time.sleep(0.01)

    start_t = time.time()
    iteration = 0
    try:
        while time.time() - start_t < duration_s:
            iteration += 1
            # 轮流让每个关节做 ±amp 摆动
            j = (iteration - 1) % num_joints
            sign = 1 if (iteration // num_joints) % 2 == 0 else -1
            q_target = q_home.copy()
            q_target[j] = q_home[j] + sign * amp_rad
            print(
                f"[MOVEJ] iter={iteration} J{j+1} -> {math.degrees(q_target[j]):+.2f}deg "
                f"(jumps so far: {jump_count.tolist()})"
            )
            _move_and_monitor(q_target)
            time.sleep(0.2)
            _move_and_monitor(q_home)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[MOVEJ] interrupted by user")

    # 收尾：回 home
    print("[MOVEJ] returning to q_home ...")
    try:
        arm.move_j(joint_angles=q_home, is_radians=True, speed=speed_pct, block=True)
    except Exception as exc:
        print(f"[MOVEJ][WARN] final move_j failed: {exc}")

    print("\n[MOVEJ] ===== summary =====")
    for i in range(num_joints):
        print(
            f"  J{i+1}: jump_events(>{jump_threshold_deg:.1f}deg)={jump_count[i]}, "
            f"max_observed_step={max_jump[i]:.3f}deg"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",
                        choices=["static", "movej", "servo-hold", "servo-sin", "servo-ramp"],
                        default="static",
                        help="static: pure feedback poll (no command). "
                             "movej: S-curve move_j sweeps over each joint. "
                             "servo-hold: servo_j with constant target (electrically holding pose). "
                             "servo-sin: servo_j with small sin perturbation. "
                             "servo-ramp: servo_j linear ramp on selected joints (simulates lift).")
    parser.add_argument("--sin-amp-deg", type=float, default=1.0,
                        help="Amplitude (deg) of sin perturbation in --mode servo-sin.")
    parser.add_argument("--sin-period-s", type=float, default=4.0,
                        help="Period (s) of sin perturbation in --mode servo-sin.")
    parser.add_argument("--ramp-joints", type=str, default="2,3",
                        help="Comma-separated 1-based joint indices to ramp (e.g. '2,3' for J2+J3).")
    parser.add_argument("--ramp-target-deg", type=float, default=30.0,
                        help="Total angle (deg) to ramp each selected joint upward.")
    parser.add_argument("--ramp-speed-deg-s", type=float, default=5.0,
                        help="Ramp speed (deg/s) per joint. Real teleop lift was 5-15 deg/s.")
    parser.add_argument("--ramp-hold-s", type=float, default=2.0,
                        help="Hold time at top of ramp (s) before returning.")
    parser.add_argument("--ramp-no-return", action="store_true", default=False,
                        help="Do not ramp back to home; stay at top until duration ends.")
    parser.add_argument("--cfg", type=str, default="",
                        help="Path to robot.cfg. Empty = auto-locate next to this script.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Total test duration in seconds.")
    parser.add_argument("--jump-threshold", type=float, default=5.0,
                        help="Print [JUMP] when adjacent meas_step exceeds this (deg).")
    parser.add_argument("--movej-speed", type=int, default=10,
                        help="move_j speed percentage (1-100). Smaller is gentler.")
    parser.add_argument("--has-gripper", action="store_true", default=True)
    parser.add_argument("--no-gripper", action="store_false", dest="has_gripper")
    parser.add_argument("--gripper-id", type=int, default=7)
    # ---- safety net knobs (used for cascade-bug isolation) -----------------
    # Default values match production; set both to 0 to test whether observed
    # "J4 drops mid-ramp" failure mode is caused by our own defenses (see
    # README / chat for the cascade explanation).
    parser.add_argument("--watchdog-ms", type=int, default=350,
                        help="ServoOpts.watchdog_ms (firmware-side timeout, "
                             "0 = disable). DANGEROUS at 0: host hang will "
                             "leave the motors powered with last-target frame.")
    parser.add_argument("--max-lag-deg", type=float, default=35.0,
                        help="ServoOpts.max_lag_rad in degrees. 0 = disable "
                             "the per-frame tracking-error rejection. Set to "
                             "0 to test if servo_j rejection cascade triggers "
                             "the J4 droop. Keep your hand on the E-stop.")
    parser.add_argument("--no-safety-nets", action="store_true",
                        help="Shortcut: equivalent to --watchdog-ms 0 "
                             "--max-lag-deg 0. Disables BOTH defenses.")
    args = parser.parse_args()

    if args.no_safety_nets:
        args.watchdog_ms = 0
        args.max_lag_deg = 0.0

    _enable_high_res_timer()

    cfg_path = args.cfg.strip() or _resolve_default_cfg()
    print(f"[INIT] cfg_path = {cfg_path}")
    if not os.path.isfile(cfg_path):
        sys.exit(f"[ERROR] cfg not found: {cfg_path}")

    FafuRobotController = _import_controller()
    arm = FafuRobotController(
        cfg_path=cfg_path,
        has_gripper=bool(args.has_gripper),
        gripper_motor_id=args.gripper_id if args.has_gripper else None,
        auto_enable=False,  # we'll call enable() ourselves
    )

    print("[INIT] calling enable() ...")
    arm.enable()

    try:
        if args.mode == "static":
            run_static(arm, args.duration, args.jump_threshold)
        elif args.mode == "movej":
            run_movej(arm, args.duration, args.jump_threshold, args.movej_speed)
        elif args.mode == "servo-hold":
            run_servo_hold(arm, args.duration, args.jump_threshold,
                           watchdog_ms=args.watchdog_ms,
                           max_lag_deg=args.max_lag_deg)
        elif args.mode == "servo-sin":
            run_servo_sin(
                arm,
                args.duration,
                args.jump_threshold,
                amp_deg=args.sin_amp_deg,
                period_s=args.sin_period_s,
                watchdog_ms=args.watchdog_ms,
                max_lag_deg=args.max_lag_deg,
            )
        elif args.mode == "servo-ramp":
            try:
                ramp_joints = [int(x.strip()) for x in args.ramp_joints.split(",") if x.strip()]
            except Exception:
                raise ValueError(f"invalid --ramp-joints: {args.ramp_joints!r}")
            if not ramp_joints:
                raise ValueError("--ramp-joints cannot be empty")
            run_servo_ramp(
                arm,
                args.duration,
                args.jump_threshold,
                ramp_joints=ramp_joints,
                ramp_target_deg=args.ramp_target_deg,
                ramp_speed_deg_s=args.ramp_speed_deg_s,
                hold_at_top_s=args.ramp_hold_s,
                return_to_home=not args.ramp_no_return,
                watchdog_ms=args.watchdog_ms,
                max_lag_deg=args.max_lag_deg,
            )
        else:
            raise ValueError(f"unknown mode: {args.mode}")
    finally:
        try:
            # Note: we keep motors in position hold; not calling disable()
            # to avoid free-fall on release. close_connection releases the port.
            arm.close_connection(joint_release="hold")
        except Exception as exc:
            print(f"[CLEANUP][WARN] close_connection failed: {exc}")


if __name__ == "__main__":
    main()
