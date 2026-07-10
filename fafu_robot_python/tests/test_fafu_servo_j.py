# -*- coding: utf-8 -*-
"""
test_fafu_servo_j.py
====================

``FafuRobotController.servo_j`` 的独立可跑 Python 例子.

跟 ``test_fafu_motion_interactive.py`` 的 [v] 子菜单是同一份逻辑, 但这里
是 **非交互**, 用 argparse 配置一次跑完, 适合:

* 第一次接触 servoJ 的 smoke test
* 在 CI / 实验记录里复跑相同参数
* 性能/抖动比对 (rate / lag)

两种 mode (``--mode``):

``sin`` (默认)
    多关节联动正弦, 相邻关节反相. 自动 ``go_home`` -> ``servo_start``
    -> 100Hz 跟踪 ``--duration`` 秒 -> ``servo_end("brake")`` -> 打印
    Hz / lag 统计.

``hold``
    把当前位置作为目标持续 stream ``--duration`` 秒. 不会让机器动.
    主要用来:
      1) 看 servo 链路通不通, 实测 Hz 多少 (Windows 下 100Hz 应该能稳到 95-100)
      2) **验证看门狗**: 跑起来后从任务管理器强杀 python.exe, 主循环没机会
         cleanup, 电机应该在 ``--watchdog-ms`` 内自动 brake.

⚠️ 第一次运行前:
    - 机器臂已上电, 桌面留出空间
    - 急停在手边, 手放在 Ctrl+C 上
    - 推荐 ``--amplitude-deg`` <= 10, ``--max-vel`` <= 1.5 rad/s

用法:
    cd fafu_robot_sdk
    python tests/test_fafu_servo_j.py
    python tests/test_fafu_servo_j.py --mode hold --duration 30
    python tests/test_fafu_servo_j.py --mode sin --amplitude-deg 12 --period-s 3
    python tests/test_fafu_servo_j.py --gripper-id 7 --no-home   # 不归零, 用当前位置作起点

输出 (示例):
    [FafuRobot] connected on COM14 @ 4000000 (6 joints + gripper M7)
    [FafuRobot] all motors enabled (position control hold).
    [FafuRobot] servo_start: watchdog=100ms, max_vel=1.5rad/s, max_step=0.026rad, max_lag=0.175rad

    --- Streaming sin for 6.0s @ 100Hz ---
        send_ok=599  send_fail=1  measured_rate=99.7Hz
        per-joint max |lag| (deg): [0.42, 0.55, 0.61, 0.34, 0.48, 0.71]
        loop jitter p99 = 2.1ms
    [FafuRobot] servo_end (brake): 600 ticks in 6.02s (~99.7 Hz)
    [FafuRobot] connection closed (joints=stop, gripper=brake).
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import traceback
from typing import Iterable, List

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ----------------------------------------------------------------------------
#  Windows high-resolution timer (critical for 100Hz+ servoJ).
#
#  Default Windows scheduler tick = 15.625 ms, which means time.sleep(0.005)
#  actually returns after ~15 ms — this single fact is why every untuned
#  Python servo loop on Windows shows ~16 ms jitter at 100 Hz.  Calling
#  timeBeginPeriod(1) raises the timer resolution to 1 ms for the whole
#  process (and the rest of the system, while we are alive).
#
#  We call this at import time so any caller of the helpers below gets the
#  benefit; timeEndPeriod is registered with atexit for symmetry.
# ----------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        import ctypes
        _winmm = ctypes.WinDLL("winmm")
        _winmm.timeBeginPeriod(1)
        import atexit
        atexit.register(_winmm.timeEndPeriod, 1)
    except Exception as _e:
        print(f"[test_fafu_servo_j] WARN: timeBeginPeriod(1) failed: {_e}")


def _sleep_until(t_target: float, busy_threshold: float = 0.0015) -> None:
    """Sleep until ``time.monotonic() >= t_target`` with sub-ms accuracy.

    Hybrid strategy: ``time.sleep`` most of the way (releases CPU, doesn't
    burn battery), then busy-spin the last ~1.5 ms because even with
    timeBeginPeriod(1) Windows' sleep can overshoot by 1-2 ms on a busy
    box.  Net per-tick jitter ~50-300us on a normal desktop.
    """
    now = time.monotonic()
    sleep_for = t_target - now - busy_threshold
    if sleep_for > 0.0:
        time.sleep(sleep_for)
    while time.monotonic() < t_target:
        pass


def _sleep_print(*a, **kw):
    """Background polling thread can interleave its log lines; nudge
    it out of the way before printing."""
    time.sleep(0.02)
    kw.setdefault("flush", True)
    print(*a, **kw)


def _deg_str(angles_rad: Iterable[float]) -> str:
    return "[" + ", ".join(f"{math.degrees(v):+7.2f}" for v in angles_rad) + "]"


def _run_servo_loop(
    arm,
    target_fn,                 # callable(t_s: float) -> np.ndarray of num_joints rad
    duration_s: float,
    rate_hz: float,
) -> dict:
    """Drive ``servo_j`` at ``rate_hz`` for ``duration_s`` seconds and
    return a stats dict. Caller is responsible for servo_start/servo_end."""
    period_s = 1.0 / max(1.0, rate_hz)
    n_total = max(1, int(duration_s * rate_hz))

    send_ok = 0
    send_fail = 0
    tick_jitter_ms: List[float] = []   # actual - planned, capped via deque-ish
    n_joints = arm.num_joints
    max_abs_lag_turns = np.zeros(n_joints, dtype=float)   # tracking error

    t_start = time.monotonic()
    next_tick = t_start
    for k in range(n_total):
        t_now = time.monotonic()
        # jitter: how late we are vs the planned tick time
        if k > 0:
            tick_jitter_ms.append((t_now - next_tick) * 1000.0)

        target = target_fn(k * period_s)
        if arm.servo_j(target):
            send_ok += 1
        else:
            send_fail += 1

        # Sample tracking error AFTER send (1-tick old, fine for stats)
        for j, mid in enumerate(arm.joint_motor_ids):
            s = arm.driver.get_cached_state(mid)
            if s is None:
                continue
            target_turns_j = float(target[j]) / (2.0 * math.pi)
            lag = abs(s.position - target_turns_j)
            if lag > max_abs_lag_turns[j]:
                max_abs_lag_turns[j] = lag

        next_tick += period_s
        _sleep_until(next_tick)

    elapsed = time.monotonic() - t_start
    measured_rate = (send_ok + send_fail) / max(1e-3, elapsed)

    if tick_jitter_ms:
        jitter_p50 = statistics.median(tick_jitter_ms)
        jitter_p99 = sorted(tick_jitter_ms)[int(0.99 * (len(tick_jitter_ms) - 1))]
        jitter_max = max(tick_jitter_ms)
    else:
        jitter_p50 = jitter_p99 = jitter_max = 0.0

    return {
        "send_ok": send_ok,
        "send_fail": send_fail,
        "elapsed_s": elapsed,
        "measured_rate_hz": measured_rate,
        "max_abs_lag_deg": [math.degrees(v * 2.0 * math.pi) for v in max_abs_lag_turns],
        "jitter_p50_ms": jitter_p50,
        "jitter_p99_ms": jitter_p99,
        "jitter_max_ms": jitter_max,
    }


def _print_summary(stats: dict, *, mode_label: str) -> None:
    _sleep_print(f"\n  --- {mode_label} summary ---")
    _sleep_print(f"    send_ok={stats['send_ok']}  send_fail={stats['send_fail']}  "
                 f"elapsed={stats['elapsed_s']:.2f}s")
    _sleep_print(f"    measured_rate = {stats['measured_rate_hz']:6.2f} Hz")
    _sleep_print(
        "    loop jitter    p50={:.2f}ms  p99={:.2f}ms  max={:.2f}ms".format(
            stats["jitter_p50_ms"], stats["jitter_p99_ms"], stats["jitter_max_ms"])
    )
    _sleep_print(
        "    per-joint max |lag| (deg) = [" +
        ", ".join(f"{v:.2f}" for v in stats["max_abs_lag_deg"]) + "]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone smoke test for FafuRobotController.servo_j (online streaming)."
    )
    # ---- connection ----
    parser.add_argument("--cfg", default="robot.cfg",
                        help="robot.cfg 路径 (默认: fafu_robot_sdk/robot.cfg)")
    parser.add_argument("--gripper-id", type=int, default=None,
                        help="夹爪 motor id, 不传则视为无夹爪 (servoJ 不动夹爪)")

    # ---- mode ----
    parser.add_argument("--mode", choices=("sin", "hold"), default="sin",
                        help="sin: 多关节 sin 跟踪 (默认); "
                             "hold: 持续 hold 当前位置 (用来测看门狗)")

    # ---- timing ----
    parser.add_argument("--rate", type=float, default=100.0,
                        help="上层 servo_j 调用频率 Hz (默认 100)")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="持续时长 秒 (默认 6, hold 模式建议 >=10)")

    # ---- sin shape (mode=sin only) ----
    parser.add_argument("--amplitude-deg", type=float, default=8.0,
                        help="sin 振幅 deg (默认 8)")
    parser.add_argument("--period-s", type=float, default=2.5,
                        help="sin 周期 秒 (默认 2.5)")
    parser.add_argument("--joint", type=int, default=None,
                        help="如果给, 只动这一个关节 (0..num_joints-1); "
                             "默认所有关节同 sin 相邻反相")

    # ---- ServoOpts ----
    parser.add_argument("--watchdog-ms", type=int, default=100,
                        help="固件级看门狗 ms (默认 100; 0 = 禁用★不推荐)")
    parser.add_argument("--max-vel", type=float, default=1.5,
                        help="ServoOpts.max_vel rad/s (默认 1.5)")
    parser.add_argument("--max-step-deg", type=float, default=1.5,
                        help="ServoOpts.max_step_rad 单步限幅 deg (默认 1.5)")
    parser.add_argument("--max-lag-deg", type=float, default=10.0,
                        help="ServoOpts.max_lag_rad 跟踪误差上限 deg (默认 10)")
    parser.add_argument("--lookahead-ms", type=float, default=0.0,
                        help="ServoOpts.lookahead_time 一阶低通时间常数 ms "
                             "(默认 0 = 不滤波; 手柄/teleop 噪音大时试 30-100)")
    parser.add_argument("--no-feedforward", action="store_true",
                        help="禁用速度前馈 (vel 退回到恒定 max_vel; "
                             "复现旧行为 — 通常会让电机噪音变大)")

    # ---- lifecycle ----
    parser.add_argument("--no-home", action="store_true",
                        help="不先 go_home, 直接用当前位置作起点")
    parser.add_argument("--home-on-exit", action="store_true",
                        help="退出前再 go_home 一次 (方便复跑)")
    parser.add_argument("--finish", choices=("hold", "brake", "stop"),
                        default="hold",
                        help="servo_end 后关节模式 (默认 hold = 继续位置控制 hold "
                             "最后一帧, 跟 move_j 完成后一致, 可以直接接下一个动作; "
                             "brake = 三相短路, 不耗电但抗扭; "
                             "stop = PWM off, 自由可手推)")
    parser.add_argument("--speed", type=int, default=15,
                        help="go_home 时用的 speed percent (默认 15)")

    args = parser.parse_args()

    print("=" * 72)
    print(f"  test_fafu_servo_j  mode={args.mode}  rate={args.rate}Hz  "
          f"duration={args.duration}s")
    print(f"  ServoOpts: watchdog={args.watchdog_ms}ms  max_vel={args.max_vel}rad/s  "
          f"max_step={args.max_step_deg}deg  max_lag={args.max_lag_deg}deg")
    print(f"             feedforward={'off' if args.no_feedforward else 'on'}  "
          f"lookahead={args.lookahead_ms:.0f}ms  rate_hz={args.rate:.0f}")
    print("=" * 72)

    try:
        from fafu_robot_controller import FafuRobotController, ServoOpts
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] import fafu_robot_controller 失败: {e}")
        return 1

    has_gripper = args.gripper_id is not None
    try:
        arm = FafuRobotController(
            cfg_path=args.cfg,
            has_gripper=has_gripper,
            gripper_motor_id=args.gripper_id,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] 创建 FafuRobotController 失败: {e}")
        return 1

    rc = 0
    try:
        if not args.no_home:
            _sleep_print("\n--- go_home (offline S-curve) ---")
            arm.go_home(speed=args.speed, block=True)
        else:
            _sleep_print("[--no-home] skipping go_home; "
                         "starting servo from current pose")

        q0 = arm.get_joint_values()
        _sleep_print(f"  起点 (deg): {_deg_str(q0)}")

        opts = ServoOpts(
            watchdog_ms=args.watchdog_ms,
            max_vel=args.max_vel,
            max_step_rad=math.radians(args.max_step_deg),
            max_lag_rad=math.radians(args.max_lag_deg),
            is_radians=True,
            rate_hz=args.rate,
            feedforward_vel=not args.no_feedforward,
            lookahead_time=args.lookahead_ms / 1000.0,
        )

        # ----------------------------------------------------------------
        #  Build the per-tick target function
        # ----------------------------------------------------------------
        if args.mode == "hold":
            _sleep_print(f"\n--- mode=hold: streaming current pose for "
                         f"{args.duration:.1f}s @ {args.rate:.0f}Hz ---")
            _sleep_print("  机器不应该动. Ctrl+C 走 servo_end; "
                         "想测看门狗就从任务管理器强杀 python.exe")

            def target_fn(_t):
                return q0
            mode_label = "hold"

        else:
            # mode == "sin"
            amp_rad = math.radians(args.amplitude_deg)
            freq_hz = 1.0 / max(0.1, args.period_s)
            peak_vel = 2 * math.pi * freq_hz * amp_rad
            n_joints = arm.num_joints

            if args.joint is not None:
                if not (0 <= args.joint < n_joints):
                    print(f"[FAIL] --joint {args.joint} 越界 (0..{n_joints - 1})")
                    return 1
                _sleep_print(
                    f"\n--- mode=sin (single J{args.joint}): "
                    f"±{args.amplitude_deg:.1f}deg / {args.period_s:.1f}s "
                    f"for {args.duration:.1f}s @ {args.rate:.0f}Hz ---"
                )

                def target_fn(t):
                    out = q0.copy()
                    out[args.joint] = q0[args.joint] + amp_rad * math.sin(
                        2 * math.pi * freq_hz * t)
                    return out
                mode_label = f"sin J{args.joint}"
            else:
                signs = np.array(
                    [1.0 if (j % 2 == 0) else -1.0 for j in range(n_joints)])
                _sleep_print(
                    f"\n--- mode=sin (multi, 相邻反相): "
                    f"±{args.amplitude_deg:.1f}deg / {args.period_s:.1f}s "
                    f"for {args.duration:.1f}s @ {args.rate:.0f}Hz ---"
                )

                def target_fn(t):
                    return q0 + signs * amp_rad * math.sin(
                        2 * math.pi * freq_hz * t)
                mode_label = "sin multi"

            _sleep_print(
                f"  估算峰值角速度 = {peak_vel:.2f} rad/s   "
                f"(opts.max_vel={args.max_vel:.2f} rad/s)"
            )
            if peak_vel > args.max_vel:
                _sleep_print(
                    "  ★ 峰值速度 > max_vel, 跟踪会有明显 lag. "
                    "调小振幅 / 加大周期 / 加大 max_vel."
                )
            est_step_deg = math.degrees(peak_vel) / args.rate
            _sleep_print(
                f"  估算每 tick 最大步长 ≈ {est_step_deg:.3f} deg  "
                f"(opts.max_step={args.max_step_deg:.2f} deg)"
            )
            if est_step_deg > args.max_step_deg:
                _sleep_print(
                    "  ★ 单 tick 步长 > max_step_rad, servo_j 会 clamp + 警告. "
                    "加大 --max-step-deg 或降低振幅 / 加大周期."
                )

        # ----------------------------------------------------------------
        #  Run the servo session
        # ----------------------------------------------------------------
        try:
            arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc()
            print(f"\n[FAIL] servo_start 失败: {e}")
            return 1

        stats = None
        try:
            stats = _run_servo_loop(arm, target_fn, args.duration, args.rate)
        except KeyboardInterrupt:
            _sleep_print("\n  [INTERRUPT] Ctrl+C, 提前 servo_end")
            rc = 130
        finally:
            try:
                arm.servo_end(finish_mode=args.finish)
            except Exception as e:
                print(f"  servo_end 失败: {e}")
                rc = max(rc, 1)

        if stats is not None:
            _print_summary(stats, mode_label=mode_label)

            # sanity-check thresholds for the sin case
            if args.mode == "sin":
                if stats["send_fail"] > 0.05 * (stats["send_ok"] + stats["send_fail"]):
                    _sleep_print(
                        "  [WARN] send_fail 超过 5%; lag 报警常见原因: "
                        "max_vel 太小 / 振幅过大 / max_lag_deg 太严."
                    )
                    rc = max(rc, 1)
                if stats["measured_rate_hz"] < 0.9 * args.rate:
                    _sleep_print(
                        f"  [WARN] 实测 Hz 不足 ({stats['measured_rate_hz']:.1f} < "
                        f"{args.rate * 0.9:.1f}); Windows 抖动 / 后台进程过重."
                    )
                    rc = max(rc, 1)

        # ----------------------------------------------------------------
        # Optional: go home again so the next run starts from a known pose
        # ----------------------------------------------------------------
        if args.home_on_exit:
            _sleep_print("\n--- home_on_exit: go_home ---")
            try:
                # finish_mode=hold (default) leaves motors enabled; brake/stop
                # need a re-enable before move_j can run.
                if not arm.is_enabled:
                    arm.enable()
                arm.go_home(speed=args.speed, block=True)
            except Exception as e:
                print(f"  go_home 失败: {e}")
                rc = max(rc, 1)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Ctrl+C, emergency stop...")
        try: arm.emergency_stop()
        except Exception: pass
        rc = 130
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] {e}")
        try: arm.emergency_stop()
        except Exception: pass
        rc = 1
    finally:
        try:
            arm.close_connection()
        except Exception:
            pass

    print()
    print("=" * 72)
    print(f"  结果: {'PASS' if rc == 0 else 'FAIL/WARN (rc=' + str(rc) + ')'}")
    print("=" * 72)
    return rc


if __name__ == "__main__":
    sys.exit(main())
