# -*- coding: utf-8 -*-
"""
test_fafu_grasp_calibrate.py
============================

交互式标定 ``FafuRobotController.grasp()`` 的 ``force_threshold``.

为什么需要标定?
---------------
``MotorState.torque`` 是 raw int16, 单位与电机型号相关, 没有通用的
"多少 raw 等于多少 N·m". 不同夹爪/不同电机, 空抓时的稳态力矩
都不一样, 所以阈值必须按实物量一次.

测试流程 (每次都用 ``grasp(force_threshold=很大的数)`` 强制走完,
然后看实际峰值力矩):

    1) **空抓 N 次**   → 记录 peak torque, 取最大值作为 "noise floor"
    2) **抓硬物 N 次**  → 记录 peak torque, 取最小值作为 "object floor"
    3) 推荐阈值 = (noise_floor + object_floor) / 2
       并按 --safety-factor 略向 object_floor 偏移
    4) (可选) 用推荐阈值再跑一次 grasp(), 看是否能正确判定 grasped=True

通过条件 (用于回归):
    - 空抓 peak 与 抓物 peak 有清晰 gap (>= ~ 50 raw 差距)
    - 用推荐阈值的验证一次, grasped == True, reason == 'detected_object_force'

⚠️ 第一次运行前:
    - 机器臂已上电, 夹爪已张到至少能放下你的标定物
    - 准备一个 **硬质** 标定物 (木块/螺丝刀杆 等), 不要用易碎物
    - 手放在 Ctrl+C 上

用法:
    cd fafu_robot_sdk
    python tests/test_fafu_grasp_calibrate.py --gripper-id 7
    python tests/test_fafu_grasp_calibrate.py --gripper-id 7 --samples 5 --vel 0.1
    python tests/test_fafu_grasp_calibrate.py --gripper-id 7 --skip-verify

输出 (示例):
    [empty]   trial 1 peak = 187 raw   reason=reached_target
    [empty]   trial 2 peak = 195 raw   reason=reached_target
    [object]  trial 1 peak = 712 raw   reason=detected_object_stall
    [object]  trial 2 peak = 695 raw   reason=detected_object_stall

    noise_floor  = 195
    object_floor = 695
    recommended force_threshold = 445   (safety_factor=0.5)

    [verify] grasp(force_threshold=445) -> grasped=True
             reason=detected_object_force  peak=472  closed=24.3°
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def _sleep_print(*a, **kw):
    """Background polling thread can interleave its log lines; nudge
    it out of the way before printing, like the wrapper's own demo."""
    time.sleep(0.02)
    kw.setdefault("flush", True)
    print(*a, **kw)


def _run_one_cycle(arm, label: str, vel: float, timeout: float):
    """Open, then close-with-impossibly-high-threshold (so we always
    walk to the soft limit or stall), return the GraspResult."""
    arm.open_gripper(vel=vel)
    time.sleep(0.3)
    # force_threshold so high that detect_object_force never fires;
    # the call will end via reached_target / detected_object_stall.
    r = arm.grasp(
        force_threshold=10_000_000,
        vel=vel,
        timeout=timeout,
    )
    _sleep_print(
        f"  [{label:>6s}] peak = {r.peak_torque_raw:>5d} raw   "
        f"reason={r.reason:<22s}  closed={r.closed_deg:5.1f}°  "
        f"dt={r.duration_s:.2f}s"
    )
    return r


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate force_threshold for FafuRobotController.grasp()"
    )
    parser.add_argument("--cfg", default="robot.cfg",
                        help="robot.cfg 路径 (默认: fafu_robot_sdk/robot.cfg)")
    parser.add_argument("--gripper-id", type=int, required=True,
                        help="夹爪 motor id (必填; 通常 7)")
    parser.add_argument("--samples", type=int, default=3,
                        help="每个阶段重复多少次 (默认 3)")
    parser.add_argument("--vel", type=float, default=0.15,
                        help="抓取速度 turns/s (默认 0.15 ~ 54 deg/s)")
    parser.add_argument("--timeout", type=float, default=6.0,
                        help="单次 grasp 超时 秒 (默认 6)")
    parser.add_argument("--safety-factor", type=float, default=0.5,
                        help="推荐阈值在 (noise, object) 之间的位置, "
                             "0=贴近噪声/灵敏, 1=贴近物体/保守 (默认 0.5)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过最后用推荐阈值跑一次验证抓取")
    args = parser.parse_args()

    print("=" * 64)
    print(f"  test_fafu_grasp_calibrate  (gripper M{args.gripper_id}, "
          f"samples={args.samples}, vel={args.vel})")
    print("=" * 64)

    try:
        from fafu_robot_controller import FafuRobotController
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] import fafu_robot_controller 失败: {e}")
        return 1

    try:
        arm = FafuRobotController(
            cfg_path=args.cfg,
            has_gripper=True,
            gripper_motor_id=args.gripper_id,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] 创建 FafuRobotController 失败: {e}")
        return 1

    rc = 0
    try:
        # ----------------------------------------------------------------
        # Stage 1: empty close (noise floor)
        # ----------------------------------------------------------------
        _sleep_print("\n--- Stage 1: 空抓 (建立 noise floor) ---")
        input("  确认夹爪里没有任何物体, 按 Enter 开始: ")
        empty_peaks = []
        for i in range(args.samples):
            _sleep_print(f"\n  trial {i + 1}/{args.samples}")
            r = _run_one_cycle(arm, "empty", args.vel, args.timeout)
            empty_peaks.append(r.peak_torque_raw)

        # ----------------------------------------------------------------
        # Stage 2: object close (object floor)
        # ----------------------------------------------------------------
        _sleep_print("\n--- Stage 2: 抓硬物 (建立 object floor) ---")
        input("  在夹爪里放一个硬质标定物 (木块/螺丝刀杆), 按 Enter 开始: ")
        object_peaks = []
        for i in range(args.samples):
            _sleep_print(f"\n  trial {i + 1}/{args.samples}")
            r = _run_one_cycle(arm, "object", args.vel, args.timeout)
            object_peaks.append(r.peak_torque_raw)

        # ----------------------------------------------------------------
        # Recommend
        # ----------------------------------------------------------------
        noise_floor  = max(empty_peaks)
        object_floor = min(object_peaks)
        gap = object_floor - noise_floor

        sf = max(0.0, min(1.0, args.safety_factor))
        recommended = int(round(noise_floor + sf * gap))

        _sleep_print("\n--- 标定结果 ---")
        _sleep_print(f"  empty  peaks  = {empty_peaks}")
        _sleep_print(f"  object peaks  = {object_peaks}")
        _sleep_print(f"  noise_floor   = {noise_floor}")
        _sleep_print(f"  object_floor  = {object_floor}")
        _sleep_print(f"  gap           = {gap}")
        _sleep_print(f"  recommended force_threshold = {recommended}  "
                     f"(safety_factor={sf})")

        if gap < 50:
            _sleep_print(
                "\n  [WARN] 空抓和抓物的力矩间隔太小 (<50 raw), "
                "建议:\n"
                "         1) 换更硬/更大的标定物\n"
                "         2) 把 --vel 调慢点 (默认 0.15, 试 0.08)\n"
                "         3) 检查夹爪是否真的夹到了 (而不是物体掉了)"
            )
            rc = max(rc, 1)

        # ----------------------------------------------------------------
        # Stage 3: verify
        # ----------------------------------------------------------------
        if args.skip_verify:
            _sleep_print("\n[--skip-verify] 跳过验证.")
        elif gap < 50:
            _sleep_print("\n[gap 太小] 跳过验证, 先调参再来.")
        else:
            _sleep_print("\n--- Stage 3: 用推荐阈值验证 ---")
            input("  把标定物再放回夹爪里, 按 Enter 验证: ")
            arm.open_gripper(vel=args.vel)
            time.sleep(0.3)
            v = arm.grasp(
                force_threshold=recommended,
                vel=args.vel,
                timeout=args.timeout,
            )
            _sleep_print(
                f"\n  [verify] grasped={v.grasped}  "
                f"reason={v.reason}  peak={v.peak_torque_raw}  "
                f"closed={v.closed_deg:.1f}°  dt={v.duration_s:.2f}s"
            )
            if not v.grasped:
                _sleep_print(
                    "  [FAIL] 验证未识别为抓到; 把 safety_factor 调小 "
                    "(更灵敏) 或重新标定."
                )
                rc = max(rc, 1)
            elif v.reason != "detected_object_force":
                _sleep_print(
                    f"  [WARN] grasped=True 但 reason={v.reason} (不是力矩触发); "
                    "阈值偏低, 可以调大一些."
                )

        # ----------------------------------------------------------------
        # Cleanup
        # ----------------------------------------------------------------
        _sleep_print("\n--- 收尾: 张开夹爪 ---")
        arm.open_gripper(vel=args.vel)

    except KeyboardInterrupt:
        print("\n\n  [INTERRUPT] Ctrl+C, 紧急停止...")
        try:
            arm.emergency_stop()
        except Exception:
            pass
        rc = 130
    except Exception as e:
        traceback.print_exc()
        print(f"\n  [FAIL] 测试中断: {e}")
        try:
            arm.emergency_stop()
        except Exception:
            pass
        rc = 1
    finally:
        try:
            arm.close_connection()
        except Exception:
            pass

    print()
    print("=" * 64)
    print(f"  结果: {'PASS' if rc == 0 else 'FAIL/WARN (rc=' + str(rc) + ')'}")
    print("=" * 64)
    return rc


if __name__ == "__main__":
    sys.exit(main())
