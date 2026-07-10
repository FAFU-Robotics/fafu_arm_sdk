# -*- coding: utf-8 -*-
"""
test_one_joint.py
=================

最小幅度真机测试: 让指定关节缓慢转 5 度, 然后回到起点.

通过条件:
    - 关节真的动了 (起止角度差 ~ 5 度)
    - 回到起点后误差 < 0.5 度
    - 程序自然退出 (无异常 / 资源释放)

⚠️ 第一次运行前:
    - 机械臂已上电
    - 桌面留出至少 30cm 空间
    - 手放在键盘 Ctrl+C 上, 旁边有急停按钮
    - 推荐 --speed 不超过 10

用法:
    cd fafu_robot_sdk
    python tests/test_one_joint.py                          # 默认动 motor 6, ±5 度
    python tests/test_one_joint.py --joint 1 --delta-deg 3  # 动 motor 1, ±3 度
    python tests/test_one_joint.py --gripper-id 7           # 有夹爪的版本
    python tests/test_one_joint.py --dry-run                # 只读不动 (smoke 之后的下一步)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def _fmt_deg(angles_rad) -> str:
    return "[" + ", ".join(f"{math.degrees(v):+7.2f}" for v in angles_rad) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description="single-joint safe motion test")
    parser.add_argument("--cfg", default="robot.cfg",
                        help="robot.cfg 路径 (默认: fafu_robot_sdk/robot.cfg)")
    parser.add_argument("--gripper-id", type=int, default=None,
                        help="夹爪 motor id, 不传表示无夹爪")
    parser.add_argument("--joint", type=int, default=-1,
                        help="动哪个关节 (0-based, 默认 -1 = 最后一个关节)")
    parser.add_argument("--delta-deg", type=float, default=5.0,
                        help="角度变化量, 度数 (默认 5)")
    parser.add_argument("--speed", type=int, default=10,
                        help="speed percent (默认 10, 慢)")
    parser.add_argument("--tolerance-deg", type=float, default=0.5,
                        help="到位误差容忍, 度 (默认 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只连接 / 读状态, 不发任何 move 命令")
    args = parser.parse_args()

    try:
        from fafu_robot_controller import FafuRobotController
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] import fafu_robot_controller 失败: {e}")
        print("       先跑 tests/smoke_test.py 排查环境")
        return 1

    print("=" * 60)
    print(f"  test_one_joint  (joint={args.joint}, "
          f"delta={args.delta_deg:+.2f}°, speed={args.speed})")
    print("=" * 60)

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
        n = arm.num_joints
        print(f"\n  joint_motor_ids = {arm.joint_motor_ids} ({n} joints)")
        joint_idx = args.joint if args.joint >= 0 else n + args.joint
        if not (0 <= joint_idx < n):
            print(f"  [FAIL] --joint {args.joint} 超出范围 [0, {n})")
            return 1
        print(f"  动: joint #{joint_idx} (motor M{arm.joint_motor_ids[joint_idx]})")

        q0 = arm.get_joint_values()
        print(f"\n  起始 (deg): {_fmt_deg(q0)}")
        print(f"  is_enabled: {arm.is_enabled}")

        if args.dry_run:
            print("\n  [--dry-run] 只读不动, 测试结束.")
            return 0

        delta_rad = math.radians(args.delta_deg)
        target = q0.copy()
        target[joint_idx] += delta_rad
        print(f"  目标 (deg): {_fmt_deg(target)}")

        confirm = input("\n  按 Enter 开始运动, 输入 q 取消: ").strip().lower()
        if confirm == "q":
            print("  用户取消.")
            return 0

        print(f"\n  [1/2] 慢速正向 (block=True S 曲线, speed={args.speed})...")
        t0 = time.time()
        arm.move_j(target, speed=args.speed, block=True)
        dt1 = time.time() - t0
        time.sleep(0.3)
        q1 = arm.get_joint_values()
        err1 = math.degrees(q1[joint_idx] - target[joint_idx])
        print(f"        到位 (deg): {_fmt_deg(q1)}  耗时 {dt1:.2f}s")
        print(f"        关节 #{joint_idx} 误差: {err1:+.3f}° "
              f"{'OK' if abs(err1) <= args.tolerance_deg else 'WARN'}")
        if abs(err1) > args.tolerance_deg:
            rc = 1

        print(f"\n  [2/2] 慢速回到起点...")
        t0 = time.time()
        arm.move_j(q0, speed=args.speed, block=True)
        dt2 = time.time() - t0
        time.sleep(0.3)
        q2 = arm.get_joint_values()
        err2 = math.degrees(q2[joint_idx] - q0[joint_idx])
        print(f"        终点 (deg): {_fmt_deg(q2)}  耗时 {dt2:.2f}s")
        print(f"        关节 #{joint_idx} 误差: {err2:+.3f}° "
              f"{'OK' if abs(err2) <= args.tolerance_deg else 'WARN'}")
        if abs(err2) > args.tolerance_deg:
            rc = 1

        print(f"\n  [stats] {arm.get_status().to_string()}")

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
    print("=" * 60)
    print(f"  结果: {'PASS' if rc == 0 else 'FAIL/WARN (rc=' + str(rc) + ')'}")
    print("=" * 60)
    return rc


if __name__ == "__main__":
    sys.exit(main())
