# -*- coding: utf-8 -*-
"""
visible_motion.py
=================

明显能看出动作的真机示例. 演示要点:
    1. 正确传 gripper_motor_id=7  → 6 个关节 + 1 个夹爪
    2. 把肉眼看得见的关节真的动起来 (J2 + J4 各 30 度)
    3. 夹爪也走一个完整的 开 → 关 循环

⚠️ 第一次跑前:
    - 机器臂上电
    - 桌面留 30cm 空间, 手放在 Ctrl+C 上
    - 速度已经设为 10 (慢)

用法:
    cd fafu_robot_sdk
    python examples/visible_motion.py
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback

# Make the SDK root importable when this script is run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from fafu_robot_controller import FafuRobotController  # noqa: E402


def fmt(q):
    return "[" + ", ".join(f"{math.degrees(v):+7.2f}" for v in q) + "]"


def main():
    # 这里用 with, 出错也能干净断开.
    # cfg_path 用相对路径, 由 FafuRobotController 内部自动回退到 SDK 根.
    with FafuRobotController(
        cfg_path="robot.cfg",
        has_gripper=True,
        gripper_motor_id=7,        # ← 重点: 7 号是夹爪
        auto_enable=True,
    ) as arm:
        print(f"\n  关节数 = {arm.num_joints} (joint motors: {arm.joint_motor_ids})")
        print(f"  夹爪   = M{arm.gripper_motor_id}")

        q0 = arm.get_joint_values()
        print(f"  起始 (deg): {fmt(q0)}")

        confirm = input("\n  按 Enter 开始 (q 取消): ").strip().lower()
        if confirm == "q":
            return

        # ----- 1) 让 J2 走到 30° (J2 限位 0~91°, 30° 完全合法且明显) -----
        print("\n  [1/4] J2 -> 30° (慢速)")
        target = q0.copy()
        target[1] = math.radians(30)        # joint index 1 = J2
        arm.move_j(target, speed=10, block=True)
        time.sleep(0.3)
        print(f"        到位: {fmt(arm.get_joint_values())}")

        # ----- 2) J4 走到 -30° (J4 限位 -125~91°, -30° 明显) -----
        print("\n  [2/4] J4 -> -30° (J2 保持)")
        target[3] = math.radians(-30)       # joint index 3 = J4
        arm.move_j(target, speed=10, block=True)
        time.sleep(0.3)
        print(f"        到位: {fmt(arm.get_joint_values())}")

        # ----- 3) 全部回到起点 -----
        print("\n  [3/4] 回到起点")
        arm.move_j(q0, speed=10, block=True)
        time.sleep(0.3)
        print(f"        到位: {fmt(arm.get_joint_values())}")

        # ----- 4) 夹爪开关循环 -----
        # 你的硬件: -1.83° = 开 (软限位上限), -114.98° = 关到底 (软限位下限)
        # 现在 open_gripper() / close_gripper() 默认就走对应的软限位.
        print("\n  [4/4] 夹爪 open -> close -> open")
        print("        open  (-> upper soft limit ~ -1.83°)")
        arm.open_gripper()
        time.sleep(2.0)
        gs = arm.get_gripper_state()
        print(f"        gripper pos = {math.degrees(arm._turns_to_rad(gs.position)):+.2f}°")

        print("        close (-> lower soft limit ~ -114.98°)")
        arm.close_gripper()
        time.sleep(3.0)         # 闭合行程长一些, 多等等
        gs = arm.get_gripper_state()
        print(f"        gripper pos = {math.degrees(arm._turns_to_rad(gs.position)):+.2f}°")

        print("        open  (-> upper soft limit)")
        arm.open_gripper()
        time.sleep(2.0)
        gs = arm.get_gripper_state()
        print(f"        gripper pos = {math.degrees(arm._turns_to_rad(gs.position)):+.2f}°")

        print(f"\n  [stats] {arm.get_status().to_string()}")
        print("\n  完成. 接下来会自动断开.")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Ctrl+C")
        sys.exit(130)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
