# -*- coding: utf-8 -*-
"""
move_j.py
=========

点到点关节运动示例 (move_j).

从当前位置把 J2 往前走 20°, 停一下, 再走回起点.
速度已经设为 10 (慢).

⚠️ 第一次跑前:
    - 机器臂上电, 桌面留出空间
    - 手放在 Ctrl+C 上

用法:
    cd fafu_robot_python
    python examples/move_j.py
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from fafu_robot_controller import FafuRobotController  # noqa: E402


def deg(q):
    return "[" + ", ".join(f"{math.degrees(v):+7.2f}" for v in q) + "]"


def main():
    with FafuRobotController(
        cfg_path="robot.cfg",
        has_gripper=True,
        gripper_motor_id=7,
        auto_enable=True,
    ) as arm:
        q0 = arm.get_joint_values()
        print(f"\n  起始 (deg): {deg(q0)}")

        if input("\n  按 Enter 开始 (q 取消): ").strip().lower() == "q":
            return

        target = q0.copy()
        target[1] = q0[1] + math.radians(20)   # J2 +20°

        print("\n  [1/2] J2 +20°")
        arm.move_j(target, speed=10, block=True)
        print(f"        到位: {deg(arm.get_joint_values())}")
        time.sleep(0.5)

        print("\n  [2/2] 回到起点")
        arm.move_j(q0, speed=10, block=True)
        print(f"        到位: {deg(arm.get_joint_values())}")
        print("\n  完成. 接下来会自动断开.")


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Ctrl+C")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
