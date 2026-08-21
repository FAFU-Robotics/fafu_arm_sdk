# -*- coding: utf-8 -*-
"""
go_home.py
==========

把所有关节回到 0°.

速度已经设为 10 (慢).

⚠️ 第一次跑前:
    - 机器臂上电, 回零路径上没有障碍
    - 手放在 Ctrl+C 上

用法:
    cd fafu_robot_python
    python examples/go_home.py
"""
from __future__ import annotations

import math
import os
import sys
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
        print(f"\n  起始 (deg): {deg(arm.get_joint_values())}")

        if input("\n  按 Enter 回零 (q 取消): ").strip().lower() == "q":
            return

        arm.go_home(speed=10, block=True)
        print(f"  回零 (deg): {deg(arm.get_joint_values())}")
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
