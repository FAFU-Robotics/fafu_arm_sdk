# -*- coding: utf-8 -*-
"""
gripper.py
==========

夹爪开合示例.

默认走 robot.cfg 里夹爪 (M7) 的软限位, 当前配置是 0° ~ 105°.

⚠️ 第一次跑前确认夹爪前方没有夹到不该夹的东西.

用法:
    cd fafu_robot_python
    python examples/gripper.py
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


def gripper_deg(arm):
    gs = arm.get_gripper_state()
    return math.degrees(arm._turns_to_rad(gs.position))


def main():
    with FafuRobotController(
        cfg_path="robot.cfg",
        has_gripper=True,
        gripper_motor_id=7,
        auto_enable=True,
    ) as arm:
        print(f"\n  夹爪当前位置 = {gripper_deg(arm):+.2f}°")

        if input("\n  按 Enter 开始 (q 取消): ").strip().lower() == "q":
            return

        print("\n  [1/3] 打开")
        arm.open_gripper()
        time.sleep(0.3)
        print(f"        到位 {gripper_deg(arm):+.2f}°")

        print("\n  [2/3] 闭合")
        arm.close_gripper()
        time.sleep(0.3)
        print(f"        到位 {gripper_deg(arm):+.2f}°")

        print("\n  [3/3] 再打开")
        arm.open_gripper()
        time.sleep(0.3)
        print(f"        到位 {gripper_deg(arm):+.2f}°")
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
