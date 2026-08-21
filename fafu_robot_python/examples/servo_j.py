# -*- coding: utf-8 -*-
"""
servo_j.py
==========

在线流式关节控制示例 (servo_j).

跟 move_j 的区别: move_j 是一次给目标、内部自己走完;
servo_j 要你按固定频率 (这里 100Hz) 不停地喂下一个目标点.

本例从当前位置出发, 让 J2 按一个来回的正弦摆 +8°, 约 5 秒后回到起点.

⚠️ 第一次跑前:
    - 机器臂上电, 桌面留出空间
    - 手放在 Ctrl+C 上 (固件看门狗约 100ms 后会自动刹车)

用法:
    cd fafu_robot_python
    python examples/servo_j.py
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

from fafu_robot_controller import FafuRobotController, ServoOpts  # noqa: E402

# Windows 默认定时粒度约 15ms, 不抬高的话 100Hz 会抖.
if sys.platform == "win32":
    try:
        import atexit
        import ctypes

        _winmm = ctypes.WinDLL("winmm")
        _winmm.timeBeginPeriod(1)
        atexit.register(_winmm.timeEndPeriod, 1)
    except Exception:
        pass


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

        # use_mit=False: 走位置通道, 不需要动力学模型
        arm.servo_start(ServoOpts(
            watchdog_ms=100,
            max_vel=1.0,
            max_step_rad=math.radians(2),
            max_lag_rad=math.radians(12),
            use_mit=False,
        ))

        dt = 0.01          # 100 Hz
        duration = 5.0     # 秒
        amp = math.radians(8)

        print(f"\n  100Hz 流式跟踪 {duration:.0f}s, J2 ±0~+8° ...")
        try:
            t0 = time.monotonic()
            while True:
                t = time.monotonic() - t0
                if t >= duration:
                    break
                q = q0.copy()
                # 0 → +8° → 0, 起点终点都是当前位置
                q[1] = q0[1] + amp * 0.5 * (1.0 - math.cos(2.0 * math.pi * t / duration))
                arm.servo_j(q)
                time.sleep(dt)
        finally:
            arm.servo_end("hold")

        print(f"  结束 (deg): {deg(arm.get_joint_values())}")
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
