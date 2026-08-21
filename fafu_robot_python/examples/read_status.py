# -*- coding: utf-8 -*-
"""
read_status.py
==============

只读示例: 连上机器人, 打印当前状态, 不发任何运动指令.

会打印:
    - 连接 / 关节 / 夹爪配置
    - 高层软件状态
    - 各关节角度、速度
    - 每台电机的 型号 / mode / fault / 力矩 / 缓存年龄
    - 软限位、CAN 总线、收发统计

用法:
    cd fafu_robot_python
    python examples/read_status.py
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
    # auto_enable=False: 只读, 不上电、不改电机模式
    with FafuRobotController(
        cfg_path="robot.cfg",
        has_gripper=True,
        gripper_motor_id=7,
        auto_enable=False,
    ) as arm:
        print(f"\n  端口     = {arm.port} @ {arm.baudrate}")
        print(f"  高层状态 = {arm.state}")
        print(f"  已使能   = {arm.is_enabled}")
        print(f"  关节数   = {arm.num_joints}  (motors {arm.joint_motor_ids})")
        print(f"  夹爪     = M{arm.gripper_motor_id}")

        q = arm.get_joint_values(prefer_cache=False)
        v = arm.get_joint_velocities(prefer_cache=False)
        print(f"\n  关节角 deg = {deg(q)}")
        print(f"  关节速 deg/s = {deg(v)}")

        print("\n  电机状态:")
        print("    id | type       | mode | fault | pos(deg) | vel(deg/s) | torque | age(ms)")
        for mid, s in arm.get_motor_states(prefer_cache=False).items():
            try:
                mtype = arm.cfg.find_motor_type(mid) or "-"
            except Exception:
                mtype = "-"
            try:
                age = arm.driver.state_age_ms(mid)
                age_s = f"{age:7.1f}" if age >= 0 else "    n/a"
            except Exception:
                age_s = "    n/a"
            print(
                f"    M{mid:<2}| {mtype:<10} | 0x{int(s.mode):02X} | 0x{int(s.fault):02X}  | "
                f"{math.degrees(arm._turns_to_rad(s.position)):+8.2f} | "
                f"{math.degrees(arm._turns_to_rad(s.velocity)):+9.2f} | "
                f"{int(s.torque):+6d} | {age_s}"
            )

        print("\n  软限位 (deg):")
        for mid in arm.all_motor_ids:
            lim = arm.get_limit(mid, is_radians=False)
            if lim is None:
                print(f"    M{mid}: (未设置)")
            else:
                print(f"    M{mid}: [{lim[0]:+7.2f}, {lim[1]:+7.2f}]")

        try:
            print(f"\n  CAN  = {arm.get_can_status().to_string()}")
        except Exception as e:
            print(f"\n  CAN  = 读取失败: {e}")

        print(f"  统计 = {arm.get_status().to_string()}")
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
