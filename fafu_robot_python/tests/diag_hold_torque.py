# -*- coding: utf-8 -*-
"""
diag_hold_torque.py
===================

定位"重力补偿没力气"的关键一次性测量 (只读, 不发任何力矩命令).

原理:
    连接 FafuRobotController 时, 电机会被使能并进入"位置保持"模式
    (position control hold). 此时电机为了顶住重力, 固件内部会自己输出
    一个保持力矩, 并把这个 raw 力矩值通过 MotorState.torque 反馈回来.

    我们读出 J2/J3 (受重力最大的关节) 在保持时的反馈 raw 力矩 R:
      - 已知 J3 重力约 ~3.8 Nm.
      - 若 R 是个位数 (~6) -> 命令端 coeff=0.66 量级正确, 问题在 MIT(kp=0)
        通道没真正加前馈;
      - 若 R 是几百 (~200) -> 命令端真实 LSB 远小于 0.66, raw 要发到几百才有力,
        正确 coeff ≈ 3.8 / R, 光改系数表是不够的, 需要重新标定换算.

★ 安全: 本脚本不调用任何 set_torque / apply_compensation_torque.
        机械臂只是保持在当前姿态(和你刚上电时一样), 不会突然动.
        想读出有意义的值, 请让机械臂处于"水平伸出/受重力"的姿态再跑.

用法:
    cd fafu_robot_sdk/fafu_robot_python
    python tests/diag_hold_torque.py
    python tests/diag_hold_torque.py --seconds 5 --gripper-id 7
"""
from __future__ import annotations

import argparse
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read holding (gravity) feedback torque per joint. Read-only."
    )
    parser.add_argument("--cfg", default="robot.cfg", help="robot.cfg path")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="gripper motor id (default 7; 0 = treat all as joints)")
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="how long to sample (default 4s)")
    parser.add_argument("--rate-hz", type=float, default=10.0,
                        help="print rate (default 10Hz)")
    args = parser.parse_args()

    from fafu_robot_controller import FafuRobotController  # noqa: E402

    has_gripper = bool(args.gripper_id)
    arm = FafuRobotController(
        cfg_path=args.cfg,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id if has_gripper else None,
    )

    joint_ids = list(arm.joint_motor_ids)
    print()
    print("=" * 70)
    print(" HOLD-TORQUE 诊断 (只读, 电机处于位置保持/顶重力)")
    print(" 关注 J2/J3 的 |tqe_raw| —— 这是固件顶住重力时实际输出的 raw 力矩")
    print("=" * 70)
    header = "  t(s) | " + " | ".join(f"J{i+1} pos°  tqe_raw" for i in range(len(joint_ids)))
    print(header)

    peak = {mid: 0 for mid in joint_ids}
    t0 = time.monotonic()
    period = 1.0 / max(1.0, args.rate_hz)
    try:
        while (time.monotonic() - t0) < args.seconds:
            tick = time.monotonic()
            states = arm.get_motor_states(prefer_cache=True)
            cells = []
            for mid in joint_ids:
                s = states.get(mid)
                if s is None:
                    cells.append("   --      --  ")
                    continue
                tq = int(s.torque)
                if abs(tq) > abs(peak[mid]):
                    peak[mid] = tq
                pos_deg = s.position * 360.0
                cells.append(f"{pos_deg:+7.1f} {tq:+7d}")
            print(f"  {tick - t0:4.1f} | " + " | ".join(cells))
            slp = period - (time.monotonic() - tick)
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        print("\n[diag] interrupted")
    finally:
        try:
            arm.close_connection()
        except Exception:
            pass

    print()
    print("  峰值 |tqe_raw| (保持期间各关节):")
    for i, mid in enumerate(joint_ids):
        print(f"    J{i+1} (M{mid}): {peak[mid]:+d}")
    print()
    print("  解读: J3 重力 ~3.8 Nm.")
    print("    - 若 J2/J3 峰值是个位数 -> coeff 0.66 量级对, 问题在 kp=0 MIT 通道;")
    print("    - 若 J2/J3 峰值是几百   -> 命令 LSB 远小于 0.66, 需重标定换算.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
