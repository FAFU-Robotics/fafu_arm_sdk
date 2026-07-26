# -*- coding: utf-8 -*-
"""
diag_torque_ramp.py
===================

★★★ 主动力矩诊断 —— 会让单个关节真正出力. 安全版, 但仍务必看完说明 ★★★

背景 / 教训:
    `set_torque(mid, 0)` 不是"保持不动", 而是把电机从位置保持切到 *纯力矩*
    模式、且力矩=0 => 关节完全松开. 受重力的水平关节会立刻甩落!
    (上一版默认测 J2 就因此甩动, 已修正.)

本安全版的做法:
    1. 默认只测 **J1 (基座, 竖直轴, 重力不会让它甩)**, 先安全验证:
         - 力矩通道到底有没有真正出力;
         - 大约多大 raw 能克服静摩擦让它开始转 (=> 力矩标度量级).
    2. 任一保护触发立即停: |vel| 超阈值 / 偏离起始角超 --max-dpos-deg.
    3. 退出时把关节 **重新切回位置保持** (按当前角), 不留在松开状态.
    4. raw=0 起步; 若 raw=0 就动 => 说明该通道一接管就松开保持, 立即报告并停.

★ 受重力的关节 (J2/J3/J4) 不要用纯力矩通道从 0 起步测! 会甩落.
  如果一定要测 J2/J3, 先把它摆到"连杆竖直向下"的稳定平衡位再测, 且手扶住.

用法 (建议先 J1):
    cd fafu_robot_sdk/fafu_robot_python
    python tests/diag_torque_ramp.py --joint 1 --path torque --max-raw 200 --step 5
    python tests/diag_torque_ramp.py --joint 1 --path mit    --max-raw 200 --step 5
    python tests/diag_torque_ramp.py --joint 1 --path mit-many --max-raw 200 --step 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAFE single-joint active torque ramp (MOVES THE ROBOT)."
    )
    parser.add_argument("--cfg", default="robot.cfg")
    parser.add_argument("--gripper-id", type=int, default=7)
    parser.add_argument("--joint", type=int, default=1,
                        help="1-based joint index (default 1 = J1 base, gravity-safe)")
    parser.add_argument("--path", choices=["torque", "mit", "mit-many"],
                        default="torque",
                        help=("torque=set_torque(0x0a); "
                              "mit=single set_pos_vel_tqe_kp_kd_rad(0x15); "
                              "mit-many=broadcast ID 0x8093"))
    parser.add_argument("--sign", type=float, default=1.0,
                        help="+1 or -1: torque direction")
    parser.add_argument("--max-raw", type=float, default=200.0,
                        help="max |raw| torque (start moderate)")
    parser.add_argument("--step", type=float, default=5.0,
                        help="raw increment per tick")
    parser.add_argument("--step-dt", type=float, default=0.2)
    parser.add_argument("--vel-abort", type=float, default=0.15 * math.tau,
                        help="abort when |vel| (rad/s) exceeds this")
    parser.add_argument("--max-dpos-deg", type=float, default=20.0,
                        help="abort when |pos - start| exceeds this (deg)")
    args = parser.parse_args()

    from fafu_robot_controller import FafuRobotController  # noqa: E402

    arm = FafuRobotController(
        cfg_path=args.cfg,
        has_gripper=bool(args.gripper_id),
        gripper_motor_id=args.gripper_id if args.gripper_id else None,
    )
    joint_ids = list(arm.joint_motor_ids)
    if not (1 <= args.joint <= len(joint_ids)):
        print(f"[FAIL] --joint must be 1..{len(joint_ids)}")
        arm.close_connection()
        return 1
    mid = joint_ids[args.joint - 1]
    drv = arm._ht  # private: low-level diagnostic only

    def read_state():
        return arm.get_motor_states(prefer_cache=False).get(mid)

    def send_torque(raw: float):
        r = float(raw) * args.sign
        # Empty model uses coeff=1, so one raw count equals 0.01 Nm.
        if args.path == "torque":
            drv.set_torque(mid, r * 0.01, "")
        elif args.path == "mit":
            st = read_state()
            pos_rad = st.position * math.tau if st is not None else 0.0
            drv.set_pos_vel_tqe_kp_kd_rad(
                mid, pos_rad, 0.0, r * 0.01, 0.0, 0.0, "")
        else:
            st = read_state()
            pos_rad = st.position * math.tau if st is not None else 0.0
            if not hasattr(drv, "set_many_mit_rad"):
                raise RuntimeError(
                    "当前 fafu_motor.pyd 还没有 "
                    "set_many_mit_rad; 请先重新编译 C++ 扩展")
            # 一拖多 MIT/PD (ID=0x8093): 帧内每电机 10 字节, CAN-FD 单帧最大
            # 64 字节 => 一帧最多 6 个电机. 诊断只测单关节, 槽位数只需覆盖到
            # 被测电机 (mid) 即可, 前面的槽位自动填 0x8000 no-op.
            # (J7 是夹爪, 不走 MIT; 它用 set_pos_vel_tqe 位置/速度/最大力矩.)
            max_mid = mid
            if raw <= 1e-9:
                print(f"  [mit-many] mid={mid}, max_mid={max_mid}, "
                      f"pos_rad={pos_rad:+.5f} rad")
            drv.set_many_mit_rad(
                [mid], [pos_rad], [0.0], [int(round(r))],
                [0], [0], max_mid, 0.05)

    def rehold():
        """把关节切回位置保持(按当前角), 避免松开下垂."""
        st = read_state()
        if st is None:
            try:
                drv.stop(mid)
            except Exception:
                pass
            return
        try:
            drv.set_pos_vel_acc_rad(
                mid, st.position * math.tau,
                0.3 * math.tau, 2.0 * math.tau)
        except Exception:
            try:
                drv.stop(mid)
            except Exception:
                pass

    if args.joint != 1 and args.path in ("torque", "mit", "mit-many"):
        print("  ⚠ 你在测非 J1 关节. 若该关节受重力, raw=0/切换通道可能会甩落!")
        print("    请确认已摆到稳定平衡位并手扶住.")

    print()
    print("=" * 70)
    print(f" 安全力矩斜坡: J{args.joint} (M{mid}), 通道={args.path}, "
          f"sign={args.sign:+.0f}, max_raw={args.max_raw}, "
          f"vel_abort={args.vel_abort}, max_dpos={args.max_dpos_deg}°")
    print("=" * 70)
    if args.path == "mit-many":
        print("  mit-many 使用协议文档的一拖多 MIT/PD 帧: CAN ID=0x8093, "
              "[pos,vel,tqe,kp,kd]*N + 0x17 0x01")
        print("  只填被测关节槽位, 其它电机槽位为 0x8000 no-op; kp=kd=0, "
              "tqe_raw 直接使用 --max-raw/--step 的 raw 值.")
    try:
        input(" 确认安全后按 Enter 开始 (Ctrl+C 取消)... ")
    except KeyboardInterrupt:
        print("\n[cancel]")
        arm.close_connection()
        return 0

    st0 = read_state()
    pos0 = (st0.position * 360.0) if st0 else 0.0
    print(f"  起始 J{args.joint} = {pos0:+.1f}°")
    print("  raw  | pos°    | vel(rad/s) | dpos°")
    print("  -----+---------+----------+--------")

    raw = 0.0
    move_raw = None
    reason = "reached_max_no_motion"
    try:
        while raw <= args.max_raw + 1e-9:
            send_torque(raw)
            time.sleep(args.step_dt)
            st = read_state()
            if st is None:
                print(f"  {raw:4.0f} |   (no state)")
                raw += args.step
                continue
            pos_deg = st.position * 360.0
            vel = st.velocity * math.tau
            dpos = pos_deg - pos0
            print(f"  {raw:4.0f} | {pos_deg:+7.1f} | {vel:+8.3f} | {dpos:+6.1f}")

            if abs(vel) > args.vel_abort or abs(dpos) > args.max_dpos_deg:
                if raw <= 1e-9:
                    reason = "moved_at_raw0_hold_released"
                    print(f"  ★ raw=0 就动了 (vel={vel:+.3f}, dpos={dpos:+.1f}°) "
                          f"=> 此通道一接管就松开保持. 立即停.")
                else:
                    reason = "started_moving"
                    move_raw = raw
                    print(f"  ★ 开始运动 @ raw={raw:.0f} "
                          f"(vel={vel:+.3f}, dpos={dpos:+.1f}°) => 此 raw 克服了阻力/重力. 停.")
                break
            raw += args.step
    except KeyboardInterrupt:
        print("\n[diag] Ctrl+C")
        reason = "ctrl_c"
    finally:
        rehold()
        time.sleep(0.2)
        try:
            arm.close_connection()
        except Exception:
            pass

    print()
    print(f"  结果: {reason}")
    if move_raw is not None:
        print(f"  起动 raw ≈ {move_raw:.0f}")
        print(f"  -> 该通道有效, 力矩标度量级: 起动需 raw ~{move_raw:.0f}.")
        print(f"     若起动 raw 是几百 => 命令端 LSB 远小于 0.66, coeff/换算需重标定.")
        print(f"     若起动 raw 是个位/十几 => 当前 coeff 量级基本对.")
    elif reason == "reached_max_no_motion":
        print(f"  到 max_raw={args.max_raw:.0f} 完全不动:")
        print(f"     若 path=torque 不动而 mit 动(或反之) => 只有动的那条通道有效.")
        print(f"     两条都不动 => 加大 --max-raw 再试 (LSB 极小).")
    elif reason == "moved_at_raw0_hold_released":
        print(f"  此通道一发命令就脱离保持. 说明它确实接管了电机(通道有效),")
        print(f"  但无法用'从0斜坡'测起动点. 改用 J1 (竖直轴) 测起动 raw.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
