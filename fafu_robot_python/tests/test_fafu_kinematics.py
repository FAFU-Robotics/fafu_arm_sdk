# -*- coding: utf-8 -*-
"""
test_fafu_kinematics.py
=======================

正/逆运动学 (FK/IK) + 笛卡尔运动 (move_p / move_l) 的真机/半离线测试.

封装在 FafuRobotController 里, 复用厂商 follower URDF (pinocchio):

    arm.setup_dynamics(eef_frame="tool_link")   # 一次: 载入 URDF
    fk = arm.forward_kinematics(q)               # 正解
    q  = arm.inverse_kinematics(pos, rot)         # 逆解 (阻尼最小二乘)
    arm.move_p([x, y, z], rpy, is_euler=True)     # 笛卡尔点到点
    arm.move_l([x, y, z], rpy, is_euler=True)     # 笛卡尔直线

前置条件:
    1. pinocchio (FK/IK 都要). Windows 建议 conda-forge / WSL.
    2. 一个带运动学链的 URDF. 不传 --urdf 时自动定位:
         <package>/fafu_robot_description/*.urdf  (已随包内置 follower)

测试内容:
    [1] FK: 打印当前位姿 (位置 m + RPY deg).
    [2] FK<->IK 往返自洽: 随机若干组关节角 -> FK 得位姿 -> IK 回解 ->
        再 FK, 比较末端笛卡尔误差 (默认阈值 1mm / 0.5deg). 不动机械臂.
    [3] (可选 --move) 真机笛卡尔运动: 当前位姿基础上沿 +Z 抬升 --dz 米,
        move_p 过去再 move_l 回来. ⚠️ 会真实运动, 手扶 + 急停待命.

用法:
    cd fafu_robot_sdk/fafu_robot_python
    # 只跑 FK + 往返自洽 (不动机械臂)
    python tests/test_fafu_kinematics.py --samples 20
    # 附带真机笛卡尔运动 (抬升 3cm 再直线回来)
    python tests/test_fafu_kinematics.py --move --dz 0.03 --speed 20
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def _fmt_pose(pos, rpy_rad) -> str:
    p = np.asarray(pos, dtype=float)
    r = np.rad2deg(np.asarray(rpy_rad, dtype=float))
    return (f"pos(m)=[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]  "
            f"rpy(deg)=[{r[0]:+.2f}, {r[1]:+.2f}, {r[2]:+.2f}]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FK / IK / move_p / move_l test")
    parser.add_argument("--cfg", default="robot.cfg",
                        help="robot.cfg 路径 (默认: fafu_robot_python/robot.cfg)")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="夹爪 motor id (默认 7); 传 0 表示无夹爪")
    parser.add_argument("--urdf", default="",
                        help="URDF 路径; 留空则自动定位内置 follower URDF")
    parser.add_argument("--eef-frame", default="tool_link",
                        help="末端坐标系 frame 名 (默认 tool_link)")
    parser.add_argument("--samples", type=int, default=20,
                        help="FK<->IK 往返自洽的随机采样组数 (默认 20)")
    parser.add_argument("--pos-tol", type=float, default=1e-3,
                        help="往返位置误差阈值 m (默认 1e-3 = 1mm)")
    parser.add_argument("--rot-tol-deg", type=float, default=0.5,
                        help="往返姿态误差阈值 deg (默认 0.5)")
    parser.add_argument("--move", action="store_true",
                        help="★ 真机笛卡尔运动测试 (move_p 抬升 + move_l 返回)")
    parser.add_argument("--dz", type=float, default=0.03,
                        help="--move 时末端沿 +Z 抬升的高度 m (默认 0.03)")
    parser.add_argument("--speed", type=int, default=20,
                        help="--move 时的速度百分比 (默认 20)")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机数种子 (默认 0)")
    args = parser.parse_args()

    from fafu_robot_controller import FafuRobotController

    rng = np.random.default_rng(args.seed)
    has_gripper = args.gripper_id > 0

    arm = FafuRobotController(
        cfg_path=args.cfg,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id if has_gripper else 7,
    )

    try:
        arm.setup_dynamics(
            urdf_path=args.urdf or None,
            eef_frame=args.eef_frame,
        )

        # -------------------------------------------------- [1] FK
        print("\n===== [1] 当前位姿 (FK) =====")
        q0 = arm.get_joint_values()
        fk0 = arm.forward_kinematics(q0)
        print(f"  q(deg) = {np.rad2deg(q0).round(2).tolist()}")
        print(f"  {_fmt_pose(fk0['position'], fk0['rpy'])}")

        # -------------------------------------------------- [2] FK<->IK
        print(f"\n===== [2] FK<->IK 往返自洽 ({args.samples} 组) =====")
        lim = arm._joint_limits_rad()  # noqa: SLF001 (test introspection)
        if lim is not None:
            lo, hi = lim
        else:
            # 没配限位: 用一个保守范围采样
            lo = np.full(arm.num_joints, -1.0)
            hi = np.full(arm.num_joints, 1.0)
            print("  (未配置关节限位, 采样范围回退到 ±1 rad)")

        n_pass = 0
        worst_p = 0.0
        worst_r = 0.0
        for k in range(args.samples):
            q_ref = rng.uniform(lo, hi)
            fk = arm.forward_kinematics(q_ref)
            q_sol = arm.inverse_kinematics(
                fk["position"], fk["rotation"], multi_init=True,
            )
            if q_sol is None:
                print(f"  [{k + 1:>2}] IK 未收敛 (目标可能近奇异)")
                continue
            fk2 = arm.forward_kinematics(q_sol)
            ep = float(np.linalg.norm(fk2["position"] - fk["position"]))
            # 姿态误差: 旋转差的角度
            dR = fk["rotation"].T @ fk2["rotation"]
            cos = (np.trace(dR) - 1.0) / 2.0
            er = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
            worst_p = max(worst_p, ep)
            worst_r = max(worst_r, er)
            ok = ep <= args.pos_tol and er <= args.rot_tol_deg
            n_pass += int(ok)
            flag = "OK " if ok else "BAD"
            print(f"  [{k + 1:>2}] {flag} pos_err={ep * 1e3:6.3f}mm  "
                  f"rot_err={er:6.3f}deg")
        print(f"  --> {n_pass}/{args.samples} 通过; "
              f"最差 pos={worst_p * 1e3:.3f}mm, rot={worst_r:.3f}deg")

        # -------------------------------------------------- [3] move
        if args.move:
            print(f"\n===== [3] 真机笛卡尔运动 (抬升 {args.dz * 1e3:.0f}mm) =====")
            print("  ⚠️ 机械臂即将运动, 手扶 + 急停待命! 3 秒后开始...")
            import time
            time.sleep(3.0)

            arm.enable()
            pos0, rot0 = arm.get_pose()
            target = pos0.copy()
            target[2] += args.dz
            print(f"  move_p -> {_fmt_pose(target, arm.forward_kinematics(q0)['rpy'])}")
            arm.move_p(target, rot0, speed=args.speed, block=True)

            posN, _ = arm.get_pose()
            print(f"  到位: pos(m)=[{posN[0]:+.4f}, {posN[1]:+.4f}, {posN[2]:+.4f}]")

            print(f"  move_l -> 直线返回起点")
            arm.move_l(pos0, rot0, speed=args.speed, steps=20)
            posB, _ = arm.get_pose()
            back_err = float(np.linalg.norm(posB - pos0))
            print(f"  返回误差: {back_err * 1e3:.2f}mm")

        print("\n[TEST] done.")
        return 0

    except Exception as e:  # noqa: BLE001
        print(f"\n[TEST][FAIL] {e}")
        traceback.print_exc()
        return 1
    finally:
        try:
            arm.close_connection(joint_release="brake" if args.move else "stop")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
