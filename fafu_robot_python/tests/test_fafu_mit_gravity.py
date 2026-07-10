# -*- coding: utf-8 -*-
"""
test_fafu_mit_gravity.py
========================

验证 **6 关节同帧一拖多 MIT** (CAN ID 0x8093) 的纯重力补偿:

    每 tick:
        q   = get_joint_values()
        tau = compute_compensation_torque(q, v, friction=False)  # = clip(G(q), ±tau_limit)
        arm.move_MIT(q, 0, tau, kp=0, kd=0)               # 一帧下发 J1-J6
        sleep(1/rate)

和 test_fafu_gravity_only.py 的唯一差别:
    - gravity_only 走 set_torque (0x0A) 逐关节单发 (每关节一帧);
    - 本脚本走 move_MIT -> set_many_mit (0x8093) 六关节 **一帧** 下发,
      对齐官方 pos_vel_tqe_kp_kd(pos,vel,gravity,kp=0,kd=0) 的一拖多语义.

目的:
    确认整组 MIT 通道对 J1-J6 全部有效 (之前 diag_torque_ramp --path mit-many
    只验证了单关节 J1). 若六关节都能失重浮空/托得住, 说明可以把重力补偿 /
    拖动示教 / 回放正式切到一拖多 MIT.

⚠️ 安全: LIVE 模式机械臂会失重, 重关节松手会下坠, 手扶 + 急停待命.
    第一次务必先 --dry-run 看力矩/raw 量级是否合理.

前置条件:
    1. pinocchio (重力项).  -> 建议在 py3.10 conda 环境里装.
    2. 内置 follower URDF (已随包) 或 --urdf 指定.
    3. --gripper-id 7 (默认): 夹爪不进 MIT 帧 (一帧最多 6 个电机).

用法:
    conda activate <py310-env>
    cd fafu_robot_sdk/fafu_robot_python
    # 1) 干跑, 只算不发, 看每关节力矩 / raw
    python tests/test_fafu_mit_gravity.py --dry-run
    # 2) 真机: 六关节同帧 MIT 纯重力浮空
    python tests/test_fafu_mit_gravity.py
    # 3) 换第二只臂预设
    python tests/test_fafu_mit_gravity.py --arm 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# 每只实物臂的经验 torque_scale 预设 (与 test_fafu_gravity_only.py 保持一致).
_ARM_TORQUE_SCALE = {
    1: [1.0, 1.0, 1.0, 1.3, 1.0, 1.0],
    2: [1.0, 0.85, 1.15, 1.3, 1.0, 1.0],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="6-joint group MIT (0x8093) pure gravity compensation")
    parser.add_argument("--cfg", default="robot.cfg")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="夹爪 motor id (默认 7); MIT 帧只发 J1-J6, 夹爪不进帧")
    parser.add_argument("--urdf", default="",
                        help="URDF 路径; 留空自动定位内置 follower URDF")
    parser.add_argument("--motor-models",
                        default="M5036_02,M6036_02,M6036_02,M5036_02,M4438_30,M4438_30",
                        help="逗号分隔, 每关节电机型号 (Nm->raw). 默认= Fafu 实物配置.")
    parser.add_argument("--tau-limit", default="15,30,30,15,5,5",
                        help="逗号分隔力矩限幅 Nm (默认 厂商值 15,30,30,15,5,5)")
    parser.add_argument("--arm", type=int, default=1, choices=(1, 2),
                        help="实物臂编号, 选 torque_scale 预设 (1 或 2); "
                             "--torque-scale 显式传值时覆盖")
    parser.add_argument("--torque-scale", default="",
                        help="经验增益, 标量或逗号分隔每关节; 留空用 --arm 预设")
    parser.add_argument("--rate", type=float, default=200.0,
                        help="控制循环频率 Hz (默认 200; 一拖多带回读, 别太高)")
    parser.add_argument("--print-hz", type=float, default=10.0,
                        help="力矩打印频率 Hz (默认 10; 0 = 每 tick 都打)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行秒数 (默认 0 = 一直到 Ctrl+C)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算+打印, 不向电机下发力矩")
    parser.add_argument("--exit-stop", action="store_true",
                        help="退出时掉电变软; 默认是刹车(更安全)")
    args = parser.parse_args()

    from fafu_robot_controller import FafuRobotController

    motor_models = [s.strip() for s in args.motor_models.split(",") if s.strip()]
    tau_limit = [float(s) for s in args.tau_limit.split(",") if s.strip()]
    if args.torque_scale.strip():
        ts = [float(s) for s in args.torque_scale.split(",") if s.strip()]
        torque_scale = ts[0] if len(ts) == 1 else ts
    else:
        torque_scale = list(_ARM_TORQUE_SCALE[args.arm])

    has_gripper = args.gripper_id > 0
    arm = FafuRobotController(
        cfg_path=args.cfg,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id if has_gripper else 7,
    )

    try:
        n = arm.num_joints
        if n > 6:
            print(f"[TEST][FAIL] 一拖多 MIT 单帧最多 6 个电机, 当前 num_joints={n}; "
                  f"请用 --gripper-id 7 把夹爪排除在关节之外")
            return 1
        if len(motor_models) != n:
            print(f"[TEST][FAIL] --motor-models 需要 {n} 个, 收到 {len(motor_models)}")
            return 1
        if len(tau_limit) != n:
            print(f"[TEST][FAIL] --tau-limit 需要 {n} 个, 收到 {len(tau_limit)}")
            return 1

        arm.setup_dynamics(
            urdf_path=args.urdf or None,
            motor_models=motor_models,
            tau_limit=tau_limit,
            torque_scale=torque_scale,
        )

        mode = "DRY-RUN (只算不发)" if args.dry_run else "LIVE (真机下发)"
        print("\n" + "=" * 60)
        print(f"六关节同帧 MIT (0x8093) 纯重力补偿 [{mode}]  ~{args.rate:.0f}Hz")
        print(f"  motor_ids    = {arm.joint_motor_ids}")
        print(f"  motor_models = {motor_models}")
        print(f"  tau_limit    = {tau_limit} Nm")
        print(f"  torque_scale = {torque_scale}  (arm {args.arm})")
        print(f"  通道         = move_MIT -> set_many_mit (kp=kd=0 纯前馈)")
        if not args.dry_run:
            print("  ⚠️ 机械臂即将失重, 手扶 + 急停待命! 3 秒后开始...")
        print("=" * 60)
        if not args.dry_run:
            time.sleep(3.0)
            arm.enable()

        zeros = [0.0] * n
        period = 1.0 / max(1.0, args.rate)
        print_every = (0 if args.print_hz <= 0
                       else max(1, int(round(args.rate / args.print_hz))))
        t_start = time.monotonic()
        tick = 0
        while True:
            t0 = time.monotonic()
            q = arm.get_joint_values()
            v = arm.get_joint_velocities()
            tau = arm.compute_compensation_torque(q, v, friction=False)

            if not args.dry_run:
                # kp=kd=0 => 纯力矩前馈 (= 官方 pos_vel_tqe_kp_kd(q,0,tau,0,0));
                # move_MIT 内部乘 torque_scale 并 Nm->raw, 一帧下发 J1-J6.
                arm.move_MIT(q, zeros, tau, kp=0.0, kd=0.0,
                             is_radians=True, apply_torque_scale=True,
                             timeout=0.0)

            if print_every == 0 or tick % print_every == 0:
                raw = arm.tau_to_raw(tau * np.asarray(arm._dyn_torque_scale))
                tau_str = ", ".join(f"{x:+6.2f}" for x in tau)
                raw_str = ", ".join(f"{int(r):+6d}" for r in raw)
                print(f"\rMIT重力(Nm)=[{tau_str}]  raw=[{raw_str}]   ", end="")

            tick += 1
            if args.duration > 0 and (time.monotonic() - t_start) >= args.duration:
                print("\n[TEST] 到达 --duration, 结束.")
                break

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

        return 0

    except KeyboardInterrupt:
        print("\n\n[TEST] 程序被中断")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"\n[TEST][FAIL] {e}")
        traceback.print_exc()
        return 1
    finally:
        try:
            arm.close_connection(
                joint_release="stop" if args.exit_stop else "brake")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
