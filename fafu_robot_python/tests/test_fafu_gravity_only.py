# -*- coding: utf-8 -*-
"""
test_fafu_gravity_only.py
=========================

严格照搬厂商 ``2_gravity_compensation_control.py`` 的纯重力补偿:

    厂商:
        tor = robot.get_Gravity()
        tor = np.clip(tor, -tau_limit, tau_limit)     # [15,30,30,15,5,5]
        robot.pos_vel_tqe_kp_kd(0, 0, tor, kp=0, kd=0) # 纯力矩前馈
        print("重力补偿力矩：", tor)
        time.sleep(0.002)                              # ~500Hz

    本脚本 (fafu 等价):
        tau = arm.gravity_compensation_step(friction=False)  # = clip(G(q), ±tau_limit) 并下发
        time.sleep(0.002)

特点 (与厂商一致):
    - 只补重力项 G(q), 无摩擦, 无阻抗 (K/B/I 全不参与).
    - 力矩限幅默认 [15, 30, 30, 15, 5, 5] Nm.
    - ~500Hz 循环, 每个 tick 重算并下发.
    - 实时打印每关节力矩 (默认按 --print-hz 节流, 免刷屏).
    - torque_scale 默认 1.0,1.0,1.0,1.3,1.0,1.0: J4 重力力矩天生最小
      (~0.8Nm), 纯重力模式无摩擦补偿, 易被静摩擦拖住下垂, 故单独补 1.3;
      其余关节保持 1.0 (实测稳定). 想完全照搬厂商等比例, 传 --torque-scale 1.0.

与厂商的唯一差别:
    - 力矩走 fafu 的 set_torque (0x0A) 通道, 不走 MIT pos_vel_tqe_kp_kd
      (这批电机固件 MIT 通道不响应; 效果相同).
    - 退出 (正常 / Ctrl+C) 默认 **刹车** 而非掉电变软 (更安全);
      想完全照搬厂商"掉电变软", 传 --exit-stop.

前置条件:
    1. pinocchio (重力项).  -> 建议在 py3.10 conda 环境里装.
    2. 内置 follower URDF (已随包) 或 --urdf 指定.
    3. --motor-models 已默认 Fafu 实物配置.

⚠️ 安全: LIVE 模式机械臂会失重, 重关节松手会下坠, 手扶 + 急停待命.
    第一次务必先 --dry-run 看力矩量级是否合理.

用法:
    conda activate <py310-env>
    cd fafu_robot_sdk/fafu_robot_python
    # 1) 干跑, 只算不发, 看每关节力矩 / raw
    python tests/test_fafu_gravity_only.py --dry-run
    # 2) 真机纯重力浮空 (第一只臂)
    python tests/test_fafu_gravity_only.py
    # 3) 第二只臂 (J2 过补/J3 欠补 -> 用 arm2 预设)
    python tests/test_fafu_gravity_only.py --arm 2
    # 4) 手动标定每关节增益 (覆盖预设)
    python tests/test_fafu_gravity_only.py --torque-scale 1,0.85,1.15,1.3,1,1
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


# 每只实物臂的经验 torque_scale 预设 (重力前馈增益, 每关节).
# 两只臂运动学/URDF 相同, 但电机力矩常数/静摩擦/零点有个体差异,
# 故纯重力下需各自标定. --torque-scale 显式传值时覆盖本预设.
#   arm 1: 初版臂 (COM17), 实测稳定.
#   arm 2: J2 过补翘起 -> 调小; J3 欠补下垂 -> 调大 (起步值, 需现场微调).
_ARM_TORQUE_SCALE = {
    1: [1.0, 1.0, 1.0, 1.3, 1.0, 1.0],
    2: [1.0, 0.85, 1.15, 1.3, 1.0, 1.0],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="pure gravity compensation (照搬厂商 2_gravity_compensation_control)")
    parser.add_argument("--cfg", default="robot.cfg")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="夹爪 motor id (默认 7); 传 0 表示无夹爪")
    parser.add_argument("--urdf", default="",
                        help="URDF 路径; 留空自动定位内置 follower URDF")
    parser.add_argument("--motor-models",
                        default="M5036_02,M6036_02,M6036_02,M5036_02,M4438_30,M4438_30",
                        help="逗号分隔, 每关节电机型号 (Nm->raw). 默认= Fafu 实物配置.")
    parser.add_argument("--tau-limit", default="15,30,30,15,5,5",
                        help="逗号分隔力矩限幅 Nm (默认 厂商值 15,30,30,15,5,5)")
    parser.add_argument("--arm", type=int, default=1, choices=(1, 2),
                        help="实物臂编号, 选 torque_scale 预设 (1 或 2); "
                             "两只臂电机 Kt/摩擦有个体差异需各自标定. "
                             "--torque-scale 显式传值时覆盖本预设")
    parser.add_argument("--torque-scale", default="",
                        help="经验增益, 标量或逗号分隔每关节; 留空用 --arm 预设 "
                             "(arm1=1,1,1,1.3,1,1; arm2=1,0.85,1.15,1.3,1,1: "
                             "arm2 的 J2 过补翘起故调小, J3 欠补下垂故调大)")
    parser.add_argument("--rate", type=float, default=500.0,
                        help="控制循环频率 Hz (默认 500, 即 sleep 0.002)")
    parser.add_argument("--print-hz", type=float, default=10.0,
                        help="力矩打印频率 Hz (默认 10; 0 = 每 tick 都打, 照搬厂商)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="运行秒数 (默认 0 = 一直到 Ctrl+C)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算+打印, 不向电机下发力矩")
    parser.add_argument("--exit-stop", action="store_true",
                        help="退出时掉电变软(照搬厂商); 默认是刹车(更安全)")
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
        print(f"纯重力补偿 [{mode}]  ~{args.rate:.0f}Hz")
        print(f"  motor_models = {motor_models}")
        print(f"  tau_limit    = {tau_limit} Nm")
        print(f"  torque_scale = {torque_scale}  (arm {args.arm})")
        if not args.dry_run:
            print("  ⚠️ 机械臂即将失重, 手扶 + 急停待命! 3 秒后开始...")
        print("=" * 60)
        if not args.dry_run:
            time.sleep(3.0)
            arm.enable()

        period = 1.0 / max(1.0, args.rate)
        print_every = (0 if args.print_hz <= 0
                       else max(1, int(round(args.rate / args.print_hz))))
        t_start = time.monotonic()
        tick = 0
        while True:
            t0 = time.monotonic()
            tau = arm.gravity_compensation_step(
                friction=False, dry_run=args.dry_run)

            if print_every == 0 or tick % print_every == 0:
                raw = arm.tau_to_raw(tau * np.asarray(arm._dyn_torque_scale))
                tau_str = ", ".join(f"{x:+6.2f}" for x in tau)
                raw_str = ", ".join(f"{int(r):+6d}" for r in raw)
                print(f"\r重力补偿力矩(Nm)=[{tau_str}]  raw=[{raw_str}]   ", end="")

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
