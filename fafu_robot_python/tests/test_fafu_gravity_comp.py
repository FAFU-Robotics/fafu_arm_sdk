# -*- coding: utf-8 -*-
"""
test_fafu_gravity_comp.py
=========================

重力 + 摩擦力补偿 ("浮空 / 拖动w'w's's's's'swwsssss示教") 真机示例.

重力 + 摩擦力前馈补偿, 封装到 FafuRobotController 后只需几行:

    arm.setup_dynamics(motor_models=[...])      # 一次: 载入 URDF (pinocchio)
    arm.start_gravity_compensation()            # 循环: 机械臂变"失重", 可手拖

控制律:
    tau = clip( G(q) + [ fc*sign(v) + fv*v ],  ±tau_limit )   (单位 Nm)
经 MIT 模式 (pos=vel=kp=kd=0) 作为纯前馈力矩下发到每个关节.

前置条件:
    1. pip / conda 安装 pinocchio   (Windows 建议 conda-forge 或 WSL)
       重力项必须用它; 仅摩擦补偿不需要.
    2. 一个带 <inertial> 的 URDF. 不传 --urdf 时自动定位:
         <package>/fafu_robot_description/*.urdf (已随包)
    3. ★ 每个关节的电机型号 (--motor-models), 否则力矩 Nm->raw 不被正确
       缩放, 机械臂会"欠驱动 / 下垂" (安全但没用).

⚠️ 安全:
    - 第一次务必先 --dry-run, 看打印的 tau 数值量级是否合理 (远超
      tau_limit 说明 motor_models / URDF 不对).
    - LIVE 模式机械臂会失重, 重的关节会在你松手时下坠 —— 手扶住 + 急停待命.
    - 退出 (正常 / Ctrl+C / 异常) 时所有关节进入 stop (自由) 模式.

用法:
    cd fafu_robot_sdk/fafu_robot_python
    # 1) 干跑, 只算不发, 看力矩量级
    python tests/test_fafu_gravity_comp.py --dry-run --verbose --duration 5
    # 2) 仅摩擦补偿 (不需要 pinocchio, 验证 MIT 下发通路)
    python tests/test_fafu_gravity_comp.py --no-gravity
    # 3) 真机重力+摩擦浮空 (--motor-models 已默认 Fafu 实物配置, 可不传)
    python tests/test_fafu_gravity_comp.py --verbose

标定流程:
    ★ 力矩 Nm->raw 换算已修正 (firmware LSB = coeff*0.01 Nm), torque_scale
      默认 1.0 即为物理正确值, 不再需要调到几十. 若仍想微调:
    a) 先 dry-run 看每关节 raw int16 量级 (现在应是几百, 不再是个位数):
       python tests/test_fafu_gravity_comp.py --dry-run --verbose --duration 5
    b) 若略欠/略过, 在 1.0 附近小幅调 --torque-scale (如 0.9 / 1.1 / 1.2),
       手扶着机械臂, 观察到"刚好浮起不下垂"即可.
       ⚠️ 不要再用 ~90 这种大增益! 换算修正后那会放大 100x 导致失控.
    c) 不同关节负载差别大时, 用逗号给每关节单独增益: --torque-scale 1.0,1.2,1.2,1.0,1.0,1.0
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="gravity + friction compensation (float / teach mode)")
    parser.add_argument("--cfg", default="robot.cfg",
                        help="robot.cfg 路径 (默认: fafu_robot_python/robot.cfg)")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="夹爪 motor id (默认 7); 传 0 表示无夹爪")
    parser.add_argument("--urdf", default="",
                        help="URDF 路径; 空 = 自动定位")
    parser.add_argument("--motor-models",
                        default="M5036_02,M6036_02,M6036_02,M5036_02,M4438_30,M4438_30",
                        help="逗号分隔, 每个关节一个电机型号 (Nm->raw 换算). "
                             "默认= Fafu 实物配置 J1..J6 = "
                             "M5036_02,M6036_02,M6036_02,M5036_02,M4438_30,M4438_30. "
                             "传空字符串则力矩不缩放(机械臂会下垂).")
    parser.add_argument("--tau-limit", default="",
                        help="逗号分隔, 每关节力矩上限 Nm. 空 = [15,30,30,15,5,5].")
    parser.add_argument("--torque-scale", default="",
                        help="标定增益: 单个数(全关节) 或逗号分隔(每关节). "
                             "Nm->raw 系数不确定时, 从 1.0 起调大到机械臂刚好浮起. "
                             "建议先 --dry-run 看 raw 量级再上真机.")
    parser.add_argument("--rate-hz", type=float, default=200.0,
                        help="控制环频率 (默认 200Hz)")
    parser.add_argument("--duration", type=float, default=None,
                        help="运行秒数; 不传则一直跑到 Ctrl+C")
    parser.add_argument("--damping-kd", type=float, default=0.0,
                        help="固件速度阻尼 kd (>=0, 默认 0 = 与原版一致)")
    parser.add_argument("--no-gravity", action="store_true",
                        help="只做摩擦补偿 (不需要 pinocchio)")
    parser.add_argument("--no-friction", action="store_true",
                        help="只做重力补偿, 不加摩擦项")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算+打印力矩, 不下发 (强烈建议先跑这个)")
    parser.add_argument("--verbose", action="store_true",
                        help="每 ~0.5s 打印 q / v / tau")
    parser.add_argument("--k-soft", default="",
                        help="软件 PD 安全网刚度 K (Nm/rad), 逗号分隔每关节. "
                             "调 torque-scale 时强烈建议加上(防失控), 如 4,10,10,2,2,1")
    parser.add_argument("--b-soft", default="",
                        help="软件 PD 安全网阻尼 B (Nm·s/rad), 逗号分隔, 如 0.5,0.8,0.8,0.2,0.2,0.1")
    parser.add_argument("--tau-lpf", type=float, default=0.4,
                        help="力矩一阶低通系数 alpha (0~1, 越小越平滑, 1=不滤波). "
                             "高 torque-scale 抽搐时调小, 如 0.3")
    parser.add_argument("--tau-slew", type=float, default=40.0,
                        help="力矩变化率上限 Nm/s (防抽搐). 0 或负=不限. "
                             "抽搐严重时调小, 如 20")
    parser.add_argument("--i-soft", default="",
                        help="软件积分增益 Ki (Nm/(rad·s)), 逗号分隔或单值广播. "
                             "用它(而非加大 K)消除静摩擦死区/回弹下垂, 如 0.5")
    parser.add_argument("--i-clamp", type=float, default=3.0,
                        help="每关节积分力矩上限 Nm (防积分饱和). 默认 3. "
                             "太小会让模型/摩擦缺口大的关节积分提前饱和、持续缓慢下沉")
    parser.add_argument("--vel-abort", type=float, default=4.0,
                        help="失速保护: 任一关节 |速度| 超过此值(rev/s)立即停环复位. "
                             "0=关闭. 默认 4")
    parser.add_argument("--vel-lpf", type=float, default=0.3,
                        help="阻尼项 B 用的速度低通系数 (0~1, 越小越平滑). 默认 0.3")
    parser.add_argument("--no-hold", action="store_true",
                        help="关闭'拖到哪停哪'示教模式, 改回固定 q_des(推开会弹回起始姿态)")
    parser.add_argument("--move-vel", type=float, default=0.15,
                        help="判定'正在被拖动'的速度阈值 rev/s (拖动示教用). 默认 0.15. "
                             "无人碰时若缓慢漂移就调大")
    parser.add_argument("--home-on-exit", action="store_true",
                        help="退出时(正常/Ctrl+C)先停力矩再慢速回零位再退出")
    parser.add_argument("--home-speed", type=int, default=15,
                        help="--home-on-exit 回零位的速度百分比 (0~100]. 默认 15(慢)")
    parser.add_argument("--home-pause", type=float, default=0.0,
                        help="--home-on-exit 刹车后、开始回零前的停顿秒数. 默认 0(刹车后立即回零)")
    args = parser.parse_args()

    from fafu_robot_controller import FafuRobotController, FrictionParams  # noqa: E402

    has_gripper = bool(args.gripper_id)
    motor_models = (
        [s.strip() for s in args.motor_models.split(",") if s.strip()]
        if args.motor_models else None
    )
    tau_limit = (
        [float(s) for s in args.tau_limit.split(",") if s.strip()]
        if args.tau_limit else None
    )
    if args.torque_scale:
        ts = [float(s) for s in args.torque_scale.split(",") if s.strip()]
        torque_scale = ts[0] if len(ts) == 1 else ts
    else:
        torque_scale = None

    arm = FafuRobotController(
        cfg_path=args.cfg,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id if has_gripper else None,
    )

    try:
        # --- 仅摩擦补偿: 不需要 pinocchio / URDF ---
        if args.no_gravity:
            print("[TEST] gravity DISABLED — friction-only mode "
                  "(no pinocchio needed).")
            fp = FrictionParams.reference_6dof()
            arm._friction_params = fp  # 没有 setup_dynamics 时手动塞默认摩擦
            # motor_models / tau_limit 在这条路径上手动设置
            if motor_models is not None:
                arm._dyn_motor_models = motor_models
            arm._dyn_tau_limit = (
                np.asarray(tau_limit, dtype=float) if tau_limit is not None
                else np.array([15.0, 30.0, 30.0, 15.0, 5.0, 5.0])[:arm.num_joints]
            )
            if torque_scale is not None:
                arm.set_torque_scale(torque_scale)
            if not arm.is_enabled:
                arm.enable()
            import time as _t
            t0 = _t.monotonic()
            last = t0
            try:
                while args.duration is None or (_t.monotonic() - t0) < args.duration:
                    tick = _t.monotonic()
                    v = arm.get_joint_velocities()
                    tau = arm.get_friction_compensation(v)
                    tau = np.clip(tau, -arm._dyn_tau_limit, arm._dyn_tau_limit)
                    if not args.dry_run:
                        arm.apply_compensation_torque(tau, damping_kd=args.damping_kd)
                    if args.verbose and (tick - last) >= 0.5:
                        last = tick
                        raw = arm.tau_to_raw(tau * arm._dyn_torque_scale)
                        print(f"[fric] v={v.round(3).tolist()} "
                              f"tau(Nm)={tau.round(3).tolist()} "
                              f"-> raw={raw.tolist()}")
                    s = 1.0 / max(1.0, args.rate_hz) - (_t.monotonic() - tick)
                    if s > 0:
                        _t.sleep(s)
            except KeyboardInterrupt:
                print("\n[TEST] interrupted")
            finally:
                for mid in arm.joint_motor_ids:
                    try:
                        arm.driver.stop(mid)
                    except Exception:
                        pass
            return 0

        # --- 重力(+摩擦)补偿: 需要 pinocchio + URDF ---
        try:
            arm.setup_dynamics(
                urdf_path=args.urdf or None,
                motor_models=motor_models,
                tau_limit=tau_limit,
                torque_scale=torque_scale,
            )
        except RuntimeError as e:
            print(f"[TEST][FAIL] setup_dynamics: {e}")
            print("提示: 缺 pinocchio 可先用 --no-gravity 仅测摩擦补偿通路.")
            return 2

        k_soft = ([float(s) for s in args.k_soft.split(",") if s.strip()]
                  if args.k_soft else None)
        b_soft = ([float(s) for s in args.b_soft.split(",") if s.strip()]
                  if args.b_soft else None)
        i_soft = ([float(s) for s in args.i_soft.split(",") if s.strip()]
                  if args.i_soft else None)

        arm.start_gravity_compensation(
            friction=not args.no_friction,
            rate_hz=args.rate_hz,
            duration=args.duration,
            damping_kd=args.damping_kd,
            dry_run=args.dry_run,
            verbose=args.verbose,
            k_soft=k_soft,
            b_soft=b_soft,
            i_soft=i_soft,
            i_clamp=args.i_clamp,
            vel_abort_rps=args.vel_abort,
            vel_lpf_alpha=args.vel_lpf,
            hold_on_release=not args.no_hold,
            move_vel_thresh=args.move_vel,
            tau_lpf_alpha=args.tau_lpf,
            tau_slew_per_s=(args.tau_slew if args.tau_slew > 0 else None),
            home_on_exit=args.home_on_exit,
            home_speed=args.home_speed,
            home_brake_pause=args.home_pause,
        )
        return 0

    except Exception:
        traceback.print_exc()
        return 1
    finally:
        try:
            # Keep joints in brake mode on exit (matches gravity-comp teardown)
            # instead of "stop", so they freeze-in-place rather than go limp.
            arm.close_connection(joint_release="brake")
        except Exception as exc:
            print(f"[CLEANUP][WARN] close_connection failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
