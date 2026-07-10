# -*- coding: utf-8 -*-
"""
test_fafu_keyboard_cartesian.py
===============================

机械臂笛卡尔空间键盘控制 —— **1:1 照搬厂商** ``7_keyboard_cartesian_pos_control.py``,
仅替换一处底层控制通道 (见下方"唯一差别")。

功能 (与厂商一致):
    1. 启动时机械臂缓慢移动到安全位置 [0.0, 0.5, 0.6, 0.0, 0.0, 0.0] (rad)
    2. 键盘控制末端在笛卡尔空间移动 (只在目标改变时才重算 IK)
    3. 实时显示 目标位置 vs 当前位置
    4. 退出 (ESC / Ctrl+C) 返回零位

键盘控制 (与厂商一致):
    W/S: X 轴前后        A/D: Y 轴左右        Q/E: Z 轴上下
    1/2: 绕 X 轴旋转 +/-  3/4: 绕 Y 轴旋转 +/-  5/6: 绕 Z 轴旋转 +/-
    ESC: 退出程序
    (旋转增量右乘 = 欧拉角法, 每次基于当前已旋转坐标系, 与厂商默认一致)

★ 与厂商的唯一差别 (硬件所迫) ★
    厂商运动核心是 MIT 模式 ``pos_vel_tqe_kp_kd`` (mode 0x15) + 重力前馈 + kp/kd:
        robot_torque = np.array(robot.get_Gravity())
        robot.pos_vel_tqe_kp_kd(joint_pos, [0]*n, robot_torque, kp, kd)
    但**本批电机固件 mode 0x15 (MIT) 被静默忽略** (这也是之前重力补偿"没力气"的根因),
    所以本脚本把这一步换成 fafu 的**位置流 servo 通道** (set_many_pos_vel_tqe, mode 0x0A):
        arm.servo_j(joint_pos)
    位置环本身刚性保持, 不需要重力前馈; 自带固件看门狗 + 步长钳制 + 滞后监控, 更安全。
    其余 (安全位初始化 / 读 FK 当初始目标 / 仅变化时算 IK / inverse_kinematics
    multi_init=False 用上次解做种子 / 100Hz 循环 / 退出回零) 与厂商完全相同。

前置条件:
    1. pinocchio (FK/IK).  Windows 建议 conda-forge / WSL.
    2. pynput (键盘监听):  pip install pynput
    3. 内置 follower URDF (已随包) 或 --urdf 指定.

⚠️ 安全:
    - 机械臂会实时跟随键盘运动, 远离工作空间, 急停待命.
    - 增量很小 (位置 5mm / 旋转 ~1.7deg), 多按几次才明显.
    - 某些姿态下 IK 多解/无解, 关节可能突变; 建议在工作空间中心操作.

用法:
    conda activate <py310-env>
    cd fafu_robot_sdk/fafu_robot_python
    python tests/test_fafu_keyboard_cartesian.py
    # 跳过安全位初始化 / 自定义增量 / 速率
    python tests/test_fafu_keyboard_cartesian.py --no-safe --pos-delta 0.003 --rate 100
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


# ==========================================================================
#  全局变量 (照搬厂商: 主循环读取, 键盘监听线程写入)
# ==========================================================================
target_position = np.array([0.24, 0.0, 0.15])              # 初始目标位置 (m), 启动后被 FK 覆盖
target_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)  # 初始姿态, 同上
position_delta = 0.005                                     # 位置增量 5mm
rotation_delta = 0.03                                      # 旋转增量 ~1.7deg
running = True
position_changed = False                                   # 标记目标是否改变


# --------------------------------------------------------------------------
#  增量旋转矩阵 (与厂商脚本完全一致)
# --------------------------------------------------------------------------
def rotation_matrix_x(angle):
    """绕 X 轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle):
    """绕 Y 轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle):
    """绕 Z 轴旋转矩阵"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def on_press(key):
    """键盘按下事件处理 (照搬厂商)."""
    global target_position, target_rotation, running, position_changed

    if hasattr(key, "char") and key.char:
        ch = key.char.lower()
        # 字母键控制位置
        if ch == "w":
            target_position[0] += position_delta           # X 轴向前
            position_changed = True
        elif ch == "s":
            target_position[0] -= position_delta           # X 轴向后
            position_changed = True
        elif ch == "a":
            target_position[1] += position_delta           # Y 轴向左
            position_changed = True
        elif ch == "d":
            target_position[1] -= position_delta           # Y 轴向右
            position_changed = True
        elif ch == "q":
            target_position[2] += position_delta           # Z 轴向上
            position_changed = True
        elif ch == "e":
            target_position[2] -= position_delta           # Z 轴向下
            position_changed = True
        # 数字键控制旋转 (增量右乘, 欧拉角法: 每次基于当前已旋转坐标系)
        elif ch == "1":
            target_rotation = target_rotation @ rotation_matrix_x(rotation_delta)
            position_changed = True
        elif ch == "2":
            target_rotation = target_rotation @ rotation_matrix_x(-rotation_delta)
            position_changed = True
        elif ch == "3":
            target_rotation = target_rotation @ rotation_matrix_y(rotation_delta)
            position_changed = True
        elif ch == "4":
            target_rotation = target_rotation @ rotation_matrix_y(-rotation_delta)
            position_changed = True
        elif ch == "5":
            target_rotation = target_rotation @ rotation_matrix_z(rotation_delta)
            position_changed = True
        elif ch == "6":
            target_rotation = target_rotation @ rotation_matrix_z(-rotation_delta)
            position_changed = True


def on_release(key):
    """键盘释放事件处理 (照搬厂商): ESC 退出."""
    global running
    from pynput import keyboard
    if key == keyboard.Key.esc:
        print("\n检测到 ESC 键, 准备退出...")
        running = False
        return False  # 停止监听


def move_to_safe_position(arm, safe_joint_pos):
    """启动时缓慢移动到安全位置 (照搬厂商; 关节空间)."""
    print("\n" + "=" * 60)
    print("正在移动到安全位置...")
    print("=" * 60)
    print("移动中...")
    try:
        arm.move_j(safe_joint_pos, is_radians=True, speed=20, block=True)
    except Exception as e:  # noqa: BLE001
        print(f"✗ 移动到安全位置失败: {e}")
        return False
    print("✓ 已到达安全位置")
    time.sleep(0.5)
    return True


def main() -> int:
    global target_position, target_rotation, running, position_changed
    global position_delta, rotation_delta

    parser = argparse.ArgumentParser(
        description="keyboard cartesian teleop (照搬厂商 7_keyboard_cartesian_pos_control)")
    parser.add_argument("--cfg", default="robot.cfg")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="夹爪 motor id (默认 7); 传 0 表示无夹爪")
    parser.add_argument("--urdf", default="",
                        help="URDF 路径; 留空自动定位内置 follower URDF")
    parser.add_argument("--eef-frame", default="tool_link")
    parser.add_argument("--pos-delta", type=float, default=0.005,
                        help="每次按键的位置增量 m (默认 0.005 = 5mm, 同厂商)")
    parser.add_argument("--rot-delta", type=float, default=0.03,
                        help="每次按键的旋转增量 rad (默认 0.03 ~ 1.7deg, 同厂商)")
    parser.add_argument("--rate", type=float, default=100.0,
                        help="控制循环频率 Hz (默认 100, 同厂商 control_rate=0.01)")
    parser.add_argument("--safe-pos", default="0.0,0.5,0.6,0.0,0.0,0.0",
                        help="安全关节位 (rad, 逗号分隔, 同厂商 [0,0.5,0.6,0,0,0])")
    parser.add_argument("--no-safe", action="store_true",
                        help="跳过启动时移动到安全位 (厂商默认会移动)")
    args = parser.parse_args()

    position_delta = args.pos_delta
    rotation_delta = args.rot_delta

    try:
        from pynput import keyboard
    except Exception:
        print("[TEST][FAIL] 需要 pynput: pip install pynput")
        return 1

    from fafu_robot_controller import FafuRobotController, ServoOpts

    print("=" * 60)
    print("机械臂笛卡尔空间键盘控制程序 (fafu, 照搬厂商)")
    print("=" * 60)

    print("\n初始化机械臂...")
    has_gripper = args.gripper_id > 0
    arm = FafuRobotController(
        cfg_path=args.cfg,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id if has_gripper else 7,
    )

    listener = None
    servo_started = False
    try:
        arm.setup_dynamics(urdf_path=args.urdf or None, eef_frame=args.eef_frame)
        arm.enable()

        # 1) 移动到安全位置 (厂商默认行为)
        safe_joint_pos = [float(x) for x in args.safe_pos.split(",") if x.strip()]
        if len(safe_joint_pos) != arm.num_joints:
            print(f"[TEST][FAIL] --safe-pos 需要 {arm.num_joints} 个值, "
                  f"收到 {len(safe_joint_pos)}")
            return 1
        if not args.no_safe:
            if not move_to_safe_position(arm, safe_joint_pos):
                print("初始化失败, 退出程序")
                return 1

        # 2) 读当前位姿作为初始目标 (照搬厂商: forward_kinematics()['position'/'rotation'])
        current_fk = arm.forward_kinematics()
        target_position = np.array(current_fk["position"], dtype=float)
        target_rotation = np.array(current_fk["rotation"], dtype=float)

        print(f"\n初始位置: [{target_position[0]:.3f}, "
              f"{target_position[1]:.3f}, {target_position[2]:.3f}] m")
        print("\n" + "=" * 60)
        print("键盘控制说明:")
        print("  W/S: X 轴前后    A/D: Y 轴左右    Q/E: Z 轴上下")
        print("  1/2: 绕 X 旋转   3/4: 绕 Y 旋转   5/6: 绕 Z 旋转")
        print("  ESC: 退出程序")
        print("=" * 60)
        print("\n开始控制, 请小心操作!\n")

        # 3) 启动键盘监听
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # 4) 控制循环 (照搬厂商: 仅 position_changed 时算 IK)
        #    唯一差别: 厂商用 MIT pos_vel_tqe_kp_kd, 这里用位置流 servo_j (见模块 docstring)。
        last_valid_joint_pos = arm.get_joint_values()
        control_rate = 1.0 / max(1.0, args.rate)            # 厂商 0.01s = 100Hz

        # servo 会话替代厂商"先发一次当前位置稳定机械臂"的那步
        arm.servo_start(ServoOpts(rate_hz=args.rate, watchdog_ms=100,
                                  max_step_rad=0.10, max_lag_rad=0.3))
        servo_started = True
        arm.servo_j(last_valid_joint_pos)
        time.sleep(0.2)

        no_sol = False
        while running:
            t0 = time.monotonic()

            # 只在目标改变时才重新计算逆运动学 (照搬厂商)
            if position_changed:
                joint_pos = arm.inverse_kinematics(
                    target_position.tolist(),
                    target_rotation,
                    init_q=last_valid_joint_pos,
                    multi_init=False,
                )
                if joint_pos is not None:
                    arm.servo_j(joint_pos)                  # ← 厂商此处为 pos_vel_tqe_kp_kd(MIT)
                    last_valid_joint_pos = joint_pos
                    no_sol = False
                else:
                    if not no_sol:
                        print("\r逆运动学无解, 保持当前位置", end="")
                    no_sol = True
                position_changed = False                    # 重置标志 (照搬厂商)
            else:
                # 位置流通道有固件看门狗, 空闲也要持续喂帧保持当前位置
                # (厂商走 MIT 靠电机自身保持, 不需要重发; 这是位置流的必要差别)
                arm.servo_j(last_valid_joint_pos)

            # 获取当前实际位置并显示 (照搬厂商)
            current_pos = arm.forward_kinematics()["position"]
            print(f"\r目标位置: [{target_position[0]:.3f}, "
                  f"{target_position[1]:.3f}, {target_position[2]:.3f}] | "
                  f"当前位置: [{current_pos[0]:.3f}, "
                  f"{current_pos[1]:.3f}, {current_pos[2]:.3f}]", end="")

            dt = time.monotonic() - t0
            if dt < control_rate:
                time.sleep(control_rate - dt)

        print("\n[TEST] ESC, 退出中...")
        return 0

    except KeyboardInterrupt:
        print("\n\n程序被中断")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"\n[TEST][FAIL] {e}")
        traceback.print_exc()
        return 1
    finally:
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if servo_started:
            try:
                arm.servo_end("hold")
            except Exception:
                pass
        # 返回零位 (照搬厂商 finally)
        try:
            print("\n\n返回零位...")
            arm.go_home(speed=20, block=True)
            print("所有电机已停止")
        except Exception as e:  # noqa: BLE001
            print(f"[TEST] 返回零位失败: {e}")
        try:
            arm.close_connection(joint_release="brake")
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
