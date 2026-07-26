# -*- coding: utf-8 -*-
"""
test_fafu_motion_interactive.py
===============================

带菜单的交互式实验脚本 (基于 fafu_robot_controller 的高层封装).
不用反复改代码就能跑遍 FafuRobotController 的各种功能
(单关节 / 多关节 / 夹爪 / 软限位 / 状态监视 / 示教复现).

⚠️ 第一次跑前:
    - 机器臂上电
    - 桌面留出空间, 手放在键盘 Ctrl+C 上, 急停就在旁边
    - 推荐 --speed 不超过 15

用法:
    cd fafu_robot_sdk
    python tests/test_fafu_motion_interactive.py                      # 默认 7 号为夹爪
    python tests/test_fafu_motion_interactive.py --gripper-id 0       # 无夹爪 (7 关节)
    python tests/test_fafu_motion_interactive.py --teach-friction     # 示教启用摩擦补偿
    python tests/test_fafu_motion_interactive.py --speed 10           # 全局降速
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional

try:
    import msvcrt  # Windows 非阻塞按键 (录制时键控夹爪)
except ImportError:  # pragma: no cover - 非 Windows
    msvcrt = None


def _poll_key() -> Optional[str]:
    """非阻塞读取一个按键 (Windows msvcrt); 无按键返回 None.

    用于示教录制时一边徒手拖动手臂, 一边用键盘指令夹爪开合.
    """
    if msvcrt is None or not msvcrt.kbhit():
        return None
    try:
        ch = msvcrt.getwch()
    except Exception:
        return None
    # 方向键/功能键是双字节, 读掉第二个字节避免污染下次
    if ch in ("\x00", "\xe0"):
        try:
            msvcrt.getwch()
        except Exception:
            pass
        return None
    return ch

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


# ============================================================================
#  Helpers
# ============================================================================
def deg_str(angles_rad) -> str:
    return "[" + ", ".join(f"{math.degrees(v):+7.2f}" for v in angles_rad) + "]"


def parse_floats(s: str, sep: Optional[str] = None) -> List[float]:
    """'10 20 30' / '10,20,30' / '10, 20, 30' -> [10.0, 20.0, 30.0]."""
    s = s.replace(",", " ").strip()
    out: List[float] = []
    for tok in s.split():
        try:
            out.append(float(tok))
        except ValueError:
            raise ValueError(f"无法解析为数字: {tok!r}")
    return out


def prompt(text: str) -> str:
    try:
        return input(text).strip()
    except EOFError:
        return ""


def yes(s: str) -> bool:
    return s.lower() in ("y", "yes", "ok", "ok!", "")


_SETTLE_VEL_DEG_S = 2.0
_SETTLE_TIMEOUT_S = 4.0
_ORIGIN_REPLAY_SPEED_RATIO = 0.35   # 回原点段约为用户速度的 35%, 下限 5%


def _refresh_all_motor_states(arm) -> None:
    """主动查询刷新 cache, 避免 move_j 用到示教结束前的旧状态."""
    for mid in arm.all_motor_ids:
        arm._ht.read_motor_state(mid, 0.1)


def _wait_joints_settled(
    arm,
    *,
    timeout_s: float = _SETTLE_TIMEOUT_S,
    vel_thresh_deg_s: float = _SETTLE_VEL_DEG_S,
) -> bool:
    """轮询关节速度直到低于阈值 (示教结束 / 复现前的停稳等待)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _refresh_all_motor_states(arm)
        try:
            qd = arm.get_joint_velocities(prefer_cache=True)
        except Exception:
            time.sleep(0.05)
            continue
        peak = max(abs(math.degrees(v)) for v in qd)
        if peak <= vel_thresh_deg_s:
            time.sleep(0.15)
            _refresh_all_motor_states(arm)
            qd2 = arm.get_joint_velocities(prefer_cache=True)
            if max(abs(math.degrees(v)) for v in qd2) <= vel_thresh_deg_s:
                return True
        time.sleep(0.05)
    print(f"  [警告] {timeout_s:.0f}s 内未完全停稳, 继续后续步骤")
    return False


def _origin_replay_speed(speed: int) -> int:
    """复现时回原点用较低速度."""
    return max(5, min(speed, int(speed * _ORIGIN_REPLAY_SPEED_RATIO)))


_CFG_GRIPPER_KEY = "gripper_max_torque_raw"


def _read_cfg_int(cfg_path: Optional[str], key: str,
                  default: int) -> int:
    """从 robot.cfg 文本里读一个 `key = int` (底层 RobotConfig 忽略未知键,
    故这类交互脚本自定义参数由本函数自行解析). 读不到返回 default."""
    if not cfg_path or not os.path.isfile(cfg_path):
        return default
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                for c in ("#", ";"):
                    p = line.find(c)
                    if p != -1:
                        line = line[:p]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip().lower() == key.lower():
                    return int(float(v.strip()))
    except Exception:
        pass
    return default


def _write_cfg_int(cfg_path: Optional[str], key: str, value: int) -> bool:
    """把 `key = value` 写回 robot.cfg: 若已有该键则原地替换, 否则追加一行.
    保留其余内容/注释. 成功返回 True."""
    if not cfg_path:
        return False
    try:
        lines: List[str] = []
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        replaced = False
        for i, line in enumerate(lines):
            # 只看非注释部分是否是目标键 (保留行尾注释无意义, 这里整行替换)
            code = line
            for c in ("#", ";"):
                p = code.find(c)
                if p != -1:
                    code = code[:p]
            if "=" in code and code.split("=", 1)[0].strip().lower() == key.lower():
                lines[i] = f"{key} = {value}\n"
                replaced = True
                break
        if not replaced:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f"{key} = {value}\n")
        with open(cfg_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print(f"  写回 robot.cfg 失败: {e}")
        return False


# ============================================================================
#  示教录制 / 复现 (仿照官方 5_record_trajectory.py / 5_replay_trajectory.py)
# ============================================================================
# 重力补偿(浮空)所需参数, 与官方 Follower.yaml / 5_record_trajectory.py 对齐,
# 并按 Fafu 实物电机型号/系数填写 (J4 单独补 1.3 抗下垂).
_TEACH_MOTOR_MODELS = ["M5036_02", "M6036_02", "M6036_02",
                       "M5036_02", "M4438_30", "M4438_30"]
_TEACH_TAU_LIMIT = [15.0, 30.0, 30.0, 15.0, 5.0, 5.0]
_TEACH_TORQUE_SCALE = [1.0, 1.0, 1.0, 1.3, 1.0, 1.0]
# 摩擦补偿 (照搬官方 5_record_trajectory.py 的 Fc / Fv / vel_threshold)
_TEACH_FC = [0.15, 0.12, 0.12, 0.12, 0.04, 0.04]
_TEACH_FV = [0.05, 0.05, 0.05, 0.03, 0.02, 0.02]
_TEACH_VEL_THRESHOLD = 0.02
_TEACH_RECORD_HZ = 100.0
_TEACH_DIR = os.path.join(HERE, "recordings")

# 回放 (MIT 模式) 每关节 kp/kd —— ★物理量★, 与官方 5_replay_trajectory.py 完全一致.
# move_MIT 内部按厂商 kp_float2int (radian_2pi) 公式转成 0x8093 帧的
# rkp/rkd int16: raw = int16( (kp / coeff) * 10 * 2π ). 所以这里直接照抄官方数值.
_REPLAY_KP = [30.0, 40.0, 55.0, 15.0, 7.0, 5.0]
_REPLAY_KD = [3.0, 4.0, 5.5, 1.5, 0.7, 0.5]

# 夹爪控制通道: "位置+速度+最大力矩" (set_pos_vel_tqe). effort = 最大力矩上限
# (原始 int16, LSB = coeff*0.01 Nm; 夹爪 M4438_30 coeff≈0.5256 => 1 raw≈0.0053 Nm).
# 该值只是"上限", 固件按需取用: 空载开合只用一点, 夹到物体时才顶到上限当夹持力.
# 官方 gripper_control 用 0.5Nm(≈95 raw)偏软; 这里默认 ~1.6Nm 保证可靠开合+适中夹持.
_TEACH_GRIPPER_EFFORT = 300
# 夹爪电机 M4438_30 力矩系数 (raw->Nm: raw * coeff * 0.01), 仅用于菜单里显示 Nm 参考.
_GRIPPER_COEFF = 0.5256
# MIT 回放跑飞保护: 用"相对速度"判据 (实测速度 vs 轨迹目标速度).
#   跑飞(不稳定): 实测速度 >> 轨迹要求的速度 -> 超出量大; 触发刹车.
#   正常快速跟踪: 实测 ≈ 目标速度 (示教时手挥得快, 腕关节 J5/J6 本就会快) -> 不误刹.
#   滞后(kp太弱): 实测 < 目标, 超出量为负 -> 不误刹.
# 判据: 任一关节 (|实测速度| - |目标速度|) 超过该值(rad/s) 立即刹车.
_REPLAY_VEL_EXCESS_RPS = 2.5
# 绝对硬上限(rad/s): 无论目标多快, 实测超过它一定刹车 (物理安全兜底).
_REPLAY_VEL_HARD_RPS = 6.0


class TrajectoryRecorder:
    """轨迹记录器 / 回放器.

    把示教轨迹按帧写成 ``.jsonl`` (每行一帧 关节位置/速度 + 夹爪位置/速度 + 时间戳),
    并支持按记录时的真实时间间隔回放.

    与官方的差别 (硬件所迫):
        官方回放走 MIT ``pos_vel_tqe_kp_kd`` (位置+速度+重力前馈+kp/kd);
        本批电机固件 MIT(0x15) 被静默忽略, 故回放改走 Fafu 位置流
        ``servo_j`` (0x0A), 见 :meth:`App._replay_trajectory_file`.
    """

    def __init__(self, filepath: Optional[str] = None):
        os.makedirs(_TEACH_DIR, exist_ok=True)
        if not filepath:
            filepath = time.strftime("trajectory_%Y%m%d_%H%M%S.jsonl")
        if not os.path.isabs(filepath):
            filepath = os.path.join(_TEACH_DIR, filepath)
        if not filepath.endswith(".jsonl"):
            filepath += ".jsonl"
        self.filepath = filepath
        self._fh = open(filepath, "w", encoding="utf-8")
        self._t0 = time.monotonic()
        self.count = 0

    def log(self, pos, vel, gripper_pos=None, gripper_vel=None) -> None:
        rec = {
            "t": time.monotonic() - self._t0,
            "pos": [float(x) for x in pos],
            "vel": [float(x) for x in vel],
            "gpos": None if gripper_pos is None else float(gripper_pos),
            "gvel": None if gripper_vel is None else float(gripper_vel),
        }
        self._fh.write(json.dumps(rec) + "\n")
        self.count += 1

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    @staticmethod
    def load(filepath: str) -> List[dict]:
        out: List[dict] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    @staticmethod
    def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
        """列向滑动平均 (edge padding), 照搬官方 recorder._moving_average."""
        if window <= 1:
            return values
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(values, [(pad, pad), (0, 0)], mode="edge")
        kernel = np.ones(window) / window
        return np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="valid"), 0, padded)

    @staticmethod
    def prepare_playback(samples: List[dict],
                         playback_dt: float = 0.01,
                         smooth_window: int = 7) -> List[dict]:
        """轨迹预处理 (照搬官方 recorder._prepare_playback_frames).

        MIT (kp/kd) 回放对喂入信号的平滑度很敏感: 原始录制帧带抖动 + 电机反馈
        速度有量化噪声, 直接喂 kp*(pos-q)+kd*(vel-qd) 环路会激起高惯量关节
        (如 J3) 震荡/发散. 官方的关键做法是回放前:
          1) 均匀重采样到 playback_dt (默认 100Hz);
          2) 位置做 smooth_window 点滑动平均;
          3) 速度用平滑后位置的 np.gradient 重新算 (丢弃录制的噪声速度).
        夹爪位置同样重采样+平滑, 速度重算.
        """
        if playback_dt <= 0:
            raise ValueError("playback_dt 必须大于 0")
        if len(samples) < 2:
            return samples

        t = np.array([s["t"] for s in samples], dtype=float)
        t = t - t[0]
        pos = np.array([s["pos"] for s in samples], dtype=float)

        # 去掉重复/倒退时间戳, 避免插值异常
        keep = np.concatenate(([True], np.diff(t) > 1e-6))
        t = t[keep]
        pos = pos[keep]
        kept = [s for s, k in zip(samples, keep) if k]
        if len(t) < 2:
            return kept

        new_t = np.arange(0.0, t[-1] + playback_dt * 0.5, playback_dt)
        new_pos = np.column_stack([
            np.interp(new_t, t, pos[:, i]) for i in range(pos.shape[1])
        ])
        new_pos = TrajectoryRecorder._moving_average(new_pos, smooth_window)
        new_vel = np.gradient(new_pos, new_t, axis=0)

        has_g = kept[0].get("gpos") is not None
        if has_g:
            gpos = np.array([float(s.get("gpos") or 0.0) for s in kept])
            gpos = gpos[keep] if gpos.shape[0] != len(kept) else gpos
            new_g = np.interp(new_t, t, gpos)
            new_g = TrajectoryRecorder._moving_average(
                new_g[:, None], smooth_window)[:, 0]
            new_gvel = np.gradient(new_g, new_t)

        out: List[dict] = []
        for i, ts in enumerate(new_t):
            item = {
                "t": float(ts),
                "pos": new_pos[i].tolist(),
                "vel": new_vel[i].tolist(),
            }
            if has_g:
                item["gpos"] = float(new_g[i])
                item["gvel"] = float(new_gvel[i])
            out.append(item)
        return out


# ============================================================================
#  Menu items
# ============================================================================
class App:
    def __init__(self, arm, default_speed: int,
                 teach_torque_scale: Optional[List[float]] = None,
                 teach_motor_models: Optional[List[str]] = None,
                 teach_friction: bool = True,
                 teach_fc: Optional[List[float]] = None,
                 teach_fv: Optional[List[float]] = None,
                 teach_vel_threshold: float = _TEACH_VEL_THRESHOLD):
        from fafu_robot_controller import FafuRobotController
        assert isinstance(arm, FafuRobotController)
        self.arm = arm
        self.default_speed = default_speed
        # [j]/[a] 单/多关节运动的控制模式 (记住上次选择):
        #   "movej"  -> move_j 固件位置模式, 固定用 linear (官方 jointsSyncArrival
        #               风格, 多次实测重复精度好); 默认.
        #   "mithold"-> MIT(servo_j) 逼近后持续发帧保持, Ctrl+C -> 刹车.
        #   "servoj" -> servo_j 流式逼近目标, 到位后结束会话 (不持续保持).
        self.move_mode = "movej"
        # 示教: 最近一次录制的轨迹文件路径 (jsonl), 供 [p] 复现默认使用
        self.taught_file: Optional[str] = None
        # 示教重力补偿标定参数 (可被 CLI / [t] 内现场覆盖, 以适配不同的臂)
        self.teach_motor_models = (list(teach_motor_models)
                                   if teach_motor_models
                                   else list(_TEACH_MOTOR_MODELS))
        self.teach_torque_scale = (list(teach_torque_scale)
                                   if teach_torque_scale
                                   else list(_TEACH_TORQUE_SCALE))
        self.teach_friction = bool(teach_friction)
        self.teach_fc = list(teach_fc) if teach_fc else list(_TEACH_FC)
        self.teach_fv = list(teach_fv) if teach_fv else list(_TEACH_FV)
        self.teach_vel_threshold = float(teach_vel_threshold)
        # 夹爪最大力矩 (位置+速度+最大力矩模式的 effort 上限, raw int16).
        # 从 robot.cfg 的 gripper_max_torque_raw 读取, [g] 子菜单可改并写回.
        self._cfg_path = getattr(self.arm, "_cfg_path", None)
        self.gripper_effort = _read_cfg_int(
            self._cfg_path, _CFG_GRIPPER_KEY, _TEACH_GRIPPER_EFFORT)
        # 自定义书签 name -> (joint_angles_rad, gripper_angle_rad_or_None)
        self.bookmarks: Dict[str, tuple] = {}

    # ----- 状态信息 -----

    def show_state(self):
        q = self.arm.get_joint_values()
        print(f"\n  joint motor ids: {self.arm.joint_motor_ids}")
        print(f"  joint angles   : {deg_str(q)} (deg)")
        try:
            qd = self.arm.get_joint_velocities()
            print(f"  joint vel      : {deg_str(qd)} (deg/s)")
        except Exception as e:
            print(f"  joint vel      : <读取失败: {e}>")

        if self.arm.has_gripper:
            try:
                gs = self.arm.get_gripper_state()
                ang_deg = math.degrees(self.arm._turns_to_rad(gs.position))
                print(f"  gripper M{self.arm.gripper_motor_id:<2}    : {ang_deg:+.2f} deg "
                      f"(mode=0x{gs.mode:02X}, fault=0x{gs.fault:02X})")
            except Exception as e:
                print(f"  gripper        : <读取失败: {e}>")

        print(f"  is_enabled     : {self.arm.is_enabled}")
        print(f"  stats          : {self.arm.get_status().to_string()}")

    def show_limits(self):
        print()
        for mid in self.arm.all_motor_ids:
            lim = self.arm.get_limit(mid)
            tag = " (gripper)" if (self.arm.has_gripper
                                   and mid == self.arm.gripper_motor_id) else ""
            if lim is None:
                print(f"  M{mid}{tag}: 未设软限位")
            else:
                lo, hi = lim
                print(f"  M{mid}{tag}: [{math.degrees(lo):+8.2f}, "
                      f"{math.degrees(hi):+8.2f}] deg")

    # ----- 软重启 -----

    def soft_reboot(self):
        """对所有关节电机执行固件级软重启 (motor_reset), 再切回位置模式.

        用途: MIT 持续保持结束后电机卡在 MIT 残留模式 (0x0B), 裸写模式寄存器
        切不回去 (本固件特性), 只有软重启能清掉; 或任何 enable/回零发不动、
        疑似模式锁死时一键复位. 软重启后电机先进入停止模式(三相悬空)会瞬时
        松一下, 随即 enable() 切回位置模式(0x0A) 持稳. (夹爪不动.)
        """
        if not yes(prompt("  软重启所有关节电机 (会瞬时松一下再切回位置模式)? [Y/n]: ")):
            print("  取消."); return
        print("  motor_reset 软重启中 ...")
        for mid in self.arm.joint_motor_ids:
            try:
                self.arm._ht.motor_reset(mid)
            except Exception as e:
                print(f"  motor {mid}: motor_reset 失败: {e}")
        time.sleep(0.6)
        try:
            self.arm.enable()
            print("  软重启完成, 已切回位置模式.")
        except Exception as e:
            print(f"  软重启后 enable 失败: {e}")

    # ----- 运动控制模式选择 (moveJ / MIT 持续保持) -----

    def _ask_move_mode(self) -> str:
        """询问本次运动用哪种控制模式, 记住上次选择 (回车沿用上次).

        - movej  : move_j 固件位置模式, 固定 linear 风格 (官方 jointsSyncArrival,
                   实测重复精度好); 带积分, 到位后固件位置环持稳. (默认)
        - mithold: MIT(servo_j, kp/kd + 重力前馈) 逼近后 **持续发帧保持**
                   (~100Hz 不停发), 靠不断刷新维持位置. 保持直到 Ctrl+C,
                   **暂停后进入刹车模式 (阻尼保持)**.
        - servoj : servo_j (~100Hz, use_mit=False / 0x8090) 流式逼近,
                   结束后 move_j(linear) 回锁持稳 (0x8090 停发会下垂, 不能只消看门狗).
        - servohold: servo_j (~100Hz, use_mit=False / 0x8090) 流式逼近, 到位后
                   **持续发位置帧保持** (不停刷新喂看门狗). Ctrl+C 结束 -> 清看门狗
                   留在 0x0A 保持最后一帧 (位置环带积分, 不残留 0x0B).
        """
        cur = self.move_mode
        default_map = {"movej": "1", "mithold": "2", "servoj": "3",
                       "servohold": "4"}
        default_tag = default_map.get(cur, "1")
        s = prompt(f"  控制模式 [1] moveJ(linear)  "
                   f"[2] MIT持续保持(Ctrl+C->刹车)  "
                   f"[3] ServoJ(到位结束)  "
                   f"[4] ServoJ持续保持(Ctrl+C结束) (默认 {default_tag}): ").strip()
        if s == "1":
            self.move_mode = "movej"
        elif s == "2":
            self.move_mode = "mithold"
        elif s == "3":
            self.move_mode = "servoj"
        elif s == "4":
            self.move_mode = "servohold"
        # 其它输入(含回车) -> 沿用上次
        tag = {"movej": "moveJ(linear)",
               "mithold": "MIT(持续保持->刹车)",
               "servoj": "ServoJ(到位结束)",
               "servohold": "ServoJ(持续保持->0x0A)"}[self.move_mode]
        print(f"  -> 使用 {tag}")
        return self.move_mode

    def _move_to(self, target_rad, speed: int, mode: str) -> None:
        """按指定模式移动到目标关节角 (target_rad: 弧度数组).

        - "mithold"-> MIT 逼近 + 到位后持续发帧保持 (Ctrl+C 结束 -> 刹车保持);
        - "servoj" -> servo_j 流式逼近, 到位后结束会话;
        - "servohold"-> servo_j 位置通道逼近 + 到位后持续发位置帧保持 (Ctrl+C 结束);
        - 其它("movej") -> move_j 固件位置模式, 固定 linear 风格.
        """
        if mode == "mithold":
            self._move_MIT_to(target_rad, speed)
        elif mode == "servoj":
            self._move_servo_j_to(target_rad, speed)
        elif mode == "servohold":
            self._move_servo_hold_to(target_rad, speed)
        else:
            self.arm.move_j(target_rad, is_radians=True, speed=speed,
                            block=True, style="linear")

    # ----- 单关节运动 -----

    def _move_MIT_to(self, target_rad, speed: int) -> bool:
        """MIT 持续保持: servo_j (MIT 通道, kp/kd + 重力前馈) 柔顺逼近目标,
        到位后 **持续以 ~100Hz 发 MIT 帧保持** (PD + 每拍重力前馈), 靠不断刷新
        维持位置 -> 不下垂. 保持期间每秒打印一次当前各关节角度. 一直保持到用户
        按 Ctrl+C; **暂停后直接进入刹车模式** (servo_end('brake')), 不做
        motor_reset + move_j 回锁, 从而避开软重启窗口的下垂+回弹.

        注意: MIT 会把电机留在 MIT 残留模式 (0x0B), 本固件裸写模式寄存器切不回
        位置模式, 后续 [h] 回零 / moveJ (0x8090) 会发不动. 继续运动前请先用
        [y] 软重启 (或 [r] 恢复) 把电机切回位置模式.
        """
        from fafu_robot_controller import ServoOpts

        self._ensure_dynamics()      # 有动力学 -> MIT 带重力前馈; 无则 kp/kd-only

        target = np.asarray(target_rad, dtype=float)
        q0 = np.asarray(self.arm.get_joint_values(), dtype=float)
        max_dq = float(np.max(np.abs(target - q0)))
        if max_dq < math.radians(0.1):
            print("  已在目标位姿附近, 无需移动")
            return True

        # speed% -> 逼近角速度 (100% ≈ 90°/s). 每 tick 步长 = vel/rate, 决定快慢.
        rate = 100.0
        vel_rad_s = max(0.05, (speed / 100.0) * math.radians(90.0))
        opts = ServoOpts(
            watchdog_ms=100, rate_hz=rate,
            max_vel=vel_rad_s * 1.5,
            max_step_rad=vel_rad_s / rate,          # 逼近速度的真正上限
            max_lag_rad=math.radians(30.0),
            is_radians=True,                        # use_mit 默认 True -> 走 MIT
        )
        dt = 1.0 / rate
        tol = math.radians(2.0)
        deadline = time.monotonic() + max(3.0, max_dq / vel_rad_s + 2.0)

        print(f"  servo_j(MIT) 逼近目标 (speed={speed}%, "
              f"~{math.degrees(vel_rad_s):.0f}°/s)")
        self.arm.servo_start(opts)
        settle = 0
        reached = False
        last_report = 0.0                           # 保持中: 每秒打印一次角度
        try:
            while True:
                self.arm.servo_j(target)
                cur = np.asarray(
                    self.arm.get_joint_values(prefer_cache=True), dtype=float)
                near = float(np.max(np.abs(cur - target))) <= tol
                settle = settle + 1 if near else 0
                # 连续接近几拍 = 到位; 逼近超时也算到位, 都转入持续保持
                if not reached and (settle >= 5 or time.monotonic() > deadline):
                    reached = True
                    head = "已到位" if settle >= 5 else "逼近超时, 转为"
                    print(f"  {head} MIT 持续发帧保持中 (PD + 重力前馈, ~100Hz). "
                          "按 Ctrl+C 结束 -> 进入刹车保持")
                    last_report = time.monotonic()
                if reached:
                    now = time.monotonic()
                    if now - last_report >= 1.0:
                        last_report = now
                        print(f"  [保持中] 当前 (deg): {deg_str(cur)}")
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\n  [结束保持]")
        finally:
            # 暂停即进入刹车 (阻尼保持): servo_end('brake') 先清看门狗再对每个
            # 关节发刹车(0x0F). 跳过 motor_reset + move_j 回锁, 避开软重启窗口的
            # 下垂+回弹. (回读 mode 可能仍是 0x0B, 但看门狗已关, 保持不受影响.)
            self.arm.servo_end("brake")

        print("  已结束保持, 进入刹车模式. 继续运动前请先 [y] 软重启 / [r] 恢复 / [h] 回零.")
        return True

    def _move_servo_j_to(self, target_rad, speed: int) -> bool:
        """ServoJ 到位结束: servo_j (~100Hz, use_mit=False / 0x8090) 流式逼近,
        清看门狗结束会话后, 再用 move_j(linear) 回锁目标持稳.

        为何不能只消看门狗就停:
          0x8090 (pos_vel_MAXtqe) 是**流式**通道且无积分; 停发帧后重关节
          (J2/J3) 在重力下会下垂. move_j(linear) 会持续发最终目标直到到位,
          固件位置环才能稳住 — 这和 [1] moveJ 持稳方式一致.
        """
        from fafu_robot_controller import ServoOpts

        target = np.asarray(target_rad, dtype=float)
        q0 = np.asarray(self.arm.get_joint_values(), dtype=float)
        max_dq = float(np.max(np.abs(target - q0)))
        if max_dq < math.radians(0.1):
            print("  已在目标位姿附近, 无需移动")
            return True

        rate = 100.0
        vel_rad_s = max(0.05, (speed / 100.0) * math.radians(90.0))
        opts = ServoOpts(
            watchdog_ms=100, rate_hz=rate,
            max_vel=vel_rad_s,
            max_step_rad=vel_rad_s / rate,
            max_lag_rad=math.radians(30.0),
            is_radians=True,
            use_mit=False,                          # 与 move_j 同通道 (0x8090)
            # 该交互恢复路径选择固定 max_vel. 默认 True 现在也会依据实测
            # 误差持续追赶, 只在 position deadband 内把速度上限归零.
            feedforward_vel=False,
        )
        dt = 1.0 / rate
        tol = math.radians(2.0)
        deadline = time.monotonic() + max(5.0, max_dq / vel_rad_s + 3.0)

        print(f"  servo_j(pos/0x8090) 逼近目标 (speed={speed}%, "
              f"~{math.degrees(vel_rad_s):.0f}°/s) -> 结束会话后 move_j 回锁")
        self.arm.servo_start(opts)
        settle = 0
        reached = False
        try:
            while True:
                self.arm.servo_j(target)
                # ★ 必须用缓存 (async_rx 维护): 若在这里做阻塞 read_motor_state,
                #   循环周期会超过 watchdog_ms, 固件看门狗在两帧之间刹车 ->
                #   剧烈抖动 + 重关节爬不动. 缓存读几乎零耗时, 循环保持 ~100Hz.
                cur = np.asarray(
                    self.arm.get_joint_values(prefer_cache=True), dtype=float)
                near = float(np.max(np.abs(cur - target))) <= tol
                settle = settle + 1 if near else 0
                if settle >= 5:
                    reached = True
                    # 到位后再流式钉住一小段, 减少立刻清看门狗时的瞬时下垂
                    hold_until = time.monotonic() + 0.3
                    while time.monotonic() < hold_until:
                        self.arm.servo_j(target)
                        time.sleep(dt)
                    break
                if time.monotonic() > deadline:
                    print("  逼近超时, 结束 ServoJ 会话并改用 move_j 回锁")
                    break
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\n  [中断 ServoJ]")
        finally:
            # 只清看门狗 + 结束会话; 真正持稳交给下面的 move_j
            self.arm.servo_end("hold")

        # ★ servo_end 的 set_timeout(mid,0) 是即发即忘, 重关节上可能残留
        #   ~100ms 看门狗; 若紧接着 move_j (发帧+阻塞读, 间隙偶尔>100ms) 就会
        #   被看门狗刹车 -> J4 下垂 / J2/J3 不动. 这里显式再清一遍所有关节看门狗
        #   并停顿, 让残留看门狗彻底失效后再回锁 (等效于隔几秒手动按 [1] moveJ).
        for mid in self.arm.joint_motor_ids:
            try:
                self.arm._ht.set_timeout(mid, 0)
            except Exception:
                pass
        time.sleep(0.5)

        print("  ServoJ 逼近结束, move_j(linear) 回锁目标持稳 ...")
        try:
            self.arm.move_j(target, is_radians=True, speed=speed,
                            block=True, style="linear")
        except Exception as e:
            print(f"  move_j 回锁失败: {e}")
            return False
        if not reached:
            print("  (ServoJ 段未完全到位, 已由 move_j 补到位)")
        return True

    def _move_servo_hold_to(self, target_rad, speed: int) -> bool:
        """ServoJ 持续保持: servo_j (~100Hz, use_mit=False / 0x8090 位置通道)
        流式逼近目标, 到位后 **持续发位置帧保持** (不停刷新喂看门狗 -> 不被刹车,
        重关节靠固件位置环顶住). 保持期间每秒打印一次当前角度. 一直保持到用户
        按 Ctrl+C; 结束时 servo_end('hold') 清看门狗并留在位置模式 (0x0A) 保持
        最后一帧 (位置环带积分, 不残留 0x0B, 后续 moveJ/回零可直接用).

        与 [2] MIT持续保持的区别: 走 move_j 同一条位置通道 (非 MIT), 结束后
        直接 0x0A 持稳, 不进刹车、不残留 0x0B, 无需 [y] 软重启.
        """
        from fafu_robot_controller import ServoOpts

        target = np.asarray(target_rad, dtype=float)
        q0 = np.asarray(self.arm.get_joint_values(), dtype=float)
        max_dq = float(np.max(np.abs(target - q0)))

        rate = 100.0
        vel_rad_s = max(0.05, (speed / 100.0) * math.radians(90.0))
        opts = ServoOpts(
            watchdog_ms=100, rate_hz=rate,
            max_vel=vel_rad_s,
            max_step_rad=vel_rad_s / rate,
            max_lag_rad=math.radians(30.0),
            is_radians=True,
            use_mit=False,                          # 与 move_j 同通道 (0x8090)
            # 固定 max_vel 便于这个交互路径预测到位时间; 默认 True 已保留
            # Position 正速度上限、实测误差追赶和 deadband.
            feedforward_vel=False,
        )
        dt = 1.0 / rate
        tol = math.radians(2.0)
        deadline = time.monotonic() + max(5.0, max_dq / vel_rad_s + 3.0)

        print(f"  servo_j(pos/0x8090) 逼近目标 (speed={speed}%, "
              f"~{math.degrees(vel_rad_s):.0f}°/s) -> 到位后持续发帧保持")
        self.arm.servo_start(opts)
        settle = 0
        reached = False
        last_report = 0.0
        try:
            while True:
                self.arm.servo_j(target)
                cur = np.asarray(
                    self.arm.get_joint_values(prefer_cache=True), dtype=float)
                near = float(np.max(np.abs(cur - target))) <= tol
                settle = settle + 1 if near else 0
                # 连续接近几拍 = 到位; 逼近超时也算到位, 都转入持续保持
                if not reached and (settle >= 5 or time.monotonic() > deadline):
                    reached = True
                    head = "已到位" if settle >= 5 else "逼近超时, 转为"
                    print(f"  {head} 持续发位置帧保持中 (0x8090, ~100Hz). "
                          "按 Ctrl+C 结束 -> 留在位置模式(0x0A)保持")
                    last_report = time.monotonic()
                if reached:
                    now = time.monotonic()
                    if now - last_report >= 1.0:
                        last_report = now
                        print(f"  [保持中] 当前 (deg): {deg_str(cur)}")
                time.sleep(dt)
        except KeyboardInterrupt:
            print("\n  [结束保持]")
        finally:
            # 位置通道结束: servo_end('hold') 清看门狗 + 留在 0x0A 保持最后一帧.
            # 与 MIT 不同, 这里不残留 0x0B, 后续 moveJ/回零可直接用.
            self.arm.servo_end("hold")

        print("  已结束保持, 停在位置模式(0x0A). 可直接 [h] 回零 / 继续运动.")
        return True

    def move_one_joint(self):
        n = self.arm.num_joints
        s = prompt(f"  动哪个关节 (0..{n - 1}, 默认 {n - 1}): ")
        try:
            idx = int(s) if s else (n - 1)
        except ValueError:
            print("  非法编号"); return
        if not (0 <= idx < n):
            print(f"  超出范围 [0, {n})"); return

        s = prompt("  目标角度 (度, 如 +30): ")
        try:
            target_deg = float(s)
        except ValueError:
            print("  非法数字"); return

        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed

        mode = self._ask_move_mode()

        q0 = self.arm.get_joint_values()
        target = q0.copy()
        target[idx] = math.radians(target_deg)
        print(f"\n  当前 (deg): {deg_str(q0)}")
        print(f"  目标 (deg): {deg_str(target)}")
        if not yes(prompt("  执行? [Y/n]: ")):
            print("  取消."); return
        try:
            self._move_to(target, speed, mode)
        except Exception as e:
            traceback.print_exc()
            print(f"  [失败] {e}"); return
        time.sleep(0.2)
        q1 = self.arm.get_joint_values()
        err = math.degrees(q1[idx] - target[idx])
        tag = {"mithold": "MIT-hold", "servoj": "ServoJ",
               "servohold": "ServoJ-hold",
               "movej": "moveJ:linear"}.get(mode, mode)
        print(f"  到位 (deg): {deg_str(q1)}  关节 {idx} 误差 {err:+.3f}° [{tag}]")

    # ----- 多关节运动 -----

    def move_all_joints(self):
        n = self.arm.num_joints
        print(f"  输入 {n} 个角度 (度, 空格或逗号分隔):")
        s = prompt(f"  > ")
        try:
            vals = parse_floats(s)
        except ValueError as e:
            print(f"  {e}"); return
        if len(vals) != n:
            print(f"  需要 {n} 个值, 给了 {len(vals)} 个"); return

        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed

        mode = self._ask_move_mode()

        target = np.radians(vals)
        q0 = self.arm.get_joint_values()
        print(f"\n  当前 (deg): {deg_str(q0)}")
        print(f"  目标 (deg): {deg_str(target)}")
        if not yes(prompt("  执行? [Y/n]: ")):
            print("  取消."); return
        try:
            self._move_to(target, speed, mode)
        except Exception as e:
            traceback.print_exc()
            print(f"  [失败] {e}"); return
        time.sleep(0.2)
        q1 = self.arm.get_joint_values()
        tag = {"mithold": "MIT-hold", "servoj": "ServoJ",
               "servohold": "ServoJ-hold",
               "movej": "moveJ:linear"}.get(mode, mode)
        print(f"  到位 (deg): {deg_str(q1)}  [{tag}]")
        print(f"  各关节误差 (deg): {deg_str(np.asarray(q1) - np.asarray(target))}")

    # ----- 关节空间路径 (move_jntspace_path) -----

    def move_jntspace_path_menu(self):
        """输入多组关节角度, 用 move_jntspace_path 按序经过, 停在最后一点并打印误差."""
        n = self.arm.num_joints
        print(f"\n  关节空间路径 (move_jntspace_path / TOPPRA + move_j)")
        print(f"  每行输入 {n} 个角度 (度, 空格或逗号分隔); 至少 1 组.")
        print("  空行结束输入.")

        waypoints_deg: List[List[float]] = []
        while True:
            s = prompt(f"  路径点 {len(waypoints_deg) + 1}> ")
            if not s:
                break
            try:
                vals = parse_floats(s)
            except ValueError as e:
                print(f"  {e}; 重新输入该点"); continue
            if len(vals) != n:
                print(f"  需要 {n} 个值, 给了 {len(vals)} 个; 重新输入该点")
                continue
            waypoints_deg.append(vals)

        if not waypoints_deg:
            print("  未输入任何路径点, 取消."); return

        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed

        print(f"\n  共 {len(waypoints_deg)} 个路径点:")
        for i, wp in enumerate(waypoints_deg, 1):
            print(f"    [{i}] {deg_str(np.radians(wp))}")
        q0 = self.arm.get_joint_values()
        print(f"  当前 (deg): {deg_str(q0)}")
        print(f"  终点 (deg): {deg_str(np.radians(waypoints_deg[-1]))}")
        if not yes(prompt("  执行? [Y/n]: ")):
            print("  取消."); return

        # 当前位姿接到路径前, 让 TOPPRA 从实际起点平滑起步;
        # start_frame_id=1 跳过插值后的首帧 (即当前位姿附近).
        path = np.vstack([
            np.asarray(q0, dtype=float),
            np.radians(np.asarray(waypoints_deg, dtype=float)),
        ])
        goal = path[-1].copy()

        try:
            self.arm.move_jntspace_path(
                path, is_radians=True, speed=speed, start_frame_id=1)
        except NotImplementedError as e:
            print(f"  [失败] TOPPRA/wrs 不可用: {e}")
            print("         请安装 wrs, 或改用 [a] 逐点 move_j.")
            return
        except Exception as e:
            traceback.print_exc()
            print(f"  [失败] {e}"); return

        # 流式 move_j(block=False) 结束后稍等到位再读反馈
        time.sleep(0.4)
        _refresh_all_motor_states(self.arm)
        q1 = self.arm.get_joint_values()
        err = np.asarray(q1, dtype=float) - goal
        print(f"  到位 (deg): {deg_str(q1)}  [move_jntspace_path]")
        print(f"  目标终点 (deg): {deg_str(goal)}")
        print(f"  各关节误差 (deg): {deg_str(err)}")
        print(f"  最大 |误差|: {math.degrees(float(np.max(np.abs(err)))):+.3f}°")

    def go_home(self):
        s = prompt(f"  speed (默认 {self.default_speed}): ")
        speed = int(s) if s else self.default_speed
        if not yes(prompt("  全部关节回零? [Y/n]: ")):
            print("  取消."); return
        try:
            self.arm.go_home(speed=speed, block=True)
        except Exception as e:
            traceback.print_exc()
            print(f"  [失败] {e}"); return
        print(f"  到位 (deg): {deg_str(self.arm.get_joint_values())}")

    # ----- 夹爪 -----

    def gripper_menu(self):
        if not self.arm.has_gripper:
            print("\n  没有配置夹爪 (启动时未传 --gripper-id)")
            return
        while True:
            try:
                gs = self.arm.get_gripper_state()
                cur_deg = math.degrees(self.arm._turns_to_rad(gs.position))
                print(f"\n  当前夹爪角度: {cur_deg:+.2f} deg")
            except Exception as e:
                print(f"\n  读取失败: {e}")
            eff_nm = self.gripper_effort * _GRIPPER_COEFF * 0.01
            print(f"  控制模式: 位置+速度+最大力矩  "
                  f"(effort={self.gripper_effort} raw ≈ {eff_nm:.2f} Nm)")
            print("    [o] open  (软限位上限, 阻塞)")
            print("    [c] close (软限位下限, 阻塞)")
            print("    [a] 自定义角度")
            print("    [t] 改最大力矩 effort (写回 robot.cfg)")
            print("    [b] 返回主菜单")
            cmd = prompt("  > ").lower()
            if cmd in ("b", "back", "q", "exit", ""):
                return
            elif cmd == "o":
                try:
                    self.arm.open_gripper(effort=self.gripper_effort)
                except Exception as e:
                    print(f"  [失败] {e}")
            elif cmd == "c":
                try:
                    self.arm.close_gripper(effort=self.gripper_effort)
                except Exception as e:
                    print(f"  [失败] {e}")
            elif cmd == "a":
                s = prompt("  目标角度 (度): ")
                try:
                    deg = float(s)
                except ValueError:
                    print("  非法数字"); continue
                try:
                    self.arm.gripper_control(angle=math.radians(deg),
                                             effort=self.gripper_effort)
                except Exception as e:
                    print(f"  [失败] {e}")
            elif cmd == "t":
                self._edit_gripper_effort()
            else:
                print(f"  未识别 {cmd!r}")

    def _edit_gripper_effort(self):
        """改夹爪最大力矩 effort (raw int16) 并写回 robot.cfg."""
        eff_nm = self.gripper_effort * _GRIPPER_COEFF * 0.01
        print(f"  当前 effort = {self.gripper_effort} raw ≈ {eff_nm:.2f} Nm "
              f"(1 raw ≈ {_GRIPPER_COEFF * 0.01:.4f} Nm)")
        print("  参考: 95≈0.5Nm(官方,软) / 300≈1.6Nm(默认) / 500≈2.6Nm / 800≈4.2Nm")
        s = prompt("  新 effort (raw int16, 回车不改): ").strip()
        if not s:
            return
        try:
            v = int(float(s))
        except ValueError:
            print("  非法数字"); return
        v = max(0, min(v, 32767))
        self.gripper_effort = v
        new_nm = v * _GRIPPER_COEFF * 0.01
        if _write_cfg_int(self._cfg_path, _CFG_GRIPPER_KEY, v):
            print(f"  已更新 effort = {v} raw ≈ {new_nm:.2f} Nm, 并写回 "
                  f"{os.path.basename(self._cfg_path or 'robot.cfg')}")
        else:
            print(f"  已更新 effort = {v} (本次会话生效, 但写回 cfg 失败)")

    # ----- 软限位 -----

    def limit_menu(self):
        while True:
            print("\n  软限位")
            self.show_limits()
            print("\n    [s] set   <id> <lo_deg> <hi_deg>")
            print("    [d] disable <id>")
            print("    [c] clear all")
            print("    [b] 返回")
            cmd = prompt("  > ").lower()
            if cmd in ("b", "back", "q", "exit", ""):
                return
            if cmd == "c":
                if yes(prompt("  确认清空全部? [Y/n]: ")):
                    self.arm.clear_limits()
                    print("  已清除")
                continue
            tokens = cmd.split()
            if len(tokens) >= 2 and tokens[0] in ("d", "disable"):
                try:
                    mid = int(tokens[1])
                    self.arm.disable_limit(mid)
                    print(f"  已禁用 M{mid}")
                except Exception as e:
                    print(f"  失败: {e}")
                continue
            if len(tokens) == 4 and tokens[0] in ("s", "set"):
                try:
                    mid = int(tokens[1])
                    lo = float(tokens[2])
                    hi = float(tokens[3])
                    self.arm.set_limit(mid, lo=lo, hi=hi, is_radians=False)
                    print(f"  已设 M{mid}: [{lo:+.2f}, {hi:+.2f}] deg")
                except Exception as e:
                    print(f"  失败: {e}")
                continue
            print(f"  未识别 {cmd!r}")

    # ----- 状态实时监视 (Ctrl+C 退出) -----

    def monitor(self, hz: float = 4.0):
        print("\n  [实时监视] Ctrl+C 退出")
        period = 1.0 / max(0.5, hz)
        try:
            while True:
                q = self.arm.get_joint_values()
                line = "  " + deg_str(q)
                if self.arm.has_gripper:
                    try:
                        gs = self.arm.get_gripper_state()
                        line += f"  | grip {math.degrees(self.arm._turns_to_rad(gs.position)):+6.1f}°"
                    except Exception:
                        pass
                print(line, flush=True)
                time.sleep(period)
        except KeyboardInterrupt:
            print("\n  [退出监视]")

    # ----- 示教 / 复现 (仿照官方 5_record_trajectory / 5_replay_trajectory) -----

    def _ensure_dynamics(self) -> bool:
        """示教浮空需要重力补偿; 确保 setup_dynamics 已就绪."""
        if self.arm.has_dynamics:
            return True
        if self.arm.num_joints != 6:
            print(f"  [失败] 内置重力补偿参数仅支持 6 关节, 当前 "
                  f"{self.arm.num_joints}.")
            if self.arm.num_joints == 7 and not self.arm.has_gripper:
                print("  >> 多半是启动时漏了 --gripper-id 7: 第 7 号夹爪被当成关节了.")
                print("     重新启动: python tests/test_fafu_motion_interactive.py "
                      "--gripper-id 7")
            else:
                print("  请先自行 setup_dynamics (6-DoF URDF).")
            return False
        try:
            self.arm.setup_dynamics(
                motor_models=list(self.teach_motor_models),
                tau_limit=list(_TEACH_TAU_LIMIT),
                torque_scale=list(self.teach_torque_scale),
                friction=self._make_friction(),
            )
            return True
        except Exception as e:
            print(f"  [失败] setup_dynamics: {e}")
            print("  (重力补偿示教需要 pinocchio: "
                  "conda install -c conda-forge pinocchio)")
            return False

    def _make_friction(self):
        """按当前 teach_fc / teach_fv / vel_threshold 构造 FrictionParams."""
        from fafu_robot_controller import FrictionParams
        return FrictionParams(
            fc=np.array(self.teach_fc, dtype=float),
            fv=np.array(self.teach_fv, dtype=float),
            vel_threshold=self.teach_vel_threshold)

    def _apply_friction(self) -> None:
        """把当前摩擦参数热更新到控制器 (无需重载 URDF)."""
        try:
            self.arm._friction_params = self._make_friction()
        except Exception as e:
            print(f"  摩擦参数更新失败: {e}")

    def _latest_recording(self) -> Optional[str]:
        try:
            files = [os.path.join(_TEACH_DIR, f)
                     for f in os.listdir(_TEACH_DIR) if f.endswith(".jsonl")]
        except FileNotFoundError:
            return None
        return max(files, key=os.path.getmtime) if files else None

    @staticmethod
    def _resolve_traj_path(name: str) -> Optional[str]:
        """把用户输入的文件名解析成实际路径: 支持省略 .jsonl / 录制目录相对 / 绝对路径."""
        names = [name] if name.endswith(".jsonl") else [name, name + ".jsonl"]
        cands: List[str] = []
        for nm in names:
            if os.path.isabs(nm):
                cands.append(nm)
            else:
                cands.append(os.path.join(_TEACH_DIR, nm))   # 优先录制目录
                cands.append(nm)                              # 再试当前工作目录
        for c in cands:
            if os.path.isfile(c):
                return c
        return None

    def teach_record(self):
        """重力补偿浮空 + 实时连续记录 (仿照官方 5_record_trajectory.py).

        - 进入纯重力(+摩擦)补偿, 机械臂失重, 可徒手拖动;
        - 以 ~100Hz 连续记录 关节位置/速度 (+夹爪指令) 到 jsonl 文件;
        - 夹爪保持位置使能, 按【空格】切换开/合, 记录"指令角度"
          (自由模式下夹爪位置反馈会被固件冻结, 徒手示教记不到变化);
        - Ctrl+C 结束记录并保存.
        """
        print("\n  示教录制 (重力补偿浮空 + 连续记录, 仿官方 5_record):")
        print("    1) 机械臂进入重力补偿后会失重 -> 先扶稳!")
        print("    2) 徒手拖动机械臂演示轨迹, 程序按 ~100Hz 连续记录")
        print("    3) 夹爪按【空格】切换开/合 (记录指令角度, 回放可还原)")
        print("    4) Ctrl+C 结束记录并保存到 jsonl 文件")
        if not self._ensure_dynamics():
            return

        # 现场标定 (适配不同的臂): 每关节 torque_scale + 摩擦开关.
        #   J2 自己往前转 -> 把该关节 scale 调小 (过补偿 / 摩擦库仑项自激);
        #   J3 托不住没力 -> 把该关节 scale 调大 (欠补偿).
        print(f"\n  当前 torque_scale = {self.teach_torque_scale}")
        s = prompt("  调每关节 torque_scale? (回车不改; 或输 6 个逗号值 "
                   "如 1,0.8,1.3,1.3,1,1): ")
        if s.strip():
            try:
                ts = parse_floats(s)
                if len(ts) != self.arm.num_joints:
                    print(f"  需要 {self.arm.num_joints} 个值, 忽略本次输入")
                else:
                    self.teach_torque_scale = ts
                    self.arm.set_torque_scale(ts)
                    print(f"  已更新 torque_scale = {ts}")
            except ValueError as e:
                print(f"  解析失败, 沿用旧值: {e}")
        else:
            # 沿用(可能被上次修改过的) self.teach_torque_scale
            try:
                self.arm.set_torque_scale(list(self.teach_torque_scale))
            except Exception:
                pass

        fs = prompt(f"  启用摩擦补偿? (当前 {'开' if self.teach_friction else '关'}) "
                    "[Y/n, n=纯重力便于排查 J2 自激]: ")
        if fs.strip():
            self.teach_friction = yes(fs)
        print(f"  摩擦补偿: {'开' if self.teach_friction else '关 (纯重力)'}")

        # 每关节库仑 Fc 现场微调: 某关节自激往前转 -> 把它的 Fc 调小,
        # 既保留其它关节的摩擦补偿 (拖起来轻), 又压住自激.
        if self.teach_friction:
            print(f"  当前每关节 Fc = {self.teach_fc}")
            cs = prompt("  调每关节 Fc? (回车不改; 或输 6 个逗号值, "
                        "J2 自激就把第 2 个调小如 0.04): ")
            if cs.strip():
                try:
                    fc = parse_floats(cs)
                    if len(fc) != self.arm.num_joints:
                        print(f"  需要 {self.arm.num_joints} 个值, 忽略本次输入")
                    else:
                        self.teach_fc = fc
                        self._apply_friction()
                        print(f"  已更新 Fc = {fc}")
                except ValueError as e:
                    print(f"  解析失败, 沿用旧值: {e}")
            else:
                self._apply_friction()

        fname = prompt("\n  文件名 (回车=自动 trajectory_时间戳.jsonl): ")
        if not yes(prompt("  机械臂即将失重, 已扶稳? [Y/n]: ")):
            return

        try:
            self.arm.enable()
        except Exception as e:
            print(f"  enable 失败: {e}"); return

        # 夹爪不徒手示教: 自由模式 (0x00) 下固件会冻结位置反馈, 手掰也记不到
        # 变化 (gpos 恒值, gvel 只有量化噪声). 改为夹爪保持"位置使能", 录制时
        # 按【空格】在 开/合 之间切换, 记录的是"指令角度"; 回放再用 gripper_control
        # 还原. open=上软限位 (开口最大), close=下软限位 (闭合最紧).
        grip_open_rad = grip_close_rad = grip_cmd_rad = None
        grip_is_open = False
        if self.arm.has_gripper:
            lo_t, hi_t = self.arm._gripper_limit_turns()
            grip_open_rad = (self.arm._turns_to_rad(hi_t)
                             if hi_t is not None else math.radians(90.0))
            grip_close_rad = (self.arm._turns_to_rad(lo_t)
                              if lo_t is not None else math.radians(-90.0))
            # 初始指令 = 当前实际位置 (第一次按空格前忠实记录当前开口)
            try:
                gs = self.arm.get_gripper_state()
                grip_cmd_rad = self.arm._turns_to_rad(gs.position)
            except Exception:
                grip_cmd_rad = grip_close_rad
            # 离哪个限位近就当作当前状态, 第一次空格切到另一边
            grip_is_open = (abs(grip_cmd_rad - grip_open_rad)
                            < abs(grip_cmd_rad - grip_close_rad))
            print(f"  夹爪: 位置+速度+最大力矩(effort={self.gripper_effort}); "
                  f"按【空格】开/合 "
                  f"(开 {math.degrees(grip_open_rad):+.0f}° / "
                  f"合 {math.degrees(grip_close_rad):+.0f}°)")

        rec = TrajectoryRecorder(fname or None)
        period = 1.0 / _TEACH_RECORD_HZ
        print(f"\n  开始记录 -> {rec.filepath}")
        print("  (徒手拖动机械臂; 【空格】切换夹爪开/合; Ctrl+C 结束)\n")
        try:
            while True:
                t0 = time.monotonic()
                # 纯重力(+摩擦)前馈, 浮空可拖动 (= 官方 zero kp/kd + gravity + friction)
                self.arm.gravity_compensation_step(friction=self.teach_friction)

                # 键控夹爪: 空格切换开/合, 记录指令角度 (而非被冻结的反馈)
                if self.arm.has_gripper and _poll_key() == " ":
                    grip_is_open = not grip_is_open
                    grip_cmd_rad = grip_open_rad if grip_is_open else grip_close_rad
                    try:
                        self.arm.gripper_control(
                            grip_cmd_rad, effort=self.gripper_effort,
                            block=False)
                    except Exception:
                        pass
                    # 另起一行打印切换事件 (\r 状态行会盖掉, 故先换行)
                    print(f"\n  [空格] 夹爪 -> {'开' if grip_is_open else '合'} "
                          f"({math.degrees(grip_cmd_rad):+.0f}°)")

                q = self.arm.get_joint_values(prefer_cache=True)
                qd = self.arm.get_joint_velocities(prefer_cache=True)
                rec.log(q, qd, grip_cmd_rad,
                        0.0 if grip_cmd_rad is not None else None)
                gtag = (("开" if grip_is_open else "合")
                        if self.arm.has_gripper else "-")
                print(f"\r  已记录 {rec.count:5d} 帧 | 爪[{gtag}] | {deg_str(q)}   ",
                      end="", flush=True)
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\n  结束记录...")
        finally:
            rec.close()
            self.taught_file = rec.filepath
            print(f"  已保存 {rec.count} 帧 -> {rec.filepath}")
            try:
                self.arm.brake()
                print("  已刹车保持. ([p] 复现 / [h] 回零 / [r] 恢复 enable)")
            except Exception as e:
                print(f"  brake 失败: {e}")

    def teach_replay(self):
        """回放 jsonl 轨迹 (仿照官方 2.0 5_replay_trajectory.py).

        两种通道 (对齐官方 recorder.play 的 mode):
          [1] MIT (默认): 一拖多 0x8093, 每帧下发 位置+速度+重力前馈+kp/kd,
              硬件阻抗跟踪, 手感更柔 (= 官方 mode="mit").
          [2] servo_j: 纯位置流 (0x0A), 位置精度更高 (= 官方 mode="posvel").
        """
        default_file = self.taught_file or self._latest_recording()
        hint = (f" (默认 {os.path.basename(default_file)})"
                if default_file else "")
        fname = prompt(f"\n  回放文件名{hint}: ") or default_file
        if not fname:
            print("  未指定文件, 先用 [t] 录制"); return
        resolved = self._resolve_traj_path(fname)
        if resolved is None:
            print(f"  文件不存在: {fname} (已在 {_TEACH_DIR} 下找过, "
                  "自动补 .jsonl 也没有)"); return
        fname = resolved
        try:
            samples = TrajectoryRecorder.load(fname)
        except Exception as e:
            print(f"  读取失败: {e}"); return
        if len(samples) < 2:
            print("  轨迹帧数太少 (<2)"); return

        m = prompt("  回放通道 [1]MIT(位置+速度+重力+kp/kd, 默认) "
                   "[2]servo_j(纯位置): ")
        mode = "servo" if m.strip() == "2" else "mit"

        kp_scale: "float | list" = 1.0
        if mode == "mit":
            if not self._ensure_dynamics():
                print("  MIT 回放需要重力前馈 (setup_dynamics); 改用 servo_j")
                mode = "servo"
            else:
                print(f"     官方 kp={_REPLAY_KP} kd={_REPLAY_KD} (物理量, "
                      "内部按厂商公式转 raw); scale=1.0 即完全照抄官方")
                print("     跑飞保护(相对速度): 实测比目标快 > "
                      f"{_REPLAY_VEL_EXCESS_RPS:.1f} rad/s 或超硬限 "
                      f"{_REPLAY_VEL_HARD_RPS:.1f} rad/s 自动刹车")
                ks = prompt("  kp/kd 强度缩放 (默认 1.0=官方; 单值=全关节; 也可输 6 个"
                            "逗号值单独调某关节, 如 1,1,0.7,1,1,1; "
                            "0=纯重力浮空不跟踪): ")
                if ks.strip():
                    try:
                        vals = parse_floats(ks)
                        if len(vals) == 1:
                            kp_scale = max(0.0, min(vals[0], 3.0))
                        elif len(vals) == self.arm.num_joints:
                            kp_scale = [max(0.0, min(v, 3.0)) for v in vals]
                        else:
                            print(f"  需要 1 或 {self.arm.num_joints} 个值, "
                                  "用默认 1.0"); kp_scale = 1.0
                    except ValueError:
                        kp_scale = 1.0

        s = prompt("  回放速度倍率 (默认 1.0, 范围 0.5~2.0): ")
        try:
            rate_scale = float(s) if s else 1.0
        except ValueError:
            rate_scale = 1.0
        rate_scale = max(0.5, min(rate_scale, 2.0))

        dur = samples[-1]["t"]
        print(f"\n  {os.path.basename(fname)}: {len(samples)} 帧, "
              f"时长 {dur:.1f}s (x{rate_scale:.1f} => {dur / rate_scale:.1f}s)")
        print(f"  通道={mode}"
              + (f" (kp_scale={kp_scale})" if mode == "mit" else "")
              + "; 将先慢速移动到轨迹起点, 再按真实时间间隔回放")
        if not yes(prompt("  开始? [Y/n]: ")):
            return
        try:
            if mode == "mit":
                self._replay_trajectory_mit(samples, rate_scale, kp_scale)
            else:
                self._replay_trajectory_file(samples, rate_scale)
        except KeyboardInterrupt:
            print("\n  [中断]")
            try:
                self.arm.servo_end("brake")
            except Exception:
                pass
            try:
                self.arm.brake()
            except Exception:
                pass
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}")

    def _replay_trajectory_mit(self, samples: List[dict],
                               rate_scale: float = 1.0,
                               kp_scale: "float | list" = 0.5) -> None:
        """MIT 回放: 一拖多 0x8093, 每帧 位置+速度+重力前馈+kp/kd (官方 mode='mit').

        - 先慢速 move_j 到起点, 再按记录的真实时间间隔逐帧 move_MIT;
        - 重力前馈按每帧目标位姿现算 (= 官方 get_Gravity(f['pos']));
        - kp/kd = _REPLAY_KP/_REPLAY_KD * kp_scale (物理量, 与官方一致);
          move_MIT 内部按厂商公式转 raw. kp_scale 可标量或每关节列表;
        - 跑飞保护(实测速度): 任一关节 |实测速度| > 阈值 立即刹车中止;
        - 夹爪仍单独走 gripper_control (节流), 不进 MIT 帧 (一帧最多 6 电机);
        - 靠固件看门狗兜底: 循环异常/中断 -> watchdog_ms 内自动停.
        """
        n = self.arm.num_joints
        if np.isscalar(kp_scale):
            ksv = [float(kp_scale)] * n
        else:
            ksv = [float(x) for x in kp_scale][:n]
            ksv += [ksv[-1] if ksv else 1.0] * (n - len(ksv))
        kp = [k * ksv[j] for j, k in enumerate(_REPLAY_KP[:n])]
        kd = [k * ksv[j] for j, k in enumerate(_REPLAY_KD[:n])]
        print(f"  kp={kp} kd={kd} (物理量, 官方=x1.0); "
              f"跑飞保护: 实测比目标快>{_REPLAY_VEL_EXCESS_RPS:.1f} 或 "
              f">{_REPLAY_VEL_HARD_RPS:.1f} rad/s")

        # 官方关键做法: MIT 回放前对轨迹重采样(100Hz)+滑动平均+梯度重算速度.
        # 干净的 pos/vel 大幅降低 kp/kd 环路激起 J3 等高惯量关节震荡的风险.
        raw_n = len(samples)
        samples = TrajectoryRecorder.prepare_playback(
            samples, playback_dt=0.01, smooth_window=7)
        print(f"  轨迹预处理: {raw_n} 帧 -> {len(samples)} 帧 "
              f"(重采样100Hz + 7点平滑 + 梯度重算速度)")

        start_q = samples[0]["pos"]
        self.arm.enable()
        _refresh_all_motor_states(self.arm)
        _wait_joints_settled(self.arm, timeout_s=1.5)
        o_spd = _origin_replay_speed(self.default_speed)
        print(f"  预备: 移动到轨迹起点 (speed={o_spd}%) {deg_str(start_q)}")
        self.arm.move_j(start_q, speed=o_spd, block=True)
        time.sleep(0.3)
        if self.arm.has_gripper and samples[0].get("gpos") is not None:
            try:
                self.arm.gripper_control(samples[0]["gpos"],
                                         effort=self.gripper_effort,
                                         block=True)
            except Exception:
                pass

        # 武装固件看门狗: 回放期间上位机崩溃 / Ctrl+C 后 100ms 内自动刹车.
        # (group-MIT 会话 API 已移除; move_MIT 直接循环下发即可。)
        _WD_MS = 100
        for _mid in self.arm._joint_motor_ids:
            try:
                self.arm._ht.set_timeout(_mid, _WD_MS)
            except Exception:
                pass

        print("  MIT 回放中... (Ctrl+C 中断)")
        last_grip: Optional[float] = None
        t_play0 = time.monotonic()
        try:
            for i, smp in enumerate(samples):
                target_t = smp["t"] / rate_scale
                q_t = smp["pos"]
                qd_t = smp.get("vel", [0.0] * n)
                # 重力(+可选摩擦)前馈按目标位姿现算, 再交给 move_MIT
                # (内部乘 torque_scale, Nm->raw, 与 kp/kd 一同打包成一帧 0x8093).
                tau_ff = self.arm.compute_compensation_torque(
                    q_t, qd_t, friction=self.teach_friction)

                def _send():
                    self.arm.move_MIT(
                        q_t, qd_t, tau_ff, kp=kp, kd=kd,
                        is_radians=True, apply_torque_scale=True, timeout=0.0)

                # 按真实时间对齐; 等待期间持续喂帧, 避免触发看门狗
                while True:
                    ahead = target_t - (time.monotonic() - t_play0)
                    if ahead <= 0:
                        break
                    _send()
                    time.sleep(min(ahead, 0.01))
                _send()

                # 跑飞保护(相对速度): 比较实测速度 vs 轨迹目标速度.
                #   跑飞 -> 实测远超目标 (excess 大); 正常快速跟踪 -> 实测≈目标
                #   (示教手挥快时腕关节本就会快, 不该误刹); 滞后 -> 实测<目标.
                qd_meas = self.arm.get_joint_velocities(prefer_cache=True)
                jbad, worst = -1, 0.0
                for j in range(n):
                    excess = abs(qd_meas[j]) - abs(qd_t[j])
                    if excess > worst:
                        worst, jbad = excess, j
                hard = max(range(n), key=lambda j: abs(qd_meas[j]))
                if worst > _REPLAY_VEL_EXCESS_RPS or \
                        abs(qd_meas[hard]) > _REPLAY_VEL_HARD_RPS:
                    jb = jbad if worst > _REPLAY_VEL_EXCESS_RPS else hard
                    print(f"\n  [跑飞保护] J{jb + 1} 实测 {qd_meas[jb]:+.2f} rad/s "
                          f"(目标 {qd_t[jb]:+.2f}, 超出 {worst:+.2f}), 立即刹车. "
                          f"(把该关节 kp_scale 调小)")
                    break

                g = smp.get("gpos")
                if (self.arm.has_gripper and g is not None
                        and (last_grip is None
                             or abs(g - last_grip) > math.radians(2.0))):
                    try:
                        self.arm.gripper_control(
                            g, effort=self.gripper_effort, block=False)
                    except Exception:
                        pass
                    last_grip = g
                if i % 20 == 0:
                    cur = self.arm.get_joint_values(prefer_cache=True)
                    print(f"\r  [{i + 1}/{len(samples)}] {deg_str(cur)}   ",
                          end="", flush=True)
        finally:
            # 清看门狗 + 刹车保持 (-> DISABLED). group-MIT 残留模式 (0x0B) 下
            # 刹车 (0x0F) 可直接切到, 无需 motor_reset.
            for _mid in self.arm._joint_motor_ids:
                try:
                    self.arm._ht.set_timeout(_mid, 0)
                except Exception:
                    pass
            try:
                self.arm.brake()
                print("\n  回放结束, 已刹车保持.")
            except Exception as e:
                print(f"\n  刹车失败: {e}")
        print("  回放完成")

    def _replay_trajectory_file(self, samples: List[dict],
                                rate_scale: float = 1.0) -> None:
        """先慢速到起点, 再用 servo_j 按真实时间间隔逐帧回放 (+夹爪节流)."""
        from fafu_robot_controller import ServoOpts

        start_q = samples[0]["pos"]
        self.arm.enable()
        _refresh_all_motor_states(self.arm)
        _wait_joints_settled(self.arm, timeout_s=1.5)
        o_spd = _origin_replay_speed(self.default_speed)
        print(f"  预备: 移动到轨迹起点 (speed={o_spd}%) {deg_str(start_q)}")
        self.arm.move_j(start_q, speed=o_spd, block=True)
        time.sleep(0.3)
        if self.arm.has_gripper and samples[0].get("gpos") is not None:
            try:
                self.arm.gripper_control(samples[0]["gpos"],
                                         effort=self.gripper_effort,
                                         block=True)
            except Exception:
                pass

        # 这是回放菜单 [2] "servo_j 纯位置" 回退路径, 显式走位置模式 (0x8090),
        # 不受 ServoOpts.use_mit 默认(现已改 MIT)影响.
        opts = ServoOpts(watchdog_ms=100, max_vel=2.0,
                         max_step_rad=math.radians(8.0),
                         max_lag_rad=math.radians(20.0), is_radians=True,
                         use_mit=False)
        self.arm.servo_start(opts)
        print("  回放中... (Ctrl+C 中断)")
        send_ok = send_fail = 0
        last_grip: Optional[float] = None
        t_play0 = time.monotonic()
        try:
            for i, smp in enumerate(samples):
                target_t = smp["t"] / rate_scale
                # 按真实时间对齐; 等待期间持续喂帧, 避免触发看门狗
                while True:
                    ahead = target_t - (time.monotonic() - t_play0)
                    if ahead <= 0:
                        break
                    self.arm.servo_j(smp["pos"])
                    time.sleep(min(ahead, 0.01))
                if self.arm.servo_j(smp["pos"]):
                    send_ok += 1
                else:
                    send_fail += 1
                g = smp.get("gpos")
                if (self.arm.has_gripper and g is not None
                        and (last_grip is None
                             or abs(g - last_grip) > math.radians(2.0))):
                    try:
                        self.arm.gripper_control(
                            g, effort=self.gripper_effort, block=False)
                    except Exception:
                        pass
                    last_grip = g
                if i % 20 == 0:
                    cur = self.arm.get_joint_values(prefer_cache=True)
                    print(f"\r  [{i + 1}/{len(samples)}] {deg_str(cur)}   ",
                          end="", flush=True)
        finally:
            print(f"\n  sent OK={send_ok} fail={send_fail}")
            try:
                self.arm.servo_end("hold")
            except Exception as e:
                print(f"  servo_end 失败: {e}")
        print("  回放完成")

    # ----- ServoJ (online streaming) -----

    def servo_menu(self):
        """servo_j sub-menu. 比 move_j 实时性高, 适合 teleop / 视觉伺服 demo."""
        from fafu_robot_controller import ServoOpts

        # 当前 session 的默认参数, 可以在菜单 [s] 里改
        watchdog_ms  = 100
        max_vel      = 1.0          # rad/s
        max_step_deg = 1.5          # deg, 比 C++ 例程稍宽松点便于上手
        max_lag_deg  = 10.0         # deg

        def _opts() -> ServoOpts:
            return ServoOpts(
                watchdog_ms=watchdog_ms,
                max_vel=max_vel,
                max_step_rad=math.radians(max_step_deg),
                max_lag_rad=math.radians(max_lag_deg),
                is_radians=True,
            )

        while True:
            print("\n  ServoJ (online streaming) 子菜单")
            print(f"    当前参数: watchdog={watchdog_ms}ms, "
                  f"max_vel={max_vel:.2f}rad/s, "
                  f"step<={max_step_deg:.1f}deg, lag<={max_lag_deg:.1f}deg")
            print(f"    is_servoing = {self.arm.is_servoing}")
            print("    [1] hold-in-place 测试  (推荐先跑这个, 验证看门狗)")
            print("    [2] 单关节正弦跟踪     (100Hz, 选编号+振幅+持续时间)")
            print("    [3] 多关节联动正弦     (100Hz, 相邻关节反相)")
            print("    (示教轨迹 servo 复现见主菜单 [p], 已是 servo 流式回放)")
            print("    [s] 改默认参数")
            print("    [b] 返回")
            cmd = prompt("  > ").lower()
            if cmd in ("b", "back", "q", "exit", ""):
                return
            if cmd == "1":
                self._servo_hold_test(_opts())
            elif cmd == "2":
                self._servo_single_joint_sin(_opts())
            elif cmd == "3":
                self._servo_multi_joint_sin(_opts())
            elif cmd == "s":
                s = prompt(f"  watchdog_ms (当前 {watchdog_ms}, "
                           "0=禁用): ")
                if s:
                    try: watchdog_ms = int(s)
                    except ValueError: print("  非法整数")
                s = prompt(f"  max_vel rad/s (当前 {max_vel:.2f}): ")
                if s:
                    try: max_vel = float(s)
                    except ValueError: print("  非法数字")
                s = prompt(f"  max_step_deg (当前 {max_step_deg:.2f}): ")
                if s:
                    try: max_step_deg = float(s)
                    except ValueError: print("  非法数字")
                s = prompt(f"  max_lag_deg (当前 {max_lag_deg:.2f}): ")
                if s:
                    try: max_lag_deg = float(s)
                    except ValueError: print("  非法数字")
            else:
                print(f"  未识别 {cmd!r}")

    def _servo_hold_test(self, opts):
        """[1] 开 servo 会话, 持续把当前位置 stream 出去 N 秒.

        关键观察:
          - servo_end 后会打印实际平均频率 (应该接近 100Hz)
          - 中途按 Ctrl+C 会触发 finally -> servo_end, 看动作平滑
          - 想真正验证看门狗: 改完代码后从 task manager 强杀 python.exe,
            机械臂应该在 watchdog_ms 内自动 brake (不会保持最后一帧冲)
        """
        s = prompt("  持续时长 秒 (默认 5): ")
        try: duration = float(s) if s else 5.0
        except ValueError: print("  非法数字"); return
        period_s = 0.01    # 100 Hz
        n_total = int(duration / period_s)

        q0 = self.arm.get_joint_values()
        print(f"\n  起点 (deg): {deg_str(q0)}")
        print(f"  会持续给这个目标点 {duration:.1f}s @ {1/period_s:.0f}Hz")
        if not yes(prompt("  开始? [Y/n]: ")):
            return

        try:
            self.arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}"); return

        send_ok, send_fail = 0, 0
        t_start = time.monotonic()
        next_tick = t_start
        try:
            for _ in range(n_total):
                if self.arm.servo_j(q0):
                    send_ok += 1
                else:
                    send_fail += 1
                next_tick += period_s
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n  [中断] Ctrl+C 收到, 走 servo_end")
        finally:
            print(f"  sent OK={send_ok}  fail={send_fail}")
            try: self.arm.servo_end("hold")
            except Exception as e: print(f"  servo_end 失败: {e}")

    def _servo_single_joint_sin(self, opts):
        """[2] 单关节 sin 跟踪. 默认 J6 (末端), 振幅 10deg, 周期 2s."""
        n = self.arm.num_joints
        s = prompt(f"  动哪个关节 (0..{n-1}, 默认 {n-1}): ")
        try: idx = int(s) if s else (n - 1)
        except ValueError: print("  非法编号"); return
        if not (0 <= idx < n): print("  超出范围"); return

        s = prompt("  振幅 deg (默认 10): ")
        try: amp_deg = float(s) if s else 10.0
        except ValueError: print("  非法数字"); return

        s = prompt("  正弦周期 秒 (默认 2.0): ")
        try: period_sin = float(s) if s else 2.0
        except ValueError: print("  非法数字"); return

        s = prompt("  持续时长 秒 (默认 6): ")
        try: duration = float(s) if s else 6.0
        except ValueError: print("  非法数字"); return

        q0 = self.arm.get_joint_values()
        amp = math.radians(amp_deg)
        freq = 1.0 / max(0.1, period_sin)
        peak_vel = 2 * math.pi * freq * amp     # rad/s, 估算
        print(f"\n  起点 (deg): {deg_str(q0)}")
        print(f"  J{idx} 围绕 q0[{idx}] 摆动 ±{amp_deg:.1f}deg / "
              f"{period_sin:.1f}s sin")
        print(f"  估算峰值速度 {peak_vel:.2f} rad/s  "
              f"(opts.max_vel={opts.max_vel:.2f} rad/s, "
              f"step≈{math.degrees(2*math.pi*freq*amp*0.01):.2f}deg/tick)")
        if peak_vel > opts.max_vel:
            print(f"  ⚠ 峰值速度可能高于 max_vel, 会出现 lag, 调小振幅或加大周期 ⚠")
        if not yes(prompt("  开始? [Y/n]: ")):
            return

        try:
            self.arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}"); return

        period_s = 0.01   # 100Hz
        n_total = int(duration / period_s)
        send_ok, send_fail = 0, 0
        t_start = time.monotonic()
        next_tick = t_start
        try:
            for k in range(n_total):
                t = k * period_s
                target = q0.copy()
                target[idx] = q0[idx] + amp * math.sin(2 * math.pi * freq * t)
                if self.arm.servo_j(target):
                    send_ok += 1
                else:
                    send_fail += 1
                next_tick += period_s
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n  [中断]")
        finally:
            print(f"  sent OK={send_ok}  fail={send_fail}")
            try: self.arm.servo_end("hold")
            except Exception as e: print(f"  servo_end 失败: {e}")

    def _servo_multi_joint_sin(self, opts):
        """[3] 所有关节同 sin, 相邻关节反相, 视觉上更明显."""
        s = prompt("  振幅 deg (默认 8): ")
        try: amp_deg = float(s) if s else 8.0
        except ValueError: print("  非法数字"); return
        s = prompt("  正弦周期 秒 (默认 2.5): ")
        try: period_sin = float(s) if s else 2.5
        except ValueError: print("  非法数字"); return
        s = prompt("  持续时长 秒 (默认 6): ")
        try: duration = float(s) if s else 6.0
        except ValueError: print("  非法数字"); return

        q0 = self.arm.get_joint_values()
        amp = math.radians(amp_deg)
        freq = 1.0 / max(0.1, period_sin)
        n = self.arm.num_joints
        signs = np.array([1.0 if (j % 2 == 0) else -1.0 for j in range(n)])
        peak_vel = 2 * math.pi * freq * amp
        print(f"\n  起点 (deg): {deg_str(q0)}")
        print(f"  所有关节同 sin (相邻反相), ±{amp_deg:.1f}deg / {period_sin:.1f}s")
        print(f"  估算峰值速度 {peak_vel:.2f} rad/s "
              f"(opts.max_vel={opts.max_vel:.2f} rad/s)")
        if not yes(prompt("  开始? [Y/n]: ")):
            return

        try:
            self.arm.servo_start(opts)
        except Exception as e:
            traceback.print_exc(); print(f"  [失败] {e}"); return

        period_s = 0.01
        n_total = int(duration / period_s)
        send_ok, send_fail = 0, 0
        next_tick = time.monotonic()
        try:
            for k in range(n_total):
                t = k * period_s
                phase = math.sin(2 * math.pi * freq * t)
                target = q0 + signs * amp * phase
                if self.arm.servo_j(target):
                    send_ok += 1
                else:
                    send_fail += 1
                next_tick += period_s
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\n  [中断]")
        finally:
            print(f"  sent OK={send_ok}  fail={send_fail}")
            try: self.arm.servo_end("hold")
            except Exception as e: print(f"  servo_end 失败: {e}")

    # ----- 安全 -----

    def estop(self):
        if yes(prompt("\n  确认急停 (mode=0x00)? [Y/n]: ")):
            self.arm.emergency_stop()
            print("  急停已发. 按 'r' 恢复.")

    def resume(self):
        if yes(prompt("\n  确认恢复 (重新 enable)? [Y/n]: ")):
            try:
                self.arm.resume()
            except Exception as e:
                print(f"  失败: {e}")


# ============================================================================
#  Main loop
# ============================================================================
MENU = """
+--------------------------------------------------+
|  FafuRobotController 交互式测试                  |
+--------------------------------------------------+
  [s]  显示当前状态 (角度 / 速度 / 夹爪 / 统计)
  [j]  单关节移动 (选编号 + 目标角度; 模式 moveJ / MIT保持 / ServoJ / ServoJ保持)
  [a]  全部关节移动 (输入 N 个角度; 模式 moveJ / MIT保持 / ServoJ / ServoJ保持)
  [w]  关节路径 (多组角度 -> move_jntspace_path 按序经过, 停终点并打印误差)
  [y]  软重启 (motor_reset 所有关节电机 + 切回位置模式; MIT保持后用它复位)
  [h]  回零 (go_home)
  [g]  夹爪子菜单
  [l]  软限位子菜单
  [t]  示教录制 (重力补偿浮空 + ~100Hz 连续记录到 jsonl, Ctrl+C 结束)
  [p]  示教复现 (读 jsonl, servo 按真实时间间隔流式回放)
  [v]  ServoJ (online streaming, 100Hz 跟踪, 含看门狗)
  [m]  实时状态监视 (Ctrl+C 退出)
  [e]  急停
  [r]  从急停恢复 (enable)
  [q]  退出
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="interactive test menu")
    _default_cfg = os.path.join(PARENT, "robot.cfg")
    parser.add_argument("--cfg", default=_default_cfg,
                        help=f"robot.cfg 路径 (默认: {_default_cfg})")
    parser.add_argument("--gripper-id", type=int, default=7,
                        help="夹爪 motor id (默认 7); 传 0 或负数视为无夹爪")
    parser.add_argument("--speed", type=int, default=15,
                        help="默认 speed percent (默认 15)")
    parser.add_argument("--teach-torque-scale",
                        default=",".join(str(x) for x in _TEACH_TORQUE_SCALE),
                        help="示教重力补偿每关节经验增益 (逗号分隔, 6 个). "
                             "J2 自转就调小该关节, J3 没力就调大. "
                             f"默认 {_TEACH_TORQUE_SCALE}")
    parser.add_argument("--teach-motor-models",
                        default=",".join(_TEACH_MOTOR_MODELS),
                        help="示教重力补偿每关节电机型号 (逗号分隔, 6 个, Nm->raw 系数)")
    parser.add_argument("--teach-friction", action="store_true",
                        help="示教录制启用摩擦补偿 (默认关: 纯重力, 避免 J2 库仑项自激跑飞)")
    parser.add_argument("--teach-fc",
                        default=",".join(str(x) for x in _TEACH_FC),
                        help="示教摩擦补偿 每关节库仑 Fc (逗号分隔, 6 个, Nm). "
                             "某关节自激跑飞就调小它. "
                             f"默认 {_TEACH_FC}")
    parser.add_argument("--teach-fv",
                        default=",".join(str(x) for x in _TEACH_FV),
                        help="示教摩擦补偿 每关节粘滞 Fv (逗号分隔, 6 个, Nm*s/rad). "
                             f"默认 {_TEACH_FV}")
    parser.add_argument("--teach-vel-threshold", type=float,
                        default=_TEACH_VEL_THRESHOLD,
                        help="低速死区 rad/s, 低于此速不加库仑项 (调大可压自激). "
                             f"默认 {_TEACH_VEL_THRESHOLD}")
    args = parser.parse_args()

    # 容错: 传入的 --cfg 在当前目录找不到时, 回退到脚本旁同名文件
    # (避免在工作区根目录运行时报 "无法打开配置文件").
    if not os.path.isfile(args.cfg):
        fallback = os.path.join(PARENT, os.path.basename(args.cfg))
        if os.path.isfile(fallback):
            print(f"[cfg] {args.cfg!r} 不存在, 改用 {fallback!r}")
            args.cfg = fallback

    try:
        from fafu_robot_controller import FafuRobotController
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] import fafu_robot_controller 失败: {e}")
        print("       先跑 tests/smoke_test.py 排查环境")
        return 1

    has_gripper = args.gripper_id is not None and args.gripper_id > 0
    try:
        arm = FafuRobotController(
            cfg_path=args.cfg,
            has_gripper=has_gripper,
            gripper_motor_id=args.gripper_id if has_gripper else None,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] 创建 FafuRobotController 失败: {e}")
        return 1

    try:
        teach_scale = parse_floats(args.teach_torque_scale)
    except ValueError:
        teach_scale = list(_TEACH_TORQUE_SCALE)
    teach_models = [s.strip() for s in args.teach_motor_models.split(",")
                    if s.strip()]
    try:
        teach_fc = parse_floats(args.teach_fc)
    except ValueError:
        teach_fc = list(_TEACH_FC)
    try:
        teach_fv = parse_floats(args.teach_fv)
    except ValueError:
        teach_fv = list(_TEACH_FV)
    app = App(arm, default_speed=args.speed,
              teach_torque_scale=teach_scale,
              teach_motor_models=teach_models,
              teach_friction=args.teach_friction,
              teach_fc=teach_fc,
              teach_fv=teach_fv,
              teach_vel_threshold=args.teach_vel_threshold)
    print("\n  当前运行时软限位:")
    app.show_limits()
    rc = 0
    try:
        while True:
            print(MENU)
            cmd = prompt("  >>> ").lower()
            if cmd in ("q", "quit", "exit"):
                break
            # Per-command guard: a handler raising (e.g. RobotStateError
            # when a motion is issued while the arm is braked / estopped /
            # not enabled) should only print a message and keep the menu
            # alive, not tear down the whole session.  KeyboardInterrupt is
            # NOT caught here so Ctrl+C still routes to the outer handler
            # (emergency_stop + clean exit).
            try:
                if cmd == "s":
                    app.show_state()
                elif cmd == "j":
                    app.move_one_joint()
                elif cmd in ("a", "all"):
                    app.move_all_joints()
                elif cmd in ("w", "path", "waypoints"):
                    app.move_jntspace_path_menu()
                elif cmd == "y":
                    app.soft_reboot()
                elif cmd in ("h", "home"):
                    app.go_home()
                elif cmd == "g":
                    app.gripper_menu()
                elif cmd == "l":
                    app.limit_menu()
                elif cmd == "t":
                    app.teach_record()
                elif cmd == "p":
                    app.teach_replay()
                elif cmd == "v":
                    app.servo_menu()
                elif cmd == "m":
                    app.monitor()
                elif cmd == "e":
                    app.estop()
                elif cmd == "r":
                    app.resume()
                elif cmd == "":
                    continue
                else:
                    print(f"  未识别 {cmd!r}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                traceback.print_exc()
                print(f"  [命令 {cmd!r} 出错] {e}  (菜单继续; 若为状态错误请先 "
                      "enable / resume)")
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Ctrl+C, 紧急停止..")
        try:
            arm.emergency_stop()
        except Exception:
            pass
        rc = 130
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] {e}")
        try:
            arm.emergency_stop()
        except Exception:
            pass
        rc = 1
    finally:
        try:
            arm.close_connection()
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    sys.exit(main())
