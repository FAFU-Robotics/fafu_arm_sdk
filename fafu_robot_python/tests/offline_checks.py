#!/usr/bin/env python3
"""不需要连电机就能跑的健全性检查 (CI 用, 本地也能跑).

    python fafu_robot_python/tests/offline_checks.py

覆盖五件事:
  1. 编出来的 fafu_motor 导出面是否齐全 —— 抓 bindings.cpp 漏导出 / 改名。
  2. TORQUE_COEFF 里本臂三个型号的值有没有漂 —— 这是安全关键量, 填错会让实发
     力矩成倍偏差, 历史上已经错过两次 (漏 *0.01; 用了主机侧 SDK 的 0.5256)。
  3. 随包发布的 robot.cfg 能不能解析, 每个关节是不是都配了认识的电机型号。
  4. 故障表照厂商表3 收全了没有, 表外的码有没有静默变成空串。
  5. 一拖多梯形帧 (0x80AD) 的结构: 帧长落在 DLC 表上、尾部 [0x17,0x01]、填充
     只用 0x50、未用槽位按官方约定、超长抛异常。

退出码 0 = 全过; 1 = 有失败项 (失败信息打到 stderr)。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PY_DIR = REPO_ROOT / "fafu_robot_python"
sys.path.insert(0, str(PY_DIR))

import fafu_motor as pm  # noqa: E402  (path 必须先设好)


MODULE_ATTRS = [
    "HightorqueSerial", "RobotConfig", "MotorState", "Stats", "PosUnit",
    "TORQUE_COEFF", "parse_motor_state_int16", "to_turns", "from_turns",
    "list_serial_ports", "find_likely_debug_boards",
    "FaultInfo", "FAULT_TABLE", "describe_fault",
    "build_many_pos_vel_acc_int16", "build_read_int16", "parse_int16_registers",
    "GROUP_CAN_ID_POS_VEL_TQE", "GROUP_CAN_ID_POS_VEL_ACC",
    "GROUP_CAN_ID_MIT_LEGACY", "GROUP_CAN_ID_MIT_RECOMMENDED",
]

DRIVER_METHODS = [
    # 下发通道
    "set_pos_vel_tqe", "set_pos_vel_acc", "set_pos_vel_tqe_kp_kd",
    "set_many_pos_vel_tqe", "set_many_pos_vel_acc", "set_many_mit",
    "set_torque", "set_velocity", "set_timeout",
    "read_registers_int16", "diagnostic_query",
    # 状态读取 / 异步 RX
    "enable_async_rx", "get_cached_state", "get_states", "state_age_ms",
    "wait_state", "get_stats",
    # 软限位
    "enable_position_limit", "clear_all_position_limits",
    # 一拖多 MIT 通道选择
    "set_group_mit_can_id", "group_mit_can_id",
]

# 权威来源: 固件工程 fdcan_h730 的 motor_tqe_adj。
# 主机侧 livelybot_serial 有一张同名但数值不同的表, 别拿它来"订正"。
EXPECTED_COEFF = {
    "M5036_02": 0.67,    # J1, J4
    "M6036_02": 0.66,    # J2, J3
    "M4438_30": 0.64,    # J5, J6, J7
}

# 厂商表3 收录的全部故障码: 0 (正常) + 1-7 (DMA/UART) + 32-47 (电机层)。
# 8-31 在表里标「保留」, 不该出现在 FAULT_TABLE 里。
EXPECTED_FAULT_CODES = {0} | set(range(1, 8)) | set(range(32, 48))

# CAN-FD 的 DLC 只有这些档位, 帧长必须正好落在其中一档。
CANFD_DLC_SIZES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64)

NAN_INT16 = -32768      # 0x8000, 协议的"无操作"标记
PADDING = 0x50


def check_fault_table(failures: list[str]) -> None:
    """故障表要照厂商表3 收全, 且表外的码不能静默返回空串。"""
    got = set(pm.FAULT_TABLE)
    missing = sorted(EXPECTED_FAULT_CODES - got)
    if missing:
        failures.append(f"FAULT_TABLE 缺少厂商表3 的故障码: {missing}")
    extra = sorted(got - EXPECTED_FAULT_CODES)
    if extra:
        failures.append(
            f"FAULT_TABLE 多出厂商表3 没有的码: {extra} "
            f"(8-31 是表里标「保留」的, 不该收进来)")

    for code in sorted(got):
        info = pm.FAULT_TABLE[code]
        if not info.name:
            failures.append(f"FAULT_TABLE[{code}] 的 name 是空的")
        desc = pm.describe_fault(code)
        if not desc or str(code) not in desc:
            failures.append(
                f"describe_fault({code}) 没有给出带码的描述: {desc!r}")

    # 表外的码要明确说"没收录", 不能返回空串让日志里出现一行空白。
    for code in (8, 31, 200):
        desc = pm.describe_fault(code)
        if "未定义故障码" not in desc:
            failures.append(
                f"describe_fault({code}) 应回落成「未定义故障码」, 实际: {desc!r}")

    # 这两条 hint 是本项目自己补的排查线索, 掉了就等于故障日志少了半句话。
    for code in (41, 45):
        if code in pm.FAULT_TABLE and not pm.FAULT_TABLE[code].hint:
            failures.append(f"FAULT_TABLE[{code}] 的 hint 不该为空")


def check_group_acc_frame(failures: list[str]) -> None:
    """一拖多梯形帧 (0x80AD) 的结构。

    这个函数在没有硬件时唯一能验的就是字节布局, 所以把能钉的都钉住: 一旦有人
    改了填充字节 / 尾部 / 槽位约定, 这里立刻红。
    """
    if pm.GROUP_CAN_ID_POS_VEL_ACC != 0x80AD:
        failures.append(
            f"GROUP_CAN_ID_POS_VEL_ACC = {pm.GROUP_CAN_ID_POS_VEL_ACC:#x}, 应为 0x80AD")

    # 2 个电机: 12 字节数据 + 2 字节尾部 = 14 -> 补到 DLC 档位 16
    frame = pm.build_many_pos_vel_acc_int16([100, 200], [10, 20], [5, 6])
    if len(frame) not in CANFD_DLC_SIZES:
        failures.append(f"梯形帧长 {len(frame)} 不在 CAN-FD DLC 档位上")
    if frame[-2:] != bytes([0x17, 0x01]):
        failures.append(f"梯形帧尾部应为 [0x17,0x01], 实际 {frame[-2:].hex()}")

    # 前 12 字节是两个电机的 pos/vel/acc, 小端
    want_head = bytes([100, 0, 10, 0, 5, 0, 200, 0, 20, 0, 6, 0])
    if frame[:12] != want_head:
        failures.append(
            f"梯形帧数据段错: 期望 {want_head.hex()}, 实际 {frame[:12].hex()}")

    # 数据段和尾部之间只能是 0x50 填充
    pad = frame[12:-2]
    if any(b != PADDING for b in pad):
        failures.append(f"梯形帧填充应全是 0x50, 实际 {pad.hex()}")

    # 未用槽位: 官方 serial_struct.h 是 pos=0x8000 / vel=0 / acc=0。
    # 注意这跟老的 pos_vel_tqe 一拖多 (三个字段全 0x8000) 不一样, 是刻意的。
    idle = pm.build_many_pos_vel_acc_int16([NAN_INT16], [0], [0])
    if idle[:6] != bytes([0x00, 0x80, 0x00, 0x00, 0x00, 0x00]):
        failures.append(
            f"梯形帧未用槽位应为 pos=0x8000/vel=0/acc=0, 实际 {idle[:6].hex()}")

    # 每电机 6 字节, 10 个正好 62 字节; 11 个就超了 64 必须抛, 不能默默截断。
    try:
        pm.build_many_pos_vel_acc_int16([0] * 10, [0] * 10, [0] * 10)
    except Exception as exc:
        failures.append(f"梯形帧 10 个电机不该抛异常: {exc}")
    try:
        pm.build_many_pos_vel_acc_int16([0] * 11, [0] * 11, [0] * 11)
    except Exception:
        pass
    else:
        failures.append("梯形帧 11 个电机 (>64 字节) 应该抛异常, 实际没抛")

    # 三个数组长度不一致也必须抛 —— 静默错位会让力矩发到别的关节上。
    try:
        pm.build_many_pos_vel_acc_int16([0, 0], [0], [0])
    except Exception:
        pass
    else:
        failures.append("梯形帧 pos/vel/acc 长度不一致时应该抛异常, 实际没抛")


def check_read_int16_frame(failures: list[str]) -> None:
    """读寄存器子帧: 个数 1-3 走模式一, 4+ 走模式二; 回包能抽出 raw."""
    # 读 2 个 (MIT kp/kd @ 0x2B): cmd = 0x14 | 2 = 0x16
    frame = pm.build_read_int16(0x2B, 2)
    if frame != bytes([0x16, 0x2B]):
        failures.append(f"build_read_int16(0x2B,2) 应为 162B, 实际 {frame.hex()}")

    # 读 10 个 (0x20..0x29): 模式二 0x14, count, addr
    frame = pm.build_read_int16(0x20, 10)
    if frame != bytes([0x14, 0x0A, 0x20]):
        failures.append(f"build_read_int16(0x20,10) 应为 140A20, 实际 {frame.hex()}")

    try:
        pm.build_read_int16(0x20, 0)
    except Exception:
        pass
    else:
        failures.append("build_read_int16 count=0 应该抛异常")

    # 回包模式一: reply int16 ×2 @ 0x2B, kp=1234 kd=56
    regs = pm.parse_int16_registers(bytes([0x26, 0x2B, 0xD2, 0x04, 0x38, 0x00]))
    if regs.get(0x2B) != 1234 or regs.get(0x2C) != 56:
        failures.append(f"parse_int16_registers 模式一解错: {regs}")

    # 回包模式二: reply int16 ×2 @ 0x23
    regs = pm.parse_int16_registers(bytes([0x24, 0x02, 0x23, 0x64, 0x00, 0x0A, 0x00]))
    if regs.get(0x23) != 100 or regs.get(0x24) != 10:
        failures.append(f"parse_int16_registers 模式二解错: {regs}")


def main() -> int:
    failures: list[str] = []

    for name in MODULE_ATTRS:
        if not hasattr(pm, name):
            failures.append(f"fafu_motor 缺少 {name}")

    for name in DRIVER_METHODS:
        if not hasattr(pm.HightorqueSerial, name):
            failures.append(f"HightorqueSerial 缺少 {name}()")

    for model, want in EXPECTED_COEFF.items():
        got = pm.TORQUE_COEFF.get(model)
        if got is None:
            failures.append(f"TORQUE_COEFF 缺少型号 {model}")
        elif abs(got - want) > 1e-9:
            failures.append(
                f"TORQUE_COEFF[{model}] = {got}, 应为 {want} "
                f"(fdcan_h730 motor_tqe_adj)")

    check_fault_table(failures)
    check_group_acc_frame(failures)
    check_read_int16_frame(failures)

    cfg_path = PY_DIR / "robot.cfg"
    try:
        cfg = pm.RobotConfig.load(str(cfg_path))
    except Exception as exc:
        failures.append(f"robot.cfg 解析失败: {exc}")
    else:
        no_type = [i for i in cfg.motor_ids if not cfg.find_motor_type(i)]
        if no_type:
            failures.append(f"robot.cfg 这些关节没配 motor_type: {no_type}")
        unknown = sorted({t for t in cfg.motor_type_list()
                          if t not in pm.TORQUE_COEFF})
        if unknown:
            failures.append(f"robot.cfg 里的型号不在 TORQUE_COEFF 表中: {unknown}")

    if failures:
        print("offline_checks: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"offline_checks: OK  ({pm.__file__})")
    print(f"  TORQUE_COEFF     : {len(pm.TORQUE_COEFF)} 个型号")
    print(f"  FAULT_TABLE      : {len(pm.FAULT_TABLE)} 个故障码 (厂商表3)")
    print(f"  一拖多 MIT 通道    : 默认 {pm.GROUP_CAN_ID_MIT_LEGACY:#x} "
          f"(厂商推荐 {pm.GROUP_CAN_ID_MIT_RECOMMENDED:#x})")
    print(f"  robot.cfg 关节型号: {dict(cfg.motor_types)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
