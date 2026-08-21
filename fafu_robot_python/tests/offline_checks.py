#!/usr/bin/env python3
"""不需要连电机就能跑的健全性检查 (CI 用, 本地也能跑).

    python fafu_robot_python/tests/offline_checks.py

覆盖三件事:
  1. 编出来的 fafu_motor 导出面是否齐全 —— 抓 bindings.cpp 漏导出 / 改名。
  2. TORQUE_COEFF 里本臂三个型号的值有没有漂 —— 这是安全关键量, 填错会让实发
     力矩成倍偏差, 历史上已经错过两次 (漏 *0.01; 用了主机侧 SDK 的 0.5256)。
  3. 随包发布的 robot.cfg 能不能解析, 每个关节是不是都配了认识的电机型号。

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
]

DRIVER_METHODS = [
    # 下发通道
    "set_pos_vel_tqe", "set_pos_vel_acc", "set_pos_vel_tqe_kp_kd",
    "set_many_pos_vel_tqe", "set_many_mit", "set_torque", "set_velocity",
    "set_timeout",
    # 状态读取 / 异步 RX
    "enable_async_rx", "get_cached_state", "get_states", "state_age_ms",
    "wait_state", "get_stats",
    # 软限位
    "enable_position_limit", "clear_all_position_limits",
]

# 权威来源: 固件工程 fdcan_h730 的 motor_tqe_adj。
# 主机侧 livelybot_serial 有一张同名但数值不同的表, 别拿它来"订正"。
EXPECTED_COEFF = {
    "M5036_02": 0.67,    # J1, J4
    "M6036_02": 0.66,    # J2, J3
    "M4438_30": 0.64,    # J5, J6, J7
}


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
    print(f"  robot.cfg 关节型号: {dict(cfg.motor_types)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
