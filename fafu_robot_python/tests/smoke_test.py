# -*- coding: utf-8 -*-
"""
smoke_test.py
=============

无硬件 / 半硬件烟雾测试 (Smoke Test).
按顺序跑下面 5 个检查, 哪一步先红就先修哪一步:

    1. Python ABI tag 与 fafu_motor*.pyd 是否匹配
    2. import fafu_motor 成功
    3. import fafu_robot_controller 成功
    4. robot.cfg 找得到 + 能解析
    5. find_likely_debug_boards() 列出 USB 调试板 (机器臂可不上电)

通过条件: 全部出现 [PASS]; 任何 [WARN] / [FAIL] 按提示处理.

用法:
    cd fafu_robot_sdk
    python tests/smoke_test.py
    python tests/smoke_test.py --cfg robot.cfg     # 显式指定
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback


HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)


def section(title: str) -> None:
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def passed(msg: str) -> None:
    print(f"  [PASS] {msg}")


def warned(msg: str) -> None:
    print(f"  [WARN] {msg}")


def failed(msg: str) -> int:
    print(f"  [FAIL] {msg}")
    return 1


def step_python_version() -> int:
    section("1. Python 版本")
    import glob

    abi_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print(f"    sys.executable: {sys.executable}")
    print(f"    sys.version   : {sys.version.splitlines()[0]}")
    print(f"    ABI tag       : {abi_tag}")

    pyds = sorted(glob.glob(os.path.join(PARENT, "fafu_motor*.pyd")))
    if not pyds:
        return failed(f"未在 {PARENT} 找到任何 fafu_motor*.pyd, 先 build.bat")
    matched = [p for p in pyds if abi_tag in os.path.basename(p)]
    for p in pyds:
        tag = "← 匹配本 Python" if p in matched else "ABI 不匹配 (无法加载)"
        print(f"    pyd: {os.path.basename(p)}  {tag}")
    if not matched:
        return failed(
            f"没有任何 .pyd 匹配当前 Python ABI {abi_tag}; "
            f"切到对应版本 Python 或重新 build.bat"
        )
    passed(f"找到匹配的 .pyd: {os.path.basename(matched[0])}")
    return 0


def step_import_pm() -> int:
    section("2. import fafu_motor")
    try:
        import fafu_motor as pm  # noqa: F401
    except Exception as e:
        traceback.print_exc()
        return failed(f"import fafu_motor 失败: {e}")
    passed(f"import OK ({pm.__file__})")

    expected = ["HightorqueSerial", "RobotConfig", "ManyMotorCmd",
                "PosUnit", "find_likely_debug_boards", "to_turns", "from_turns"]
    miss = [name for name in expected if not hasattr(pm, name)]
    if miss:
        return failed(f"fafu_motor 缺少符号: {miss} (pyd 不完整?)")
    passed(f"必备符号齐全: {expected}")
    return 0


def step_import_controller() -> int:
    section("3. import fafu_robot_controller")
    try:
        from fafu_robot_controller import FafuRobotController
    except Exception as e:
        traceback.print_exc()
        return failed(f"import fafu_robot_controller 失败: {e}")
    passed("import OK")

    expected_methods = [
        "enable", "disable", "brake", "is_enabled",
        "move_j", "go_home", "move_jntspace_path",
        "get_joint_values", "get_joint_velocities", "get_motor_states",
        "open_gripper", "close_gripper", "gripper_control",
        "set_limit", "get_limit", "clear_limits",
        "emergency_stop", "resume", "close_connection",
    ]
    miss = [m for m in expected_methods if not hasattr(FafuRobotController, m)]
    if miss:
        return failed(f"FafuRobotController 缺少方法: {miss}")
    passed(f"FafuRobotController 接口完整 ({len(expected_methods)} 个方法)")

    if hasattr(FafuRobotController, "_rad_to_turns"):
        return failed("控制器不应暴露 radians -> turns 下发转换 helper")
    rad2 = FafuRobotController._turns_to_rad(0.5)
    if abs(rad2 - 3.141592653589793) > 1e-9:
        return failed(f"_turns_to_rad(0.5) 应该 = pi, 得到 {rad2}")
    passed("radians API 边界与反馈单位换算正确")
    return 0


def step_cfg_load(cfg_path: str) -> int:  # noqa: D401
    section(f"4. 解析配置 {cfg_path!r}")
    if not os.path.isabs(cfg_path):
        candidate = os.path.join(PARENT, cfg_path)
        if os.path.exists(candidate):
            cfg_path = candidate
    if not os.path.exists(cfg_path):
        return failed(f"配置文件不存在: {cfg_path}")
    print(f"    实际路径: {cfg_path}")
    try:
        import fafu_motor as pm
        cfg = pm.RobotConfig.load(cfg_path)
    except Exception as e:
        traceback.print_exc()
        return failed(f"RobotConfig.load 失败: {e}")
    print(f"    port           : {cfg.port}")
    print(f"    baudrate       : {cfg.baudrate}")
    print(f"    motor_ids      : {list(cfg.motor_ids)}")
    print(f"    pos_unit       : {cfg.pos_unit}")
    print(f"    control_rate_hz: {cfg.control_rate_hz}")
    print(f"    use_async_rx   : {cfg.use_async_rx}")
    print(f"    limits         : {dict(cfg.limits)}")
    if not cfg.motor_ids:
        return failed("motor_ids 为空, 至少需要一个电机")
    passed("配置文件解析成功")
    return 0


def step_enumerate_ports() -> int:
    section("5. 枚举 USB 调试板")
    import fafu_motor as pm
    try:
        all_ports = pm.list_serial_ports()
        candidates = pm.find_likely_debug_boards()
    except Exception as e:
        traceback.print_exc()
        return failed(f"枚举串口失败: {e}")
    cand_set = {p.port for p in candidates}
    print(f"    系统全部串口 ({len(all_ports)}):")
    for p in all_ports:
        flag = "★" if p.port in cand_set else " "
        print(f"      {flag} {p.port:<10} {p.description}  [{p.hardware_id}]")
    if not candidates:
        warned("未识别到 USB 调试板候选; 如果调试板已插好, 检查驱动 (CH340/CP210x 等)")
    else:
        passed(f"识别到调试板候选: {[p.port for p in candidates]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="fafu_robot_controller smoke test")
    parser.add_argument("--cfg", default="robot.cfg",
                        help="robot.cfg 路径 (默认: fafu_robot_sdk/robot.cfg)")
    args = parser.parse_args()

    print("=" * 60)
    print("  fafu_robot_controller smoke test")
    print("=" * 60)

    rc = 0
    rc |= step_python_version()
    if rc:
        return rc
    rc |= step_import_pm()
    if rc:
        return rc
    rc |= step_import_controller()
    if rc:
        return rc
    rc |= step_cfg_load(args.cfg)
    if rc:
        return rc
    rc |= step_enumerate_ports()

    section("结论")
    if rc == 0:
        print("  全部检查通过, 可以继续 test_one_joint.py 做真机小幅运动测试.")
    else:
        print("  有失败项, 请先修复再继续.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
