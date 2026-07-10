# -*- coding: utf-8 -*-
"""
diag_motors.py
==============

电机 / 总线快速诊断工具.

只读 / 不动机器, 主要用来在 ``enable failed`` 之类的错误时定位:
    - 哪个电机 motor_id 没回响应 (CAN 总线上找不到 / 供电断 / 配错 ID)
    - 哪个电机的 fault 字段 != 0 (电机自己报警了 — 过流 / 过温 / encoder 错 / ...)
    - CAN 总线本身是不是正常 (BusOff / ErrorPassive 之类)

输出示例 (一切正常):
    USB ports:
      COM14   (VID:PID = 0x1234:0x5678)
    [connect] COM14 @ 4000000
    [CAN status] fault=Ok  lec=0  tx_err=0  rx_err=0

    motor    | mode  | fault | pos (turns) |   vel    |  tqe  | tag
    ---------+-------+-------+-------------+----------+-------+------
       M1    | 0x0A  | 0x00  |    +0.0123  |  +0.0000 |    +5 | OK
       M2    | 0x0A  | 0x00  |    -0.0045  |  +0.0000 |    +3 | OK
       M3    | 0x0F  | 0x00  |    +0.0000  |  +0.0000 |    +0 | OK
       M7    | 0x0A  | 0x00  |    -0.3500  |  +0.0000 |   +12 | OK

    summary: 7/7 motors responding, 0 with fault

异常示例:
    M3    | ----  | ----  |     -        |    -     |   -   | ★ NO RESPONSE
    M5    | 0x0A  | 0x04  |   +1.2300    |  +0.0000 |  +18  | ★ FAULT 0x04 — power-cycle the motor

用法:
    cd fafu_robot_sdk
    python fafu_robot_python/tests/diag_motors.py
    python fafu_robot_python/tests/diag_motors.py --cfg path/to/robot.cfg
    python fafu_robot_python/tests/diag_motors.py --port COM14    # 指定端口
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only motor / CAN diagnostic. No motion issued."
    )
    parser.add_argument("--cfg",
                        default=os.path.join(PARENT, "robot.cfg"),
                        help="robot.cfg path (default: fafu_robot_python/robot.cfg)")
    parser.add_argument("--port", default=None,
                        help="override cfg.port; useful when auto-detect picks the wrong one")
    parser.add_argument("--timeout", type=float, default=0.3,
                        help="per-motor read timeout in seconds (default 0.3)")
    args = parser.parse_args()

    try:
        import fafu_motor as pm
    except Exception as e:
        traceback.print_exc()
        print(f"\n[FAIL] import fafu_motor failed: {e}")
        print("       缺 vcredist / serial_cmake.dll / Python 版本与 pyd 不匹配?")
        return 1

    # 1) cfg
    try:
        cfg = pm.RobotConfig.load(args.cfg)
    except Exception as e:
        print(f"[FAIL] cannot load {args.cfg!r}: {e}")
        return 1

    # 2) port
    ports = pm.find_likely_debug_boards()
    print("USB ports:")
    if not ports:
        print("  (none) — 调试板没插 / 驱动没装 / VID 不在白名单")
    for p in ports:
        # PortInfo.vid / pid 是底层透传的 hex 字符串 (例: "1A86"), 不是 int
        print(f"  {p.port}   (VID:PID = {p.vid}:{p.pid})   {p.description}")

    use_port = args.port or (ports[0].port if ports else cfg.port)
    if not use_port or use_port.lower() == "auto":
        print("[FAIL] no usable port; pass --port COMXX or fix cfg.port")
        return 1

    print(f"[connect] {use_port} @ {cfg.baudrate}")
    try:
        ht = pm.HightorqueSerial(use_port, cfg.baudrate)
    except Exception as e:
        print(f"[FAIL] open serial port {use_port!r} failed: {e}")
        return 1

    rc = 0
    try:
        # 3) CAN bus status
        try:
            cs = ht.read_can_status()
            tag = "OK" if cs.fault == pm.CanFault.Ok else "★ ABNORMAL"
            print(f"[CAN status] fault={cs.fault}  lec={cs.lec}  "
                  f"tx_err={cs.tx_err_count}  rx_err={cs.rx_err_count}   {tag}")
            if cs.fault != pm.CanFault.Ok:
                print("           ★ CAN 总线异常 — 检查接线 / 终端电阻 / 共地")
                rc = max(rc, 1)
        except Exception as e:
            print(f"[CAN status] read failed: {e}")

        # 4) per-motor read
        print()
        print("  motor  | mode  | fault | pos (turns) |   vel    |  tqe  | tag")
        print("  -------+-------+-------+-------------+----------+-------+------")
        responding = 0
        with_fault = 0
        for mid in cfg.motor_ids:
            try:
                s = ht.read_motor_state(mid, args.timeout)
            except Exception as e:
                s = None
                err_msg = str(e)
            else:
                err_msg = None

            if s is None:
                print(f"   M{mid:<2}   | ----  | ----  |     -       |    -     |   -   | "
                      f"★ NO RESPONSE" + (f"  ({err_msg})" if err_msg else ""))
                rc = max(rc, 1)
                continue

            responding += 1
            if s.fault != 0:
                with_fault += 1
                tag = f"★ FAULT 0x{s.fault:02X} — power-cycle the motor"
                rc = max(rc, 1)
            else:
                tag = "OK"

            print(f"   M{mid:<2}   | 0x{s.mode:02X}  | 0x{s.fault:02X}  | "
                  f"{s.position:+11.4f}  | {s.velocity:+7.3f}  | "
                  f"{int(s.torque):+5d} | {tag}")

        total = len(cfg.motor_ids)
        print()
        print(f"  summary: {responding}/{total} motors responding, "
              f"{with_fault} with non-zero fault")
        if responding < total:
            print(f"  ★ {total - responding} 个电机没回响应 — 检查供电 / 接线 / cfg.motor_ids")
        if with_fault > 0:
            print(f"  ★ 有电机进入 fault 状态 — 通常需要给电机模块断电 5 秒再上电")

    finally:
        try:
            ht.close()
        except Exception:
            pass

    print()
    print("=" * 64)
    print(f"  结果: {'PASS' if rc == 0 else 'PROBLEM DETECTED (rc=' + str(rc) + ')'}")
    print("=" * 64)
    return rc


if __name__ == "__main__":
    sys.exit(main())
