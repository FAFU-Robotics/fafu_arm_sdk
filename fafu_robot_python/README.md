# fafu_robot_python

**Fafu 机器人手臂** 的 Python SDK，提供 Piper / Realman 风格的高层 API。
底层是同仓库 `fafu_robot_cpp/` 用 pybind11 构建出来的 `fafu_motor` 扩展模块。

```
你的脚本  ──Python──▶  FafuRobotController  ──pybind11──▶  HightorqueSerial (C++)  ──USB串口──▶  调试板
```

> SDK 总览见上一级目录 [`../README.md`](../README.md)；构建 `.pyd` 见同级 [`../fafu_robot_cpp/README.md`](../fafu_robot_cpp/README.md)；
> 架构 / 协议栈 / 动力学等实现细节见 [`../技术文档.md`](../技术文档.md)。

---

## 目录结构

```
fafu_robot_python/
├── README.md                           ← 你正在看的文件
├── __init__.py                         包入口, 暴露 FafuRobotController / GraspResult
├── fafu_robot_controller.py            主控制器 (约 1700 行)
├── robot.cfg                           默认配置 (端口/波特率/电机ID/软限位/控制率)
├── fafu_motor.cpXY-win_amd64.pyd   底层 C++ 绑定 (由 ../fafu_robot_cpp/ 构建)
├── serial_cmake.dll                    Windows 运行时依赖 (同上)
├── requirements.txt
├── requirements-build.txt               构建 C++ 扩展时使用
├── fafu_robot_description/             vendored follower URDF (FK/IK 用)
├── examples/
│   ├── README.md                        基础示例索引
│   ├── 01_connect_and_read.py           连接并读取状态
│   ├── 02_move_joint.py                 最小关节运动
│   ├── 03_gripper.py                    夹爪开合
│   ├── 04_enable_disable.py             使能 / 制动 / 释放
│   ├── 05_servo_joint.py                Servo 流式控制
│   ├── 06_emergency_stop.py             急停与恢复
│   └── visible_motion.py                视觉可见的综合运动 demo
└── tests/
    ├── smoke_test.py                   无硬件环境检查
    ├── test_one_joint.py               单关节 ±5° 安全测试
    ├── test_fafu_motion_interactive.py 交互式菜单 (运动 / 夹爪 / 软限位 / 示教)
    ├── test_fafu_grasp_calibrate.py    力控抓取阈值标定
    ├── test_fafu_servo_j.py            servoJ 在线流式跟踪 + 看门狗
    ├── test_fafu_kinematics.py         FK / IK + move_p / move_l 自检
    ├── test_fafu_keyboard_cartesian.py 键盘笛卡尔 teleop
    ├── test_fafu_gravity_only.py       纯重力补偿 (照搬厂商脚本)
    ├── test_fafu_gravity_comp.py       重力+摩擦+阻抗 拖动示教
    └── diag_*.py                       诊断脚本 (motors / hold_torque / torque_ramp)
```

---

## 命名约定

| 概念 | 名字 | 备注 |
|---|---|---|
| 顶层 Python 类 | `FafuRobotController` | 用户直接使用的入口 |
| Python 模块文件 | `fafu_robot_controller` | 同目录可直接 import |
| Python 包名 | `fafu_robot_python` | `from fafu_robot_python import FafuRobotController` |
| 底层 C++ 绑定 | `fafu_motor` | 不改名（C++ 编译产物名，改名要重编 `bindings.cpp`） |
| 日志前缀 | `[FafuRobot]` | 控制器内部打印 |

---

## 安装

```bash
# 1. 装 Python 依赖
cd fafu_robot_sdk/fafu_robot_python
pip install -r requirements.txt

# 2. 确认 fafu_motor.cpXX-win_amd64.pyd 的 ABI 与你的 Python 匹配
python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
# 输出 cp310 → 用 fafu_motor.cp310-win_amd64.pyd
# 输出 cp38  → 进 fafu_robot_cpp/ 用 cp38 环境重 build 一份

# 3. 跑烟雾测试 (不需要连机器人)
python tests/smoke_test.py
```

如果 `.pyd` 跟当前 Python ABI 对不上，去 `../fafu_robot_cpp/` 跑 `build.bat`
重新编一份。如果在 conda / 不同 Python 环境下，**先激活目标环境再 pip install + build**。

---

## 5 分钟快速上手

### 方式 A — 作为包（如果 `fafu_robot_sdk` 的上一级在 sys.path）

```python
import math
from fafu_robot_python import FafuRobotController

with FafuRobotController(cfg_path="fafu_robot_sdk/fafu_robot_python/robot.cfg",
                         has_gripper=True, gripper_motor_id=7) as arm:
    arm.go_home(speed=15)
    arm.move_j([0, math.radians(20), math.radians(40), 0, 0, 0], speed=15)
    arm.open_gripper()
    r = arm.grasp(force_threshold=500)
    if r.grasped:
        print(f"got it: closed {r.closed_deg:.1f}° peak {r.peak_torque_raw}")
```

### 方式 B — 作为脚本（同目录扁平 import）

```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))   # 指向 fafu_robot_python/
from fafu_robot_controller import FafuRobotController
```

`tests/` 和 `examples/` 下的脚本都按方式 B 写，可以直接运行。
最简用法请从 [`examples/README.md`](examples/README.md) 和编号示例开始。

---

## API 速查

### 连接 / 电源

```python
arm = FafuRobotController(cfg_path="robot.cfg",
                          has_gripper=True, gripper_motor_id=7)
arm.is_enabled                                    # → bool
arm.enable() / arm.disable() / arm.brake()
arm.close_connection(joint_release="stop",
                     gripper_release="brake")     # 退出时电机模式
```

### 关节运动

```python
arm.move_j([j1, j2, ..., jn], is_radians=True,
           speed=50, block=True, tolerance=0.01)
arm.go_home(speed=20)
arm.move_jntspace_path(path, speed=50)            # 需要 wrs (TOPPRA)
arm.get_joint_values()                            # → numpy array, 弧度
arm.get_joint_velocities()                        # → numpy array, rad/s
arm.get_motor_states()                            # → dict[int, MotorState]
```

### servoJ（在线流式控制，100~200Hz）

给上层在线规划 / teleop / 视觉伺服用，自带固件看门狗等四道安全防线：

```python
from fafu_robot_controller import ServoOpts
arm.servo_start(ServoOpts(watchdog_ms=100, max_vel=1.5,
                          max_step_rad=0.02, max_lag_rad=0.15))
while running:
    arm.servo_j(target_angles)                    # 非阻塞, 每 tick 调一次
arm.servo_end(finish_mode="brake")                # 清看门狗 + 收尾
arm.servo_lag_count()                             # 跟踪误差超限计数
```

### 笛卡尔运动 / FK·IK（需要 pinocchio）

```python
arm.setup_dynamics(motor_models=[...], eef_frame="tool_link")  # 先加载 URDF
pos, rot = arm.get_pose()                         # 当前末端位姿 (返回元组)
fk = arm.forward_kinematics(q)                    # FK → dict: position/rotation/rpy/transform/q
q = arm.inverse_kinematics(fk["position"], fk["rotation"])     # IK (阻尼最小二乘+多初值)
arm.move_p(pos, rot, speed=20)                    # 笛卡尔点到点
arm.move_l(pos, rot, speed=20)                    # 笛卡尔直线 (SE3 测地线)
```

### 重力 / 摩擦补偿 + 拖动示教（需要 pinocchio）

```python
arm.setup_dynamics(motor_models=["M5036_02", "M6036_02", ...],
                   tau_limit=[15,30,30,15,5,5], torque_scale=1.0)
arm.get_gravity(q)                                # 重力项 G(q), Nm
arm.gravity_compensation_step(friction=False)     # 单步纯重力 (照搬厂商)
arm.start_gravity_compensation(...)               # 拖动示教 (K/B/I 阻抗软保持)
arm.tau_to_raw(tau)                               # Nm → raw int16 (干跑预览)
```

> 动力学需要 `pinocchio`（Windows 建议 conda-forge / WSL）。未装时这些方法报清晰错误，
> 其余功能（move_j / servo_j / 夹爪）照常工作。详见 `tests/test_fafu_gravity_*.py`。

### 夹爪（4 种调用方式）

```python
arm.open_gripper(effort=None)                     # 张到上软限位
arm.close_gripper(effort=None)                    # 关到下软限位
arm.gripper_control(angle=math.radians(-90),      # Piper 风格: 任意角度
                    effort=500)
arm.grasp(force_threshold=500,                    # 力控抓取
          effort=None, vel=0.15, timeout=5.0)     # → GraspResult
```

`GraspResult` 字段：

| 字段 | 含义 |
|---|---|
| `grasped` | `True` = 抓到物体 |
| `reason` | `'detected_object_force'` / `'detected_object_stall'` / `'reached_target'` / `'no_movement'` / `'timeout'` |
| `angle_rad` | 停止时夹爪角度 |
| `closed_deg` | 从起始位置闭合了多少度 |
| `peak_torque_raw` | 整个过程力矩峰值（raw int16） |
| `duration_s` | 耗时秒数 |

### 软限位 / 安全

```python
arm.set_limit(motor_id, lo, hi, is_radians=True)
arm.get_limit(motor_id) / arm.disable_limit(motor_id) / arm.clear_limits()
arm.emergency_stop() / arm.resume()
```

详细的 `grasp()` / 力控原理见 `fafu_robot_controller.py` 顶部 docstring 与
`tests/test_fafu_grasp_calibrate.py` 注释。

---

## 测试流程（推荐顺序）

| 阶段 | 命令 | 通过条件 |
|---|---|---|
| 1 | `python tests/smoke_test.py` | 全部 `[PASS]`；ABI 匹配、import 成功、cfg 解析 OK、能枚举到 USB 调试板 |
| 2 | `python tests/test_one_joint.py --joint 5 --delta-deg 5 --speed 10` | 关节移动 ±5°，回到起点误差 < 0.5° |
| 3 | `python examples/visible_motion.py` | J2 / J4 / 夹爪都看得见动作 |
| 4 | `python tests/test_fafu_motion_interactive.py --gripper-id 7` | 菜单驱动全功能跑一遍 |
| 5 | `python tests/test_fafu_grasp_calibrate.py --gripper-id 7` | 拿到推荐的 `force_threshold` 数值 |
| 6 | `python tests/test_fafu_servo_j.py` | servoJ 跟踪 sin 轨迹，lag/clamp 计数可控，看门狗生效 |
| 7 | `python tests/test_fafu_kinematics.py` | FK→IK 往返自洽，`move_p` / `move_l` 落点误差达标 |
| 8 | `python tests/test_fafu_gravity_only.py --dry-run` | 干跑看每关节力矩 / raw 量级合理（先确认再上真机） |

---

## 与 piper.py 的差异

`FafuRobotController` 的接口刻意对齐 `PiperArmController`，所以大部分代码可以
逐行迁移：

| Piper | Fafu | 备注 |
|---|---|---|
| `PiperArmController(can_name="can0")` | `FafuRobotController(cfg_path="robot.cfg")` | 一个走 CAN，一个走 USB→CAN-FD 调试板 |
| `arm.move_j(angles, speed=...)` | 同名同语义 | ✅ |
| `arm.move_p(...) / move_l(...)` | ✅ 已实现（Pinocchio FK/IK） | 需先 `setup_dynamics()` 加载 URDF + 装 `pinocchio` |
| `arm.open_gripper(width)` | `arm.open_gripper(angle)` | Fafu 夹爪是旋转关节，单位是角度而非宽度 |
| `arm.close_gripper(effort)` | `arm.close_gripper(effort=N)` | `effort` 在 Piper 是 N·m，在 Fafu 是 raw int16 |
| ─ | `arm.grasp(force_threshold=...)` | Fafu 独有：Python 侧力矩监测+早停 |
| ─ | `arm.servo_j(...)` + 看门狗 | Fafu 独有：在线流式控制（C++ 侧也有） |
| ─ | `arm.start_gravity_compensation(...)` | Fafu 独有：重力/摩擦补偿拖动示教 |

---

## 安全提醒

- 第一次跑先把 `robot.cfg` 里的 `motor_ids` 改成单关节，确认 OK 再加。
- `limits.*` 是保守初值，**必须按你机器臂实际行程修改 `robot.cfg`**。
- `close_connection()` 默认对手臂关节是 `stop`（PWM off），**重力下会下坠** —
  肩 / 肘等承重关节先用手或外力托住。
- 夹爪默认 `gripper_release="brake"`，断开后能短路阻尼挂住物体，但**不会**保持驱动力矩，
  没有自锁机构的物体仍可能滑落。
- 调试时手边备好"拔 USB"的能力。

---

## 排错

| 现象 | 处理 |
|---|---|
| `ImportError: DLL load failed`（导入 `fafu_motor` 时） | 看 `serial_cmake.dll` 是否在本目录；装 VC++ 2015-2022 Redistributable |
| `ModuleNotFoundError: fafu_motor` | `.pyd` 不在本目录，或 ABI tag (cp310 / cp38) 与当前 Python 不匹配；去 `../fafu_robot_cpp/` 重 build |
| 连接失败 / 串口打开失败 | `robot.cfg` 里 `port = auto`，或显式写 `COMxx`；端口被其它程序占着也会失败 |
| 所有电机 `mode 0 / fault ≠ 0` | 调试板→电机的 CAN 没通：检查 24V / 终端电阻 / `motor_ids` 是否对得上 |
| 夹爪不响应 | 确认构造时传了 `has_gripper=True, gripper_motor_id=N`，且 N 在 `cfg.motor_ids` 列表里 |

---

## 版本

- `fafu_robot_python` `0.1.0`
- 底层 `fafu_motor` pybind11 绑定: 由 `../fafu_robot_cpp/bindings.cpp` 构建
