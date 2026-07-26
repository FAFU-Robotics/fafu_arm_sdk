# fafu_robot_sdk

**Fafu 机器人手臂** 的完整 SDK，分两侧：

```
fafu_robot_sdk/
├── README.md               ← 你正在看的文件 (总览)
│
├── fafu_robot_cpp/         ← C++ 侧 (★ 完全自包含 ★)
│   ├── README.md
│   ├── bindings.cpp        pybind11 绑定 (产 fafu_motor.pyd)
│   ├── CMakeLists.txt      顶层构建配置
│   ├── build.bat           Windows 一键构建
│   ├── include/, src/      vendored 调试板协议层
│   ├── third_part/serial_cmake/   vendored 跨平台串口库
│   └── sdk/                ← 原生 C++ SDK (fafu_robot_sdk.lib + 例程)
│       ├── README.md
│       ├── include/fafu/fafu_robot_controller.hpp
│       ├── src/fafu_robot_controller.cpp
│       └── examples/       01_smoke / 02_move_j / 03_gripper
│
└── fafu_robot_python/      ← Python SDK (用户日常使用)
    ├── README.md
    ├── __init__.py
    ├── fafu_robot_controller.py   高层 API、安全状态与硬件编排
    ├── _api_types.py              公开枚举、异常与配置/结果类型
    ├── _dynamics.py               Pinocchio 动力学与运动学实现
    ├── robot.cfg                  默认配置
    ├── fafu_motor.cpXY-win_amd64.pyd   ← 由 fafu_robot_cpp 构建出来
    ├── serial_cmake.dll                    ← 由 fafu_robot_cpp 构建出来
    ├── requirements.txt
    ├── examples/
    │   ├── 01_read_state.py ... 06_emergency_stop.py
    │   ├── 07_full_demo.py
    │   └── visible_motion.py
    └── tests/
        ├── smoke_test.py
        ├── test_one_joint.py
        ├── test_fafu_motion_interactive.py
        └── test_fafu_grasp_calibrate.py
```

---

## 两侧的分工

| 侧 | 角色 | 输出 | 用户是否需要碰 |
|---|---|---|---|
| **`fafu_robot_cpp/`** | 调试板 USB 串口协议 + pybind11 绑定 + **原生 C++ SDK** (所有源码已 vendor, 无需任何外部仓库) | `fafu_motor.cpXY-win_amd64.pyd` + `serial_cmake.dll` + `fafu_robot_sdk.lib` + 3 个例程 exe | 在 **第一次部署 / 改 binding / 切换 Python 版本 / 写纯 C++ 程序** 时需要构建 |
| **`fafu_robot_python/`** | 在 `fafu_motor` 之上做机器人级别的封装：关节运动、夹爪、力控抓取、软限位、急停、配置加载 | 暴露 `FafuRobotController` / `GraspResult` 等高层类 | **日常使用都在这里** |

---

## 快速开始

### 情况 A — `.pyd` 已经在 `fafu_robot_python/` 里 (大多数用户)

仓库里已经带好了 Python 3.10 / 3.12 对应的预编译 `.pyd`：

```bash
cd fafu_robot_sdk/fafu_robot_python
pip install -r requirements.txt
python tests/smoke_test.py
```

然后写代码：

```python
import sys
sys.path.insert(0, "path/to/fafu_robot_sdk/fafu_robot_python")

from fafu_robot_controller import FafuRobotController

with FafuRobotController(cfg_path="robot.cfg",
                         has_gripper=True, gripper_motor_id=7) as arm:
    arm.go_home(speed=15)
    arm.grasp(force_threshold=500)
```

### 情况 B — 需要重新构建 `.pyd` (换了 Python 版本 / 改了 bindings.cpp)

```bat
REM 前置: 装 VS2022 + CMake 3.18+ + pybind11
cd fafu_robot_sdk\fafu_robot_cpp
.\build.bat
```

`build.bat` 会自动把新的 `.pyd` 拷到 `..\fafu_robot_python\` 下，
然后跑一次 import 验证。

构建完后再走 **情况 A** 的步骤。

---

## 命名约定

| 概念 | 名字 | 备注 |
|---|---|---|
| SDK 仓库根 | `fafu_robot_sdk` | 包含 C++ + Python 两侧 |
| C++ 子项目 | `fafu_robot_cpp` | 协议层 + pybind11 绑定 + 原生 C++ SDK |
| Python 子项目 / 包 | `fafu_robot_python` | 高层封装 + 预编译 .pyd |
| 高层 Python 类 | `FafuRobotController` | 用户的主入口 |
| Python 高层入口 | `fafu_robot_controller.py` | 控制器、安全状态与硬件编排 |
| Python 支撑模块 | `_api_types.py`, `_dynamics.py` | 必须随 SDK 一起分发，不应只复制入口文件 |
| 底层 pybind11 模块 | `fafu_motor` (`.pyd`) | C++ 编译产物, 改名要重编 bindings.cpp |
| 日志前缀 | `[FafuRobot]` | 控制器内部打印 |

---

## 依赖关系 / 可移植性

整个 `fafu_robot_sdk/` 是 **完全自包含** 的, 不依赖外部仓库:

```
fafu_robot_sdk/                     ← 拷到任何地方都能独立构建 / 使用
├── fafu_robot_cpp/
│   ├── bindings.cpp                pybind11 绑定
│   ├── include/                    vendored 协议层头文件
│   ├── src/                        vendored 协议层实现
│   └── third_part/serial_cmake/    vendored 跨平台串口库
│
└── fafu_robot_python/              Python 高层封装 + 预编译 .pyd
```

- **C++ 侧 (`fafu_robot_cpp/`)** 构建依赖: CMake 3.18+ / VS2022 (或 GCC/Clang) / pybind11.
  所有 C++ 源码都已 vendor 进本目录, **不需要任何外部仓库** 就能 build.

- **Python 侧 (`fafu_robot_python/`)** runtime only requires `numpy` (see
  `fafu_robot_python/requirements.txt`) plus an ABI-matched `.pyd` and `.dll`.
  Native rebuilds use `fafu_robot_python/requirements-build.txt`.

> vendored C++ 源码的上游对照表见 [`fafu_robot_cpp/README.md`](fafu_robot_cpp/README.md)
> 的 "vendored 源码来自哪里" 一节. 如果厂家升级了底层协议库, 直接覆盖回来重新构建即可.

---

## 详细文档

- **C++ 侧总览 (构建 / 改 binding)** → 看 [`fafu_robot_cpp/README.md`](fafu_robot_cpp/README.md)
- **原生 C++ SDK (类参考 / 例程)**  → 看 [`fafu_robot_cpp/sdk/README.md`](fafu_robot_cpp/sdk/README.md)
- **Python 侧 (API / 测试 / 用法)** → 看 [`fafu_robot_python/README.md`](fafu_robot_python/README.md)
