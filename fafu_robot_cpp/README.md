# fafu_robot_cpp — C++ binding source for `fafu_motor.pyd`

这是 **Fafu robot SDK** 的 **C++ 侧**, 包含两套并列的产物:

1. **`fafu_motor.pyd`** (pybind11 模块) — 给 `../fafu_robot_python/`
   那一侧的 `FafuRobotController` (Python) 当底层驱动用.
2. **`fafu_robot_sdk.lib`** (静态库, 在 `sdk/` 子目录) — 原生 C++ 高层 SDK
   `fafu_robot::FafuRobotController`, 不依赖 Python 也能跑. 详见
   [`sdk/README.md`](sdk/README.md).

> **★ 完全自包含 ★**
> 本目录已经把所有需要的 C++ 源码 (协议层 + serial_cmake 三方串口库) 都
> vendor 进来了, 不引用任何外部仓库. 你可以把整个
> `fafu_robot_sdk/` 拷到任何地方, 仍然能在本目录内独立编出 `.pyd`.

> 架构 / 协议栈 / 力矩换算等实现细节见 [`../技术文档.md`](../技术文档.md).

```
你的代码 (FafuRobotController, Python)
              │  import fafu_motor
              ▼
   fafu_motor.cpXY-win_amd64.pyd          ← 这个 .pyd 由本目录构建
              │  pybind11
              ▼
   hightorque_serial_debug.lib                ← src/hightorque_serial.cpp (vendored)
              │  serial_cmake.dll              ← third_part/serial_cmake/  (vendored)
              ▼
              USB 串口  ──▶  调试板  ──▶  电机
```

---

## 目录结构

```
fafu_robot_cpp/
├── README.md                                ← 你正在看的文件
├── bindings.cpp                             ← pybind11 绑定 (主入口, 产 fafu_motor.pyd)
├── CMakeLists.txt                           ← 顶层构建配置
├── build.bat                                ← Windows 一键构建脚本 (编 .pyd + .lib + 3 个 .exe)
│
├── include/                                 ← vendored 头文件 (协议层, 双侧共用)
│   ├── hightorque_serial.hpp                ← HightorqueSerial 类 + 数据结构
│   └── robot_config.hpp                     ← robot.cfg 解析
│
├── src/                                     ← vendored 实现 (协议层, 双侧共用)
│   └── hightorque_serial.cpp                ← HightorqueSerial 实现
│
├── third_part/
│   └── serial_cmake/                        ← vendored 跨平台串口库 (双侧共用)
│
└── sdk/                                     ← ★ 原生 C++ SDK ★
    ├── README.md                            详细文档 (强烈推荐先看这个)
    ├── CMakeLists.txt                       static lib + examples 构建
    ├── include/fafu/
    │   └── fafu_robot_controller.hpp        唯一公开头文件
    ├── src/
    │   └── fafu_robot_controller.cpp        实现
    └── examples/                            01_smoke / 02_move_j / 03_gripper / 04_servo_j
```

构建产物 (Release):

```
build/bin/Release/fafu_motor.cpXY-win_amd64.pyd     ← 自动 copy 到 ../fafu_robot_python/
build/bin/Release/serial_cmake.dll                       ← 自动 copy 到 ../fafu_robot_python/
build/sdk/Release/fafu_robot_sdk.lib                     ← 原生 C++ SDK 静态库
build/bin/Release/01_smoke.exe                           ← 已带 robot.cfg + serial_cmake.dll
build/bin/Release/02_move_j.exe                          ← 已带 robot.cfg + serial_cmake.dll
build/bin/Release/03_gripper.exe                         ← 已带 robot.cfg + serial_cmake.dll
build/bin/Release/04_servo_j.exe                         ← 已带 robot.cfg + serial_cmake.dll
```

Python 侧不用配置 PYTHONPATH, 直接
`from fafu_robot_python import FafuRobotController` 就能用.

原生 C++ 侧, 自己的项目里 `target_link_libraries(my_app PRIVATE fafu_robot_sdk)`
即可 — 详见 [`sdk/README.md`](sdk/README.md).

---

## 前置依赖

| 工具 | 推荐版本 | 备注 |
|---|---|---|
| CMake | 3.18+ | `cmake --version` |
| Visual Studio | 2019 / 2022 | 含 "C++ 桌面开发" workload (Windows) |
| Python | 3.7+ | 装在哪个环境, `.pyd` 就给哪个 ABI 编 |
| pybind11 | `pip install pybind11` | `build.bat` 会自动找它的 cmake_dir |

> **不需要任何外部仓库**. 所有源码已 vendor 到本目录.

---

## 一键构建 (Windows)

```bat
cd fafu_robot_sdk\fafu_robot_cpp
build.bat           REM Release (推荐)
build.bat Debug     REM Debug
```

成功后会自动:

1. 把 `fafu_motor.cpXY-win_amd64.pyd` copy 到 `..\fafu_robot_python\`
2. 把 `serial_cmake.dll` copy 到 `..\fafu_robot_python\`
3. 切到 `..\fafu_robot_python\` 跑一次 `import fafu_motor` 验证

---

## 手工构建 (跨平台 / 用别的 IDE)

```bash
cd fafu_robot_sdk/fafu_robot_cpp
PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build --config Release -j
```

## Ubuntu / Linux

**推荐**：用 `linux/` 子目录里的一键脚本，涵盖装依赖、USB udev、libserial_cmake.so
Linux fixup、servoJ 实时优先级等 Ubuntu 专属配套。详见 [`linux/README.md`](linux/README.md)。

```bash
cd fafu_robot_sdk/fafu_robot_cpp
bash linux/install_deps.sh          # apt 装 build-essential/cmake/python3-dev + pip 装 pybind11
bash linux/setup_udev.sh            # (推荐) 装 udev 规则, 免 sudo 访问 USB 串口
bash linux/build.sh                 # 编 fafu_motor.so + 静态库 + 4 个例程到 build_linux/
./build_linux/bin/01_smoke          # 最小连通性测试
```

> **⚠ Linux 特有的 fixup**：当前 CMakeLists 的 POST_BUILD 只在 `if(WIN32)` 分支
> 复制 `serial_cmake.dll` 到 `../fafu_robot_python/`。Linux 下这一步漏掉，
> `import fafu_motor` 会报 `libserial_cmake.so: cannot open shared object file`。
> `linux/build.sh` 已经在编译后自动补做：`cp libserial_cmake.so` +
> `patchelf --set-rpath '$ORIGIN'`。

可选 cmake 参数:

| 参数 | 默认值 | 含义 |
|---|---|---|
| `-Dpybind11_DIR=<cmake_dir>` | 由 build.bat 自动定位 | pybind11 cmake 配置目录 |
| `-DFAFU_ROBOT_PYTHON_DIR=<path>` | `../fafu_robot_python` | 产物 (.pyd + dll) 复制目标 |
| `-DCMAKE_BUILD_TYPE=Debug` | `Release` | 调试构建 |

---

## 重新构建一份不同 Python ABI 的 .pyd

`.pyd` 是按 Python ABI 编的, 只能被同 ABI 的 Python 加载:

- 当前 Python: `python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"`
- 文件名后缀: `fafu_motor.cpXY-win_amd64.pyd`, 必须匹配上面这个 `cpXY`

切到不同 Python:

```bat
conda activate <目标环境>     REM 或 venv
pip install pybind11
.\build.bat
```

`build.bat` 会用 *当前激活的 Python* 重新编一份, 输出名就是该环境的 ABI tag.

---

## 模块导出了什么 (`bindings.cpp` 速览)

| 类别 | 名字 | 说明 |
|---|---|---|
| 枚举 | `PosUnit`, `CanFault` | 位置单位 / CAN 总线状态 |
| 结构体 | `MotorState`, `CanStatus`, `PortInfo`, `Stats`, `RobotConfig`, `ManyMotorCmd` | 数据载体 |
| 函数 | `list_serial_ports`, `find_likely_debug_boards`, `to_turns`, `from_turns`, `parse_motor_state_int16` | 串口枚举 / 单位换算 / 解析 |
| 主类 | `HightorqueSerial` | 驱动核心: 打开串口 / 异步 RX / 后台轮询 / 各种 `set_*` 控制指令 / `run_control_loop` |

详见 `bindings.cpp` 的注释, 每一项都标了对应的 C++ 头文件方法签名.

---

## 跟 FafuRobotController 的协作

`fafu_robot_python/fafu_robot_controller.py` 是这个 C++ 模块的**用户**, 它:

1. `import fafu_motor as pm` (从 SDK 根目录加载本目录构建出来的 .pyd)
2. 把 `pm.HightorqueSerial` / `pm.RobotConfig` 包装成机器人级别的高层 API
   (`move_j` / `open_gripper` / `grasp` / 等等)
3. 处理单位换算 (rad ↔ turns)、阻塞 / 非阻塞、力控抓取的 Python 侧状态机...

所以 **改动 bindings.cpp 或 vendored 协议层源码后, 必须重新构建**,
高层 Python 代码才能看到新接口.

---

## vendored 源码说明

为了 self-contained, 本目录 vendor 了下列源码:

| 本目录路径 | 作用 |
|---|---|
| `include/hightorque_serial.hpp` | 调试板 USB 串口协议驱动 (头文件) |
| `include/robot_config.hpp` | robot.cfg 解析 / RobotConfig |
| `src/hightorque_serial.cpp` | 调试板 USB 串口协议驱动 (实现) |
| `third_part/serial_cmake/` | 跨平台串口库 (wjwwood/serial fork) |

如果底层协议库有升级, 用新版覆盖对应文件后重新构建即可.

---

## 排错

| 现象 | 处理 |
|---|---|
| `未找到 third_part/serial_cmake/` (cmake configure 失败) | vendor 目录被误删了, 按上一节"vendored 源码来自哪里"补回来 |
| `pybind11_DIR` 找不到 | 当前环境没装 pybind11; `pip install pybind11` 后重跑 |
| 链接报 `LNK2019: HightorqueSerial::xxx` 未解析 | `src/hightorque_serial.cpp` 缺失或没被加入 lib; 检查文件树和 CMakeLists |
| 构建成功但 `..\fafu_robot_python\` 下没有 .pyd | `FAFU_ROBOT_PYTHON_DIR` 被改到别处了; 看 cmake configure 阶段的 summary 输出 |
| 旧的 python 进程 (Flask app, Python REPL) 抓着 .dll, 拷贝失败 (MSB3073) | 先 `Stop-Process -Name python -Force` 杀掉所有 python 进程, 再重 build |
