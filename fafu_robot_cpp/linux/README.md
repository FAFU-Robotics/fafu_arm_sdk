# fafu_robot_cpp — Ubuntu 部署指南

`fafu_robot_cpp/` 在 Ubuntu 上跑起来的一站式配套。**不修改任何 C++ 源码 / CMakeLists**，
所有 Linux 特定的东西集中在 `linux/` 子目录，Windows 用 `build.bat` 的路径完全不受影响。

## 支持的发行版

| 系统 | 内核 | g++ | cmake | 状态 |
|------|------|-----|-------|------|
| Ubuntu 22.04 LTS | 5.15+ | 11.x | 3.22 | 推荐 |
| Ubuntu 20.04 LTS | 5.4+ | 9.x  | 3.16 | 需装 Kitware 源升级 cmake ≥ 3.18 |
| Ubuntu 24.04 LTS | 6.5+ | 13.x | 3.28 | OK |
| Ubuntu 18.04 LTS | 4.15+ | 7.x  | 3.10 | 需装 Kitware 源升级 cmake |
| Debian 11 / 12 | — | — | — | 同等 |

最低要求：**C++17（g++ ≥ 7）+ cmake ≥ 3.18 + Python ≥ 3.7 + pybind11**（对比老 SDK 的 3.12，这里因为用了 pybind11 提升了）。

## 从 0 到 1 部署（6 步）

```bash
# 0) 进项目目录
cd fafu_robot_sdk/fafu_robot_cpp

# 1) 装依赖: build-essential / cmake / python3-dev / pybind11, 加入 dialout 组
bash linux/install_deps.sh

# 2) (推荐) 装 udev 规则, USB 串口固定到 /dev/fafu_debug_board 且任何用户可读写
bash linux/setup_udev.sh

# 3) 插上调试板 USB, 检查能不能看到
ls -l /dev/ttyUSB* /dev/fafu_debug_board 2>/dev/null

# 4) 一键编译 (pybind11 模块 + 原生 SDK + 4 个例程)
bash linux/build.sh

# 5) 跑原生 C++ 例程 (先跑 01_smoke 验证通信, 不动电机)
./build_linux/bin/01_smoke

# 6) Python 侧验证
cd ../fafu_robot_python
python3 -c "import fafu_motor; print(fafu_motor.__file__)"
```

第一次接机械臂前**强烈建议**先跑 `01_smoke`（只读状态、不上电、安全），确认串口
和电机都正常再上 `02_move_j / 03_gripper / 04_servo_j`（这三个会真实驱动关节）。

## 文件一览

```
linux/
├── README.md                            你正在看的文件
├── install_deps.sh                      apt 装编译链 + Python + pybind11 + dialout 组
├── 99-fafu-debug-board.rules        udev 规则源文件
├── setup_udev.sh                        装 / 卸载 / 检查 udev 规则
├── build.sh                             cmake 一键编译到 build_linux/
├── build_module_only.sh                 快速重编 fafu_motor.so (换 Python 环境用)
├── run_rt.sh                            SCHED_FIFO + CPU affinity 启动 (servoJ 减抖必备)
└── robot.linux.cfg                      Linux 版示例配置 (含 gripper_max_torque_raw)
```

产出统一到 `fafu_robot_cpp/build_linux/bin/`，与 Windows 的 `build/` 完全隔离，
同一份代码可以在 WSL 和 Windows 上并存两套构建。

## Linux 配套要点

| 方面 | 要求 |
|------|------|
| cmake | **≥ 3.18** |
| pybind11 | **需要** (`pip install pybind11`) |
| python3-dev | **需要** (编 `fafu_motor.so`) |
| 例程 | `01_smoke / 02_move_j / 03_gripper / 04_servo_j` |
| 产物 | pybind11 模块 + 静态库 + 4 个 exe |
| 串口依赖 | 静态链接，无额外 `.so` / RPATH 部署 |
| RT 优先级 | **servoJ 100Hz 强烈建议** (超 watchdog_ms 会固件 brake) |

## 1. 安装依赖 `install_deps.sh`

```bash
bash linux/install_deps.sh                       # 全量 (含 pybind11)
bash linux/install_deps.sh --no-pybind11         # 只装原生 SDK 依赖, 不装 Python 侧
bash linux/install_deps.sh --python python3.10   # 指定 Python (多版本环境)
bash linux/install_deps.sh --skip-group          # 不动用户组 (Docker / CI)
```

做的事：

1. `apt install` 装 `build-essential / cmake / python3-dev / python3-pip / pkg-config / git`
2. 校验 gcc ≥ 7、cmake ≥ 3.18（不满足直接退出，18.04 会提示装 Kitware 源）
3. `pip install --user pybind11`（`--user` 装到用户 site-packages，不动系统 Python）
4. 把 pybind11 cmake 路径写到 `~/.fafu_robot_cpp.env`，`build.sh` 会自动读
5. 加入 `dialout` 组（**必须重新登录**才生效，或临时 `newgrp dialout`）

## 2. USB 权限 `setup_udev.sh`

跟老版一样，装完之后：

- 调试板插上自动出现 `/dev/fafu_debug_board`（→ 真实的 `/dev/ttyUSBx`）
- 权限 `0666`（任何用户能读写，无需 dialout 组）
- 拔插换 USB 口路径都不变

```bash
bash linux/setup_udev.sh                # 安装
bash linux/setup_udev.sh --verify       # 只检查不改
bash linux/setup_udev.sh --uninstall    # 卸载
```

覆盖了 ST/FTDI/CH340/CP210x/PL2303 五大 VID（与 `find_likely_debug_boards`
默认白名单一致）。VID 不在表里的调试板，先 `lsusb` 查到 VID，往
`99-fafu-debug-board.rules` 里照葫芦画瓢加一条，重跑 setup_udev.sh。

## 3. 编译 `build.sh`

```bash
bash linux/build.sh                       # 全部: pybind11 + SDK + examples (Release)
bash linux/build.sh --no-python           # 只编原生 SDK + examples (不依赖 pybind11)
bash linux/build.sh --no-examples         # 只编模块 + 静态库
bash linux/build.sh --module-only         # 只编 fafu_motor.so (最快)
bash linux/build.sh --clean               # 删 build_linux/ 重头编
bash linux/build.sh --debug               # Debug 构建 (-O0 -g)
bash linux/build.sh --jobs 4              # 限制 4 个并行
bash linux/build.sh --python python3.11   # 指定 Python (多 ABI 环境, 跟 build_module_only.sh 一样)
```

产出 `build_linux/bin/`：

```
build_linux/bin/
├── 01_smoke                     ★ 最小连通性测试 (不上电, 安全)
├── 02_move_j                    go_home + 多关节 S-curve (会运动)
├── 03_gripper                   open / close / grasp / release (会闭合)
├── 04_servo_j                   100Hz sin 跟踪 + 看门狗 (推荐配 run_rt.sh)
└── robot.cfg                    从 fafu_robot_python/robot.cfg copy 过来
```

同时 `../fafu_robot_python/` 会有：

```
fafu_robot_python/
├── fafu_motor.cpython-3xx-x86_64-linux-gnu.so   ← 由 build.sh 复制过来 (POST_BUILD)
└── (你原来的 fafu_robot_controller.py 等)
```

### Linux 部署

`serial_cmake` 已静态链接进 `fafu_motor.so` 和原生 SDK，不需要复制
`libserial_cmake.so`、设置 RPATH 或运行 `patchelf`。

## 4. Python 环境切换 (多 ABI)

`fafu_motor.so` 按 Python ABI 编，文件名后缀就是 ABI 标签，比如
`fafu_motor.cpython-310-x86_64-linux-gnu.so`。**只能被同 ABI 的 Python 加载**。

切到不同 Python 环境后，快速重编：

```bash
# conda: activate 你的环境
conda activate my_py311
pip install pybind11        # 每个环境都要装
bash linux/build_module_only.sh    # 用当前 python3 快速编

# 或指定 Python
bash linux/build_module_only.sh --python python3.11
```

`build_module_only.sh` 不会重编原生 SDK 和例程，很快（几秒钟）。

## 5. servoJ 与实时优先级 `run_rt.sh`

`04_servo_j` 是 **100Hz 在线 servoJ**，跟老 SDK 里的离线 S-curve 完全不同：
上层每 10ms 送一次目标关节角，**如果卡超过 `watchdog_ms`（默认 100ms）固件会自动 brake**。

这意味着：普通 SCHED_OTHER 调度下 CPU 一忙抖动上来，servoJ 循环挂 100ms 就被固件掐了。
物理 Ubuntu 上强烈推荐 SCHED_FIFO：

```bash
# 一次性授权 (推荐)
sudo apt install util-linux
sudo setcap cap_sys_nice=eip build_linux/bin/04_servo_j

# 跑
bash linux/run_rt.sh 04_servo_j                   # 默认绑 CPU 0, SCHED_FIFO prio=80
bash linux/run_rt.sh 04_servo_j --cpu 2           # 绑到 CPU 2
bash linux/run_rt.sh 04_servo_j --no-rt           # 只 affinity, 不 FIFO
bash linux/run_rt.sh 04_servo_j -- robot.cfg 6    # 透传给例程 (-- 后是例程参数)
```

进一步降抖（可选，按需追加）：

| 措施 | 效果 | 代价 |
|------|------|------|
| `setcap cap_sys_nice=eip` | 不用 root 也能 RT 优先级 | 一次性，二进制升级后要重来 |
| `taskset -c N`（CPU 亲和） | 绑核，cache 利用率高 | 占一个核 |
| `chrt -f 80`（SCHED_FIFO） | 抢占普通用户进程 | 内核 RT 调度 |
| CPU governor performance | 频率锁高 | 功耗高 |
| `isolcpus=N` 内核 cmdline | 独占 CPU N | 改 GRUB 重启 |
| PREEMPT_RT 内核 | 内核也可抢占 | 装第三方内核 |

普通 Ubuntu + setcap + FIFO + taskset 对 100Hz servoJ 一般足够。看
`04_servo_j` 自己打印的 `max_lag_rad` / `stats.max_jitter_ms` 判断是否稳定。

## 6. 例程速览

| exe | 会不会动 | 用途 | 阻塞时长 |
|-----|--------|------|---------|
| `01_smoke` | 不动 | 连接 → 列电机 → 读一次状态 → 干净断开。第一次接机械臂用这个先验证 | ~1 秒 |
| `02_move_j` | ★ 会 | `go_home` + 多关节小幅 S-curve，Fafu 侧移植自 `move_j` | ~20 秒 |
| `03_gripper` | ★ 会 (夹爪) | `open / close / grasp / release`，验证 M7 力控闭合 | ~15 秒 |
| `04_servo_j` | ★ 会 | 100Hz sin 跟踪 6 秒 + 看门狗 / 步长 / lag 报警 | ~6 秒 (可命令行改) |

每个例程都可以传 `robot.cfg` 路径：

```bash
./build_linux/bin/02_move_j linux/robot.linux.cfg
./build_linux/bin/04_servo_j robot.cfg 10        # 跑 10 秒 sin
```

## 7. 配置文件 `robot.linux.cfg`

`build_linux/bin/` 里 CMakeLists 已经自动 copy 了 `../fafu_robot_python/robot.cfg`
过来，所以例程运行时**默认用同目录下的 `robot.cfg`**（跟 Windows 一致）。

三种玩法：

```bash
# A. 用项目现有 robot.cfg 完全不改 (最简)
./build_linux/bin/02_move_j

# B. 单独传 Linux 版, 不污染 fafu_robot_python/robot.cfg
./build_linux/bin/02_move_j linux/robot.linux.cfg

# C. 用 udev 软链后, 改一行 port
bash linux/setup_udev.sh
sed -i 's|^port.*|port = /dev/fafu_debug_board|' ../fafu_robot_python/robot.cfg
```

`linux/robot.linux.cfg` 唯一的差别是 `port` 注释里推荐 Linux 路径（`/dev/fafu_debug_board`）
和 `gripper_max_torque_raw` 保留（新 SDK 用它）。

## 8. 故障排查

### 8.1 `import fafu_motor` 报 `undefined symbol: _Py...`

Python ABI 不对：编 `.so` 时用的 Python 版本 ≠ import 时用的 Python 版本。

```bash
# 看 .so 到底是给哪个 Python 编的 (文件名里就有)
ls ../fafu_robot_python/fafu_motor*.so
# fafu_motor.cpython-310-x86_64-linux-gnu.so ← 只能被 python3.10 加载

# 看当前 Python
python3 --version                    # 应该跟上面 cpXX 匹配

# 匹配的编一份:
bash linux/build_module_only.sh --python python3.10
```

### 8.1 `Permission denied: '/dev/ttyUSB0'`

三种方案任选（跟老 SDK 一样）：

| 方案 | 命令 | 持久 |
|------|------|------|
| A. udev 规则（推荐） | `bash linux/setup_udev.sh` | 是 |
| B. 加入 dialout 组 | `sudo usermod -aG dialout $USER` + **重新登录** | 是 |
| C. 临时 chmod | `sudo chmod 0666 /dev/ttyUSB0` | 重启失效 |

### 8.1 编译报 `Could NOT find pybind11`

pybind11 没装、装到别的 Python 里、或 build.sh 没找到：

```bash
# 检查当前 python3 里的 pybind11
python3 -c "import pybind11; print(pybind11.__version__, pybind11.get_cmake_dir())"

# 没有就装
python3 -m pip install --user pybind11

# 明确告诉 build.sh 用哪个 Python
bash linux/build.sh --python python3.10
```

### 8.1 `04_servo_j` 跑到中途机器人突然 brake、没有报错

触发了固件看门狗（`watchdog_ms` 内没收到新指令，固件自动刹车）。原因：

- 上层 servoJ 循环卡了 > 100ms（磁盘 IO、GC、调度抖动、Python REPL 等）
- 应对：
  1. 上 `bash linux/run_rt.sh 04_servo_j`（SCHED_FIFO + CPU affinity）
  2. 加大 `ServoOpts.watchdog_ms`（例程里能命令行改，或改 04_servo_j.cpp）
  3. 关掉后台 CPU hog（浏览器 / Docker daemon / IDE）
  4. CPU governor 调 performance

### 8.1 `04_servo_j` 报 `lag too large` 或 `servo_j returned false`

跟踪误差超过 `max_lag_rad`（默认 0.15 rad ≈ 8.6°）。原因：

- 上层送的目标步长太大（`max_step_rad` clamp 之后仍跟不上）
- 电机负载重、电压低
- `max_vel` 给太小
- 应对：先 `move_j` 慢慢热身、调低 `max_vel`、加大 `max_lag_rad`

### 8.1 `cmake configure` 报 `未找到 third_part/serial_cmake`

按理不会发生（SDK 是 self-contained 的），如果发生了说明 vendor 的三方目录被误删。
用底层串口库源码 (`third_part/serial_cmake/`) 覆盖回来即可。

### 8.1 ModemManager 抢串口

Ubuntu 桌面版常见坑：Network Manager 会自动 probe 串口看是不是 3G/4G modem，
把调试板占住导致我们打不开。

```bash
sudo lsof /dev/ttyUSB0                            # 确认是谁占的
sudo systemctl stop ModemManager
sudo systemctl disable ModemManager               # 永久禁用 (不用 3G/4G 的话)
```

## 9. 与 Windows build.bat 的对应关系

| Windows | Linux |
|---------|-------|
| `build.bat` | `bash linux/build.sh` |
| `build.bat Debug` | `bash linux/build.sh --debug` |
| `pip install pybind11` | 由 `install_deps.sh` 帮做 |
| `build\bin\Release\01_smoke.exe` | `build_linux/bin/01_smoke` |
| 串口库静态链接 | 串口库静态链接 |
| VS Console 的 UTF-8 (`/utf-8`) | Linux 终端原生 UTF-8 |

## 10. 卸载

```bash
# 1. 删 build 产出
rm -rf build_linux/

# 2. 删 fafu_robot_python 里的 Linux 扩展
rm -f ../fafu_robot_python/fafu_motor*.so

# 3. 卸载 udev 规则
bash linux/setup_udev.sh --uninstall

# 4. 移除 dialout 组成员 (可选)
sudo gpasswd -d $USER dialout

# 5. 删 env 文件
rm -f ~/.fafu_robot_cpp.env
```
