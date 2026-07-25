# fafu_robot_sdk (原生 C++) — `sdk/`

Fafu 机械臂的 **原生 C++ SDK**, 从 `fafu_robot_python/fafu_robot_controller.py`
1:1 移植过来.

> 范围: **关节空间** 控制 + 夹爪 (含力控抓取). **不包含** 笛卡尔运动
> (move_p / move_l)、URDF / IK / Pinocchio 等. 笛卡尔规划留给 Python 侧
> (用 wrs 等库做完之后, 把关节路径喂给本 SDK 执行).

> 线程安全、多进程端口独占与多机械臂边界见
> [P0_THREADING_AND_MULTI_ARM.md](../docs/P0_THREADING_AND_MULTI_ARM.md)。

---

## 目录

```
sdk/
├── README.md                                       ← 你正在看的文件
├── CMakeLists.txt                                  ← 由父 CMakeLists 通过 add_subdirectory(sdk) 引入
│
├── include/
│   └── fafu/
│       └── fafu_robot_controller.hpp               ← 唯一公开头, 用户写 `#include "fafu/fafu_robot_controller.hpp"`
│
├── src/
│   └── fafu_robot_controller.cpp                   ← 实现 (依赖 ../include/hightorque_serial.hpp + robot_config.hpp)
│
└── examples/                                       ← 四个独立 .exe, 演示典型用法
    ├── 01_smoke.cpp        连接 / 列电机 / 读一次状态 / 干净断开 (不上电, 安全)
    ├── 02_move_j.cpp       go_home + 小幅多关节运动 (offline S-curve)   (会让机械臂运动)
    ├── 03_gripper.cpp      open / close / grasp / release                (会让夹爪闭合)
    └── 04_servo_j.cpp      servoJ 100Hz sin 跟踪 + 看门狗 / 步长 / lag    (会让关节连续运动)
```

构建产物 (Release, Windows):

```
build/sdk/Release/fafu_robot_sdk.lib                ← 静态库, 上层可以链接它做自己的控制程序
build/bin/Release/01_smoke.exe                       ← + robot.cfg（串口库已静态链接）
build/bin/Release/02_move_j.exe                      ← + robot.cfg（串口库已静态链接）
build/bin/Release/03_gripper.exe                     ← + robot.cfg（串口库已静态链接）
build/bin/Release/04_servo_j.exe                     ← + robot.cfg（串口库已静态链接）
```

---

## 怎么 build

跟 pybind11 模块共用顶层 `fafu_robot_cpp/build.bat`. 一行命令编全部:

```bat
cd fafu_robot_sdk\fafu_robot_cpp
build.bat
```

只想编 SDK / 不要 pybind11 (不依赖 Python 也能编):

```bash
cd fafu_robot_sdk/fafu_robot_cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release ^
      -DFAFU_BUILD_PYBIND11_MODULE=OFF ^
      -DFAFU_BUILD_NATIVE_SDK=ON ^
      -DFAFU_BUILD_SDK_EXAMPLES=ON
cmake --build build --config Release -j
```

Ubuntu / Linux 一键 (装依赖 + 编 SDK + 例程):

```bash
bash ../linux/install_deps.sh --no-pybind11    # 不要 pybind11 时加这个
bash ../linux/build.sh --no-python             # 只编原生 SDK + 4 个例程
./../build_linux/bin/01_smoke                  # 最小连通性测试

# servoJ (04_servo_j) 强烈建议配 SCHED_FIFO 减抖:
sudo setcap cap_sys_nice=eip ../build_linux/bin/04_servo_j
bash ../linux/run_rt.sh 04_servo_j
```

完整说明见 [`../linux/README.md`](../linux/README.md).

可选 CMake 选项:

| 选项 | 默认 | 含义 |
|---|---|---|
| `FAFU_BUILD_PYBIND11_MODULE` | ON | 是否编 `fafu_motor.pyd` (依赖 pybind11) |
| `FAFU_BUILD_NATIVE_SDK`      | ON | 是否编 `fafu_robot_sdk.lib` |
| `FAFU_BUILD_SDK_EXAMPLES`    | ON | 是否编 `01_smoke / 02_move_j / 03_gripper / 04_servo_j` 例程 |

---

## API 速览

完整签名见 [`include/fafu/fafu_robot_controller.hpp`](include/fafu/fafu_robot_controller.hpp).
跟 Python 侧 `fafu_robot_controller.py` 一一对应:

| 类别 | C++                                                | Python                          |
|---|---|---|
| 构造  | `FafuRobotController(cfg_path, Options)`             | `FafuRobotController(cfg_path, **kwargs)` |
| 电源  | `enable / disable / brake`                           | 同名                            |
| 关节  | `move_j(angles, MoveOpts)`                           | `move_j(angles, speed=, block=, is_radians=, tolerance=)` |
|      | `go_home(speed, block)`                              | 同名                            |
| Servo | `servo_start(ServoOpts)` / `servo_j(angles)` / `servo_end(mode)` | (C++ 独有, Python 侧暂未实现) |
| 状态  | `get_joint_values(prefer_cache)`                     | 同名                            |
|      | `get_joint_velocities(prefer_cache)`                 | 同名                            |
|      | `get_motor_states(prefer_cache)`                     | 同名                            |
| 夹爪  | `gripper_control(angle, GripperOpts) -> GraspResult?`| 同名 (signature 一致)           |
|      | `open_gripper / close_gripper / release`             | 同名                            |
|      | `grasp(GraspOpts) -> GraspResult`                    | 同名                            |
|      | `get_gripper_state() -> MotorState`                  | 同名                            |
| 限位  | `set_limit / get_limit / disable_limit / clear_limits`| 同名                           |
| 急停  | `emergency_stop / resume`                            | 同名                            |
| 杂项  | `get_can_status / reset_zero(id, confirm)`           | 同名                            |
| 关闭  | `close_connection(joint_release, gripper_release)`   | 同名 (枚举 vs 字符串)           |

C++ 的 `MoveOpts / GripperOpts / GraspOpts` struct 等价于 Python 的关键字
参数, 让函数签名更短.

---

## 最小用例

```cpp
#include "fafu/fafu_robot_controller.hpp"
#include <iostream>

int main() {
    using namespace fafu_robot;

    FafuRobotController::Options opts;
    opts.has_gripper      = true;
    opts.gripper_motor_id = 7;
    // opts.port / opts.baudrate 都留默认, 用 cfg 里的

    FafuRobotController arm("robot.cfg", opts);

    arm.go_home(/*speed=*/15);              // 阻塞 S-curve 归零

    arm.open_gripper();                     // 走到夹爪软限位上限
    auto r = arm.grasp({.force_threshold = 500, .vel = 0.15});
    if (r.grasped) std::cout << "got it: " << r.to_string() << "\n";
    else           std::cout << "miss: "   << r.reason     << "\n";

    arm.release();
    // 析构里自动 close_connection(joints=Stop, gripper=Brake)
    return 0;
}
```

链接:

```cmake
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE fafu_robot_sdk)
```

只要 `fafu_robot_sdk/` 在你的 build 树里, 父 CMakeLists 已经把它的 include
路径用 `PUBLIC` 暴露出来了, 你直接 link 就能 include.

---

## 电机模式 (mode 字节)

调试板协议层的电机模式 **只有 3 种**, 写到电机寄存器 `0x00`:

| 常量 (`fafu_robot::`) | mode 字节 | 含义 | 行为 |
|---|---|---|---|
| `MODE_STOP`     | `0x00` | 停止 / 自由 | PWM 关断, 电机可手推, 不耗电 |
| `MODE_ACTIVE`   | `0x0A` | 主动控制   | 闭环工作, 接受位 / 速 / 力 / 电压 / 电流 / MIT 各种命令 |
| `MODE_BRAKE`    | `0x0F` | 短路刹车   | 三相短路, 抗扭, 不出力, 不耗电 |

> `MODE_POSITION` 是 `MODE_ACTIVE` 的旧别名, 兼容用. 实际上 "位置 / 速度 / 力矩 / MIT"
> 都共用同一个 `0x0A` mode, 区别在子帧字段不在 mode 字节.

应用层在 `MODE_ACTIVE` 下能调出来的控制方式有 6+ 种 (都由底层
`hightorque::HightorqueSerial` 提供, 本 SDK 用到的画 ★):

| 控制方式 | 底层 API | 本 SDK 用到的地方 |
|---|---|---|
| pos + vel + max_torque  | `set_pos_vel_tqe`       | ★ `gripper_control / grasp` 力限抓取 |
| pos + vel + acceleration | `set_pos_vel_acc`       | ★ `move_j` 单关节段, `gripper_control` 无力限路径 |
| 多电机 pos+vel+tqe 一拖多 | `set_many_pos_vel_tqe`  | ★ `move_j` 多关节同步, `servo_j` |
| pos+vel+tqe+kp+kd (MIT) | `set_pos_vel_tqe_kp_kd` | — (留底层 API, 未上层暴露) |
| 纯速度 / 纯力矩 / 纯电流 / 纯电压 | `set_velocity / set_torque / ...` | — |

---

## servoJ (online streaming)

跟 `move_j` 的离线 S-curve 阻塞规划不同, `servo_j` 是给"上层在线规划 / teleop /
VR / 视觉伺服"用的连续 streaming 接口. 典型上层周期 **100-200 Hz**.

### 四道安全防线

| # | 防线 | 由什么实现 | 默认值 |
|---|---|---|---|
| 1 | **固件看门狗 (硬保险)** | `set_timeout(mid, watchdog_ms)` 写到调试板, 固件 watchdog_ms 内没新指令就自动停 | 100 ms |
| 2 | 单步限幅 | `target - last_target` 每分量 clamp 到 ±`max_step_rad` | 0.05 rad (~2.9°) |
| 3 | 跟踪误差 | `measured - target` 超 `max_lag_rad` 返回 false | 0.2 rad (~11.5°) |
| 4 | 软限位 | 复用 `enable_position_limit` (底层 `set_many_pos_vel_tqe` 自动 clamp) | 取自 cfg |

防线 1 是最关键的: **上位机崩溃 / Ctrl-C / USB 拔出, 电机会在 watchdog_ms 内
自动 brake, 不会保持最后一帧目标继续冲到限位.** 强烈建议永远不要把
`watchdog_ms` 设到 0.

### 用法骨架

```cpp
arm.servo_start({
    .watchdog_ms   = 100,    // ms, 固件级超时
    .max_vel       = 1.5,    // rad/s, 每关节最大速度
    .max_step_rad  = 0.02,   // rad, 单步最大跳变 (~1.15deg)
    .max_lag_rad   = 0.15,   // rad, 跟踪误差上限 (~8.6deg)
    .is_radians    = true,
});

auto next_tick = std::chrono::steady_clock::now();
const auto period = std::chrono::milliseconds(10);   // 100 Hz

while (!quit) {
    auto target = compute_next_target();    // 上层 IK / 规划 / teleop
    if (!arm.servo_j(target)) {
        // 长度错 / lag 超限 / NaN / 通信失败 — 排查
        break;
    }
    next_tick += period;
    std::this_thread::sleep_until(next_tick);
}

arm.servo_end(ReleaseMode::Brake);   // 清看门狗 + brake
```

完整可跑的例程见 `examples/04_servo_j.cpp`. 它跑一个 0.5 Hz / ±10° 的 sin 轨迹,
持续 6 秒 (可以命令行改时长). 中途 `Ctrl+C` 会触发 `servo_end + Brake`; 即使
**暴力 kill 进程**, 100 ms 后固件也会自己 brake.

### servoJ 跟 move_j 怎么选

| 场景 | 用什么 |
|---|---|
| 一次性走到某个固定目标 (示教点 / 离线规划终点) | `move_j` (阻塞, S-curve, 一行就够) |
| 实时跟随上层轨迹 (VR teleop / 视觉伺服 / 在线 IK / wrs 规划流式输出) | `servo_j` (非阻塞, 自带看门狗) |
| 多段路点 (上层把路径切片后一段段送) | 都行, 单段较长用 move_j, 单段 < 100ms 用 servo_j |

---

## 单位 / 约定

- **关节角**: 默认 **弧度** (`is_radians = true`); `degree` 模式传 `false`
  即可.
- **角速度**: `get_joint_velocities()` 返回 rad/s.
- **夹爪 angle**: 跟关节同步, 默认弧度. Fafu 夹爪是 *角度越大越开*
  (软限位上限 = 张开).
- **力矩 (raw)**: int16, 跟 `MotorState.torque` 同步. `GraspResult.peak_torque_raw`
  也是这个单位. 标定方法见 `fafu_robot_python/tests/test_fafu_grasp_calibrate.py`
  (Python 侧的标定脚本, 标出来的阈值可以直接拿到 C++ 这边用).
- **闭合速度**: turns/s. 默认 0.15 (~ 54 deg/s, 比 open_gripper 慢, 接触更柔).

---

## 跟 Python 侧的关系

```
                            (用户代码)
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                     ▼
   Python: FafuRobotController        C++: fafu_robot::FafuRobotController
       (fafu_robot_python/)              (fafu_robot_cpp/sdk/)
              │                                     │
              │   import fafu_motor             │   #include "fafu/..."
              ▼                                     ▼
       fafu_motor.pyd                   fafu_robot_sdk.lib
        (pybind11)                          (C++ 静态库)
              │                                     │
              └──────────────┬──────────────────────┘
                             ▼
                hightorque::HightorqueSerial   (调试板协议层, 同一份 vendored 源码)
                             │
                             ▼
                  serial_cmake (.dll)         (跨平台串口库, 同一份)
                             │
                             ▼
                          USB / 调试板 / 电机
```

两侧调用同一个底层 `HightorqueSerial`, 所以**软限位 / 状态缓存 / 异步 RX**
等行为在两侧 100% 一致. 你可以混用 (例如用 Python 做高层规划, 把关键
关节路径段交给 C++ 程序实时跑) 而不用担心 fight 同一根串口.

> 不过同一台机器上 **同一时刻** 只能有一个进程占串口. 如果 Python 在跑,
> C++ exe 启动会 fail (port busy), 反之亦然.

---

## 排错

| 现象 | 处理 |
|---|---|
| `[ERROR] 加载配置失败 robot.cfg: 无法打开配置文件` | exe 工作目录里没 cfg. POST_BUILD 已经 copy 到了 exe 同目录, cd 进去再跑 |
| `[ERROR] auto: 未找到候选 USB 调试板, 检查 USB / 驱动` | 没插调试板 / 驱动没装 / VID 不在白名单. 把 cfg 里 `port = COM14` 写死也行 |
| `[ERROR] 通信预检失败: 电机 N 不响应 (timeout 300ms)` | 电机没上电 / motor_id 配错 / CAN 总线断 |
| 链接报 `LNK2019 hightorque::*` | 没 link `hightorque_serial_debug`. 你应该 link `fafu_robot_sdk`, 它会传递依赖 |
| 串口打开提示 already in use | 同一端口已被另一个 SDK 进程占用；为每个实例配置不同端口 |
| `servo_j` 一直返回 false / lag 报警 | 上层周期太长 / `max_vel` 给太小 / 机械负载过大 / 改大 `max_lag_rad` 或先 `move_j` 慢慢热身 |
| servoJ 中途机器突然 brake, 没有 Ctrl+C | 触发了固件看门狗 — 上层规划循环卡了 (磁盘 IO / GC / 调度抖动) 超过 `watchdog_ms`. 调大 watchdog 或加 RT 优先级 |
