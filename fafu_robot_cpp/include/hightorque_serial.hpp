// =============================================================================
//  hightorque_serial.hpp
//  C++ 端口 (1:1 移植自 motor_example/hightorque_serial.py)
//
//  调试板 USB 串口驱动
//  通过调试板的 ASCII 命令协议 (`can send <ID> <DATA>\r\n`) 透传 CAN-FD 帧给电机.
//
//  CAN 帧协议 (livelybot FDCAN):
//    CAN ID:  0x8000 | motor_id  (高位=1 表示需要回复)
//    Payload: 由子帧组成, 每个子帧 = cmd + addr + data...
//
//  单位 (int16):
//    位置: 0.0001 转   (5000 = 0.5 圈)
//    速度: 0.00025 转/秒 (400 = 0.1 转/秒)
//    力矩: 原始 raw, 需乘以电机力矩系数得 Nm
// =============================================================================
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace serial { class Serial; }   // 前向声明 (避免在头文件暴露 serial.h)

namespace hightorque {

// ---------------------------------------------------------------------------
//  位置单位制
// ---------------------------------------------------------------------------

enum class PosUnit {
    Turns,    // 圈 (协议原生)
    Radians,  // 弧度
    Degrees,  // 角度
};

// 把任意单位的位置值换算成"圈" (协议原生单位)
double to_turns(double value, PosUnit unit);

// 反向: "圈" -> 任意单位
double from_turns(double turns, PosUnit unit);

// ---------------------------------------------------------------------------
//  常量
// ---------------------------------------------------------------------------

inline constexpr int16_t  NAN_INT16 = static_cast<int16_t>(0x8000);   // -32768
inline constexpr uint32_t NAN_INT32 = 0x80000000u;
inline constexpr uint8_t  PADDING   = 0x50;

// CAN-FD DLC 对应的有效字节数
extern const std::vector<std::size_t> CANFD_DLC_SIZES;

// 力矩系数表 (文档 2.3): tqe_Nm = raw * coeff
extern const std::map<std::string, double> TORQUE_COEFF;

// ---------------------------------------------------------------------------
//  字节工具
// ---------------------------------------------------------------------------

// 按 CAN-FD DLC 规则补 0x50 填充
std::vector<uint8_t> canfd_pad(const std::vector<uint8_t>& data);

// bytes <-> ASCII hex (大写, 无空格)
std::string bytes_to_hex(const std::vector<uint8_t>& data);
std::vector<uint8_t> hex_to_bytes(const std::string& hex);

// ---------------------------------------------------------------------------
//  int16 单位转换 (协议文档 2.6 / 2.7)
// ---------------------------------------------------------------------------

int16_t turns_to_int16(double turns);
double  int16_to_turns(int16_t val);

int16_t rps_to_int16(double rps);
double  int16_to_rps(int16_t val);

int16_t rad_to_int16(double rad);
double  int16_to_rad(int16_t val);

int16_t rad_s_to_int16(double rad_s);
double  int16_to_rad_s(int16_t val);

// ---------------------------------------------------------------------------
//  CAN 帧 payload 构建 (基于 livelybot_fdcan.c)
// ---------------------------------------------------------------------------

std::vector<uint8_t> build_read_state_int16();
std::vector<uint8_t> build_stop_int16();
std::vector<uint8_t> build_brake_int16();
// 只切电机模式 (写 1 字节到 0x00 寄存器). 0x00=停止, 0x0A=位置/速度/力矩, 0x0F=刹车.
std::vector<uint8_t> build_set_mode_int16(uint8_t mode);
std::vector<uint8_t> build_pos_int16(int16_t pos);
std::vector<uint8_t> build_vel_int16(int16_t vel);
std::vector<uint8_t> build_pos_vel_tqe_int16(int16_t pos, int16_t vel, int16_t tqe);
std::vector<uint8_t> build_pos_velmax_acc_int16(int16_t pos, int16_t vel_max, int16_t acc);
// 速度+加速度模式 (协议文档 3.1.9): 等价于 build_pos_velmax_acc_int16(NAN_INT16, vel, acc),
// 即"位置不限制 + 限速 + 限加速度". 单独导出便于上层直接对应协议章节.
std::vector<uint8_t> build_vel_acc_int16(int16_t vel, int16_t acc);
std::vector<uint8_t> build_torque_int16(int16_t tqe);
std::vector<uint8_t> build_voltage_int16(int16_t volt);
std::vector<uint8_t> build_current_int16(int16_t cur);
std::vector<uint8_t> build_pos_vel_tqe_kp_kd_int16(int16_t pos, int16_t vel, int16_t tqe,
                                                    int16_t kp, int16_t kd);

// 一拖多: pos+vel+tqe 模式 (CAN ID = 0x8090, 协议文档 1.3.1.3)
//   pos_arr/vel_arr/tqe_arr: 长度 = max_motor_id, 索引 i 对应 motor_id (i+1)
//   未参与的槽位填 NAN_INT16 (0x8000), 数据末尾固定为 [0x17, 0x01] 查询状态.
std::vector<uint8_t> build_many_pos_vel_tqe_int16(const std::vector<int16_t>& pos_arr,
                                                  const std::vector<int16_t>& vel_arr,
                                                  const std::vector<int16_t>& tqe_arr);

std::vector<uint8_t> build_motor_reset();
std::vector<uint8_t> build_conf_write();
std::vector<uint8_t> build_set_zero();
std::vector<uint8_t> build_read_version();
std::vector<uint8_t> build_set_timeout_int16(int16_t timeout_ms);

// ---------------------------------------------------------------------------
//  电机状态结构 + 解析
// ---------------------------------------------------------------------------

struct MotorState {
    int     id       = 0;
    int     mode     = 0;
    int     fault    = 0;
    double  position = 0.0;   // 圈
    double  velocity = 0.0;   // 转/秒
    double  torque   = 0.0;   // raw int16 (转 Nm 需乘电机系数)

    // 软限位标志: 0=未触发, +1=超出上限, -1=超出下限
    // 当 enable_position_limit() 启用且最近一次 set_pos* 触发限位时被置位
    int     pos_limit_flag = 0;

    std::string to_string() const;
};

// ---------------------------------------------------------------------------
//  CAN 错误码 (调试板 `can status` 回复解析结果)
// ---------------------------------------------------------------------------

enum class CanFault {
    Unknown      = -1,   // 状态无效
    Ok           = 0,    // 正常
    ErrorWarning,        // 错误警告 (位错误/CRC/ACK/格式, 可自恢复)
    ErrorPassive,        // 被动错误 (不发送, 可接收)
    BusOff,              // 总线关闭 (无法收发)
};

struct CanStatus {
    CanFault    fault   = CanFault::Unknown;  // 综合状态
    int         lec     = -1;                 // last error code (协议原值)
    int         tx_err_count = 0;             // 发送错误计数
    int         rx_err_count = 0;             // 接收错误计数
    std::string raw;                          // 原始回复字符串 (调试用)

    bool is_ok() const { return fault == CanFault::Ok; }
    std::string to_string() const;
};

// ---------------------------------------------------------------------------
//  串口枚举
// ---------------------------------------------------------------------------

struct PortInfo {
    std::string port;          // e.g. "COM14" / "/dev/ttyUSB0"
    std::string description;   // 设备描述
    std::string hardware_id;   // e.g. "USB VID:PID=0483:5740"

    // 解析出的 USB VID/PID (大写 hex), 解析失败为 "" / ""
    std::string vid;
    std::string pid;
};

// 列出系统全部串口
std::vector<PortInfo> list_serial_ports();

// 按常见 USB-Serial 调试板 VID 过滤候选
//   ST     0483 (STM32 内置 USB-CDC)
//   FTDI   0403
//   CH340  1A86 (沁恒)
//   SiLabs 10C4
//   Prolific 067B
// 如果你的调试板 VID 不在表中, 可以在 known_vids 里补
std::vector<PortInfo> find_likely_debug_boards(
    const std::vector<std::string>& known_vids = {"0483", "0403", "1A86", "10C4", "067B"});

// 解析电机回复的 int16 状态帧
std::optional<MotorState> parse_motor_state_int16(const std::vector<uint8_t>& can_data);

// ---------------------------------------------------------------------------
//  收发统计 (用于上层展示控制环健康度)
// ---------------------------------------------------------------------------

struct Stats {
    uint64_t tx_frames    = 0;       // 累计发送的 CAN 帧数
    uint64_t rx_frames    = 0;       // 累计收到的 'rcv' 行
    uint64_t rx_parsed    = 0;       // 成功解析为 MotorState 的帧数
    uint64_t rx_dropped   = 0;       // 解析失败 / 串口噪声
    double   last_rx_age_ms = 0.0;   // 最后一帧 RX 距今多少 ms (no rx => -1)
    double   avg_tx_period_ms = 0.0; // 滑动窗口估算的发送周期
    double   max_tx_jitter_ms = 0.0; // 历史最大抖动 (相邻两次 send 的间隔偏差)

    std::string to_string() const;
};

// ---------------------------------------------------------------------------
//  调试板串口驱动
// ---------------------------------------------------------------------------

class HightorqueSerial {
public:
    HightorqueSerial(const std::string& port, uint32_t baudrate = 4'000'000);
    ~HightorqueSerial();

    HightorqueSerial(const HightorqueSerial&)            = delete;
    HightorqueSerial& operator=(const HightorqueSerial&) = delete;

    void close();
    bool is_open() const;

    // ---- 异步收发模式 (高频控制必备) ----
    //
    // 启用后:
    //   1. 内部启动 RX 线程, 持续 read 串口、按 'rcv' 行解析、写入 state_cache_,
    //      唤醒任何在 wait_state() 上等待的调用者.
    //   2. send_can_and_recv_ 不再阻塞等回复, 变成"只发不等" (返回空 RcvList);
    //      所以 set_pos_*/set_many_* 在 async 模式下:
    //        - 发送耗时 ~0.5ms 而非 5-15ms
    //        - 返回的 MotorState 来自 cache (上一次 RX 收到的, 不是本次回执)
    //   3. 上层应该用 get_state(id) / get_states(ids) 读最新缓存.
    //   4. 控制循环里推荐 set_many_pos_vel_tqe (一拖多) + get_states.
    //
    // 不启用时所有行为 100% 兼容旧版.
    void enable_async_rx();
    void disable_async_rx();
    bool is_async_rx() const { return async_rx_enabled_.load(); }
    // Milliseconds since this motor's last real feedback. Returns +infinity
    // when no feedback has ever been received. Unlike the global RX age,
    // healthy motors cannot hide a stale motor here.
    double get_state_age_ms(int motor_id) const;

    // ---- 状态缓存读取 (任何模式下都能用; async 模式更有意义) ----
    std::optional<MotorState> get_state(int motor_id) const;     // = get_cached_state 别名
    std::map<int, MotorState> get_states(const std::vector<int>& motor_ids) const;

    // 阻塞等待 motor_id 的下一次状态更新 (要 RX 线程在跑才有效).
    // 同步模式下退化为 read_motor_state(id, timeout).
    std::optional<MotorState> wait_state(int motor_id, double timeout_s = 0.1);

    // 收发统计
    Stats get_stats() const;
    void  reset_stats();

    // 诊断: 打开后, RX 线程会把"非 rcv"行 (ok / err / can status 输出) 也通过
    // 这个回调通知上层. 用于排查"调试板有没有 ack / 报错"等问题.
    // 设 nullptr 关闭. 同时影响 send_can_and_recv_ 同步路径下的丢弃行为.
    void set_debug_line_handler(std::function<void(const std::string&)> handler);

    // ---- 控制循环顶层封装 ----
    //
    // 把"开 RX → 固定频率 tick → 抖动统计 → Ctrl+Q/abort 检查 → 异常兜底 stop"
    // 这一套机械流程封装起来. 用户只需要在 on_tick 里写业务逻辑:
    //
    //     ht.run_control_loop({ .rate_hz = 100.0, .stop_motor_ids = {1,2,3,4,5,6,7} },
    //         [&](int tick, double period_ms) {
    //             ht.set_many_pos_vel_tqe(cmds);          // 1 帧广播
    //             auto states = ht.get_states({1,...,7}); // 拿 cache
    //             // ... 业务逻辑 ...
    //             return !user_quit;                       // false 则退出
    //         });
    //
    // 行为:
    //   1. 进入前 enable_async_rx (除非已开)
    //   2. 每 tick 用 steady_clock 校准, 抖动写入 stats
    //   3. on_tick 返回 false / 抛异常 / abort_check 返回 true → 退出
    //   4. 退出时按下面规则给 stop_motor_ids 里的电机发 stop:
    //        - on_tick 返回 false (正常完成):  stop_on_finish 控制 (默认 false → 不停, 保持 mode=10)
    //        - abort_check 返回 true (Ctrl+Q): stop_on_abort  控制 (默认 true  → 立刻 stop)
    //        - 业务异常:                         stop_on_abort  控制
    //      理由: 一拖多 0x8090 不能再次切 mode, stop 后 mode=0 ⇒ 下次命令电机不响应.
    //            正常完成时让电机保持 mode=10 + 最后位置 hold, 紧急/异常退出才完全停掉.
    //   5. 函数本身不抛异常 (业务异常在 on_exception 里通知)
    struct ControlLoopOptions {
        double                       rate_hz             = 100.0;
        std::vector<int>             stop_motor_ids;
        std::function<bool()>        abort_check;            // 返回 true → 退出 (e.g. Ctrl+Q)
        std::function<void(const std::exception&)> on_exception;
        bool                         stop_on_finish      = false;   // 正常完成是否 stop (默认 false: 保持 mode=10)
        bool                         stop_on_abort       = true;    // Ctrl+Q / 异常时是否 stop (默认 true)
    };
    using ControlTickFn = std::function<bool(int tick, double period_ms)>;

    // 阻塞运行直到结束. 返回值: 0=正常退出, 1=abort_check, 2=异常.
    int run_control_loop(const ControlLoopOptions& opt, ControlTickFn on_tick);

    // -- 调试板命令 --
    std::string can_status();          // 返回原始字符串
    CanStatus   read_can_status();     // 解析后的结构体
    std::string can_config();

    // -- 软限位 (可选, 启用后所有 set_pos* 自动 clamp 并设置 MotorState::pos_limit_flag) --
    void enable_position_limit(int motor_id, double lo, double hi,
                               PosUnit unit = PosUnit::Turns);
    void disable_position_limit(int motor_id);
    void clear_all_position_limits();
    // 返回 true=已启用, 同时填充 lo_turns/hi_turns (圈)
    bool get_position_limit_turns(int motor_id, double& lo_turns, double& hi_turns) const;

    // -- 电机控制 --
    std::optional<MotorState> read_motor_state(int motor_id, double timeout_s = 0.5);
    std::optional<MotorState> stop(int motor_id);
    std::optional<MotorState> brake(int motor_id);

    // 切电机模式 (写 0x00 寄存器), 不带 pos/vel/tqe.
    //   mode: 0x00=停止, 0x0A(=10)=位置/速度/力矩, 0x0F=刹车
    // 用法: 一拖多 set_many_pos_vel_tqe 之前必须把所有电机切到 mode=10,
    //       否则电机仍在 mode=0 (停止) 不响应控制命令.
    std::optional<MotorState> set_motor_mode(int motor_id, uint8_t mode);

    // 注意: set_position 使用最大速度/最大力矩, 在机械臂上动作激烈, 推荐用 set_pos_vel_acc.
    std::optional<MotorState> set_position(int motor_id, double pos,
                                           PosUnit unit = PosUnit::Turns);
    std::optional<MotorState> set_velocity(int motor_id, double vel_rps);
    std::optional<MotorState> set_pos_vel_tqe(int motor_id, double pos,
                                              double vel_rps, int tqe_raw,
                                              PosUnit unit = PosUnit::Turns);
    // 推荐的位置控制方式 (梯形): 位置 + 最大速度 + 加速度
    //   pos:           默认圈, 可指定 unit 为 Radians / Degrees
    //   vel_max_rps:   转/秒
    //   acc_rpss:      转/秒^2
    std::optional<MotorState> set_pos_vel_acc(int motor_id, double pos,
                                              double vel_max_rps, double acc_rpss,
                                              PosUnit unit = PosUnit::Turns);

    // 速度 + 加速度模式 (协议文档 3.1.9):
    //   以 acc_rpss 的恒定加速度加速到 vel_rps, 然后保持该速度持续转动.
    //   与 set_velocity 的区别: set_velocity 用"最大加速度", 这里可以指定加速度上限,
    //   适合需要平滑启停的场景 (例如调试齿轮箱时不希望瞬间冲击).
    std::optional<MotorState> set_vel_acc(int motor_id, double vel_rps, double acc_rpss);

    // -- 一拖多 (单帧多电机) --
    //
    // 协议: ID 0x8090, 每电机槽位 6 字节 (pos+vel+tqe int16), 末尾 2 字节 [0x17,0x01] 查询状态.
    // 一帧最多 ~10 个电机 (受 CAN-FD 64 字节单帧上限).
    //
    // 优势:
    //   - 1 次 USB 串口往返 vs N 次, 总耗时 ~3-5 ms (vs 10-20 ms 串行)
    //   - 所有电机命令在同一 CAN-FD 帧, 时间一致性 100%
    //   - 控制频率可推到 100 Hz 稳定
    //
    // 限位: 每个 cmd 都会经过 apply_position_limit_(motor_id), pos_limit_flag 写入回执
    struct ManyMotorCmd {
        int    motor_id;     // 1..max_motor_id (槽位索引依据)
        double pos;          // 默认圈, 用 unit 换算
        double vel_rps;      // 转/秒
        int    tqe_raw = 0;  // 原始 int16 (与 set_pos_vel_tqe 对称); 给 0 = 不限力矩
    };

    // 同步版: 发出去 + 等所有回复, 返回 motor_id -> 状态.
    // 没回复的 motor 不在 map 里. 后续会有异步版 (run_control_loop / 后台 RX 线程).
    //   max_motor_id = 0 时自动取 cmds 里最大 ID
    //   timeout_s    默认 50ms (够 10 个电机回复, 100Hz 周期内)
    std::map<int, MotorState> set_many_pos_vel_tqe(
        const std::vector<ManyMotorCmd>& cmds,
        PosUnit pos_unit     = PosUnit::Turns,
        int     max_motor_id = 0,
        double  timeout_s    = 0.05);

    // ----- "Partial broadcast" 控制循环热路径专用 (servoJ / teleop) -----
    //
    // 对 active_ids 列出的电机发 (pos, vel, tqe) 主动控制, 同时对 hold_ids
    // 列出的电机用各自的 ★ 缓存当前位置 ★ 作为 target (vel=0). 这样:
    //   * watchdog 不会因为某些电机被"漏发"而触发刹车
    //   * 上层 (Python servo_j / teleop loop) 不需要每 tick 重新拼 cmd vector
    //   * Python binding 可以走 numpy buffer 一次 memcpy, 比 list of
    //     ManyMotorCmd 对象 marshalling 快 100x (~5us vs ~200us / tick)
    //
    // 参数:
    //   active_ids/active_pos/active_vel  同长度 N. 默认按 PosUnit::Turns 解释 pos,
    //                                     vel 单位永远是 turns/s.
    //   tqe_raw                           原始 int16 力矩上限, 应用于 active 电机.
    //                                     0 = 用电机默认最大力矩 (NAN_INT16 语义).
    //   hold_ids                          这些电机用 get_cached_state(id) 拿到的
    //                                     position 作 target, vel=0, tqe 同上.
    //                                     cache 还没数据的电机会被静默跳过.
    //   max_motor_id                      帧 slot 数. 0 = 自动按出现的最大 id.
    //   timeout_s                         回包等待上限. async_rx 模式下传 0 可
    //                                     立即返回 (不等回包), 返回空 map.
    std::map<int, MotorState> set_many_pos_vel_tqe_partial(
        const std::vector<int>&    active_ids,
        const std::vector<double>& active_pos,
        const std::vector<double>& active_vel,
        int16_t                    tqe_raw,
        const std::vector<int>&    hold_ids,
        PosUnit pos_unit     = PosUnit::Turns,
        int     max_motor_id = 0,
        double  timeout_s    = 0.0);

    // 一拖多 MIT/PD 诊断路径 (CAN ID 0x8093). 与协议文档的
    // motor_many_pos_vel_tqe_kp_kd_2 格式一致:
    // [pos, vel, tqe, kp, kd] * motor_slot + [0x17, 0x01].
    // tqe/kp/kd 均为 raw int16, 不做 Nm 或增益单位换算.
    std::map<int, MotorState> set_many_pos_vel_tqe_kp_kd_partial(
        const std::vector<int>&    active_ids,
        const std::vector<double>& active_pos,
        const std::vector<double>& active_vel,
        int16_t                    tqe_raw,
        int16_t                    kp_raw,
        int16_t                    kd_raw,
        PosUnit pos_unit     = PosUnit::Turns,
        int     max_motor_id = 0,
        double  timeout_s    = 0.0);

    // ★ 每关节独立 tqe/kp/kd 的一拖多 MIT (CAN ID 0x8093). 这是重力补偿 /
    //   拖动示教 / 回放的正式底层通道 (官方 pos_vel_tqe_kp_kd 的一拖多等价).
    //   motor_ids/pos/vel_rps/tqe_raw/kp_raw/kd_raw 同长度 N (逐关节).
    //   tqe/kp/kd 为原始 int16 (上层负责 Nm/增益换算); pos 走软限位, vel=turns/s.
    //   单帧最多 6 个电机 (每电机 10 字节, >64B build 函数抛错).
    std::map<int, MotorState> set_many_mit(
        const std::vector<int>&    motor_ids,
        const std::vector<double>& pos,
        const std::vector<double>& vel_rps,
        const std::vector<int>&    tqe_raw,
        const std::vector<int>&    kp_raw,
        const std::vector<int>&    kd_raw,
        PosUnit pos_unit     = PosUnit::Turns,
        int     max_motor_id = 0,
        double  timeout_s    = 0.05);

    // motor_model 用于查 TORQUE_COEFF 表换算 Nm; 留空则当作系数 1.0
    std::optional<MotorState> set_torque(int motor_id, double tqe_nm,
                                         const std::string& motor_model = "");
    std::optional<MotorState> set_voltage(int motor_id, double voltage_v);
    std::optional<MotorState> set_current(int motor_id, double current_a);

    std::optional<MotorState> set_pos_vel_tqe_kp_kd(int motor_id,
                                                    double pos, double vel_rps,
                                                    double tqe_nm, double kp, double kd,
                                                    const std::string& motor_model = "",
                                                    PosUnit unit = PosUnit::Turns);

    std::string reset_zero(int motor_id);
    std::string save_config(int motor_id);
    std::string motor_reset(int motor_id);
    std::optional<std::string> read_version(int motor_id);
    std::string set_timeout(int motor_id, int16_t timeout_ms);

    // -- 后台状态轮询 (可选) --
    //
    // start_state_polling: 启动后台线程, 以 rate_hz 频率轮询所有 motor_ids 的状态,
    //   缓存最新一次成功的 MotorState, 上层用 get_cached_state(id) 读取.
    //   on_update 是可选回调, 每轮全部电机刷新完成后调用 (传入读到的 ID 列表).
    // 同一时间只能有一个轮询线程在跑; 重复调用会先停掉旧线程.
    void start_state_polling(const std::vector<int>& motor_ids,
                             double rate_hz = 50.0,
                             std::function<void(const std::vector<int>&)> on_update = {});
    void stop_state_polling();
    bool is_polling() const;

    // 读取后台线程缓存的最新状态; 没拿到过就返回 std::nullopt.
    std::optional<MotorState> get_cached_state(int motor_id) const;

private:
    using RcvList = std::vector<std::pair<int, std::vector<uint8_t>>>;

    void        serial_write_(const std::vector<uint8_t>& data, int retries = 3);
    std::string send_cmd_(const std::string& cmd, double timeout_s = 0.5);
    // expected_replies: 期望收到几条 'rcv' 行后才退出 (除非超时);
    //   单发场景下默认 1, 一拖多 N 个电机时传 N.
    RcvList     send_can_and_recv_(int can_id, const std::vector<uint8_t>& data,
                                   double timeout_s     = 0.5,
                                   int    expected_replies = 1);
    static RcvList parse_rcv_lines_(const std::string& text);

    // 限位检查; 返回 (clamped_pos_turns, flag): flag = 0/+1/-1
    std::pair<double, int> apply_position_limit_(int motor_id, double pos_turns) const;

    // RX 线程主循环: 阻塞 read 串口 → 行拼接 → dispatch (parse + cache + cv notify)
    void rx_loop_();
    // 把一行 'rcv 0xID HEX...' 解析后写 cache + 通知 wait_state
    void dispatch_rcv_line_(const std::string& line);
    // TX 计数 + 周期/抖动统计 (由 send_can_and_recv_ 调用)
    void note_tx_();

    std::unique_ptr<serial::Serial> ser_;

    // ---- 锁分层 (异步模式必要) ----
    // tx_mtx_:   保护对 ser_->write 的并发访问 (任何模式都用)
    // cache_mtx_: 保护 state_cache_ + update_seq_ + cv_
    // limits_mtx_/stats_mtx_: 各自独立, 见下
    std::mutex                      tx_mtx_;
    mutable std::mutex              cache_mtx_;
    std::condition_variable         cache_cv_;
    std::map<int, double>           state_update_time_;
    std::map<int, MotorState>       state_cache_;
    std::map<int, uint64_t>         update_seq_;        // motor_id -> 单调递增序号

    // 软限位表 (圈)
    struct Range { double lo = 0.0; double hi = 0.0; };
    std::map<int, Range>            limits_;
    mutable std::mutex              limits_mtx_;

    // 后台 *轮询* 线程 (老接口, 单线程从串口轮询)
    std::thread                     poll_thread_;
    std::atomic<bool>               poll_running_{false};

    // 后台 *接收* 线程 (新接口, 异步 RX)
    std::thread                     rx_thread_;
    std::atomic<bool>               rx_running_{false};
    std::atomic<bool>               async_rx_enabled_{false};
    std::string                     rx_buffer_;          // 行拼接 (RX 线程独占)

    // Stats
    mutable std::mutex              stats_mtx_;
    Stats                           stats_;
    double                          last_tx_time_   = 0.0;     // 用于 jitter
    double                          last_rx_time_   = -1.0;    // -1 = 还没 RX
    static constexpr int            kTxWindow_      = 32;
    std::vector<double>             tx_periods_;               // 滑动窗

    // 诊断回调
    mutable std::mutex                              dbg_mtx_;
    std::function<void(const std::string&)>         debug_line_handler_;
};

} // namespace hightorque
