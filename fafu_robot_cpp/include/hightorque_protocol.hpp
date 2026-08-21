// =============================================================================
//  hightorque_protocol.hpp
//  livelybot FDCAN 协议层: 纯数据, 不碰串口, 不碰线程.
//
//  这一层只做三件事:
//    1. 单位换算 (圈/弧度/角度 <-> 协议原生 int16)
//    2. 把控制意图打包成 CAN-FD payload 字节 (build_*)
//    3. 把电机回包字节解析成 MotorState (parse_motor_state_int16)
//
//  拆出来的理由: 这些函数全是"输入字节 -> 输出字节"的纯函数, 不需要硬件就能
//  测试和审查, 而 hightorque_serial.cpp 里混着串口 IO / 线程 / 锁, 两者的
//  修改频率和风险等级完全不同.
//
//  CAN 帧协议:
//    CAN ID:  0x8000 | motor_id  (高位=1 表示需要回复)
//    Payload: 由子帧组成, 每个子帧 = cmd + addr + data...
//             cmd 高 4 位 = 操作 (0=写 int8, 1=写 int16, 2=回复, 5=读 int16 ...),
//             低 2 位 = 寄存器个数, 0 表示变长 (下一字节给个数).
//
//  单位 (int16):
//    位置: 0.0001 转     (5000 = 0.5 圈)
//    速度: 0.00025 转/秒 (400  = 0.1 转/秒)
//    加速度: 0.001 转/秒²
//    力矩: 原始 raw, 实际标度 = 电机系数 * 0.01 Nm/count
//
//  ★ 本头文件被 hightorque_serial.hpp 包含, 外部调用方不需要直接 include.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

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

// 力矩系数表: 实际标度 = coeff * 0.01 Nm/count (见 hightorque_protocol.cpp 注释)
extern const std::map<std::string, double> TORQUE_COEFF;

// ---------------------------------------------------------------------------
//  字节工具
// ---------------------------------------------------------------------------

// 按 CAN-FD DLC 规则补 0x50 填充
std::vector<uint8_t> canfd_pad(const std::vector<uint8_t>& data);

// bytes <-> ASCII hex (大写, 无空格)
std::string bytes_to_hex(const std::vector<uint8_t>& data);
std::vector<uint8_t> hex_to_bytes(const std::string& hex);

namespace detail {

// 饱和截断到 int16. 协议层和驱动层都要用 (驱动层把 Nm / 电压 / 电流换算成 raw
// 时也得夹住), 所以放在这里共享而不是各自复制一份.
template <typename T>
inline int16_t saturate_to_i16(T value) {
    if (value >  32767) return  32767;
    if (value < -32768) return -32768;
    return static_cast<int16_t>(value);
}

}  // namespace detail

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
//   每电机 6 字节, 单帧最多 10 个电机, 超了抛 std::invalid_argument.
std::vector<uint8_t> build_many_pos_vel_tqe_int16(const std::vector<int16_t>& pos_arr,
                                                  const std::vector<int16_t>& vel_arr,
                                                  const std::vector<int16_t>& tqe_arr);

// 一拖多: MIT/PD 模式 (CAN ID = 0x8093).
//   每电机 10 字节 (pos/vel/tqe/kp/kd), 单帧最多 6 个电机, 超了抛异常.
std::vector<uint8_t> build_many_pos_vel_tqe_kp_kd_int16(
        const std::vector<int16_t>& pos_arr,
        const std::vector<int16_t>& vel_arr,
        const std::vector<int16_t>& tqe_arr,
        const std::vector<int16_t>& kp_arr,
        const std::vector<int16_t>& kd_arr);

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

    // 本帧解析完成的时刻 (steady_clock 秒). 0 = 未填 (例如手工构造的空状态).
    //
    // Stats::last_rx_age_ms 是**全局**的: 只要还有任意一个电机在回帧就算新鲜,
    // 所以「J5 掉线但 J1-J4 正常」这种情况它发现不了. 每帧带上时间戳后, 上层
    // 可以按关节判断新鲜度 (见 HightorqueSerial::state_age_ms).
    //
    // 由驱动层填写 (协议层是纯函数, 不取时钟).
    double  rx_time_s = 0.0;

    std::string to_string() const;
};

// 解析电机回复的 int16 状态帧. 一个寄存器都没解析出来时返回 nullopt
// (而不是全零的 MotorState —— 那会让上层以为电机瞬移到原点).
std::optional<MotorState> parse_motor_state_int16(const std::vector<uint8_t>& can_data);

}  // namespace hightorque
