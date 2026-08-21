// =============================================================================
//  hightorque_protocol.cpp
//  FDCAN 协议层实现: 单位换算 / 帧构建 / 帧解析. 纯函数, 不碰串口和线程.
//
//  从 hightorque_serial.cpp 原样搬迁而来 (逐行拷贝, 未改逻辑), 目的是把
//  "输入字节 -> 输出字节" 的可测试部分与串口 IO / 线程 / 锁隔开.
// =============================================================================

#include "hightorque_protocol.hpp"

#include <cstring>      // std::memcpy (read_le_f32)
#include <initializer_list>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace hightorque {

// saturate_to_i16 现在住在 hightorque_protocol.hpp 的 detail 里 (驱动层也要用),
// 这里拉进当前命名空间, 保持原有的非限定调用写法不变.
using detail::saturate_to_i16;

// ---------------------------------------------------------------------------
//  位置单位制 (前置)
// ---------------------------------------------------------------------------

namespace { constexpr double kPi_ = 3.14159265358979323846; }

double to_turns(double value, PosUnit unit) {
    switch (unit) {
        case PosUnit::Turns:   return value;
        case PosUnit::Radians: return value / (2.0 * kPi_);
        case PosUnit::Degrees: return value / 360.0;
    }
    return value;
}

double from_turns(double turns, PosUnit unit) {
    switch (unit) {
        case PosUnit::Turns:   return turns;
        case PosUnit::Radians: return turns * 2.0 * kPi_;
        case PosUnit::Degrees: return turns * 360.0;
    }
    return turns;
}

// ---------------------------------------------------------------------------
//  常量
// ---------------------------------------------------------------------------

const std::vector<std::size_t> CANFD_DLC_SIZES = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64
};

// 力矩系数表 (每型号一个 coeff; 实际标度 = coeff * 0.01 Nm/count, 见 set_torque).
//
// 协议文档给的换算是  真实扭矩 = k * tqe + d :
//   tqe 是帧里的原始 int16, k 就是本表的 coeff * 0.01;
//   d 是电机自身静摩擦的补偿项, fdcan_h730 自 v3.0.5 起不再加 (固件表里 d 恒为
//   0.0f), 所以这里只存 k。
//
// ★ 权威来源是**固件工程 fdcan_h730** 的 motor_tqe_adj 表 —— 协议文档原话
//   "扭矩系数以 fdcan_h730 中的 motor_tqe_adj 为准"。
//
// ⚠ 存在同名的第二张表: 主机侧 ROS SDK livelybot_serial/include/hardware/motor.h
//   里也叫 motor_tqe_adj, 但和固件表大面积对不上 (4438_30 是 0.5256 而固件是
//   0.64; 60BM_35 是 0.7942 而固件是 0.64), 而且完全没有 "新名称" 那一批型号。
//   本表已整体倒向固件表, 不要再拿主机 SDK 的数值来"订正"这里。
const std::map<std::string, double> TORQUE_COEFF = {
    // ---- fdcan_h730 "新名称" 型号 ----
    {"M3508_02", 0.37},
    {"M3516_02", 0.37},
    {"M3532_02", 0.37},
    {"M4530_02", 0.62},
    {"M5009_02", 0.71},
    {"M5036_02", 0.67},     // ★ Fafu arm J1, J4
    {"M6036_02", 0.66},     // ★ Fafu arm J2, J3
    {"M7033_04", 0.84},
    {"M7535_02", 0.73},
    // 换 8353 编码器的版本, 系数与同名基础型号不同, 别混用
    {"M3532_02_8353", 0.61},
    {"M4530_02_8353", 0.64},
    {"M5036_02_8353", 0.70},

    // ---- fdcan_h730 "旧名称" 型号 ----
    {"M3536_32", 0.35},
    {"M4438_30", 0.64},     // ★ Fafu arm J5, J6, J7
    {"M4438_32", 0.64},
    {"M5043_20", 0.96},
    {"M5047_36", 0.64},
    {"M6056_36", 0.66},
    {"M7256_35", 0.66},
    {"M60BM_35", 0.64},

    // ---- 固件表里的兜底档 ----
    {"MGENERAL", 0.65},     // 力矩已在电机内部修正过
    {"MNONE",    1.0},      // 不换算 (raw 当 Nm 用); 等同于查不到型号时的默认值

    // ---- ⚠ 固件表里没有的型号 ----
    // 下面几个只在主机侧 livelybot_serial 里出现过, 固件表查无此型号, 数值未经
    // 固件确认。本臂不用; 真要上这些电机, 先找厂商要 fdcan_h730 的值再上电。
    {"M4538_19", 0.4450},
    {"M5046_20", 0.5280},
    {"M5047_09", 0.5330},
    {"M60SG_35", 0.7942},
};

namespace {

constexpr double kPi = 3.14159265358979323846;

// 把 int16 按小端追加到 buf 末尾 (协议固定小端, x86/x64/ARM-LE 通用)
inline void push_le_i16(std::vector<uint8_t>& buf, int16_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xFF));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

// 从 buf[off] 按小端读 int16; off 自增 2
inline int16_t read_le_i16(const std::vector<uint8_t>& buf, std::size_t& off) {
    int16_t v = static_cast<int16_t>(
        static_cast<uint16_t>(buf[off]) |
        (static_cast<uint16_t>(buf[off + 1]) << 8));
    off += 2;
    return v;
}

inline int32_t read_le_i32(const std::vector<uint8_t>& buf, std::size_t& off) {
    uint32_t u = static_cast<uint32_t>(buf[off]) |
                 (static_cast<uint32_t>(buf[off + 1]) << 8) |
                 (static_cast<uint32_t>(buf[off + 2]) << 16) |
                 (static_cast<uint32_t>(buf[off + 3]) << 24);
    off += 4;
    return static_cast<int32_t>(u);
}

inline int8_t read_le_i8(const std::vector<uint8_t>& buf, std::size_t& off) {
    int8_t v = static_cast<int8_t>(buf[off]);
    off += 1;
    return v;
}

inline float read_le_f32(const std::vector<uint8_t>& buf, std::size_t& off) {
    static_assert(sizeof(float) == 4, "float must be 32-bit IEEE-754");
    float f;
    std::memcpy(&f, &buf[off], 4);
    off += 4;
    return f;
}

inline int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
} // namespace

// ---------------------------------------------------------------------------
//  字节工具
// ---------------------------------------------------------------------------

std::vector<uint8_t> canfd_pad(const std::vector<uint8_t>& data) {
    std::size_t n = data.size();
    std::size_t target = 64;
    for (std::size_t s : CANFD_DLC_SIZES) {
        if (s >= n) { target = s; break; }
    }
    std::vector<uint8_t> out;
    out.reserve(target);
    out.insert(out.end(), data.begin(), data.end());
    out.insert(out.end(), target - n, PADDING);
    return out;
}

std::string bytes_to_hex(const std::vector<uint8_t>& data) {
    static constexpr char kHex[] = "0123456789ABCDEF";
    std::string out;
    out.resize(data.size() * 2);
    for (std::size_t i = 0; i < data.size(); ++i) {
        out[2 * i]     = kHex[(data[i] >> 4) & 0xF];
        out[2 * i + 1] = kHex[ data[i]       & 0xF];
    }
    return out;
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        int hi = hex_digit(hex[i]);
        int lo = hex_digit(hex[i + 1]);
        if (hi < 0 || lo < 0) break;
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return out;
}

// ---------------------------------------------------------------------------
//  int16 单位转换
// ---------------------------------------------------------------------------

int16_t turns_to_int16(double turns)  { return saturate_to_i16(static_cast<long long>(turns / 0.0001)); }
double  int16_to_turns(int16_t val)   { return val * 0.0001; }
int16_t rps_to_int16(double rps)      { return saturate_to_i16(static_cast<long long>(rps  / 0.00025)); }
double  int16_to_rps(int16_t val)     { return val * 0.00025; }
int16_t rad_to_int16(double rad)      { return turns_to_int16(rad / (2.0 * kPi)); }
double  int16_to_rad(int16_t val)     { return int16_to_turns(val) * 2.0 * kPi; }
int16_t rad_s_to_int16(double rad_s)  { return rps_to_int16(rad_s / (2.0 * kPi)); }
double  int16_to_rad_s(int16_t val)   { return int16_to_rps(val) * 2.0 * kPi; }

// ---------------------------------------------------------------------------
//  CAN 帧 payload 构建
// ---------------------------------------------------------------------------

namespace {

// 查询电机状态的子帧 (附加在控制帧末尾)
//   0x14: read, int16, mode2;  0x04: 4 个数据;  0x00: 起始地址 0
//   0x11: read, int8,  1 个数据; 0x0f: 寄存器地址 0x0f (故障码)
inline std::vector<uint8_t> query_subframe_int16() {
    return {0x14, 0x04, 0x00, 0x11, 0x0f};
}

inline std::vector<uint8_t> append(std::vector<uint8_t> a,
                                    std::initializer_list<uint8_t> b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}
inline std::vector<uint8_t> append(std::vector<uint8_t> a,
                                    const std::vector<uint8_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

} // namespace

std::vector<uint8_t> build_read_state_int16() {
    return query_subframe_int16();
}

std::vector<uint8_t> build_stop_int16() {
    auto p = std::vector<uint8_t>{0x01, 0x00, 0x00};
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

// 只切电机模式 (写 1 个 int8 到 0x00 寄存器), 不带 pos/vel/tqe.
// mode 取值: 0x00=停止, 0x0A=位置/速度/力矩, 0x0F=刹车 等.
//
// 一拖多帧 (CAN ID 0x8090) 不含 mode 设置子帧, 上电默认 mode=0 时电机不响应
// pos/vel/tqe — 必须先用此命令把每个电机切到 mode=10.
std::vector<uint8_t> build_set_mode_int16(uint8_t mode) {
    auto p = std::vector<uint8_t>{0x01, 0x00, mode};
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_brake_int16() {
    auto p = std::vector<uint8_t>{0x01, 0x00, 0x0f};
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_pos_int16(int16_t pos) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x0A, 0x05, 0x20};
    push_le_i16(p, pos);
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_vel_int16(int16_t vel) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x0A, 0x06, 0x20};
    push_le_i16(p, NAN_INT16);   // 位置 = 无限制
    push_le_i16(p, vel);
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_pos_vel_tqe_int16(int16_t pos, int16_t vel, int16_t tqe) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x0a, 0x06, 0x20};
    push_le_i16(p, NAN_INT16);   // 位置先占位
    push_le_i16(p, vel);
    p.push_back(0x06);
    p.push_back(0x25);
    push_le_i16(p, tqe);
    push_le_i16(p, pos);
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

// 0x28 / 0x29 是**全局**速度/加速度限制 (厂商表1), 写进去会一直保留, 不随控制
// 方式切换自动清除. 厂商表3 第 45 条「停止位置使用错误」明确不支持同时设置停止
// 位置和速度/加速度限制 —— 而 build_pos_vel_tqe_int16 用的正是停止位置(0x26).
// 因此先走本函数再走 pos_vel_tqe 有触发 45 号错误的风险, 待硬件验证.
std::vector<uint8_t> build_pos_velmax_acc_int16(int16_t pos, int16_t vel_max, int16_t acc) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x0A, 0x05, 0x20};
    push_le_i16(p, pos);
    p.push_back(0x06);
    p.push_back(0x28);
    push_le_i16(p, vel_max);
    push_le_i16(p, acc);        // 子帧到此完整: 0x06 = 写 2 个 int16 (0x28, 0x29)
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_vel_acc_int16(int16_t vel, int16_t acc) {
    // 协议层等价: 位置寄存器写 NAN_INT16 (无限制), 复用 pos_velmax_acc 子帧布局.
    return build_pos_velmax_acc_int16(NAN_INT16, vel, acc);
}

std::vector<uint8_t> build_torque_int16(int16_t tqe) {
    // 与控制板固件 set_torque_int16 完全一致 (livelybot_fdcan.c):
    //   模式 0x0A + mode2 子帧 (cmd=0x04, num=0x06, addr=0x20) 写 6 个 int16:
    //   [pos=NAN, vel=0, tqe, kp=0, kd=0, maxtqe=NAN] -> 寄存器 0x20..0x25.
    // ★ kp 必须为 0, 不能是 NAN(0x8000): kp=NAN 会让电机用未定义位置增益
    //   去追位置, 即使 tqe=0 也会猛烈甩动 (旧版 bug).
    std::vector<uint8_t> p = {0x01, 0x00, 0x0a, 0x04, 0x06, 0x20};
    push_le_i16(p, NAN_INT16);   // pos = 无限制 (0x8000)
    push_le_i16(p, 0);           // vel = 0
    push_le_i16(p, tqe);         // tqe (前馈力矩 raw)
    push_le_i16(p, 0);           // kp = 0  (纯力矩, 无位置增益)
    push_le_i16(p, 0);           // kd = 0
    push_le_i16(p, NAN_INT16);   // 最大力矩 = 无限制 (0x8000)
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_voltage_int16(int16_t volt) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x08, 0x06, 0x1a, 0x00, 0x00};
    push_le_i16(p, volt);
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_current_int16(int16_t cur) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x09, 0x06, 0x1C};
    push_le_i16(p, cur);
    p.push_back(0x00); p.push_back(0x00);            // d 电流 = 0
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

std::vector<uint8_t> build_pos_vel_tqe_kp_kd_int16(int16_t pos, int16_t vel, int16_t tqe,
                                                    int16_t kp, int16_t kd) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x15, 0x07, 0x20};
    push_le_i16(p, pos);
    push_le_i16(p, vel);
    push_le_i16(p, tqe);
    p.push_back(0x06);
    p.push_back(0x2b);
    push_le_i16(p, kp);
    push_le_i16(p, kd);
    p = append(std::move(p), query_subframe_int16());
    return canfd_pad(p);
}

// ---------------------------------------------------------------------------
//  一拖多 (CAN ID = 0x8090, 协议文档 1.3.1.3)
//
//  布局: [pos1, vel1, tqe1] [pos2, vel2, tqe2] ... [posN, velN, tqeN] <pad...> 0x17 0x01
//        ↑ 槽位 i (i=0,1,..) 对应 motor_id = i+1, 没参与的填 NAN_INT16
//        ↑ 末尾固定 [0x17, 0x01] = 查询状态请求 (让被广播电机各回 1 帧)
//
//  CAN-FD 总长度按 DLC 表向上取整: pad 加在 motor 数据 与 [0x17,0x01] 之间.
//
//  ★ 一拖多 CAN ID 的通用规律: can_id = 0x8000 | MODE, MODE 取自厂商 SDK
//    livelybot_serial/include/serial_struct.h。这也解释了协议文档「ID 低 8 位
//    >= 0x80 即模式三」的规则 —— 控制模式全在 0x80..0xB0, 而 per-motor 操作码
//    (reset_zero 0x01 / conf_write 0x02 / stop 0x03 ...) 全 < 0x80。
//
//      MODE_POSITION            0x80    MODE_POS_VEL_TQE_KP_KI_KD 0x98  (12B/电机)
//      MODE_VELOCITY            0x81    MODE_POS_VEL_KP_KD        0x9E  ( 8B/电机)
//      MODE_TORQUE              0x82    MODE_POS_VEL_ACC          0xAD  ( 6B/电机)
//      MODE_VOLTAGE             0x83    MODE_POS_VEL_TQE_KP_KD2   0xB0  (10B/电机)
//      MODE_CURRENT             0x84
//      MODE_TIME_OUT            0x85    (每电机 1 个 int16, 批量设看门狗)
//      MODE_POS_VEL_TQE         0x90    (6B/电机, 即本函数)
//      MODE_POS_VEL_TQE_KP_KD   0x93    (10B/电机, 见下面的 kp_kd 版本)
//
//    注意 0x93 是厂商标注「不建议使用」的旧版 MIT (位置靠积分得到, 低频控制下
//    容易出意外), 推荐用结构完全相同的 0xB0。切换只需改 CAN ID 常量。
// ---------------------------------------------------------------------------
std::vector<uint8_t> build_many_pos_vel_tqe_int16(const std::vector<int16_t>& pos_arr,
                                                  const std::vector<int16_t>& vel_arr,
                                                  const std::vector<int16_t>& tqe_arr) {
    if (pos_arr.size() != vel_arr.size() || pos_arr.size() != tqe_arr.size()) {
        throw std::invalid_argument("build_many_pos_vel_tqe_int16: pos/vel/tqe size mismatch");
    }
    const std::size_t n = pos_arr.size();

    std::vector<uint8_t> data;
    data.reserve(n * 6 + 16);
    for (std::size_t i = 0; i < n; ++i) {
        push_le_i16(data, pos_arr[i]);
        push_le_i16(data, vel_arr[i]);
        push_le_i16(data, tqe_arr[i]);
    }

    // 末尾 2 字节 = 查询状态码; pad 插中间, 让总长度落到 DLC 表上一档
    const std::size_t need = data.size() + 2;
    // 每电机 6 字节 (pos/vel/tqe), CAN-FD 单帧数据段最大 64 字节, 所以一帧最多
    // 10 个电机 (10*6+2=62). 超了必须报错, 否则下面 target-need 会下溢成巨大值
    // -> "vector too long".
    if (need > 64) {
        throw std::invalid_argument(
            "build_many_pos_vel_tqe_int16: frame > 64 bytes "
            "(一拖多 pos/vel/tqe 单帧最多 10 个电机; 减少 max_motor_id / 槽位数)");
    }
    std::size_t target = 64;
    for (std::size_t s : CANFD_DLC_SIZES) {
        if (s >= need) { target = s; break; }
    }
    data.insert(data.end(), target - need, PADDING);
    data.push_back(0x17);
    data.push_back(0x01);
    return data;
}

std::vector<uint8_t> build_many_pos_vel_tqe_kp_kd_int16(
        const std::vector<int16_t>& pos_arr,
        const std::vector<int16_t>& vel_arr,
        const std::vector<int16_t>& tqe_arr,
        const std::vector<int16_t>& kp_arr,
        const std::vector<int16_t>& kd_arr) {
    if (pos_arr.size() != vel_arr.size() || pos_arr.size() != tqe_arr.size() ||
        pos_arr.size() != kp_arr.size()  || pos_arr.size() != kd_arr.size()) {
        throw std::invalid_argument(
            "build_many_pos_vel_tqe_kp_kd_int16: pos/vel/tqe/kp/kd size mismatch");
    }
    const std::size_t n = pos_arr.size();

    std::vector<uint8_t> data;
    data.reserve(n * 10 + 16);
    for (std::size_t i = 0; i < n; ++i) {
        push_le_i16(data, pos_arr[i]);
        push_le_i16(data, vel_arr[i]);
        push_le_i16(data, tqe_arr[i]);
        push_le_i16(data, kp_arr[i]);
        push_le_i16(data, kd_arr[i]);
    }

    const std::size_t need = data.size() + 2;
    // 一拖多 MIT/PD 每电机 10 字节 (pos/vel/tqe/kp/kd), CAN-FD 单帧数据段最大
    // 64 字节, 所以一帧最多 6 个电机 (6*10+2=62). 超了必须报错, 否则下面
    // target-need 会下溢成巨大值 -> "vector too long".
    if (need > 64) {
        throw std::invalid_argument(
            "build_many_pos_vel_tqe_kp_kd_int16: frame > 64 bytes "
            "(一拖多 MIT/PD 单帧最多 6 个电机; 减少 max_motor_id / 槽位数)");
    }
    std::size_t target = 64;
    for (std::size_t s : CANFD_DLC_SIZES) {
        if (s >= need) { target = s; break; }
    }
    data.insert(data.end(), target - need, PADDING);
    data.push_back(0x17);
    data.push_back(0x01);
    return data;
}

std::vector<uint8_t> build_motor_reset() {
    return {0x40, 0x01, 0x08, 0x64, 0x20, 0x72, 0x65, 0x73,
            0x65, 0x74, 0x0A, 0x50};
}

std::vector<uint8_t> build_conf_write() {
    return canfd_pad({0x40, 0x01, 0x0B, 0x63, 0x6F, 0x6E, 0x66, 0x20,
                      0x77, 0x72, 0x69, 0x74, 0x65, 0x0A});
}

std::vector<uint8_t> build_set_zero() {
    return {0x40, 0x01, 0x15, 0x64, 0x20, 0x63, 0x66, 0x67, 0x2d, 0x73,
            0x65, 0x74, 0x2d, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x20,
            0x30, 0x2e, 0x30, 0x0a};
}

std::vector<uint8_t> build_read_version() {
    return {0x15, 0xB5, 0x02};
}

// 固件看门狗: 超过 timeout_ms 没收到新指令帧, 电机自行刹车。这是主机侧完全
// 失效 (拔 USB / Ctrl-C / 上位机崩溃) 时唯一还能生效的防线。
//
// ★ 寄存器 0x1f 未见于厂商寄存器表 (表1 在 0x01d 之后直接跳到 0x020), 但实测
//   确实生效 —— 见 tests/test_fafu_motion_interactive.py 703-705 行 (阻塞读导致
//   循环超时, 看门狗在两帧之间刹车) 与 728-731 行 (servo_end 清看门狗是即发即
//   忘, 残留会让 J4 下垂 / J2/J3 不动)。moteus 的看门狗在 0x027, 而厂商表把
//   0x027 标为「保留」, 推测就是把它挪到了 0x01f。
//
//   厂商 SDK 另有一条批量通道 MODE_TIME_OUT = 0x85 (CAN ID 0x8085, 每电机一个
//   int16), 将来若要一帧设置全部关节的看门狗可以走它。
std::vector<uint8_t> build_set_timeout_int16(int16_t timeout_ms) {
    std::vector<uint8_t> p = {0x05, 0x1f};
    push_le_i16(p, timeout_ms);
    return p;
}

// ---------------------------------------------------------------------------
//  CAN 回复解析
// ---------------------------------------------------------------------------

std::string MotorState::to_string() const {
    std::ostringstream oss;
    oss << "MotorState(id=" << id
        << ", mode=" << mode
        << ", fault=0x" << std::hex << std::uppercase << std::setw(2)
        << std::setfill('0') << fault << std::dec
        << ", pos=" << std::fixed << std::setprecision(4) << position << " turns"
        << ", vel=" << std::fixed << std::setprecision(4) << velocity << " rps"
        << ", tqe_raw=" << std::fixed << std::setprecision(1) << torque << ")";
    return oss.str();
}

std::optional<MotorState> parse_motor_state_int16(const std::vector<uint8_t>& can_data) {
    if (can_data.size() < 2) return std::nullopt;

    // -- robustness fix 2026-05 -------------------------------------------
    // 老版本在遇到 op != 2 的首字节 (即整帧不是 reply 子帧) 时, break 出
    // 循环再 `return state;` —— 返回的是一个 *零初始化* 的 MotorState
    // (id=0, mode=0, fault=0, position=0, velocity=0, torque=0). 这会让
    // 上层 cache 误以为电机突然瞬移到原点, 触发 lag-trip / safety
    // cascade. 修复: 只有真的解析到至少一个寄存器才返回 state, 否则
    // 上抛 std::nullopt 让上层把帧当 rx_dropped, cache 保持不动.
    // ---------------------------------------------------------------------
    MotorState state;
    std::size_t offset = 0;
    bool any_field_parsed = false;

    while (offset < can_data.size()) {
        if (can_data[offset] == PADDING) {
            ++offset;
            continue;
        }

        const uint8_t cmd = can_data[offset];
        const uint8_t op    = (cmd >> 4) & 0x0F;
        const uint8_t dtype = (cmd >> 2) & 0x03;
        const uint8_t count =  cmd       & 0x03;

        // 不是 reply 子帧 → 帧结构不可信. 已经解析过 1+ 寄存器就保留已有
        // 数据 (帧后半段被截断的可能性), 否则当作脏帧丢弃.
        if (op != 2) {
            if (!any_field_parsed) return std::nullopt;
            break;
        }

        ++offset;
        if (offset >= can_data.size()) break;

        if (count == 0) {              // 模式二
            if (offset >= can_data.size()) break;
            const uint8_t num = can_data[offset++];
            if (offset >= can_data.size()) break;
            const uint8_t addr = can_data[offset++];

            if (dtype == 1) {          // int16
                for (uint8_t i = 0; i < num; ++i) {
                    if (offset + 2 > can_data.size()) break;
                    const int16_t val = read_le_i16(can_data, offset);
                    const int reg = addr + i;
                    if      (reg == 0x00) { state.mode     = val;                          any_field_parsed = true; }
                    else if (reg == 0x01) { state.position = int16_to_turns(val);          any_field_parsed = true; }
                    else if (reg == 0x02) { state.velocity = int16_to_rps(val);            any_field_parsed = true; }
                    else if (reg == 0x03) { state.torque   = static_cast<double>(val);     any_field_parsed = true; }
                }
            } else if (dtype == 2) {   // int32
                for (uint8_t i = 0; i < num; ++i) {
                    if (offset + 4 > can_data.size()) break;
                    const int32_t val = read_le_i32(can_data, offset);
                    const int reg = addr + i;
                    if      (reg == 0x00) { state.mode     = val;                          any_field_parsed = true; }
                    else if (reg == 0x01) { state.position = val * 0.00001;                any_field_parsed = true; }
                    else if (reg == 0x02) { state.velocity = val * 0.00001;                any_field_parsed = true; }
                    else if (reg == 0x03) { state.torque   = static_cast<double>(val);     any_field_parsed = true; }
                }
            } else if (dtype == 0) {   // int8
                for (uint8_t i = 0; i < num; ++i) {
                    if (offset + 1 > can_data.size()) break;
                    const int8_t val = read_le_i8(can_data, offset);
                    const int reg = addr + i;
                    if (reg == 0x0f) { state.fault = static_cast<int>(val) & 0xFF;         any_field_parsed = true; }
                }
            }
        } else {                       // 模式一
            const uint8_t addr = can_data[offset++];
            const std::size_t type_size_arr[4] = {1, 2, 4, 4};
            const std::size_t type_size = type_size_arr[dtype];

            for (uint8_t i = 0; i < count; ++i) {
                if (offset + type_size > can_data.size()) break;

                double f_val = 0.0;
                int    i_val = 0;
                if      (dtype == 0) i_val = read_le_i8 (can_data, offset);
                else if (dtype == 1) i_val = read_le_i16(can_data, offset);
                else if (dtype == 2) i_val = read_le_i32(can_data, offset);
                else                 f_val = read_le_f32(can_data, offset);

                const int reg = addr + i;
                if (reg == 0x00) {
                    state.mode = (dtype == 3) ? static_cast<int>(f_val) : i_val;
                    any_field_parsed = true;
                } else if (reg == 0x01) {
                    if      (dtype == 1) state.position = int16_to_turns(static_cast<int16_t>(i_val));
                    else if (dtype == 2) state.position = i_val * 0.00001;
                    else if (dtype == 3) state.position = f_val;
                    else                 state.position = i_val;
                    any_field_parsed = true;
                } else if (reg == 0x02) {
                    if      (dtype == 1) state.velocity = int16_to_rps(static_cast<int16_t>(i_val));
                    else if (dtype == 2) state.velocity = i_val * 0.00001;
                    else if (dtype == 3) state.velocity = f_val;
                    else                 state.velocity = i_val;
                    any_field_parsed = true;
                } else if (reg == 0x03) {
                    state.torque = (dtype == 3) ? f_val : static_cast<double>(i_val);
                    any_field_parsed = true;
                } else if (reg == 0x0f) {
                    state.fault = ((dtype == 3) ? static_cast<int>(f_val) : i_val) & 0xFF;
                    any_field_parsed = true;
                }
            }
        }
    }

    if (!any_field_parsed) return std::nullopt;
    return state;
}

}  // namespace hightorque
