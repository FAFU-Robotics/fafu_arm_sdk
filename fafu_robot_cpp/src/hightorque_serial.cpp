// =============================================================================
//  hightorque_serial.cpp
//  调试板 USB 串口驱动 (1:1 移植自 hightorque_serial.py).
// =============================================================================
#include "hightorque_serial.hpp"

#include <serial/serial.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>   // std::cout (调试打印, 见 read_motor_state_with_debug)
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  #include <timeapi.h>
  #pragma comment(lib, "winmm.lib")
#endif

namespace hightorque {

// ---------------------------------------------------------------------------
//  Windows 高精度系统时钟 RAII 卫士
//
//  Windows 默认 system timer resolution = 15.6 ms (来自电源策略). 这意味着
//  std::this_thread::sleep_for(1ms) 实际会睡 ~15ms. 100Hz (10ms 周期) 已经
//  在临界, 250Hz (4ms) 必然爆 jitter. timeBeginPeriod(1) 可以临时把粒度
//  降到 1ms (会增加全系统能耗, 出 scope 自动还原).
//
//  POSIX (Linux/Mac) 默认就是高分辨率, 不需要这个.
// ---------------------------------------------------------------------------

namespace {

class HighResolutionTimer {
public:
    HighResolutionTimer() {
#ifdef _WIN32
        if (timeBeginPeriod(1) == TIMERR_NOERROR) active_ = true;
#endif
    }
    ~HighResolutionTimer() {
#ifdef _WIN32
        if (active_) timeEndPeriod(1);
#endif
    }
    HighResolutionTimer(const HighResolutionTimer&) = delete;
    HighResolutionTimer& operator=(const HighResolutionTimer&) = delete;
private:
    bool active_ = false;
};

} // namespace

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

const std::map<std::string, double> TORQUE_COEFF = {
    // ---- Fafu arm actual motors (measured, authoritative) ----
    {"M5036_02", 0.67},     // J1, J4
    {"M6036_02", 0.677},    // J2, J3
    {"M4438_30", 0.5256},   // J5, J6, J7  (vendor motor_tqe_adj value)
    // ---- legacy / other Hightorque models ----
    {"M3536_32", 0.458105},
    {"M4438_32", 0.485565},
    {"M4538_19", 0.493835},
    {"M5043_20", 0.966},
    {"M5046_20", 0.533654},
    {"M5047_09", 0.547474},
    {"M5047_36", 0.803},
    {"M6056_36", 0.677},
    {"M7256_35", 0.676524},
    {"M60SG_35", 0.7942},
    {"M60BM_35", 0.7942},
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

template <typename T>
inline int16_t saturate_to_i16(T value) {
    if (value >  32767) return  32767;
    if (value < -32768) return -32768;
    return static_cast<int16_t>(value);
}

inline int16_t finite_to_i16(double value, const char* field) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(field) + " must be finite");
    }
    if (value > 32767.0) return 32767;
    if (value < -32768.0) return -32768;
    return static_cast<int16_t>(value);
}

inline int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline std::string strip(const std::string& s) {
    auto b = s.begin();
    auto e = s.end();
    while (b != e && std::isspace(static_cast<unsigned char>(*b))) ++b;
    while (e != b && std::isspace(static_cast<unsigned char>(*(e - 1)))) --e;
    return std::string(b, e);
}

inline std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    for (std::string tok; iss >> tok;) out.push_back(std::move(tok));
    return out;
}

inline std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '\n') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

inline double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline void sleep_seconds(double s) {
    if (s <= 0) return;
    std::this_thread::sleep_for(
        std::chrono::microseconds(static_cast<long long>(s * 1'000'000)));
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

int16_t turns_to_int16(double turns)  { return finite_to_i16(turns / 0.0001, "position"); }
double  int16_to_turns(int16_t val)   { return val * 0.0001; }
int16_t rps_to_int16(double rps)      { return finite_to_i16(rps / 0.00025, "velocity"); }
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

std::vector<uint8_t> build_pos_velmax_acc_int16(int16_t pos, int16_t vel_max, int16_t acc) {
    std::vector<uint8_t> p = {0x01, 0x00, 0x0A, 0x05, 0x20};
    push_le_i16(p, pos);
    p.push_back(0x06);
    p.push_back(0x28);
    push_le_i16(p, vel_max);
    push_le_i16(p, acc);
    p.push_back(0x00);
    p.push_back(0x00);          // reserved
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
    std::size_t target = 64;
    for (std::size_t s : CANFD_DLC_SIZES) {
        if (s >= need) { target = s; break; }
    }
    if (target < need) target = 64;     // 兜底, 实际不会走到
    const std::size_t pad = target - need;
    data.insert(data.end(), pad, PADDING);
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

// ---------------------------------------------------------------------------
//  调试板串口驱动
// ---------------------------------------------------------------------------

HightorqueSerial::HightorqueSerial(const std::string& port, uint32_t baudrate) {
    auto timeout = serial::Timeout::simpleTimeout(100);   // 100 ms, 与 Python 一致
    ser_ = std::make_unique<serial::Serial>(port, baudrate, timeout);
    sleep_seconds(0.1);
    if (ser_->isOpen()) ser_->flushInput();
}

HightorqueSerial::~HightorqueSerial() {
    try {
        // 顺序很关键: 先停后台线程, 再关串口, 否则线程可能访问已销毁的 ser_
        stop_state_polling();
        disable_async_rx();
        close();
    } catch (...) {
        // 析构里吞掉异常, 避免 std::terminate
    }
}

void HightorqueSerial::close() {
    std::lock_guard<std::mutex> lifecycle(lifecycle_mtx_);
    stop_state_polling_unlocked_();
    disable_async_rx_unlocked_();

    std::lock_guard<std::mutex> tx(tx_mtx_);
    if (ser_ && ser_->isOpen()) {
        ser_->close();
    }
}

bool HightorqueSerial::is_open() const {
    std::lock_guard<std::mutex> lifecycle(lifecycle_mtx_);
    return ser_ && ser_->isOpen();
}

// -- 底层串口通信 --

void HightorqueSerial::serial_write_(const std::vector<uint8_t>& data, int retries) {
    for (int attempt = 0; attempt < retries; ++attempt) {
        try {
            std::size_t offset = 0;
            while (offset < data.size()) {
                const std::size_t written = ser_->write(
                    data.data() + offset, data.size() - offset);
                if (written == 0) {
                    throw std::runtime_error(
                        "serial write made no progress after " +
                        std::to_string(offset) + " of " +
                        std::to_string(data.size()) + " bytes");
                }
                offset += written;
            }
            // Do not flushOutput(): serial_cmake implements it as TX purge.
            return;
        } catch (const serial::SerialException&) {
            if (attempt == retries - 1) throw;
            sleep_seconds(0.05);
            ser_->flushInput();
            ser_->flushOutput();
        } catch (const serial::IOException&) {
            if (attempt == retries - 1) throw;
            sleep_seconds(0.05);
            ser_->flushInput();
            ser_->flushOutput();
        }
    }
}

// ---------------------------------------------------------------------------
//  send_cmd_ 和 send_can_and_recv_ 在两种模式下行为不同:
//
//   同步模式 (默认): 写串口 → 阻塞读直到收到目标行数 → return
//   异步模式 (async_rx_enabled_): 写串口 → return; 回复由 RX 线程异步入 cache.
//                                  send_cmd_ 仍然短同步等 ASCII 行 (because
//                                  RX 线程只 dispatch 'rcv' 行, ASCII 命令响应
//                                  不会进 cache); 但 send_can_and_recv_ 立即
//                                  return 空 RcvList.
// ---------------------------------------------------------------------------

std::string HightorqueSerial::send_cmd_(const std::string& cmd, double timeout_s) {
    std::lock_guard<std::mutex> lock(tx_mtx_);
    const bool async_mode = async_rx_enabled_.load();
    if (!async_mode) ser_->flushInput();

    const std::string line = cmd + "\r\n";
    std::vector<uint8_t> bytes(line.begin(), line.end());
    serial_write_(bytes);
    note_tx_();

    // 异步模式: ASCII 响应行也会被 RX 线程吃掉. 此函数只用于诊断命令(can status/config),
    // 启用 async 后建议尽量不用. 这里只做"短等" (50ms) 兜底.
    if (async_mode) {
        sleep_seconds(0.05);
        return std::string{};
    }

    std::string response;
    const double deadline = now_seconds() + timeout_s;
    while (now_seconds() < deadline) {
        const std::size_t n = ser_->available();
        if (n > 0) {
            std::string chunk;
            ser_->read(chunk, n);
            response.append(chunk);
            if (response.find('\n') != std::string::npos) break;
        } else {
            sleep_seconds(0.005);
        }
    }
    return strip(response);
}

HightorqueSerial::RcvList HightorqueSerial::send_can_and_recv_(
        int can_id, const std::vector<uint8_t>& data, double timeout_s, int expected_replies) {

    std::unique_lock<std::mutex> lock(tx_mtx_);
    const bool async_mode = async_rx_enabled_.load();
    if (!async_mode) ser_->flushInput();
    std::ostringstream cmd;
    cmd << "can send " << std::hex << std::uppercase << can_id
        << " " << bytes_to_hex(data) << "\r\n";
    const std::string line = cmd.str();
    std::vector<uint8_t> bytes(line.begin(), line.end());
    serial_write_(bytes);
    note_tx_();

    // 异步模式: 提前 return, 释放锁; 回复由 RX 线程异步入 cache
    if (async_mode) return {};

    // 同步模式: 持锁继续读串口直到收齐回复或超时
    if (expected_replies < 1) expected_replies = 1;

    // 退出条件: 实测调试板 'can send' 不送命令回显行, 只回 N 个 'rcv'.
    // 所以 target_newlines = expected_replies (旧版 +1 是误判, 永远等满 timeout)
    // 1ms 轮询粒度 (vs 旧版 10ms), 100Hz 控制下整体抖动 < 1ms
    const int    target_newlines = expected_replies;
    std::string  response;
    const double deadline = now_seconds() + timeout_s;
    while (now_seconds() < deadline) {
        const std::size_t n = ser_->available();
        if (n > 0) {
            std::string chunk;
            ser_->read(chunk, n);
            response.append(chunk);
            if (std::count(response.begin(), response.end(), '\n') >= target_newlines) break;
        } else {
            sleep_seconds(0.001);
        }
    }
    return parse_rcv_lines_(response);
}

// ---------------------------------------------------------------------------
//  TX 计数 + 周期/抖动统计
// ---------------------------------------------------------------------------
void HightorqueSerial::note_tx_() {
    const double now = now_seconds();
    std::lock_guard<std::mutex> lk(stats_mtx_);
    stats_.tx_frames++;
    if (last_tx_time_ > 0.0) {
        const double dt_ms = (now - last_tx_time_) * 1000.0;
        tx_periods_.push_back(dt_ms);
        if (static_cast<int>(tx_periods_.size()) > kTxWindow_) {
            tx_periods_.erase(tx_periods_.begin());
        }
        // 滑动平均
        double sum = 0.0;
        for (double v : tx_periods_) sum += v;
        stats_.avg_tx_period_ms = sum / tx_periods_.size();
        // 抖动 (相邻间隔与平均的差)
        const double jitter = std::abs(dt_ms - stats_.avg_tx_period_ms);
        if (jitter > stats_.max_tx_jitter_ms) stats_.max_tx_jitter_ms = jitter;
    }
    last_tx_time_ = now;
}

HightorqueSerial::RcvList HightorqueSerial::parse_rcv_lines_(const std::string& text) {
    RcvList results;
    for (const auto& line : split_lines(text)) {
        const std::string s = strip(line);
        if (s.rfind("rcv ", 0) != 0) continue;
        const auto parts = split_ws(s);
        if (parts.size() < 3) continue;
        try {
            int rcv_id = std::stoi(parts[1], nullptr, 16);
            std::vector<uint8_t> rcv_data = hex_to_bytes(parts[2]);
            results.emplace_back(rcv_id, std::move(rcv_data));
        } catch (const std::exception&) {
            continue;
        }
    }
    return results;
}

// -- 调试板命令 --

std::string HightorqueSerial::can_status() { return send_cmd_("can status"); }
std::string HightorqueSerial::can_config() { return send_cmd_("can config"); }

// -- 电机控制公共模板 --
namespace {

template <typename Sender>
std::optional<MotorState> control_call(Sender&& s, int motor_id, double timeout_s = 0.5) {
    const int can_id = 0x8000 | motor_id;
    auto replies = s(can_id, timeout_s);
    for (auto& [rid, rdata] : replies) {
        (void)rid;
        if (auto st = parse_motor_state_int16(rdata)) {
            st->id = motor_id;
            return st;
        }
    }
    return std::nullopt;
}

} // namespace

std::optional<MotorState> HightorqueSerial::read_motor_state(int motor_id, double timeout_s) {
    // ---- 异步模式 ----
    // send_can_and_recv_ 在 async 下立即返回空, 同步等待会永远等不到回包.
    // 正确做法: 1) 记当前 cache seq  2) 发查询帧 (RX 线程会异步收到入 cache)
    //          3) 用 cv 等 seq+1 (= 收到这个 motor 的新一帧)
    if (async_rx_enabled_.load()) {
        uint64_t seen_seq = 0;
        {
            std::lock_guard<std::mutex> lk(cache_mtx_);
            auto it = update_seq_.find(motor_id);
            if (it != update_seq_.end()) seen_seq = it->second;
        }
        // 触发查询 (async 路径立即返回, 实际收包靠 RX 线程)
        request_motor_state_(motor_id);

        std::unique_lock<std::mutex> lk(cache_mtx_);
        const auto deadline = std::chrono::steady_clock::now()
            + std::chrono::microseconds(static_cast<long long>(timeout_s * 1e6));
        cache_cv_.wait_until(lk, deadline, [&]() {
            if (!async_rx_enabled_.load()) return true;
            auto it = update_seq_.find(motor_id);
            return it != update_seq_.end() && it->second > seen_seq;
        });

        auto it = state_cache_.find(motor_id);
        if (it == state_cache_.end()) return std::nullopt;
        auto seq_it = update_seq_.find(motor_id);
        if (seq_it == update_seq_.end() || seq_it->second <= seen_seq) {
            return std::nullopt;   // 超时
        }
        return it->second;
    }

    // ---- 同步模式 (原逻辑) ----
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_read_state_int16(), t); },
        motor_id, timeout_s);
}

void HightorqueSerial::request_motor_state_(int motor_id) {
    if (!async_rx_enabled_.load()) {
        throw std::logic_error(
            "fire-and-forget state requests require asynchronous receive");
    }
    (void)send_can_and_recv_(
        0x8000 | motor_id, build_read_state_int16(), 0.0);
}

std::optional<MotorState> HightorqueSerial::stop(int motor_id) {
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_stop_int16(), t); },
        motor_id);
}

std::optional<MotorState> HightorqueSerial::brake(int motor_id) {
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_brake_int16(), t); },
        motor_id);
}

std::optional<MotorState> HightorqueSerial::set_motor_mode(int motor_id, uint8_t mode) {
    // 实测: 单独发 build_set_mode_int16 (只写 mode 寄存器, 无后续参数子帧) 部分固件
    // 不会真正切换 mode — 回包里的 mode 仍是旧值. 经验做法是用一条带"参数"的命令
    // 顺便切 mode (Python 老版本就是这么做的, 每次 set_pos_* 都隐式切到 mode=10).
    //
    // 这里特殊处理 mode=0x0A (位置/速度/力矩):
    //   1) 先 read 当前位置
    //   2) 直接构造 pos_vel_tqe 帧 (pos=电机当前真实位置, vel=0, tqe=0)
    //      — 绕过软限位 (因为电机当前位置可能已在限位外, 走 set_pos_vel_tqe
    //         会被 clamp 后自动猛动到限位内, 危险)
    //      — vel=0 且 tqe=0 ⇒ 电机切到 mode=10 但目标位置=当前位置, 实际不动
    //
    // 其他 mode (0x00 停止 / 0x0F 刹车) 用现成接口.
    if (mode == 0x0A) {
        // ★ Brake -> Active reliability fix (2026-05).
        // 某些固件版本下电机处于 MODE_BRAKE (0x0F) 时, read_motor_state
        // 会超时 (firmware 不主动回 state 报文), 上层会以为电机"refused
        // mode 0x0A", 必须断电才能恢复. 先无条件 stop 一下把它踢回
        // MODE_STOP 是已知可靠路径, 代价仅一帧 (~2ms) + 5ms 沉降.
        // 对本来就在 Active/Stop 的电机这一步是无副作用的: stop -> active
        // 是 SDK 已经长期使用的可靠路径.
        (void)stop(motor_id);
        sleep_seconds(0.005);

        auto cur = read_motor_state(motor_id, 0.2);
        if (!cur) return std::nullopt;

        // 用电机当前真实位置作目标 (绕开软限位, 避免超限电机猛动)
        const int16_t pos_i = turns_to_int16(cur->position);
        const auto    data  = build_pos_vel_tqe_int16(pos_i, /*vel=*/0, /*tqe=*/0);
        const int     can_id = 0x8000 | motor_id;

        // 发命令 (immediate ack 可能是切换前的旧 state, 不可靠)
        (void)send_can_and_recv_(can_id, data, 0.2);

        // 关键: 等 mode 在固件里实际生效 (实测 < 5ms 完成), 然后再 read 一次 ground truth
        sleep_seconds(0.01);
        return read_motor_state(motor_id, 0.2);
    }
    if (mode == 0x00) return stop(motor_id);
    if (mode == 0x0F) return brake(motor_id);

    // 其他 mode 走原始的"只写 mode 寄存器" 路径 — 留给高级用户
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_set_mode_int16(mode), t); },
        motor_id);
}

std::optional<MotorState> HightorqueSerial::set_position(int motor_id, double pos, PosUnit unit) {
    const auto [pos_t, flag] = apply_position_limit_(motor_id, to_turns(pos, unit));
    const int16_t p = turns_to_int16(pos_t);
    auto st = control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_pos_int16(p), t); },
        motor_id);
    if (st) st->pos_limit_flag = flag;
    return st;
}

std::optional<MotorState> HightorqueSerial::set_velocity(int motor_id, double vel_rps) {
    const int16_t v = rps_to_int16(vel_rps);
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_vel_int16(v), t); },
        motor_id);
}

std::optional<MotorState> HightorqueSerial::set_pos_vel_tqe(int motor_id, double pos,
                                                            double vel_rps, int tqe_raw,
                                                            PosUnit unit) {
    const auto [pos_t, flag] = apply_position_limit_(motor_id, to_turns(pos, unit));
    // tqe_raw == 0 视为 "用电机默认最大力矩" → NAN_INT16 (与 set_many_pos_vel_tqe 一致).
    const int16_t tqe_i = (tqe_raw == 0) ? NAN_INT16 : saturate_to_i16(tqe_raw);
    auto st = control_call(
        [&](int can_id, double t) {
            return send_can_and_recv_(can_id,
                build_pos_vel_tqe_int16(turns_to_int16(pos_t),
                                        rps_to_int16(vel_rps),
                                        tqe_i), t);
        },
        motor_id);
    if (st) st->pos_limit_flag = flag;
    return st;
}

std::optional<MotorState> HightorqueSerial::set_pos_vel_acc(int motor_id, double pos,
                                                            double vel_max_rps, double acc_rpss,
                                                            PosUnit unit) {
    const auto [pos_t, flag] = apply_position_limit_(motor_id, to_turns(pos, unit));
    const int16_t acc_int = finite_to_i16(acc_rpss / 0.001, "acceleration");
    auto st = control_call(
        [&](int can_id, double t) {
            return send_can_and_recv_(can_id,
                build_pos_velmax_acc_int16(turns_to_int16(pos_t),
                                           rps_to_int16(vel_max_rps),
                                           acc_int), t);
        },
        motor_id);
    if (st) st->pos_limit_flag = flag;
    return st;
}

std::optional<MotorState> HightorqueSerial::set_vel_acc(int motor_id, double vel_rps,
                                                        double acc_rpss) {
    // 速度+加速度 (3.1.9): 不带位置限制, 不经过 apply_position_limit_ (没有 pos 参数).
    // acc LSB = 0.001 转/秒² (协议 2.8), 与 set_pos_vel_acc 一致.
    const int16_t v       = rps_to_int16(vel_rps);
    const int16_t acc_int = finite_to_i16(acc_rpss / 0.001, "acceleration");
    return control_call(
        [&](int can_id, double t) {
            return send_can_and_recv_(can_id, build_vel_acc_int16(v, acc_int), t);
        },
        motor_id);
}

// ===========================================================================
//  一拖多 (单帧多电机) — 同步版
//
//  设计要点:
//    1. 槽位按 motor_id 索引, 没参与的电机槽位写 NAN_INT16 (0x8000), 电机识别为
//       "无操作". 这跟官方 motor.cpp 里 data16[i]=0x8000 的语义完全一致.
//    2. 每个 cmd 都过 apply_position_limit_, 触发限位后 pos_limit_flag 写到结果.
//    3. 单次 send + 等 max_motor_id 个 'rcv' 行回复 (timeout_s 兜底). 100Hz 周期下
//       50ms timeout 完全够用.
//    4. 没收到回复的电机不会出现在返回 map 里 — 上层判断"通信丢"可以用 .count(id).
// ===========================================================================
std::map<int, MotorState> HightorqueSerial::set_many_pos_vel_tqe(
        const std::vector<ManyMotorCmd>& cmds, PosUnit pos_unit,
        int max_motor_id, double timeout_s) {

    std::map<int, MotorState> out;
    if (cmds.empty()) return out;

    // 1) 确定槽位数量 (帧长度由它决定)
    int n_slots = max_motor_id;
    if (n_slots <= 0) {
        for (const auto& c : cmds) n_slots = std::max(n_slots, c.motor_id);
    }
    if (n_slots <= 0) return out;

    // 2) 槽位初始化为 NAN_INT16 (= "无操作")
    std::vector<int16_t> pos_arr(n_slots, NAN_INT16);
    std::vector<int16_t> vel_arr(n_slots, NAN_INT16);
    std::vector<int16_t> tqe_arr(n_slots, NAN_INT16);

    // 3) 填入指令 + 限位 (索引 i 对应 motor_id = i+1)
    std::map<int, int> limit_flags;            // motor_id -> flag
    for (const auto& c : cmds) {
        if (c.motor_id < 1 || c.motor_id > n_slots) continue;   // 越界忽略

        const auto [pos_t, flag] = apply_position_limit_(c.motor_id, to_turns(c.pos, pos_unit));
        limit_flags[c.motor_id] = flag;

        const std::size_t idx = static_cast<std::size_t>(c.motor_id - 1);
        pos_arr[idx] = turns_to_int16(pos_t);
        vel_arr[idx] = rps_to_int16(c.vel_rps);
        // tqe_raw == 0 视为 "用电机默认最大力矩" → NAN_INT16 (协议: 0x8000 = 无效/无操作).
        // 直接发 0 会让电机理解为 "最大输出力矩=0" → 完全没力气, 不动, 还可能触发保护
        // 把 mode 切回 0. 想真正限制力矩请填非 0 值 (或负数, 内部已 saturate).
        tqe_arr[idx] = (c.tqe_raw == 0) ? NAN_INT16 : saturate_to_i16(c.tqe_raw);
    }

    // 4) 发送 + 等待 N 个回复 (N = 实际有指令的电机数, 没指令的电机不会回包)
    const auto data = build_many_pos_vel_tqe_int16(pos_arr, vel_arr, tqe_arr);
    const int  expected = static_cast<int>(cmds.size());
    auto replies = send_can_and_recv_(0x8090, data, timeout_s, expected);

    // 5) 解析: 反馈帧 ID 高 8 位 = 源地址 = 电机 ID, 低 8 位是目的地址 (主机=0).
    //    最高位是 "需要回复" 标志, 用 & 0x7F 去掉.
    for (auto& [rid, rdata] : replies) {
        int motor_id = (rid >> 8) & 0x7F;
        if (motor_id == 0) motor_id = rid & 0x7F;     // 老固件兜底
        if (motor_id < 1 || motor_id > n_slots) continue;
        if (auto st = parse_motor_state_int16(rdata)) {
            st->id = motor_id;
            if (auto it = limit_flags.find(motor_id); it != limit_flags.end()) {
                st->pos_limit_flag = it->second;
            }
            out[motor_id] = *st;
        }
    }
    return out;
}

// "Partial broadcast" — servoJ / teleop 热路径.
// 与 set_many_pos_vel_tqe 共用一帧编码格式 (build_many_pos_vel_tqe_int16),
// 区别只是 cmd 是从 active_ids + hold_ids 两组 vector 合成, 而不是上层
// 拼好的 ManyMotorCmd 数组. 这样 Python binding 可以走 numpy buffer 而
// 不是 list of objects, marshalling 开销从 ~200us 降到 ~5us per tick.
std::map<int, MotorState> HightorqueSerial::set_many_pos_vel_tqe_partial(
        const std::vector<int>&    active_ids,
        const std::vector<double>& active_pos,
        const std::vector<double>& active_vel,
        int16_t                    tqe_raw,
        const std::vector<int>&    hold_ids,
        PosUnit pos_unit, int max_motor_id, double timeout_s) {

    std::map<int, MotorState> out;

    if (active_ids.size() != active_pos.size() ||
        active_ids.size() != active_vel.size()) {
        return out;     // 长度不一致, 拒绝发送
    }
    if (active_ids.empty() && hold_ids.empty()) return out;

    // 1) slot 数 (帧长度).
    int n_slots = max_motor_id;
    if (n_slots <= 0) {
        for (int id : active_ids) n_slots = std::max(n_slots, id);
        for (int id : hold_ids)   n_slots = std::max(n_slots, id);
    }
    if (n_slots <= 0) return out;

    std::vector<int16_t> pos_arr(n_slots, NAN_INT16);
    std::vector<int16_t> vel_arr(n_slots, NAN_INT16);
    std::vector<int16_t> tqe_arr(n_slots, NAN_INT16);

    // 2) Active 电机 — 应用 soft limit (与 set_many_pos_vel_tqe 行为一致).
    std::map<int, int> limit_flags;
    const int16_t tqe_slot = (tqe_raw == 0) ? NAN_INT16 : saturate_to_i16(tqe_raw);
    int expected = 0;
    for (size_t k = 0; k < active_ids.size(); ++k) {
        const int mid = active_ids[k];
        if (mid < 1 || mid > n_slots) continue;

        const auto [pos_t, flag] = apply_position_limit_(mid, to_turns(active_pos[k], pos_unit));
        limit_flags[mid] = flag;

        const std::size_t idx = static_cast<std::size_t>(mid - 1);
        pos_arr[idx] = turns_to_int16(pos_t);
        vel_arr[idx] = rps_to_int16(active_vel[k]);
        tqe_arr[idx] = tqe_slot;
        ++expected;
    }

    // 3) Hold 电机 — 用 cache 里的当前位置当 target, vel=0.
    //    cache 还没有数据的电机 (轮询线程没起 / 该电机不在 polling 列表) 静默跳过.
    for (int mid : hold_ids) {
        if (mid < 1 || mid > n_slots) continue;
        const std::size_t idx = static_cast<std::size_t>(mid - 1);
        if (pos_arr[idx] != NAN_INT16) continue;   // 已经被 active 占了, 不覆盖

        auto cur = get_cached_state(mid);
        if (!cur) continue;     // 没有缓存就跳过, 不发该电机, 它进入 watchdog

        // 注意: hold 时不走 apply_position_limit_, 因为缓存的就是真实位置,
        // 走限位再 clamp 反而可能在边界电机上来回小幅震荡.
        pos_arr[idx] = turns_to_int16(cur->position);
        vel_arr[idx] = 0;     // 显式 0 (不是 NAN_INT16): "目标 = hold pos, 速度 = 0"
        tqe_arr[idx] = tqe_slot;
        ++expected;
    }

    // 4) 发. timeout_s <= 0 直接异步发出去不等回包 (servo loop 用法).
    const auto data = build_many_pos_vel_tqe_int16(pos_arr, vel_arr, tqe_arr);
    if (timeout_s <= 0.0 || expected <= 0) {
        (void)send_can_and_recv_(0x8090, data, 0.0, 0);
        return out;     // 空 map
    }

    auto replies = send_can_and_recv_(0x8090, data, timeout_s, expected);
    for (auto& [rid, rdata] : replies) {
        int motor_id = (rid >> 8) & 0x7F;
        if (motor_id == 0) motor_id = rid & 0x7F;
        if (motor_id < 1 || motor_id > n_slots) continue;
        if (auto st = parse_motor_state_int16(rdata)) {
            st->id = motor_id;
            if (auto it = limit_flags.find(motor_id); it != limit_flags.end()) {
                st->pos_limit_flag = it->second;
            }
            out[motor_id] = *st;
        }
    }
    return out;
}

std::map<int, MotorState> HightorqueSerial::set_many_pos_vel_tqe_kp_kd_partial(
        const std::vector<int>&    active_ids,
        const std::vector<double>& active_pos,
        const std::vector<double>& active_vel,
        int16_t                    tqe_raw,
        int16_t                    kp_raw,
        int16_t                    kd_raw,
        PosUnit pos_unit, int max_motor_id, double timeout_s) {

    std::map<int, MotorState> out;
    if (active_ids.size() != active_pos.size() ||
        active_ids.size() != active_vel.size()) {
        return out;
    }
    if (active_ids.empty()) return out;

    int n_slots = max_motor_id;
    if (n_slots <= 0) {
        for (int id : active_ids) n_slots = std::max(n_slots, id);
    }
    if (n_slots <= 0) return out;

    std::vector<int16_t> pos_arr(n_slots, NAN_INT16);
    std::vector<int16_t> vel_arr(n_slots, NAN_INT16);
    std::vector<int16_t> tqe_arr(n_slots, NAN_INT16);
    std::vector<int16_t> kp_arr(n_slots, NAN_INT16);
    std::vector<int16_t> kd_arr(n_slots, NAN_INT16);

    std::map<int, int> limit_flags;
    int expected = 0;
    for (size_t k = 0; k < active_ids.size(); ++k) {
        const int mid = active_ids[k];
        if (mid < 1 || mid > n_slots) continue;

        const auto [pos_t, flag] = apply_position_limit_(mid, to_turns(active_pos[k], pos_unit));
        limit_flags[mid] = flag;

        const std::size_t idx = static_cast<std::size_t>(mid - 1);
        pos_arr[idx] = turns_to_int16(pos_t);
        vel_arr[idx] = rps_to_int16(active_vel[k]);
        tqe_arr[idx] = tqe_raw;
        kp_arr[idx]  = kp_raw;
        kd_arr[idx]  = kd_raw;
        ++expected;
    }

    const auto data = build_many_pos_vel_tqe_kp_kd_int16(
        pos_arr, vel_arr, tqe_arr, kp_arr, kd_arr);
    if (timeout_s <= 0.0 || expected <= 0) {
        (void)send_can_and_recv_(0x8093, data, 0.0, 0);
        return out;
    }

    auto replies = send_can_and_recv_(0x8093, data, timeout_s, expected);
    for (auto& [rid, rdata] : replies) {
        int motor_id = (rid >> 8) & 0x7F;
        if (motor_id == 0) motor_id = rid & 0x7F;
        if (motor_id < 1 || motor_id > n_slots) continue;
        if (auto st = parse_motor_state_int16(rdata)) {
            st->id = motor_id;
            if (auto it = limit_flags.find(motor_id); it != limit_flags.end()) {
                st->pos_limit_flag = it->second;
            }
            out[motor_id] = *st;
        }
    }
    return out;
}

std::map<int, MotorState> HightorqueSerial::set_many_mit(
        const std::vector<int>&    motor_ids,
        const std::vector<double>& pos,
        const std::vector<double>& vel_rps,
        const std::vector<int>&    tqe_raw,
        const std::vector<int>&    kp_raw,
        const std::vector<int>&    kd_raw,
        PosUnit pos_unit, int max_motor_id, double timeout_s) {

    // 每关节独立 tqe/kp/kd 的一拖多 MIT (CAN ID 0x8093). tqe/kp/kd 均为
    // 原始 int16, 不做 Nm / 增益换算 (上层负责转换, 见 diag_torque_ramp
    // 已在硬件上验证的 raw 语义). pos 走软限位; vel 单位 turns/s.
    // 帧内每电机 10 字节, 单帧最多 6 个电机 (build 函数会对 >64B 抛错).
    std::map<int, MotorState> out;
    const std::size_t n = motor_ids.size();
    if (pos.size() != n || vel_rps.size() != n || tqe_raw.size() != n ||
        kp_raw.size() != n || kd_raw.size() != n) {
        throw std::invalid_argument(
            "set_many_mit: motor_ids/pos/vel/tqe/kp/kd 长度必须一致");
    }
    if (n == 0) return out;

    int n_slots = max_motor_id;
    if (n_slots <= 0) {
        for (int id : motor_ids) n_slots = std::max(n_slots, id);
    }
    if (n_slots <= 0) return out;

    std::vector<int16_t> pos_arr(n_slots, NAN_INT16);
    std::vector<int16_t> vel_arr(n_slots, NAN_INT16);
    std::vector<int16_t> tqe_arr(n_slots, NAN_INT16);
    std::vector<int16_t> kp_arr(n_slots, NAN_INT16);
    std::vector<int16_t> kd_arr(n_slots, NAN_INT16);

    std::map<int, int> limit_flags;
    int expected = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const int mid = motor_ids[k];
        if (mid < 1 || mid > n_slots) continue;

        const auto [pos_t, flag] = apply_position_limit_(mid, to_turns(pos[k], pos_unit));
        limit_flags[mid] = flag;

        const std::size_t idx = static_cast<std::size_t>(mid - 1);
        pos_arr[idx] = turns_to_int16(pos_t);
        vel_arr[idx] = rps_to_int16(vel_rps[k]);
        tqe_arr[idx] = saturate_to_i16(static_cast<long long>(tqe_raw[k]));
        kp_arr[idx]  = saturate_to_i16(static_cast<long long>(kp_raw[k]));
        kd_arr[idx]  = saturate_to_i16(static_cast<long long>(kd_raw[k]));
        ++expected;
    }

    const auto data = build_many_pos_vel_tqe_kp_kd_int16(
        pos_arr, vel_arr, tqe_arr, kp_arr, kd_arr);
    if (timeout_s <= 0.0 || expected <= 0) {
        (void)send_can_and_recv_(0x8093, data, 0.0, 0);
        return out;
    }

    auto replies = send_can_and_recv_(0x8093, data, timeout_s, expected);
    for (auto& [rid, rdata] : replies) {
        int motor_id = (rid >> 8) & 0x7F;
        if (motor_id == 0) motor_id = rid & 0x7F;
        if (motor_id < 1 || motor_id > n_slots) continue;
        if (auto st = parse_motor_state_int16(rdata)) {
            st->id = motor_id;
            if (auto it = limit_flags.find(motor_id); it != limit_flags.end()) {
                st->pos_limit_flag = it->second;
            }
            out[motor_id] = *st;
        }
    }
    return out;
}

std::optional<MotorState> HightorqueSerial::set_torque(int motor_id, double tqe_nm,
                                                       const std::string& motor_model) {
    double coeff = 1.0;
    if (auto it = TORQUE_COEFF.find(motor_model); it != TORQUE_COEFF.end()) coeff = it->second;
    // Firmware torque LSB is (coeff * 0.01) Nm/count (matches vendor
    // motor.cpp tqe_float2int: raw = Nm / (motor_tqe_adj * 0.01)). Without
    // the *0.01 the raw is 100x too small -> arm under-drives / sags.
    const int16_t tqe_int = (coeff != 0.0)
        ? finite_to_i16(tqe_nm / (coeff * 0.01), "torque") : 0;
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_torque_int16(tqe_int), t); },
        motor_id);
}

std::optional<MotorState> HightorqueSerial::set_voltage(int motor_id, double voltage_v) {
    const int16_t v = finite_to_i16(voltage_v / 0.1, "voltage");
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_voltage_int16(v), t); },
        motor_id);
}

std::optional<MotorState> HightorqueSerial::set_current(int motor_id, double current_a) {
    const int16_t c = finite_to_i16(current_a / 0.1, "current");
    return control_call(
        [&](int can_id, double t) { return send_can_and_recv_(can_id, build_current_int16(c), t); },
        motor_id);
}

std::optional<MotorState> HightorqueSerial::set_pos_vel_tqe_kp_kd(
        int motor_id, double pos, double vel_rps, double tqe_nm,
        double kp, double kd, const std::string& motor_model, PosUnit unit) {
    const auto [pos_t, flag] = apply_position_limit_(motor_id, to_turns(pos, unit));
    double coeff = 1.0;
    if (auto it = TORQUE_COEFF.find(motor_model); it != TORQUE_COEFF.end()) coeff = it->second;
    if (!std::isfinite(kp) || !std::isfinite(kd)) {
        throw std::invalid_argument("kp and kd must be finite");
    }
    // Firmware torque LSB is (coeff * 0.01) Nm/count (see set_torque note).
    const int16_t tqe_int = (coeff != 0.0)
        ? finite_to_i16(tqe_nm / (coeff * 0.01), "torque") : 0;
    const int16_t kp_int = finite_to_i16(std::max(0.0, kp), "kp");
    const int16_t kd_int = finite_to_i16(std::max(0.0, kd * 100.0), "kd");
    auto st = control_call(
        [&](int can_id, double t) {
            return send_can_and_recv_(can_id,
                build_pos_vel_tqe_kp_kd_int16(turns_to_int16(pos_t),
                                              rps_to_int16(vel_rps),
                                              tqe_int, kp_int, kd_int), t);
        },
        motor_id);
    if (st) st->pos_limit_flag = flag;
    return st;
}

std::string HightorqueSerial::reset_zero(int motor_id) {
    const int can_id = 0x8000 | motor_id;
    std::ostringstream cmd;
    cmd << "can send " << std::hex << std::uppercase << can_id
        << " " << bytes_to_hex(build_set_zero());
    return send_cmd_(cmd.str());
}

std::string HightorqueSerial::save_config(int motor_id) {
    const int can_id = 0x8000 | motor_id;
    std::ostringstream cmd;
    cmd << "can send " << std::hex << std::uppercase << can_id
        << " " << bytes_to_hex(build_conf_write());
    return send_cmd_(cmd.str());
}

std::string HightorqueSerial::motor_reset(int motor_id) {
    const int can_id = 0x8000 | motor_id;
    std::ostringstream cmd;
    cmd << "can send " << std::hex << std::uppercase << can_id
        << " " << bytes_to_hex(build_motor_reset());
    return send_cmd_(cmd.str());
}

std::optional<std::string> HightorqueSerial::read_version(int motor_id) {
    auto replies = send_can_and_recv_(0x8000 | motor_id, build_read_version());
    for (auto& [rid, rdata] : replies) {
        (void)rid;
        if (rdata.size() >= 4 && (rdata[0] & 0xF0) == 0x20 && rdata[1] == 0xB5) {
            const std::vector<uint8_t> ver_bytes(rdata.begin() + 2, rdata.end());
            if (ver_bytes.size() >= 2) {
                const uint16_t ver16 = static_cast<uint16_t>(ver_bytes[0]) |
                                       (static_cast<uint16_t>(ver_bytes[1]) << 8);
                const int major = (ver16 >> 12) & 0xF;
                const int minor = (ver16 >>  4) & 0xFF;
                const int patch =  ver16        & 0xF;
                std::ostringstream oss;
                oss << "v" << major << "." << minor << "." << patch;
                return oss.str();
            }
        }
    }
    return std::nullopt;
}

std::string HightorqueSerial::set_timeout(int motor_id, int16_t timeout_ms) {
    const int can_id = 0x8000 | motor_id;
    std::ostringstream cmd;
    cmd << "can send " << std::hex << std::uppercase << can_id
        << " " << bytes_to_hex(build_set_timeout_int16(timeout_ms));
    return send_cmd_(cmd.str());
}

// ===========================================================================
//  软限位 (驱动层内置)
// ===========================================================================

void HightorqueSerial::enable_position_limit(int motor_id, double lo, double hi, PosUnit unit) {
    double lo_t = to_turns(lo, unit);
    double hi_t = to_turns(hi, unit);
    if (!std::isfinite(lo_t) || !std::isfinite(hi_t)) {
        throw std::invalid_argument("position limits must be finite");
    }
    if (lo_t > hi_t) std::swap(lo_t, hi_t);
    std::lock_guard<std::mutex> lk(limits_mtx_);
    limits_[motor_id] = Range{lo_t, hi_t};
}

void HightorqueSerial::disable_position_limit(int motor_id) {
    std::lock_guard<std::mutex> lk(limits_mtx_);
    limits_.erase(motor_id);
}

void HightorqueSerial::clear_all_position_limits() {
    std::lock_guard<std::mutex> lk(limits_mtx_);
    limits_.clear();
}

bool HightorqueSerial::get_position_limit_turns(int motor_id, double& lo_turns, double& hi_turns) const {
    std::lock_guard<std::mutex> lk(limits_mtx_);
    auto it = limits_.find(motor_id);
    if (it == limits_.end()) return false;
    lo_turns = it->second.lo;
    hi_turns = it->second.hi;
    return true;
}

std::pair<double, int> HightorqueSerial::apply_position_limit_(int motor_id, double pos_turns) const {
    if (!std::isfinite(pos_turns)) {
        throw std::invalid_argument("position must be finite");
    }
    std::lock_guard<std::mutex> lk(limits_mtx_);
    auto it = limits_.find(motor_id);
    if (it == limits_.end()) return {pos_turns, 0};
    if (pos_turns > it->second.hi) return {it->second.hi, +1};
    if (pos_turns < it->second.lo) return {it->second.lo, -1};
    return {pos_turns, 0};
}

// ===========================================================================
//  CAN 错误码 (调试板 `can status` 解析)
//
//  一台典型的调试板回复形如 (空格分隔的 key=value):
//      lec=0 err=0 tx=0 rx=0 bus=ok
//  字段名/格式各家板子不太一致, 我们做"宽松匹配":
//    - 任意 key=value 形式
//    - 已知 key: lec, err, tx_err, tx, rx_err, rx, bus, status, fault
//    - bus 字段值: ok / warn / passive / busoff / off  (大小写不敏感)
// ===========================================================================

namespace {

CanFault classify_bus_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (s == "ok" || s == "0")              return CanFault::Ok;
    if (s == "warn" || s == "warning")      return CanFault::ErrorWarning;
    if (s == "passive")                     return CanFault::ErrorPassive;
    if (s == "busoff" || s == "off" || s == "bus_off") return CanFault::BusOff;
    return CanFault::Unknown;
}

CanFault classify_lec(int lec) {
    // lec=0 -> OK; 大致映射, 协议各版本略有差异
    if (lec <= 0) return CanFault::Ok;
    return CanFault::ErrorWarning;
}

} // namespace

std::string CanStatus::to_string() const {
    const char* f = "Unknown";
    switch (fault) {
        case CanFault::Unknown:      f = "Unknown";      break;
        case CanFault::Ok:           f = "Ok";           break;
        case CanFault::ErrorWarning: f = "ErrorWarning"; break;
        case CanFault::ErrorPassive: f = "ErrorPassive"; break;
        case CanFault::BusOff:       f = "BusOff";       break;
    }
    std::ostringstream oss;
    oss << "CanStatus(" << f << ", lec=" << lec
        << ", tx_err=" << tx_err_count
        << ", rx_err=" << rx_err_count << ")";
    return oss.str();
}

CanStatus HightorqueSerial::read_can_status() {
    CanStatus out;
    out.raw = can_status();

    // 拆 key=value tokens
    for (const auto& tok : split_ws(out.raw)) {
        const auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string key = tok.substr(0, eq);
        std::string val = tok.substr(eq + 1);
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });

        try {
            if      (key == "lec")                 out.lec = std::stoi(val, nullptr, 0);
            else if (key == "tx_err" || key == "tx" || key == "tec")
                                                   out.tx_err_count = std::stoi(val, nullptr, 0);
            else if (key == "rx_err" || key == "rx" || key == "rec")
                                                   out.rx_err_count = std::stoi(val, nullptr, 0);
            else if (key == "bus" || key == "status" || key == "fault")
                                                   out.fault = classify_bus_str(val);
        } catch (...) {
            // 忽略单个字段解析失败, 不影响其他字段
        }
    }

    // 如果没找到 bus 字段, 但有 lec, 用 lec 推断
    if (out.fault == CanFault::Unknown && out.lec >= 0) {
        out.fault = classify_lec(out.lec);
    }
    return out;
}

// ===========================================================================
//  USB 串口枚举 + 调试板候选过滤
// ===========================================================================

namespace {

// 把 "USB VID:PID=0483:5740 ..." 之类字符串拆出 vid / pid (大写)
void parse_vid_pid(const std::string& hwid, std::string& vid, std::string& pid) {
    vid.clear();
    pid.clear();
    auto pos = hwid.find("VID:PID=");
    if (pos == std::string::npos) {
        // 也支持 "VID_0483&PID_5740" Windows 风格
        auto v = hwid.find("VID_");
        auto p = hwid.find("PID_");
        if (v != std::string::npos && v + 8 <= hwid.size()) vid = hwid.substr(v + 4, 4);
        if (p != std::string::npos && p + 8 <= hwid.size()) pid = hwid.substr(p + 4, 4);
    } else {
        pos += 8;  // skip "VID:PID="
        if (pos + 9 <= hwid.size() && hwid[pos + 4] == ':') {
            vid = hwid.substr(pos, 4);
            pid = hwid.substr(pos + 5, 4);
        }
    }
    std::transform(vid.begin(), vid.end(), vid.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    std::transform(pid.begin(), pid.end(), pid.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
}

} // namespace

std::vector<PortInfo> list_serial_ports() {
    std::vector<PortInfo> out;
    for (const auto& p : serial::list_ports()) {
        PortInfo info;
        info.port        = p.port;
        info.description = p.description;
        info.hardware_id = p.hardware_id;
        parse_vid_pid(p.hardware_id, info.vid, info.pid);
        out.push_back(std::move(info));
    }
    return out;
}

std::vector<PortInfo> find_likely_debug_boards(const std::vector<std::string>& known_vids) {
    std::vector<std::string> vids_upper;
    vids_upper.reserve(known_vids.size());
    for (const auto& v : known_vids) {
        std::string u = v;
        std::transform(u.begin(), u.end(), u.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        vids_upper.push_back(std::move(u));
    }

    std::vector<PortInfo> out;
    for (auto& info : list_serial_ports()) {
        if (info.vid.empty()) continue;
        if (std::find(vids_upper.begin(), vids_upper.end(), info.vid) != vids_upper.end()) {
            out.push_back(std::move(info));
        }
    }
    return out;
}

// ===========================================================================
//  后台状态轮询线程
// ===========================================================================

void HightorqueSerial::start_state_polling(
        const std::vector<int>& motor_ids, double rate_hz,
        std::function<void(const std::vector<int>&)> on_update) {
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    stop_state_polling_unlocked_();

    if (motor_ids.empty()) return;
    if (!std::isfinite(rate_hz) || rate_hz <= 0.0) {
        throw std::invalid_argument(
            "polling rate must be positive and finite");
    }
    if (!ser_ || !ser_->isOpen()) {
        throw std::runtime_error("cannot start polling on a closed serial port");
    }

    if (!async_rx_enabled_.load()) {
        enable_async_rx_unlocked_();
        poll_owns_async_rx_ = true;
    }

    poll_running_.store(true);
    const auto period = std::chrono::microseconds(std::max<long long>(
        1, static_cast<long long>(1'000'000.0 / rate_hz)));
    auto ids = motor_ids;
    auto callback = std::move(on_update);

    try {
        poll_thread_ = std::thread(
            [this, ids = std::move(ids), period,
             callback = std::move(callback)]() mutable {
            std::map<int, uint64_t> reported_seq;
            {
                std::lock_guard<std::mutex> cache_lock(cache_mtx_);
                for (int motor_id : ids) {
                    reported_seq[motor_id] = update_seq_[motor_id];
                }
            }
            auto next_deadline =
                std::chrono::steady_clock::now() + period;

            while (poll_running_.load()) {
                for (int motor_id : ids) {
                    if (!poll_running_.load()) return;
                    try {
                        request_motor_state_(motor_id);
                    } catch (...) {
                        // A disconnect becomes stale feedback for RobotCore;
                        // never let an exception escape a background thread.
                    }
                }

                std::this_thread::sleep_until(next_deadline);
                if (!poll_running_.load()) return;

                std::vector<int> updated;
                updated.reserve(ids.size());
                {
                    std::lock_guard<std::mutex> cache_lock(cache_mtx_);
                    for (int motor_id : ids) {
                        const uint64_t current = update_seq_[motor_id];
                        if (current > reported_seq[motor_id]) {
                            reported_seq[motor_id] = current;
                            updated.push_back(motor_id);
                        }
                    }
                }
                if (callback && !updated.empty()) {
                    try { callback(updated); } catch (...) {}
                }

                next_deadline += period;
                const auto now = std::chrono::steady_clock::now();
                if (next_deadline <= now) {
                    const auto missed =
                        (now - next_deadline) / period + 1;
                    next_deadline += period * missed;
                }
            }
            });
    } catch (...) {
        poll_running_.store(false);
        if (poll_owns_async_rx_) {
            poll_owns_async_rx_ = false;
            disable_async_rx_unlocked_();
        }
        throw;
    }
}

void HightorqueSerial::stop_state_polling_unlocked_() {
    poll_running_.store(false);
    if (poll_thread_.joinable()) {
        poll_thread_.join();
    }
    if (poll_owns_async_rx_) {
        poll_owns_async_rx_ = false;
        disable_async_rx_unlocked_();
    }
}

void HightorqueSerial::stop_state_polling() {
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    stop_state_polling_unlocked_();
}

bool HightorqueSerial::is_polling() const {
    return poll_running_.load();
}

std::optional<MotorState> HightorqueSerial::get_cached_state(int motor_id) const {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    auto it = state_cache_.find(motor_id);
    if (it == state_cache_.end()) return std::nullopt;
    return it->second;
}

// ===========================================================================
//  异步 RX 线程 (Step 2 of 重构)
//
//  设计:
//    1. RX 线程独占 read 串口 (TX 线程独占 write, 互不阻塞).
//    2. 读到的字节追加到 rx_buffer_, 按 '\n' 切行 → dispatch_rcv_line_.
//       (rx_buffer_ 仅 RX 线程访问, 不需要锁)
//    3. dispatch 仅识别 'rcv 0xID HEX...' 行, 解析为 MotorState 写入 cache,
//       update_seq_[id]++, cache_cv_.notify_all() 唤醒 wait_state.
//    4. ASCII 命令的响应行 (ok/err/can status 输出) 也会被读到, 但 dispatch
//       会跳过非 'rcv ' 开头的行 — 这是 async 模式下少用 send_cmd_ 的原因.
//    5. 串口异常 → 设 rx_running_ = false 退出, 不影响主程序 (主程序自己重连).
// ===========================================================================

void HightorqueSerial::rx_loop_() {
    rx_buffer_.clear();
    while (rx_running_.load()) {
        std::size_t avail = 0;
        try {
            avail = ser_->available();
        } catch (...) {
            sleep_seconds(0.005);
            continue;
        }

        if (avail == 0) {
            sleep_seconds(0.0005);   // 0.5ms 粒度 — 100Hz 下足够 (周期 10ms)
            continue;
        }

        std::string chunk;
        try {
            ser_->read(chunk, avail);
        } catch (...) {
            sleep_seconds(0.005);
            continue;
        }
        rx_buffer_.append(chunk);

        // 按 '\n' 切行, 行内剥掉 '\r'
        std::size_t start = 0;
        while (true) {
            const auto nl = rx_buffer_.find('\n', start);
            if (nl == std::string::npos) break;
            std::string line = rx_buffer_.substr(start, nl - start);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            dispatch_rcv_line_(line);
            start = nl + 1;
        }
        if (start > 0) rx_buffer_.erase(0, start);
        // 防止某种异常下 buffer 无限涨
        if (rx_buffer_.size() > 64 * 1024) rx_buffer_.clear();
    }
}

void HightorqueSerial::dispatch_rcv_line_(const std::string& line) {
    const std::string s = strip(line);
    if (s.rfind("rcv ", 0) != 0) {
        // 非 'rcv ' 开头: 命令回显 / ok / err / can status 输出
        // 默认丢弃; 如果上层设了 debug 回调, 通知一下用于排查
        if (!s.empty()) {
            std::function<void(const std::string&)> cb;
            {
                std::lock_guard<std::mutex> lk(dbg_mtx_);
                cb = debug_line_handler_;
            }
            if (cb) {
                try { cb(s); } catch (...) {}
            }
        }
        return;
    }
    const auto parts = split_ws(s);
    if (parts.size() < 3) {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.rx_dropped++;
        return;
    }

    int rcv_id = 0;
    std::vector<uint8_t> rcv_data;
    try {
        rcv_id   = std::stoi(parts[1], nullptr, 16);
        rcv_data = hex_to_bytes(parts[2]);
    } catch (const std::exception&) {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.rx_dropped++;
        return;
    }

    {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.rx_frames++;
        last_rx_time_ = now_seconds();
    }

    // ---- 反馈帧 ID 解码 (依据 1.2 fdcan协议解析) ----
    //
    //  ID = (源地址 << 8) | 目的地址
    //   - 源地址 = 电机 ID (有些固件会带最高位 "需要回复" 标志, 需要 mask 掉)
    //   - 目的地址 = 主机地址 (通常 0)
    //
    //  ▸ id 1-7 电机用标准帧 (11-bit), 实际 rcv_id 形如 0x0100 / 0x0200 / ...
    //  ▸ id ≥ 8 电机用扩展帧 (29-bit), 高位可能更高, 同样取高 8 位作为源地址
    //
    //  之前误写成 `motor_id = rcv_id & 0xFF` 拿到的是目的地址 (恒为 0),
    //  导致所有电机的 cache 都被塞进 state_cache_[0], 互相覆盖, 上层永远读不到.
    int motor_id = (rcv_id >> 8) & 0x7F;
    if (motor_id == 0) {
        // 极少数固件可能反过来填 (源在低 8 位); 兼容一下
        motor_id = rcv_id & 0x7F;
    }
    auto st_opt = parse_motor_state_int16(rcv_data);
    if (!st_opt) {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.rx_dropped++;
        return;
    }

    st_opt->id = motor_id;

    {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        state_cache_[motor_id] = *st_opt;
        state_update_time_[motor_id] = now_seconds();
        update_seq_[motor_id]++;
    }
    {
        std::lock_guard<std::mutex> lk(stats_mtx_);
        stats_.rx_parsed++;
    }
    cache_cv_.notify_all();
}
void HightorqueSerial::enable_async_rx_unlocked_() {
    if (async_rx_enabled_.load()) return;
    if (!ser_ || !ser_->isOpen()) {
        throw std::runtime_error(
            "cannot enable asynchronous receive on a closed serial port");
    }

    rx_running_.store(true);
    async_rx_enabled_.store(true);
    try {
        rx_thread_ = std::thread(&HightorqueSerial::rx_loop_, this);
    } catch (...) {
        async_rx_enabled_.store(false);
        rx_running_.store(false);
        throw;
    }
}

void HightorqueSerial::enable_async_rx() {
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    if (async_rx_enabled_.load()) {
        // An explicit caller takes ownership from an auto-started poller.
        poll_owns_async_rx_ = false;
        return;
    }
    enable_async_rx_unlocked_();
}

void HightorqueSerial::disable_async_rx_unlocked_() {
    async_rx_enabled_.store(false);
    rx_running_.store(false);
    if (rx_thread_.joinable()) {
        rx_thread_.join();
    }
    cache_cv_.notify_all();
}

void HightorqueSerial::disable_async_rx() {
    std::lock_guard<std::mutex> lock(lifecycle_mtx_);
    stop_state_polling_unlocked_();
    disable_async_rx_unlocked_();
}

// ---------------------------------------------------------------------------
//  状态缓存读取 (公开 API)
// ---------------------------------------------------------------------------

std::optional<MotorState> HightorqueSerial::get_state(int motor_id) const {
    return get_cached_state(motor_id);
}

std::map<int, MotorState> HightorqueSerial::get_states(const std::vector<int>& motor_ids) const {
    std::map<int, MotorState> out;
    std::lock_guard<std::mutex> lk(cache_mtx_);
    for (int mid : motor_ids) {
        auto it = state_cache_.find(mid);
        if (it != state_cache_.end()) out[mid] = it->second;
    }
    return out;
}

double HightorqueSerial::get_state_age_ms(int motor_id) const {
    std::lock_guard<std::mutex> lk(cache_mtx_);
    auto it = state_update_time_.find(motor_id);
    if (it == state_update_time_.end()) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, (now_seconds() - it->second) * 1000.0);
}

std::optional<MotorState> HightorqueSerial::wait_state(int motor_id, double timeout_s) {
    // 同步模式: 退化到原 read_motor_state (会发查询 + 同步等)
    if (!async_rx_enabled_.load()) {
        return read_motor_state(motor_id, timeout_s);
    }

    // 异步模式: 记录当前 seq, 然后等 seq 增长 (== RX 线程收到了新一帧)
    uint64_t seen_seq = 0;
    {
        std::lock_guard<std::mutex> lk(cache_mtx_);
        auto it = update_seq_.find(motor_id);
        if (it != update_seq_.end()) seen_seq = it->second;
    }

    std::unique_lock<std::mutex> lk(cache_mtx_);
    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::microseconds(static_cast<long long>(timeout_s * 1e6));
    cache_cv_.wait_until(lk, deadline, [&]() {
        if (!async_rx_enabled_.load()) return true;   // 被 disable 强制醒来
        auto it = update_seq_.find(motor_id);
        return it != update_seq_.end() && it->second > seen_seq;
    });

    auto it = state_cache_.find(motor_id);
    if (it == state_cache_.end()) return std::nullopt;
    auto cur_seq_it = update_seq_.find(motor_id);
    if (cur_seq_it == update_seq_.end() || cur_seq_it->second <= seen_seq) {
        return std::nullopt;   // 超时, 没收到新帧
    }
    return it->second;
}

// ---------------------------------------------------------------------------
//  Stats
// ---------------------------------------------------------------------------

std::string Stats::to_string() const {
    std::ostringstream oss;
    oss << "Stats(tx=" << tx_frames
        << ", rx=" << rx_frames
        << ", parsed=" << rx_parsed
        << ", drop=" << rx_dropped
        << ", last_rx_age=" << std::fixed << std::setprecision(1) << last_rx_age_ms << "ms"
        << ", avg_period=" << std::fixed << std::setprecision(2) << avg_tx_period_ms << "ms"
        << ", max_jitter=" << std::fixed << std::setprecision(2) << max_tx_jitter_ms << "ms)";
    return oss.str();
}

Stats HightorqueSerial::get_stats() const {
    std::lock_guard<std::mutex> lk(stats_mtx_);
    Stats s = stats_;
    if (last_rx_time_ > 0) {
        s.last_rx_age_ms = (now_seconds() - last_rx_time_) * 1000.0;
    } else {
        s.last_rx_age_ms = -1.0;
    }
    return s;
}

void HightorqueSerial::reset_stats() {
    std::lock_guard<std::mutex> lk(stats_mtx_);
    stats_           = Stats{};
    last_tx_time_    = 0.0;
    last_rx_time_    = -1.0;
    tx_periods_.clear();
}

void HightorqueSerial::set_debug_line_handler(std::function<void(const std::string&)> handler) {
    std::lock_guard<std::mutex> lk(dbg_mtx_);
    debug_line_handler_ = std::move(handler);
}

// ===========================================================================
//  控制循环顶层封装
//
//  关键实现:
//    - 用 steady_clock 校准, 而不是简单的 sleep_for(period)
//      (避免 jitter 累积; 漂移补偿)
//    - 异常包一层 try/catch: on_tick 用户代码挂了不让程序崩, 进入收尾流程
//    - abort_check 在 sleep_for 期间也被检查, 不会等满一个周期才响应
// ===========================================================================
int HightorqueSerial::run_control_loop(const ControlLoopOptions& opt, ControlTickFn on_tick) {
    using namespace std::chrono;
    if (!on_tick) return 0;
    if (!std::isfinite(opt.rate_hz) || opt.rate_hz <= 0.0) return 0;

    // ⚠️ 关键: Windows 上把系统时钟粒度从 15.6ms 临时调到 1ms.
    // 不这么干, std::this_thread::sleep_for(1ms) 实际会睡 ~15ms,
    // 250Hz/4ms 控制环根本跑不起来 (jitter 会到 15ms+).
    // RAII: 控制环退出时自动还原.
    HighResolutionTimer hr_timer;

    // Polling already owns the receive path. Only create a temporary async
    // receiver when neither receive mode was active, and undo only that
    // temporary change on exit.
    const bool was_async = is_async_rx();
    const bool was_polling = is_polling();
    const bool temporary_async = !was_async && !was_polling;
    if (temporary_async) enable_async_rx();
    reset_stats();

    const auto period = microseconds(std::max<long long>(
        1, static_cast<long long>(1'000'000.0 / opt.rate_hz)));

    int return_code = 0;
    int tick = 0;
    auto next_deadline = steady_clock::now() + period;
    auto last_tick_t = steady_clock::now();

    try {
        while (true) {
            if (opt.abort_check && opt.abort_check()) {
                return_code = 1;
                break;
            }

            const auto now = steady_clock::now();
            const double dt_ms =
                duration<double, std::milli>(now - last_tick_t).count();
            last_tick_t = now;

            if (!on_tick(tick++, dt_ms)) {
                break;
            }

            while (true) {
                if (opt.abort_check && opt.abort_check()) {
                    return_code = 1;
                    goto loop_exit;
                }
                const auto current = steady_clock::now();
                if (current >= next_deadline) break;
                const auto remaining = next_deadline - current;
                std::this_thread::sleep_for(std::min<microseconds>(
                    duration_cast<microseconds>(remaining),
                    microseconds(1000)));
            }

            next_deadline += period;
            if (steady_clock::now() - next_deadline > period * 2) {
                next_deadline = steady_clock::now() + period;
            }
        }
    loop_exit:;
    } catch (const std::exception& e) {
        return_code = 2;
        if (opt.on_exception) {
            try {
                opt.on_exception(e);
            } catch (...) {
            }
        }
    } catch (...) {
        return_code = 2;
        if (opt.on_exception) {
            try {
                const std::runtime_error error("unknown control-loop exception");
                opt.on_exception(error);
            } catch (...) {
            }
        }
    }

    const bool should_stop =
        return_code == 0 ? opt.stop_on_finish : opt.stop_on_abort;
    if (should_stop) {
        for (int motor_id : opt.stop_motor_ids) {
            try {
                stop(motor_id);
            } catch (...) {
            }
        }
    }

    if (temporary_async) disable_async_rx();
    return return_code;
}

} // namespace hightorque
