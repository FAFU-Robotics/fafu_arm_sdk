// =============================================================================
//  bindings.cpp
//  pybind11 绑定: 把调试板 C++ API 暴露给 Python
//
//  生成的 Python 模块: fafu_motor (import fafu_motor)
//
//  覆盖的 C++ 接口:
//    - 枚举: PosUnit, CanFault
//    - 结构体: MotorState, CanStatus, PortInfo, Stats, RobotConfig, ManyMotorCmd
//    - 自由函数: list_serial_ports, find_likely_debug_boards,
//                to_turns / from_turns, parse_motor_state_int16
//    - 类 HightorqueSerial: 全部 public 方法 (含 set_motor_mode / run_control_loop)
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // std::vector / std::map / std::optional / std::pair
#include <pybind11/functional.h>   // std::function
#include <pybind11/numpy.h>        // py::array_t (servoJ partial broadcast, zero-copy)

#include "hightorque_serial.hpp"
#include "fafu/core/bindings.hpp"
#include "robot_config.hpp"

#ifdef _WIN32
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#endif

#include <string>
#include <cstring>     // std::memcpy (numpy buffer -> std::vector fast path)
#include <algorithm>   // std::clamp

namespace py = pybind11;
using namespace hightorque;

// ----------------------------------------------------------------------------
//  把"任意编码"的 std::string 安全转成 UTF-8 std::string.
//  Windows 下系统调用 (setup API) 返回的是 ACP (中文系统就是 GBK), 直接
//  当 UTF-8 给 pybind11 会抛 UnicodeDecodeError. 这里先 ACP -> UTF-16 -> UTF-8.
//  其他平台直接原样返回.
// ----------------------------------------------------------------------------
static std::string to_utf8_safe(const std::string& s) {
#ifdef _WIN32
    if (s.empty()) return s;
    // 1) 尝试当 UTF-8 验证: MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS)
    int wn = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                 s.data(), (int)s.size(), nullptr, 0);
    if (wn > 0) return s;   // 已是合法 UTF-8

    // 2) 当 ACP (中文系统通常是 936 GBK) -> UTF-16 -> UTF-8
    int n16 = MultiByteToWideChar(CP_ACP, 0, s.data(), (int)s.size(), nullptr, 0);
    if (n16 <= 0) return std::string();
    std::wstring w(n16, L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.data(), (int)s.size(), &w[0], n16);

    int n8 = WideCharToMultiByte(CP_UTF8, 0, w.data(), n16, nullptr, 0, nullptr, nullptr);
    if (n8 <= 0) return std::string();
    std::string out(n8, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), n16, &out[0], n8, nullptr, nullptr);
    return out;
#else
    return s;
#endif
}

// 把 PortInfo 里所有可能含中文的字段做一次 to_utf8_safe
static PortInfo sanitize_port_info(PortInfo p) {
    p.description = to_utf8_safe(p.description);
    p.hardware_id = to_utf8_safe(p.hardware_id);
    return p;
}

PYBIND11_MODULE(fafu_motor, m) {
    m.doc() = "Fafu 调试板 Python 绑定 (pybind11 over 调试板串口协议 C++)";

    // ----------------------------------------------------------------------
    //  枚举
    // ----------------------------------------------------------------------
    py::enum_<PosUnit>(m, "PosUnit", "位置单位制")
        .value("Turns",   PosUnit::Turns,   "圈 (协议原生)")
        .value("Radians", PosUnit::Radians, "弧度")
        .value("Degrees", PosUnit::Degrees, "角度")
        .export_values();

    py::enum_<CanFault>(m, "CanFault", "CAN 总线状态")
        .value("Unknown",      CanFault::Unknown)
        .value("Ok",           CanFault::Ok)
        .value("ErrorWarning", CanFault::ErrorWarning)
        .value("ErrorPassive", CanFault::ErrorPassive)
        .value("BusOff",       CanFault::BusOff)
        .export_values();

    // ----------------------------------------------------------------------
    //  单位换算 + 工具函数
    // ----------------------------------------------------------------------
    m.def("to_turns",   &to_turns,   py::arg("value"), py::arg("unit"),
          "把任意单位的位置值换算成 '圈' (协议原生单位)");
    m.def("from_turns", &from_turns, py::arg("turns"), py::arg("unit"),
          "把 '圈' 换算成指定单位");

    // ----------------------------------------------------------------------
    //  结构体
    // ----------------------------------------------------------------------
    py::class_<MotorState>(m, "MotorState", "电机状态")
        .def(py::init<>())
        .def_readwrite("id",             &MotorState::id)
        .def_readwrite("mode",           &MotorState::mode)
        .def_readwrite("fault",          &MotorState::fault)
        .def_readwrite("position",       &MotorState::position, "位置 (圈)")
        .def_readwrite("velocity",       &MotorState::velocity, "速度 (转/秒)")
        .def_readwrite("torque",         &MotorState::torque,   "力矩 (raw int16)")
        .def_readwrite("pos_limit_flag", &MotorState::pos_limit_flag,
                       "软限位标志: 0=未触发, +1=超上限, -1=超下限")
        .def("to_string", &MotorState::to_string)
        .def("__repr__", [](const MotorState& s) {
            return "<MotorState id=" + std::to_string(s.id) +
                   " pos=" + std::to_string(s.position) +
                   " vel=" + std::to_string(s.velocity) +
                   " mode=" + std::to_string(s.mode) + ">";
        });

    py::class_<CanStatus>(m, "CanStatus", "CAN 总线状态结构")
        .def(py::init<>())
        .def_readwrite("fault",        &CanStatus::fault)
        .def_readwrite("lec",          &CanStatus::lec)
        .def_readwrite("tx_err_count", &CanStatus::tx_err_count)
        .def_readwrite("rx_err_count", &CanStatus::rx_err_count)
        .def_readwrite("raw",          &CanStatus::raw)
        .def("is_ok",     &CanStatus::is_ok)
        .def("to_string", &CanStatus::to_string)
        .def("__repr__", [](const CanStatus& s) { return "<CanStatus " + s.to_string() + ">"; });

    py::class_<PortInfo>(m, "PortInfo", "串口信息")
        .def(py::init<>())
        .def_readwrite("port",        &PortInfo::port)
        .def_readwrite("description", &PortInfo::description)
        .def_readwrite("hardware_id", &PortInfo::hardware_id)
        .def_readwrite("vid",         &PortInfo::vid)
        .def_readwrite("pid",         &PortInfo::pid)
        .def("__repr__", [](const PortInfo& p) {
            return "<PortInfo " + p.port + " VID=" + p.vid + " PID=" + p.pid + " " + p.description + ">";
        });

    py::class_<Stats>(m, "Stats", "收发统计")
        .def(py::init<>())
        .def_readwrite("tx_frames",        &Stats::tx_frames)
        .def_readwrite("rx_frames",        &Stats::rx_frames)
        .def_readwrite("rx_parsed",        &Stats::rx_parsed)
        .def_readwrite("rx_dropped",       &Stats::rx_dropped)
        .def_readwrite("last_rx_age_ms",   &Stats::last_rx_age_ms)
        .def_readwrite("avg_tx_period_ms", &Stats::avg_tx_period_ms)
        .def_readwrite("max_tx_jitter_ms", &Stats::max_tx_jitter_ms)
        .def("to_string", &Stats::to_string);

    // 串口枚举 (description/hardware_id 在中文 Windows 下是 GBK,
    // 这里 wrap 一层把它转成 UTF-8 再交给 pybind11)
    m.def("list_serial_ports", []() {
        auto ports = list_serial_ports();
        for (auto& p : ports) p = sanitize_port_info(std::move(p));
        return ports;
    }, "列出系统全部串口 (description/hardware_id 已转 UTF-8)");

    m.def("find_likely_debug_boards",
          [](const std::vector<std::string>& known_vids) {
              auto ports = find_likely_debug_boards(known_vids);
              for (auto& p : ports) p = sanitize_port_info(std::move(p));
              return ports;
          },
          py::arg("known_vids") = std::vector<std::string>{"0483","0403","1A86","10C4","067B"},
          "按常见 USB-Serial 调试板 VID 过滤候选");
    m.def("parse_motor_state_int16", &parse_motor_state_int16,
          py::arg("can_data"), "解析电机回复的 int16 状态帧");

    // ----------------------------------------------------------------------
    //  RobotConfig
    // ----------------------------------------------------------------------
    py::class_<RobotConfig>(m, "RobotConfig", "INI 风格的机器人配置")
        .def(py::init<>())
        .def_readwrite("port",            &RobotConfig::port)
        .def_readwrite("baudrate",        &RobotConfig::baudrate)
        .def_readwrite("motor_ids",       &RobotConfig::motor_ids)
        .def_readwrite("pos_unit",        &RobotConfig::pos_unit)
        .def_readwrite("limits",          &RobotConfig::limits,
                       "字典: motor_id -> (lo, hi) (圈)")
        .def_readwrite("control_rate_hz", &RobotConfig::control_rate_hz)
        .def_readwrite("max_torque_raw",  &RobotConfig::max_torque_raw)
        .def_readwrite("use_async_rx",    &RobotConfig::use_async_rx)
        .def_readwrite("trajectory_dt_s", &RobotConfig::trajectory_dt_s)
        .def("find_limit",      &RobotConfig::find_limit,      py::arg("motor_id"))
        .def("apply_limits_to", &RobotConfig::apply_limits_to, py::arg("ht"))
        .def("to_string",       &RobotConfig::to_string)
        .def_static("load",     &RobotConfig::load,            py::arg("path"),
                    "从文件加载 (失败抛 RuntimeError)")
        .def("__repr__", [](const RobotConfig& c) { return "<RobotConfig\n" + c.to_string() + "\n>"; });

    // ----------------------------------------------------------------------
    //  HightorqueSerial::ManyMotorCmd 嵌套类型
    // ----------------------------------------------------------------------
    py::class_<HightorqueSerial::ManyMotorCmd>(m, "ManyMotorCmd",
        "一拖多模式下的单电机指令")
        .def(py::init<>())
        .def(py::init([](int motor_id, double pos, double vel_rps, int tqe_raw) {
            HightorqueSerial::ManyMotorCmd c;
            c.motor_id = motor_id;
            c.pos      = pos;
            c.vel_rps  = vel_rps;
            c.tqe_raw  = tqe_raw;
            return c;
        }), py::arg("motor_id"), py::arg("pos"), py::arg("vel_rps"), py::arg("tqe_raw") = 0)
        .def_readwrite("motor_id", &HightorqueSerial::ManyMotorCmd::motor_id)
        .def_readwrite("pos",      &HightorqueSerial::ManyMotorCmd::pos)
        .def_readwrite("vel_rps",  &HightorqueSerial::ManyMotorCmd::vel_rps)
        .def_readwrite("tqe_raw",  &HightorqueSerial::ManyMotorCmd::tqe_raw);

    // ----------------------------------------------------------------------
    //  HightorqueSerial 主类
    //  关键: 释放 GIL 让 Web 线程能并发查询其它接口 (尤其 Flask 多请求时)
    // ----------------------------------------------------------------------
    py::class_<HightorqueSerial>(m, "HightorqueSerial",
        "调试板 USB 串口驱动 (CAN-FD 透传)")
        .def(py::init<const std::string&, uint32_t>(),
             py::arg("port"), py::arg("baudrate") = 4'000'000u,
             "打开串口, 失败抛 RuntimeError")

        // -- 基础 --
        .def("close",   &HightorqueSerial::close,   py::call_guard<py::gil_scoped_release>())
        .def("is_open", &HightorqueSerial::is_open)

        // -- 异步 RX --
        .def("enable_async_rx",  &HightorqueSerial::enable_async_rx,
             py::call_guard<py::gil_scoped_release>())
        .def("disable_async_rx", &HightorqueSerial::disable_async_rx,
             py::call_guard<py::gil_scoped_release>())
        .def("is_async_rx",      &HightorqueSerial::is_async_rx)

        .def("get_state",  &HightorqueSerial::get_state,  py::arg("motor_id"))
        .def("get_states", &HightorqueSerial::get_states, py::arg("motor_ids"))
        .def("wait_state", &HightorqueSerial::wait_state,
             py::arg("motor_id"), py::arg("timeout_s") = 0.1,
             py::call_guard<py::gil_scoped_release>())

        .def("get_stats",   &HightorqueSerial::get_stats)
        .def("reset_stats", &HightorqueSerial::reset_stats)

        // -- 调试板命令 --
        .def("can_status",      &HightorqueSerial::can_status,
             py::call_guard<py::gil_scoped_release>())
        .def("read_can_status", &HightorqueSerial::read_can_status,
             py::call_guard<py::gil_scoped_release>())
        .def("can_config",      &HightorqueSerial::can_config,
             py::call_guard<py::gil_scoped_release>())

        // -- 软限位 --
        .def("enable_position_limit",     &HightorqueSerial::enable_position_limit,
             py::arg("motor_id"), py::arg("lo"), py::arg("hi"),
             py::arg("unit") = PosUnit::Turns)
        .def("disable_position_limit",    &HightorqueSerial::disable_position_limit,
             py::arg("motor_id"))
        .def("clear_all_position_limits", &HightorqueSerial::clear_all_position_limits)
        .def("get_position_limit_turns",
             [](const HightorqueSerial& self, int motor_id)
                 -> std::optional<std::pair<double, double>> {
                 double lo = 0.0, hi = 0.0;
                 if (self.get_position_limit_turns(motor_id, lo, hi))
                     return std::make_pair(lo, hi);
                 return std::nullopt;
             }, py::arg("motor_id"),
             "返回 (lo, hi) 圈; 未设限位时返回 None")

        // -- 电机控制 (释放 GIL: 串口 IO 是慢操作) --
        .def("read_motor_state", &HightorqueSerial::read_motor_state,
             py::arg("motor_id"), py::arg("timeout_s") = 0.5,
             py::call_guard<py::gil_scoped_release>())
        .def("stop",             &HightorqueSerial::stop,  py::arg("motor_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("brake",            &HightorqueSerial::brake, py::arg("motor_id"),
             py::call_guard<py::gil_scoped_release>())

        // 上电默认 mode=0; 一拖多前须切到 0x0A(10). 须在 enable_async_rx 之前调用以便读回执验证.
        .def("set_motor_mode",   &HightorqueSerial::set_motor_mode,
             py::arg("motor_id"), py::arg("mode"),
             py::call_guard<py::gil_scoped_release>())

        .def("set_position",     &HightorqueSerial::set_position,
             py::arg("motor_id"), py::arg("pos"), py::arg("unit") = PosUnit::Turns,
             py::call_guard<py::gil_scoped_release>())
        .def("set_velocity",     &HightorqueSerial::set_velocity,
             py::arg("motor_id"), py::arg("vel_rps"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_pos_vel_tqe",  &HightorqueSerial::set_pos_vel_tqe,
             py::arg("motor_id"), py::arg("pos"), py::arg("vel_rps"), py::arg("tqe_raw"),
             py::arg("unit") = PosUnit::Turns,
             py::call_guard<py::gil_scoped_release>())
        .def("set_pos_vel_tqe_rad", &HightorqueSerial::set_pos_vel_tqe_rad,
             py::arg("motor_id"), py::arg("pos_rad"),
             py::arg("vel_rad_s"), py::arg("tqe_raw"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_pos_vel_acc",  &HightorqueSerial::set_pos_vel_acc,
             py::arg("motor_id"), py::arg("pos"), py::arg("vel_max_rps"), py::arg("acc_rpss"),
             py::arg("unit") = PosUnit::Turns,
             py::call_guard<py::gil_scoped_release>())
        .def("set_pos_vel_acc_rad", &HightorqueSerial::set_pos_vel_acc_rad,
             py::arg("motor_id"), py::arg("pos_rad"),
             py::arg("vel_max_rad_s"), py::arg("acc_rad_s2"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_vel_acc",      &HightorqueSerial::set_vel_acc,
             py::arg("motor_id"), py::arg("vel_rps"), py::arg("acc_rpss"),
             py::call_guard<py::gil_scoped_release>())

        .def("set_many_pos_vel_tqe", &HightorqueSerial::set_many_pos_vel_tqe,
             py::arg("cmds"), py::arg("pos_unit") = PosUnit::Turns,
             py::arg("max_motor_id") = 0, py::arg("timeout_s") = 0.05,
             py::call_guard<py::gil_scoped_release>())

        // -------- servoJ / teleop 热路径专用 (numpy zero-copy 风格) --------
        //
        // Python 端用法:
        //   ht.set_many_pos_vel_tqe_partial(
        //       active_ids_np,        # int32 ndarray (N,)
        //       active_pos_np,        # float64 ndarray (N,)
        //       active_vel_np,        # float64 ndarray (N,)
        //       tqe_raw,              # int (int16 range), 0 = 默认最大力矩
        //       hold_ids_np,          # int32 ndarray (M,), 用 cached pos hold
        //       pos_unit=PosUnit.Turns,
        //       max_motor_id=7,
        //       timeout_s=0.0)        # 0 = 异步发, 不等回包
        //
        // 数组通过 buffer protocol 直接拷成 std::vector (一次 memcpy);
        // 比 list of ManyMotorCmd 对象 marshalling 快 ~40x.
        .def("set_many_pos_vel_tqe_partial",
             [](HightorqueSerial& self,
                py::array_t<int,    py::array::c_style | py::array::forcecast> active_ids_arr,
                py::array_t<double, py::array::c_style | py::array::forcecast> active_pos_arr,
                py::array_t<double, py::array::c_style | py::array::forcecast> active_vel_arr,
                int                                                            tqe_raw,
                py::array_t<int,    py::array::c_style | py::array::forcecast> hold_ids_arr,
                PosUnit pos_unit, int max_motor_id, double timeout_s) {
                 if (active_ids_arr.ndim() != 1 || active_pos_arr.ndim() != 1 ||
                     active_vel_arr.ndim() != 1 || hold_ids_arr.ndim()   != 1) {
                     throw std::runtime_error(
                         "set_many_pos_vel_tqe_partial: 所有数组必须是 1-D");
                 }
                 const size_t n = static_cast<size_t>(active_ids_arr.shape(0));
                 const size_t m = static_cast<size_t>(hold_ids_arr.shape(0));
                 if (static_cast<size_t>(active_pos_arr.shape(0)) != n ||
                     static_cast<size_t>(active_vel_arr.shape(0)) != n) {
                     throw std::runtime_error(
                         "set_many_pos_vel_tqe_partial: active_ids/pos/vel 长度必须一致");
                 }
                 // memcpy 风格构造 vector (forcecast 已经保证 contiguous + dtype).
                 // 用 assign 而不是 (iter, iter) 构造, 避开 size_t/iterator overload
                 // resolution 的歧义 (MSVC 在 ssize_t 上选错 ctor).
                 std::vector<int>    active_ids(n);
                 std::vector<double> active_pos(n);
                 std::vector<double> active_vel(n);
                 std::vector<int>    hold_ids(m);
                 if (n) {
                     std::memcpy(active_ids.data(), active_ids_arr.data(), n * sizeof(int));
                     std::memcpy(active_pos.data(), active_pos_arr.data(), n * sizeof(double));
                     std::memcpy(active_vel.data(), active_vel_arr.data(), n * sizeof(double));
                 }
                 if (m) {
                     std::memcpy(hold_ids.data(), hold_ids_arr.data(), m * sizeof(int));
                 }
                 const int16_t tqe_i16 = static_cast<int16_t>(
                     std::clamp(tqe_raw, -32768, 32767));
                 py::gil_scoped_release release;
                 return self.set_many_pos_vel_tqe_partial(
                     active_ids, active_pos, active_vel, tqe_i16, hold_ids,
                     pos_unit, max_motor_id, timeout_s);
             },
             py::arg("active_ids"), py::arg("active_pos"), py::arg("active_vel"),
             py::arg("tqe_raw"), py::arg("hold_ids"),
             py::arg("pos_unit") = PosUnit::Turns,
             py::arg("max_motor_id") = 0, py::arg("timeout_s") = 0.0)

        .def("set_many_pos_vel_tqe_kp_kd_partial",
             [](HightorqueSerial& self,
                py::array_t<int,    py::array::c_style | py::array::forcecast> active_ids_arr,
                py::array_t<double, py::array::c_style | py::array::forcecast> active_pos_arr,
                py::array_t<double, py::array::c_style | py::array::forcecast> active_vel_arr,
                int tqe_raw, int kp_raw, int kd_raw,
                PosUnit pos_unit, int max_motor_id, double timeout_s) {
                 if (active_ids_arr.ndim() != 1 || active_pos_arr.ndim() != 1 ||
                     active_vel_arr.ndim() != 1) {
                     throw std::runtime_error(
                         "set_many_pos_vel_tqe_kp_kd_partial: 所有数组必须是 1-D");
                 }
                 const size_t n = static_cast<size_t>(active_ids_arr.shape(0));
                 if (static_cast<size_t>(active_pos_arr.shape(0)) != n ||
                     static_cast<size_t>(active_vel_arr.shape(0)) != n) {
                     throw std::runtime_error(
                         "set_many_pos_vel_tqe_kp_kd_partial: active_ids/pos/vel 长度必须一致");
                 }
                 std::vector<int>    active_ids(n);
                 std::vector<double> active_pos(n);
                 std::vector<double> active_vel(n);
                 if (n) {
                     std::memcpy(active_ids.data(), active_ids_arr.data(), n * sizeof(int));
                     std::memcpy(active_pos.data(), active_pos_arr.data(), n * sizeof(double));
                     std::memcpy(active_vel.data(), active_vel_arr.data(), n * sizeof(double));
                 }
                 const int16_t tqe_i16 = static_cast<int16_t>(
                     std::clamp(tqe_raw, -32768, 32767));
                 const int16_t kp_i16 = static_cast<int16_t>(
                     std::clamp(kp_raw, -32768, 32767));
                 const int16_t kd_i16 = static_cast<int16_t>(
                     std::clamp(kd_raw, -32768, 32767));
                 py::gil_scoped_release release;
                 return self.set_many_pos_vel_tqe_kp_kd_partial(
                     active_ids, active_pos, active_vel,
                     tqe_i16, kp_i16, kd_i16, pos_unit, max_motor_id, timeout_s);
             },
             py::arg("active_ids"), py::arg("active_pos"), py::arg("active_vel"),
             py::arg("tqe_raw"), py::arg("kp_raw"), py::arg("kd_raw"),
             py::arg("pos_unit") = PosUnit::Turns,
             py::arg("max_motor_id") = 0, py::arg("timeout_s") = 0.0)

        .def("set_many_pos_vel_tqe_kp_kd_one",
             [](HightorqueSerial& self,
                int motor_id, double pos, double vel_rps,
                int tqe_raw, int kp_raw, int kd_raw,
                PosUnit pos_unit, int max_motor_id, double timeout_s) {
                 // 一拖多 MIT/PD 单帧最多 6 个电机 (6*10+2=62 <= 64). 槽位数
                 // 只需覆盖到被测电机即可, 不能盲目撑到 7 (会 72 字节溢出).
                 if (max_motor_id < motor_id) max_motor_id = motor_id;
                 if (max_motor_id <= 0 || max_motor_id > 6) {
                     max_motor_id = std::min(std::max(motor_id, 1), 6);
                 }
                 const int16_t tqe_i16 = static_cast<int16_t>(
                     std::clamp(tqe_raw, -32768, 32767));
                 const int16_t kp_i16 = static_cast<int16_t>(
                     std::clamp(kp_raw, -32768, 32767));
                 const int16_t kd_i16 = static_cast<int16_t>(
                     std::clamp(kd_raw, -32768, 32767));
                 std::vector<int> active_ids{motor_id};
                 std::vector<double> active_pos{pos};
                 std::vector<double> active_vel{vel_rps};
                 py::gil_scoped_release release;
                 return self.set_many_pos_vel_tqe_kp_kd_partial(
                     active_ids, active_pos, active_vel,
                     tqe_i16, kp_i16, kd_i16, pos_unit, max_motor_id, timeout_s);
             },
             py::arg("motor_id"), py::arg("pos"), py::arg("vel_rps"),
             py::arg("tqe_raw"), py::arg("kp_raw"), py::arg("kd_raw"),
             py::arg("pos_unit") = PosUnit::Turns,
             py::arg("max_motor_id") = 0, py::arg("timeout_s") = 0.0)

        // ★ 每关节独立 tqe/kp/kd 的一拖多 MIT (CAN ID 0x8093).
        //   Python 端传 list/ndarray 均可 (stl.h 自动转 std::vector):
        //     ht.set_many_mit([1,2,3,4,5,6], pos, vel, tqe_raw, kp_raw, kd_raw,
        //                     pos_unit=PosUnit.Radians, max_motor_id=6)
        //   tqe/kp/kd 为原始 int16; 单帧最多 6 个电机.
        .def("set_many_mit", &HightorqueSerial::set_many_mit,
             py::arg("motor_ids"), py::arg("pos"), py::arg("vel_rps"),
             py::arg("tqe_raw"), py::arg("kp_raw"), py::arg("kd_raw"),
             py::arg("pos_unit") = PosUnit::Turns,
             py::arg("max_motor_id") = 0, py::arg("timeout_s") = 0.05,
             py::call_guard<py::gil_scoped_release>())
        .def("set_many_mit_rad", &HightorqueSerial::set_many_mit_rad,
             py::arg("motor_ids"), py::arg("pos_rad"),
             py::arg("vel_rad_s"), py::arg("tqe_raw"),
             py::arg("kp_raw"), py::arg("kd_raw"),
             py::arg("max_motor_id") = 0, py::arg("timeout_s") = 0.05,
             py::call_guard<py::gil_scoped_release>())

        .def("set_torque",  &HightorqueSerial::set_torque,
             py::arg("motor_id"), py::arg("tqe_nm"), py::arg("motor_model") = "",
             py::call_guard<py::gil_scoped_release>())
        .def("set_voltage", &HightorqueSerial::set_voltage,
             py::arg("motor_id"), py::arg("voltage_v"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_current", &HightorqueSerial::set_current,
             py::arg("motor_id"), py::arg("current_a"),
             py::call_guard<py::gil_scoped_release>())

        .def("set_pos_vel_tqe_kp_kd", &HightorqueSerial::set_pos_vel_tqe_kp_kd,
             py::arg("motor_id"), py::arg("pos"), py::arg("vel_rps"),
             py::arg("tqe_nm"), py::arg("kp"), py::arg("kd"),
             py::arg("motor_model") = "", py::arg("unit") = PosUnit::Turns,
             py::call_guard<py::gil_scoped_release>())
        .def("set_pos_vel_tqe_kp_kd_rad",
             &HightorqueSerial::set_pos_vel_tqe_kp_kd_rad,
             py::arg("motor_id"), py::arg("pos_rad"),
             py::arg("vel_rad_s"), py::arg("tqe_nm"),
             py::arg("kp"), py::arg("kd"), py::arg("motor_model") = "",
             py::call_guard<py::gil_scoped_release>())

        // -- 配置/版本 --
        .def("reset_zero",  &HightorqueSerial::reset_zero,  py::arg("motor_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("save_config", &HightorqueSerial::save_config, py::arg("motor_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("motor_reset", &HightorqueSerial::motor_reset, py::arg("motor_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("read_version", &HightorqueSerial::read_version, py::arg("motor_id"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_timeout", &HightorqueSerial::set_timeout,
             py::arg("motor_id"), py::arg("timeout_ms"),
             py::call_guard<py::gil_scoped_release>())

        // 固定频率控制环 (内部 steady_clock 定时 + 可选 HighResolutionTimer).
        // on_tick(tick, dt_ms) 返回 False 则正常结束. abort_check 可选, 返回 True 则紧急退出.
        .def("run_control_loop", [](HightorqueSerial& self,
                                    double rate_hz,
                                    const std::vector<int>& stop_motor_ids,
                                    py::function on_tick,
                                    py::object abort_check,
                                    py::object on_exception,
                                    bool stop_on_finish,
                                    bool stop_on_abort) -> int {
            HightorqueSerial::ControlLoopOptions opt;
            opt.rate_hz         = rate_hz;
            opt.stop_motor_ids  = stop_motor_ids;
            opt.stop_on_finish  = stop_on_finish;
            opt.stop_on_abort   = stop_on_abort;
            if (!abort_check.is_none()) {
                py::object ac = py::reinterpret_borrow<py::object>(abort_check);
                opt.abort_check = [ac]() -> bool {
                    py::gil_scoped_acquire gil;
                    return py::cast<bool>(ac());
                };
            }
            if (!on_exception.is_none()) {
                py::object ox = py::reinterpret_borrow<py::object>(on_exception);
                opt.on_exception = [ox](const std::exception& e) {
                    py::gil_scoped_acquire gil;
                    try {
                        ox(py::str(e.what()));
                    } catch (...) {}
                };
            }
            py::object ot = std::move(on_tick);
            auto tick_fn = [ot](int tick, double period_ms) -> bool {
                py::gil_scoped_acquire gil;
                py::object r = ot(tick, period_ms);
                return py::cast<bool>(r);
            };
            // Keep all py::object construction/destruction under the GIL.
            py::gil_scoped_release release;
            return self.run_control_loop(opt, std::move(tick_fn));
        },
        py::arg("rate_hz"),
        py::arg("stop_motor_ids"),
        py::arg("on_tick"),
        py::arg("abort_check") = py::none(),
        py::arg("on_exception") = py::none(),
        py::arg("stop_on_finish") = false,
        py::arg("stop_on_abort") = true)

        // -- 后台轮询 --
        .def("start_state_polling", &HightorqueSerial::start_state_polling,
             py::arg("motor_ids"), py::arg("rate_hz") = 50.0,
             py::arg("on_update") = std::function<void(const std::vector<int>&)>{},
             py::call_guard<py::gil_scoped_release>())
        .def("stop_state_polling",  &HightorqueSerial::stop_state_polling,
             py::call_guard<py::gil_scoped_release>())
        .def("is_polling",          &HightorqueSerial::is_polling)
        .def("get_cached_state",    &HightorqueSerial::get_cached_state, py::arg("motor_id"))
        .def("get_state_age_ms",    &HightorqueSerial::get_state_age_ms,
             py::arg("motor_id"));

    fafu::core::bind_core(m);
}
