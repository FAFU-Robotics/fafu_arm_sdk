// =============================================================================
//  robot_config.hpp
//  轻量配置文件 (INI / key=value 风格), header-only, 无第三方依赖.
//
//  支持的语法:
//    # 这是注释                  (整行注释, 也支持 ';')
//    port = COM14                (字符串)
//    baudrate = 4000000          (整数)
//    motor_ids = 1,2,3,4,5,6,7   (整数列表, 逗号或空格分隔)
//    pos_unit = turns            (turns / radians / degrees)
//    limits.1 = -0.40, 0.30      (浮点对; 圈)
//    limits.2 = -0.05, 0.48
//    limits.3 = 0.00, 0.47
//
//  使用:
//    #include "robot_config.hpp"
//    auto cfg = RobotConfig::load("robot.cfg");
//    for (int mid : cfg.motor_ids) ...
//    if (auto lim = cfg.find_limit(2)) auto [lo, hi] = *lim;
// =============================================================================
#pragma once

#include "hightorque_serial.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hightorque {

struct RobotConfig {
    // -- 基本字段 --
    std::string                  port      = "COM14";
    uint32_t                     baudrate  = 4'000'000u;
    std::vector<int>             motor_ids = {};
    PosUnit                      pos_unit  = PosUnit::Turns;
    std::map<int, std::pair<double, double>> limits;   // 圈; 不论 pos_unit 如何, 内部归一为圈

    // -- 控制环参数 (高频控制相关) --
    double                       control_rate_hz = 100.0;   // run_control_loop 默认频率
    int                          max_torque_raw  = 0;       // 一拖多发送时的 tqe_raw, 0 = 不限
    bool                         use_async_rx    = true;    // 11 是否启用异步 RX 100Hz 模式
    double                       trajectory_dt_s = 4.0;     // 11 单次运动总时长 (秒, 100Hz 下 = 400 tick)

    // -- 查询 --
    std::optional<std::pair<double, double>> find_limit(int motor_id) const {
        auto it = limits.find(motor_id);
        if (it == limits.end()) return std::nullopt;
        return it->second;
    }

    // -- 调试输出 --
    std::string to_string() const {
        std::ostringstream oss;
        const char* unit_str = (pos_unit == PosUnit::Turns)   ? "turns"
                             : (pos_unit == PosUnit::Radians) ? "radians" : "degrees";
        oss << "RobotConfig {\n"
            << "  port             = " << port << "\n"
            << "  baudrate         = " << baudrate << "\n"
            << "  motor_ids        = [";
        for (size_t i = 0; i < motor_ids.size(); ++i) {
            if (i) oss << ", ";
            oss << motor_ids[i];
        }
        oss << "]\n"
            << "  pos_unit         = " << unit_str << "\n"
            << "  control_rate_hz  = " << control_rate_hz << "\n"
            << "  max_torque_raw   = " << max_torque_raw << "\n"
            << "  use_async_rx     = " << (use_async_rx ? "true" : "false") << "\n"
            << "  trajectory_dt_s  = " << trajectory_dt_s << "\n"
            << "  limits           = {\n";
        for (const auto& [mid, lim] : limits) {
            oss << "    " << mid << " : [" << lim.first << ", " << lim.second << "] (圈)\n";
        }
        oss << "  }\n}";
        return oss.str();
    }

    // 把限位表灌进 driver
    void apply_limits_to(HightorqueSerial& ht) const {
        for (const auto& [mid, lim] : limits) {
            ht.enable_position_limit(mid, lim.first, lim.second, PosUnit::Turns);
        }
    }

    // ----- 解析 -----
    static RobotConfig load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("无法打开配置文件: " + path);

        RobotConfig cfg;
        std::string line;
        int line_no = 0;

        while (std::getline(f, line)) {
            ++line_no;

            // 去注释 (# 或 ; 起始的内联部分都吃掉)
            for (char c : {'#', ';'}) {
                auto pos = line.find(c);
                if (pos != std::string::npos) line = line.substr(0, pos);
            }
            // strip
            auto b = line.begin();
            auto e = line.end();
            while (b != e && std::isspace(static_cast<unsigned char>(*b))) ++b;
            while (e != b && std::isspace(static_cast<unsigned char>(*(e - 1)))) --e;
            if (b == e) continue;

            std::string content(b, e);
            const auto eq = content.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                                         " 行没有 '=': " + content);
            }
            std::string key = trim_(content.substr(0, eq));
            std::string val = trim_(content.substr(eq + 1));
            std::string key_lower = key;
            std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            try {
                if (key_lower == "port") {
                    cfg.port = val;
                } else if (key_lower == "baudrate") {
                    cfg.baudrate = static_cast<uint32_t>(std::stoul(val, nullptr, 0));
                } else if (key_lower == "motor_ids") {
                    cfg.motor_ids = parse_int_list_(val);
                } else if (key_lower == "pos_unit") {
                    std::string v = val;
                    std::transform(v.begin(), v.end(), v.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if      (v == "turns" || v == "turn")               cfg.pos_unit = PosUnit::Turns;
                    else if (v == "radians" || v == "radian" || v == "rad")
                                                                        cfg.pos_unit = PosUnit::Radians;
                    else if (v == "degrees" || v == "degree" || v == "deg")
                                                                        cfg.pos_unit = PosUnit::Degrees;
                    else throw std::runtime_error("未知 pos_unit: " + val);
                } else if (key_lower.rfind("limits.", 0) == 0) {
                    const int mid = std::stoi(key_lower.substr(7));
                    auto pair = parse_double_pair_(val);
                    cfg.limits[mid] = pair;
                } else if (key_lower == "control_rate_hz") {
                    cfg.control_rate_hz = std::stod(val);
                } else if (key_lower == "max_torque_raw") {
                    cfg.max_torque_raw = std::stoi(val, nullptr, 0);
                } else if (key_lower == "use_async_rx") {
                    std::string v = val;
                    std::transform(v.begin(), v.end(), v.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    cfg.use_async_rx = (v == "1" || v == "true" || v == "yes" || v == "on");
                } else if (key_lower == "trajectory_dt_s") {
                    cfg.trajectory_dt_s = std::stod(val);
                } else {
                    // 未知 key: 不报错, 只是忽略, 方便用户写自定义字段做注释
                }
            } catch (const std::exception& ex) {
                throw std::runtime_error("配置文件第 " + std::to_string(line_no) +
                                         " 行解析失败 [" + key + "=" + val + "]: " + ex.what());
            }
        }

        // limits 里的值如果 pos_unit 不是 Turns, 需要归一成圈
        if (cfg.pos_unit != PosUnit::Turns) {
            for (auto& [mid, lim] : cfg.limits) {
                lim.first  = to_turns(lim.first,  cfg.pos_unit);
                lim.second = to_turns(lim.second, cfg.pos_unit);
            }
        }
        return cfg;
    }

private:
    static std::string trim_(const std::string& s) {
        auto b = s.begin();
        auto e = s.end();
        while (b != e && std::isspace(static_cast<unsigned char>(*b))) ++b;
        while (e != b && std::isspace(static_cast<unsigned char>(*(e - 1)))) --e;
        return std::string(b, e);
    }

    static std::vector<int> parse_int_list_(const std::string& s) {
        std::string norm = s;
        for (char& c : norm) if (c == ',' || c == ';') c = ' ';
        std::istringstream iss(norm);
        std::vector<int> out;
        for (int v; iss >> v; ) out.push_back(v);
        if (out.empty()) throw std::runtime_error("空整数列表");
        return out;
    }

    static std::pair<double, double> parse_double_pair_(const std::string& s) {
        std::string norm = s;
        for (char& c : norm) if (c == ',' || c == ';') c = ' ';
        std::istringstream iss(norm);
        double a = 0.0, b = 0.0;
        if (!(iss >> a >> b)) throw std::runtime_error("需要两个浮点值");
        return {a, b};
    }
};

} // namespace hightorque
