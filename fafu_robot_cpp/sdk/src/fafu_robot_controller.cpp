// =============================================================================
//  fafu_robot_controller.cpp
//
//  原生 C++ 实现, 跟 fafu_robot_python/fafu_robot_controller.py 一一对应.
// =============================================================================
#include "fafu/fafu_robot_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fafu_robot {

namespace {

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;

// 统一前缀, 跟 Python 侧 [FafuRobot] 一致
constexpr const char* kPrefix = "[FafuRobot] ";

// 默认空 GripperOpts 的"打开 / 关闭"目标 fallback (没设软限位时用)
constexpr double kGripperFallbackOpenTurns  =  0.25;   // ~ +90 deg
constexpr double kGripperFallbackCloseTurns = -0.25;   // ~ -90 deg

void log_info(const std::string& msg) {
    std::cout << kPrefix << msg << std::endl;
}
void log_warn(const std::string& msg) {
    std::cerr << kPrefix << "WARN: " << msg << std::endl;
}

double monotonic_seconds() {
    using clk = std::chrono::steady_clock;
    static const auto t0 = clk::now();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

}  // anonymous namespace

// ============================================================================
//  GraspResult::to_string
// ============================================================================
std::string GraspResult::to_string() const {
    std::ostringstream oss;
    oss << "GraspResult{grasped=" << (grasped ? "true" : "false")
        << ", reason=\"" << reason << "\""
        << ", angle_rad=" << angle_rad
        << ", closed_deg=" << closed_deg
        << ", peak_torque_raw=" << peak_torque_raw
        << ", duration_s=" << duration_s
        << "}";
    return oss.str();
}

// ============================================================================
//  helpers (static)
// ============================================================================
double FafuRobotController::rad_to_turns_(double rad)   { return rad / kTwoPi; }
double FafuRobotController::turns_to_rad_(double turns) { return turns * kTwoPi; }
int    FafuRobotController::clamp_speed_(int speed) {
    if (speed < 1)   return 1;
    if (speed > 100) return 100;
    return speed;
}
std::string FafuRobotController::release_mode_name_(ReleaseMode m) {
    switch (m) {
        case ReleaseMode::Stop:  return "stop";
        case ReleaseMode::Brake: return "brake";
        case ReleaseMode::Hold:  return "hold";
    }
    return "?";
}

// ============================================================================
//  构造 / 析构
// ============================================================================
FafuRobotController::FafuRobotController(const std::string& cfg_path,
                                         const Options& opts)
{
    if (cfg_path.empty())
        throw std::runtime_error("cfg_path 不能为空");

    // 1) 加载 cfg
    try {
        cfg_ = hightorque::RobotConfig::load(cfg_path);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("加载配置失败 ") + cfg_path + ": " + e.what());
    }
    cfg_path_ = cfg_path;

    // 2) 夹爪参数校验
    has_gripper_      = opts.has_gripper;
    gripper_motor_id_ = opts.gripper_motor_id;
    if (has_gripper_) {
        if (gripper_motor_id_ <= 0)
            throw std::runtime_error("has_gripper=true 时必须设置 gripper_motor_id");
        bool found = false;
        for (int m : cfg_.motor_ids) if (m == gripper_motor_id_) { found = true; break; }
        if (!found) {
            std::ostringstream oss;
            oss << "gripper_motor_id " << gripper_motor_id_ << " 不在 cfg.motor_ids 里";
            throw std::runtime_error(oss.str());
        }
    }

    // 3) 派生关节电机表 (motor_ids - {gripper})
    joint_motor_ids_.clear();
    for (int m : cfg_.motor_ids) {
        if (has_gripper_ && m == gripper_motor_id_) continue;
        joint_motor_ids_.push_back(m);
    }
    if (joint_motor_ids_.empty())
        throw std::runtime_error("去掉夹爪后没有关节电机了, 检查 cfg.motor_ids / gripper_motor_id");

    // 4) 选串口 -> 打开 HightorqueSerial
    port_     = pick_serial_port_(opts.port.empty() ? cfg_.port : opts.port);
    baudrate_ = opts.baudrate ? opts.baudrate : cfg_.baudrate;

    try {
        ht_ = std::make_unique<hightorque::HightorqueSerial>(port_, baudrate_);
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "打开串口失败 " << port_ << " @ " << baudrate_ << ": " << e.what();
        throw std::runtime_error(oss.str());
    }

    // 5) 把 cfg 的软限位灌进 driver
    try {
        cfg_.apply_limits_to(*ht_);
    } catch (const std::exception& e) {
        log_warn(std::string("apply_limits_to 失败: ") + e.what());
    }

    // 6) 通信预检 (每个电机读一次状态)
    precheck_communication_();

    // 7) auto_enable (先 enable_async_rx 之前必须 set_motor_mode, 不能颠倒)
    if (opts.auto_enable) enable();

    // 8) async_rx
    bool use_async = opts.async_rx.has_value() ? *opts.async_rx : cfg_.use_async_rx;
    if (use_async) {
        try {
            ht_->enable_async_rx();
            owns_rx_ = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            for (int mid : cfg_.motor_ids) {
                ht_->read_motor_state(mid, 0.1);
            }
        } catch (const std::exception& e) {
            log_warn(std::string("enable_async_rx 失败: ") + e.what());
        }
    }

    // 9) auto_polling
    if (opts.auto_polling) {
        try {
            double hz = cfg_.control_rate_hz > 0 ? cfg_.control_rate_hz : 50.0;
            if (hz < 10.0) hz = 10.0;
            ht_->start_state_polling(cfg_.motor_ids, hz);
            owns_polling_ = true;
        } catch (const std::exception& e) {
            log_warn(std::string("start_state_polling 失败: ") + e.what());
        }
    }

    std::ostringstream oss;
    oss << "connected on " << port_ << " @ " << baudrate_
        << " (" << joint_motor_ids_.size() << " joints";
    if (has_gripper_) oss << " + gripper M" << gripper_motor_id_;
    oss << ")";
    log_info(oss.str());
}

FafuRobotController::~FafuRobotController() {
    try {
        close_connection();
    } catch (...) {
        // 析构里绝不抛
    }
}

// ============================================================================
//  状态
// ============================================================================
bool FafuRobotController::is_enabled() {
    if (!ht_) return false;
    for (int mid : cfg_.motor_ids) {
        auto s = ht_->get_state(mid);
        if (!s) s = ht_->read_motor_state(mid, 0.05);
        if (!s || s->mode != static_cast<int>(MODE_POSITION)) return false;
    }
    return true;
}

// ============================================================================
//  电源管理
// ============================================================================
void FafuRobotController::enable() {
    if (!switch_mode_all_(MODE_POSITION, "position", 3)) {
        throw std::runtime_error(
            "enable 失败: 至少有一个电机拒绝 mode 0x0A; 请重新上电后重试");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    log_info("all motors enabled (position control hold).");
}

void FafuRobotController::disable() {
    if (switch_mode_all_(MODE_STOP, "stop", 2)) {
        log_info("all motors disabled (free spin).");
    }
}

void FafuRobotController::brake() {
    if (switch_mode_all_(MODE_BRAKE, "brake", 2)) {
        log_info("all motors braked.");
    }
}

bool FafuRobotController::switch_mode_all_(uint8_t mode, const char* label, int max_retry) {
    bool overall_ok = true;
    for (int mid : cfg_.motor_ids) {
        bool ok = false;
        for (int attempt = 0; attempt < std::max(1, max_retry); ++attempt) {
            try {
                auto st = ht_->set_motor_mode(mid, mode);
                if (st && st->mode == static_cast<int>(mode)) {
                    ok = true;
                    break;
                }
            } catch (const std::exception&) {
                // 重试
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!ok) {
            std::ostringstream oss;
            oss << "switch_mode_all(" << label << "): motor " << mid << " 未确认 mode";
            log_warn(oss.str());
            overall_ok = false;
        }
    }
    return overall_ok;
}

// ============================================================================
//  关节空间
// ============================================================================
bool FafuRobotController::move_j(const std::vector<double>& joint_angles,
                                 const MoveOpts& opts) {
    auto angles_turns = validate_joint_angles_(joint_angles, opts.is_radians);
    if (angles_turns.empty()) return false;

    std::map<int, double> targets_turns;
    for (size_t i = 0; i < joint_motor_ids_.size(); ++i) {
        targets_turns[joint_motor_ids_[i]] = angles_turns[i];
    }
    int speed = clamp_speed_(opts.speed);

    if (opts.block) {
        try {
            move_scurve_(targets_turns, speed);
            return true;
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "move_j (block) 失败: " << e.what();
            log_warn(oss.str());
            return false;
        }
    }

    // 非阻塞: 单帧
    double v_avg = (speed / 100.0) * VEL_AVG_MAX_TPS_;
    auto cmds = build_many_cmds_holding_others_(targets_turns, v_avg);
    int max_mid = 0;
    for (int mid : cfg_.motor_ids) max_mid = std::max(max_mid, mid);
    try {
        ht_->set_many_pos_vel_tqe(cmds, hightorque::PosUnit::Turns, max_mid, 0.05);
    } catch (const std::exception& e) {
        log_warn(std::string("move_j (no block) 失败: ") + e.what());
        return false;
    }
    return true;
}

bool FafuRobotController::go_home(int speed, bool block) {
    std::vector<double> zeros(joint_motor_ids_.size(), 0.0);
    MoveOpts o;
    o.is_radians = true;
    o.speed      = speed;
    o.block      = block;
    return move_j(zeros, o);
}

// ============================================================================
//  Servo (online streaming)
// ============================================================================
void FafuRobotController::servo_start(const ServoOpts& opts) {
    if (!ht_) throw std::runtime_error("ht_ is null");
    if (servo_active_) {
        log_warn("servo_start: 已经在 servo 会话里, 先 servo_end() 再重启");
        return;
    }

    // 复制 opts, 检查/夹一些值
    servo_opts_ = opts;
    if (servo_opts_.watchdog_ms < 0) servo_opts_.watchdog_ms = 0;
    if (servo_opts_.watchdog_ms == 0) {
        log_warn("servo_start: watchdog_ms=0, 已禁用固件看门狗! "
                 "上位机崩溃 / Ctrl-C 后电机会失控冲到限位. 仅用于离线测试.");
    } else if (servo_opts_.watchdog_ms < 30) {
        log_warn("servo_start: watchdog_ms 太小 (<30ms), 100Hz 控制下可能误触发. "
                 "建议至少 50ms.");
    }
    if (servo_opts_.max_vel <= 0.0)      servo_opts_.max_vel = 1.0;
    if (servo_opts_.max_step_rad <= 0.0) servo_opts_.max_step_rad = 0.05;
    // max_lag_rad <= 0 表示禁用, 保留

    // 1) 确保关节电机都处于 ACTIVE 模式. 如果用户没 enable, 帮他做.
    if (!is_enabled()) {
        log_info("servo_start: motors not enabled, calling enable() ...");
        enable();
    }

    // 2) 强制 async_rx — servo 频率不能容忍同步 RX 的 5-15ms 串口等待
    if (!ht_->is_async_rx()) {
        try {
            ht_->enable_async_rx();
            owns_rx_ = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("servo_start: enable_async_rx 失败: ") + e.what());
        }
    }

    // 3) 抓当前位置作为 last_target_, 避免首次 servo_j 触发 step clamp
    servo_last_target_turns_.assign(joint_motor_ids_.size(), 0.0);
    for (size_t i = 0; i < joint_motor_ids_.size(); ++i) {
        int mid = joint_motor_ids_[i];
        auto s = ht_->get_state(mid);
        if (!s) s = ht_->read_motor_state(mid, 0.1);
        if (!s) {
            throw std::runtime_error("servo_start: 无法读取电机 " + std::to_string(mid)
                                     + " 起始位置");
        }
        servo_last_target_turns_[i] = s->position;
    }

    // 4) 写固件 watchdog 到每个关节电机 (★ 不写夹爪 ★)
    if (servo_opts_.watchdog_ms > 0) {
        for (int mid : joint_motor_ids_) {
            try {
                ht_->set_timeout(mid, static_cast<int16_t>(servo_opts_.watchdog_ms));
            } catch (const std::exception& e) {
                log_warn(std::string("servo_start: set_timeout(") + std::to_string(mid)
                         + ") 失败: " + e.what());
            }
        }
    }

    servo_active_     = true;
    servo_tick_count_ = 0;
    servo_started_at_ = std::chrono::steady_clock::now();

    std::ostringstream oss;
    oss << "servo_start: watchdog=" << servo_opts_.watchdog_ms << "ms"
        << ", max_vel=" << servo_opts_.max_vel << "rad/s"
        << ", max_step=" << servo_opts_.max_step_rad << "rad"
        << ", max_lag=" << servo_opts_.max_lag_rad << "rad";
    log_info(oss.str());
}

bool FafuRobotController::servo_j(const std::vector<double>& target_angles) {
    if (!servo_active_) {
        log_warn("servo_j: 没在 servo 会话里, 先调 servo_start()");
        return false;
    }
    if (target_angles.size() != joint_motor_ids_.size()) {
        std::ostringstream oss;
        oss << "servo_j: 入参长度 " << target_angles.size() << " != num_joints "
            << joint_motor_ids_.size();
        log_warn(oss.str());
        return false;
    }

    const size_t N = joint_motor_ids_.size();

    // (a) 转 turns + NaN 检查
    std::vector<double> target_turns(N);
    for (size_t i = 0; i < N; ++i) {
        double a = target_angles[i];
        if (std::isnan(a) || std::isinf(a)) {
            log_warn("servo_j: target 含 NaN/Inf, 拒绝发送");
            return false;
        }
        target_turns[i] = servo_opts_.is_radians ? rad_to_turns_(a) : (a / 360.0);
    }

    // (b) 防线 2: 单步限幅 (跟上次 target 比)
    const double max_step_turns = std::abs(servo_opts_.max_step_rad) / kTwoPi;
    bool was_clamped = false;
    for (size_t i = 0; i < N; ++i) {
        double delta = target_turns[i] - servo_last_target_turns_[i];
        if (std::abs(delta) > max_step_turns) {
            target_turns[i] = servo_last_target_turns_[i]
                            + std::copysign(max_step_turns, delta);
            was_clamped = true;
        }
    }
    if (was_clamped) {
        log_warn("servo_j: 单步幅度超过 max_step_rad, 已 clamp. "
                 "通常说明上层规划周期 / 速度配错了.");
    }

    // (c) 防线 3: 跟踪误差 (实测 vs target). 0 / 负 = 禁用
    if (servo_opts_.max_lag_rad > 0.0) {
        const double max_lag_turns = servo_opts_.max_lag_rad / kTwoPi;
        for (size_t i = 0; i < N; ++i) {
            int mid = joint_motor_ids_[i];
            auto s = ht_->get_state(mid);
            if (!s) continue;   // 缓存还没数据, 不卡死, 下个 tick 再说
            double lag = std::abs(s->position - target_turns[i]);
            if (lag > max_lag_turns) {
                std::ostringstream oss;
                oss << "servo_j: motor " << mid << " 跟踪误差 "
                    << (lag * kTwoPi) << " rad > max_lag_rad "
                    << servo_opts_.max_lag_rad << " rad. "
                    << "检查机械负载 / max_vel / 上层规划频率.";
                log_warn(oss.str());
                return false;
            }
        }
    }

    // (d) 防线 4: 软限位由底层 set_many_pos_vel_tqe 内 apply_position_limit_ 处理,
    //     这里不重复 clamp.

    // (e) 组帧: 关节电机走 target, 夹爪不动 (保持当前位置 hold)
    const double vel_rps = servo_opts_.max_vel / kTwoPi;
    std::map<int, double> targets_map;
    for (size_t i = 0; i < N; ++i) {
        targets_map[joint_motor_ids_[i]] = target_turns[i];
    }
    auto cmds = build_many_cmds_holding_others_(targets_map, vel_rps);

    int max_mid = 0;
    for (int mid : cfg_.motor_ids) max_mid = std::max(max_mid, mid);

    try {
        // timeout 给 0: async 模式下底层不阻塞等回包, 立即返回; servo 周期
        // 由上层负责 (sleep_until next_tick)
        ht_->set_many_pos_vel_tqe(cmds, hightorque::PosUnit::Turns, max_mid, 0.0);
    } catch (const std::exception& e) {
        log_warn(std::string("servo_j: 发送失败: ") + e.what());
        return false;
    }

    // 更新 last_target 必须放到成功 send 之后, 否则 step clamp 会基于
    // "曾经想发但失败了的点"再 clamp.
    servo_last_target_turns_ = target_turns;
    ++servo_tick_count_;
    return true;
}

void FafuRobotController::servo_end(ReleaseMode finish_mode) {
    if (!servo_active_) {
        log_warn("servo_end: 不在 servo 会话里, 直接返回");
        return;
    }

    // 1) 先清掉 watchdog (set_timeout 0 = 禁用), 避免在切 mode 期间被自动 brake
    if (servo_opts_.watchdog_ms > 0) {
        for (int mid : joint_motor_ids_) {
            try { ht_->set_timeout(mid, 0); } catch (...) {}
        }
    }

    // 2) 按 finish_mode 处理. 关节走 finish_mode, 夹爪不动 (servo 不管夹爪)
    for (int mid : joint_motor_ids_) {
        try {
            switch (finish_mode) {
                case ReleaseMode::Stop:  ht_->stop(mid);                       break;
                case ReleaseMode::Brake: ht_->set_motor_mode(mid, MODE_BRAKE); break;
                case ReleaseMode::Hold:  /* 保持 MODE_ACTIVE + 最后一帧 */    break;
            }
        } catch (...) {}
    }

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - servo_started_at_).count();
    double rate = elapsed > 0 ? servo_tick_count_ / elapsed : 0.0;
    std::ostringstream oss;
    oss << "servo_end (" << release_mode_name_(finish_mode) << "): "
        << servo_tick_count_ << " ticks in " << elapsed << "s "
        << "(~" << rate << " Hz)";
    log_info(oss.str());

    servo_active_     = false;
    servo_tick_count_ = 0;
    servo_last_target_turns_.clear();
}

// ============================================================================
//  状态读取
// ============================================================================
std::vector<double> FafuRobotController::get_joint_values(bool prefer_cache) {
    auto states = read_states_(joint_motor_ids_, prefer_cache);
    std::vector<double> out;
    out.reserve(joint_motor_ids_.size());
    for (int mid : joint_motor_ids_) {
        auto it = states.find(mid);
        if (it == states.end()) {
            std::ostringstream oss;
            oss << "no state for joint motor " << mid;
            throw std::runtime_error(oss.str());
        }
        out.push_back(turns_to_rad_(it->second.position));
    }
    return out;
}

std::vector<double> FafuRobotController::get_joint_velocities(bool prefer_cache) {
    auto states = read_states_(joint_motor_ids_, prefer_cache);
    std::vector<double> out;
    out.reserve(joint_motor_ids_.size());
    for (int mid : joint_motor_ids_) {
        auto it = states.find(mid);
        if (it == states.end()) {
            std::ostringstream oss;
            oss << "no state for joint motor " << mid;
            throw std::runtime_error(oss.str());
        }
        // velocity 单位是 turns/s, 转 rad/s
        out.push_back(it->second.velocity * kTwoPi);
    }
    return out;
}

std::map<int, hightorque::MotorState>
FafuRobotController::get_motor_states(bool prefer_cache) {
    return read_states_(cfg_.motor_ids, prefer_cache);
}

// ============================================================================
//  夹爪
// ============================================================================
std::optional<GraspResult>
FafuRobotController::gripper_control(double angle, const GripperOpts& opts) {
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");

    double pos_turns = opts.is_radians ? rad_to_turns_(angle) : (angle / 360.0);

    try {
        if (opts.effort.has_value()) {
            ht_->set_pos_vel_tqe(gripper_motor_id_, pos_turns, opts.vel,
                                 *opts.effort, hightorque::PosUnit::Turns);
        } else {
            ht_->set_pos_vel_acc(gripper_motor_id_, pos_turns, opts.vel,
                                 opts.acc, hightorque::PosUnit::Turns);
        }
    } catch (const std::exception& e) {
        log_warn(std::string("gripper_control send 失败: ") + e.what());
        return std::nullopt;
    }

    if (!opts.block) return std::nullopt;

    auto result = wait_until_gripper_done_(
        pos_turns,
        opts.timeout_s,
        opts.tolerance_deg / 360.0,
        opts.effort_threshold,
        std::nullopt);

    // 与 Python 侧一致: 没传 effort_threshold 时返回 nullopt 兼容老调用
    if (!opts.effort_threshold.has_value()) return std::nullopt;
    return result;
}

bool FafuRobotController::open_gripper(std::optional<double> angle,
                                       const GripperOpts& opts) {
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");

    double target_turns;
    if (angle.has_value()) {
        target_turns = opts.is_radians ? rad_to_turns_(*angle) : (*angle / 360.0);
    } else {
        auto lim = gripper_limit_turns_();
        target_turns = lim ? lim->second : kGripperFallbackOpenTurns;
    }

    GripperOpts forwarded = opts;
    forwarded.is_radians = false;     // 我们已经手动转好 deg -> turns
    auto r = gripper_control(target_turns * 360.0, forwarded);
    (void)r;
    return true;
}

bool FafuRobotController::close_gripper(std::optional<double> angle,
                                        const GripperOpts& opts) {
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");

    double target_turns;
    if (angle.has_value()) {
        target_turns = opts.is_radians ? rad_to_turns_(*angle) : (*angle / 360.0);
    } else {
        auto lim = gripper_limit_turns_();
        target_turns = lim ? lim->first : kGripperFallbackCloseTurns;
    }

    GripperOpts forwarded = opts;
    forwarded.is_radians = false;
    auto r = gripper_control(target_turns * 360.0, forwarded);
    (void)r;
    return true;
}

GraspResult FafuRobotController::grasp(const GraspOpts& opts) {
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");

    double target_turns;
    if (opts.target_angle.has_value()) {
        target_turns = opts.is_radians ? rad_to_turns_(*opts.target_angle)
                                       : (*opts.target_angle / 360.0);
    } else {
        auto lim = gripper_limit_turns_();
        target_turns = lim ? lim->first : kGripperFallbackCloseTurns;
    }

    try {
        if (opts.effort.has_value()) {
            ht_->set_pos_vel_tqe(gripper_motor_id_, target_turns, opts.vel,
                                 *opts.effort, hightorque::PosUnit::Turns);
        } else {
            ht_->set_pos_vel_acc(gripper_motor_id_, target_turns, opts.vel,
                                 opts.acc, hightorque::PosUnit::Turns);
        }
    } catch (const std::exception& e) {
        log_warn(std::string("grasp send 失败: ") + e.what());
    }

    double min_progress_turns = std::max(0.0, opts.min_close_deg) / 360.0;
    return wait_until_gripper_done_(
        target_turns,
        opts.timeout_s,
        /*tolerance_turns=*/std::nullopt,
        std::make_optional<int>(opts.force_threshold),
        min_progress_turns);
}

void FafuRobotController::release(const GripperOpts& opts) {
    open_gripper(std::nullopt, opts);
}

hightorque::MotorState FafuRobotController::get_gripper_state() {
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");
    auto s = ht_->get_state(gripper_motor_id_);
    if (!s) s = ht_->read_motor_state(gripper_motor_id_, 0.1);
    if (!s) throw std::runtime_error("no feedback from gripper motor");
    return *s;
}

// ============================================================================
//  软限位
// ============================================================================
void FafuRobotController::set_limit(int motor_id, double lo, double hi,
                                    hightorque::PosUnit unit) {
    ht_->enable_position_limit(motor_id, lo, hi, unit);
}

std::optional<std::pair<double, double>>
FafuRobotController::get_limit(int motor_id) const {
    double lo = 0.0, hi = 0.0;
    if (!ht_->get_position_limit_turns(motor_id, lo, hi)) return std::nullopt;
    return std::make_pair(lo, hi);
}

void FafuRobotController::disable_limit(int motor_id) {
    ht_->disable_position_limit(motor_id);
}

void FafuRobotController::clear_limits() {
    ht_->clear_all_position_limits();
}

// ============================================================================
//  急停 / 状态
// ============================================================================
void FafuRobotController::emergency_stop() {
    for (int mid : cfg_.motor_ids) {
        try { ht_->stop(mid); } catch (...) {}
    }
    log_warn("EMERGENCY STOP issued — all motors PWM off");
}

void FafuRobotController::resume() {
    enable();
}

hightorque::CanStatus FafuRobotController::get_can_status() {
    return ht_->read_can_status();
}

void FafuRobotController::reset_zero(int motor_id, bool confirm) {
    if (!confirm) {
        log_warn("reset_zero: confirm=false, 跳过. 这是硬件级永久标定, 请显式 confirm=true.");
        return;
    }
    auto msg = ht_->reset_zero(motor_id);
    std::ostringstream oss;
    oss << "reset_zero motor " << motor_id << ": " << msg;
    log_info(oss.str());
}

// ============================================================================
//  close_connection
// ============================================================================
void FafuRobotController::close_connection(ReleaseMode joint_release,
                                           ReleaseMode gripper_release) {
    if (!ht_) return;

    // 如果还在 servo 会话, 先优雅退出 (清看门狗, 关节按 joint_release 处理)
    if (servo_active_) {
        try { servo_end(joint_release); } catch (...) {}
    }

    try {
        if (ht_->is_polling()) ht_->stop_state_polling();
    } catch (...) {}

    try {
        if (ht_->is_async_rx()) ht_->disable_async_rx();
    } catch (...) {}

    auto mode_for = [](ReleaseMode r) -> uint8_t {
        switch (r) {
            case ReleaseMode::Stop:  return MODE_STOP;
            case ReleaseMode::Brake: return MODE_BRAKE;
            case ReleaseMode::Hold:  return MODE_POSITION;
        }
        return MODE_STOP;
    };

    for (int mid : cfg_.motor_ids) {
        ReleaseMode policy =
            (has_gripper_ && mid == gripper_motor_id_) ? gripper_release : joint_release;
        try {
            if (policy == ReleaseMode::Stop) {
                ht_->stop(mid);
            } else {
                ht_->set_motor_mode(mid, mode_for(policy));
            }
        } catch (...) {}
    }

    try { ht_->close(); } catch (...) {}

    std::ostringstream oss;
    oss << "connection closed (joints=" << release_mode_name_(joint_release)
        << ", gripper=" << release_mode_name_(gripper_release) << ").";
    log_info(oss.str());

    ht_.reset();
}

// ============================================================================
//  内部 helpers
// ============================================================================
std::string FafuRobotController::pick_serial_port_(const std::string& preferred) {
    if (preferred.empty() || preferred == "auto") {
        auto candidates = hightorque::find_likely_debug_boards();
        if (candidates.empty())
            throw std::runtime_error("auto: 未找到候选 USB 调试板, 检查 USB / 驱动");
        if (candidates.size() > 1) {
            std::ostringstream oss;
            oss << "auto: 找到多个候选端口, 取第一个: " << candidates.front().port;
            log_warn(oss.str());
        }
        return candidates.front().port;
    }
    return preferred;
}

void FafuRobotController::precheck_communication_() {
    for (int mid : cfg_.motor_ids) {
        auto s = ht_->read_motor_state(mid, 0.3);
        if (!s) {
            std::ostringstream oss;
            oss << "通信预检失败: 电机 " << mid << " 不响应 (timeout 300ms). "
                << "检查 CAN 总线 / 电源 / motor_id 是否正确.";
            throw std::runtime_error(oss.str());
        }
    }
}

std::vector<double>
FafuRobotController::validate_joint_angles_(const std::vector<double>& angles,
                                            bool is_radians) {
    if (angles.size() != joint_motor_ids_.size()) {
        std::ostringstream oss;
        oss << "joint_angles 长度必须为 " << joint_motor_ids_.size()
            << ", 实际 " << angles.size();
        throw std::runtime_error(oss.str());
    }
    std::vector<double> turns;
    turns.reserve(angles.size());
    for (double a : angles) {
        if (std::isnan(a) || std::isinf(a))
            throw std::runtime_error("joint_angles 含 NaN/Inf");
        turns.push_back(is_radians ? rad_to_turns_(a) : (a / 360.0));
    }
    return turns;
}

std::map<int, hightorque::MotorState>
FafuRobotController::read_states_(const std::vector<int>& ids, bool prefer_cache) {
    std::map<int, hightorque::MotorState> out;
    for (int mid : ids) {
        std::optional<hightorque::MotorState> s;
        if (prefer_cache) s = ht_->get_state(mid);
        if (!s)           s = ht_->read_motor_state(mid, 0.1);
        if (s) out[mid] = *s;
    }
    return out;
}

std::optional<std::pair<double, double>>
FafuRobotController::gripper_limit_turns_() const {
    if (!has_gripper_) return std::nullopt;
    double lo = 0.0, hi = 0.0;
    if (!ht_->get_position_limit_turns(gripper_motor_id_, lo, hi)) return std::nullopt;
    return std::make_pair(lo, hi);
}

// ============================================================================
//  _wait_until_gripper_done_ (mirror of Python)
// ============================================================================
GraspResult FafuRobotController::wait_until_gripper_done_(
    double target_turns,
    double timeout_s,
    std::optional<double> tolerance_turns_opt,
    std::optional<int> effort_threshold,
    std::optional<double> min_progress_turns_opt)
{
    const double tolerance_turns = tolerance_turns_opt.value_or(GRIPPER_TOLERANCE_TURNS_);
    const double min_progress_turns = min_progress_turns_opt.value_or(GRIPPER_MIN_PROGRESS_TURNS_);

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto deadline = t0 + std::chrono::duration_cast<clk::duration>(
        std::chrono::duration<double>(std::max(0.05, timeout_s)));

    auto start_state = ht_->get_state(gripper_motor_id_);
    if (!start_state) start_state = ht_->read_motor_state(gripper_motor_id_, 0.05);
    double start_pos = start_state ? start_state->position : std::nan("");
    double last_pos  = start_pos;

    std::optional<clk::time_point> stall_since;
    int peak_torque = 0;

    while (true) {
        auto now = clk::now();
        double elapsed_s = std::chrono::duration<double>(now - t0).count();
        if (now >= deadline) {
            return make_grasp_result_("timeout", false, last_pos, start_pos,
                                      peak_torque, elapsed_s);
        }

        auto s = ht_->get_state(gripper_motor_id_);
        if (!s) s = ht_->read_motor_state(gripper_motor_id_, 0.05);
        if (s) {
            last_pos = s->position;
            int t_raw = static_cast<int>(std::abs(s->torque));
            if (t_raw > peak_torque) peak_torque = t_raw;

            if (effort_threshold.has_value() && t_raw >= *effort_threshold) {
                return make_grasp_result_("detected_object_force", true,
                                          last_pos, start_pos, peak_torque, elapsed_s);
            }

            if (std::abs(s->position - target_turns) <= tolerance_turns) {
                return make_grasp_result_("reached_target", false,
                                          last_pos, start_pos, peak_torque, elapsed_s);
            }

            if (std::abs(s->velocity) < GRIPPER_STALL_VEL_TPS_) {
                if (!stall_since.has_value()) {
                    stall_since = now;
                } else {
                    double stalled_s = std::chrono::duration<double>(now - *stall_since).count();
                    if (stalled_s >= GRIPPER_STALL_PATIENCE_S_) {
                        double progress = std::abs(s->position - start_pos);
                        if (progress >= min_progress_turns) {
                            return make_grasp_result_("detected_object_stall", true,
                                                      last_pos, start_pos, peak_torque, elapsed_s);
                        }
                        return make_grasp_result_("no_movement", false,
                                                  last_pos, start_pos, peak_torque, elapsed_s);
                    }
                }
            } else {
                stall_since.reset();
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

GraspResult FafuRobotController::make_grasp_result_(
    const std::string& reason, bool grasped,
    double last_pos_turns, double start_pos_turns,
    int peak_torque, double duration_s)
{
    GraspResult r;
    r.reason          = reason;
    r.grasped         = grasped;
    r.peak_torque_raw = peak_torque;
    r.duration_s      = duration_s;
    if (std::isnan(last_pos_turns) || std::isnan(start_pos_turns)) {
        r.closed_deg = 0.0;
        r.angle_rad  = std::isnan(last_pos_turns) ? std::nan("") : turns_to_rad_(last_pos_turns);
    } else {
        r.closed_deg = std::abs(last_pos_turns - start_pos_turns) * 360.0;
        r.angle_rad  = turns_to_rad_(last_pos_turns);
    }
    return r;
}

// ============================================================================
//  build_many_cmds_holding_others_  (mirror of Python)
// ============================================================================
std::vector<hightorque::HightorqueSerial::ManyMotorCmd>
FafuRobotController::build_many_cmds_holding_others_(
    const std::map<int, double>& targets_turns, double vel_rps)
{
    std::vector<hightorque::HightorqueSerial::ManyMotorCmd> cmds;
    int max_torque = cfg_.max_torque_raw;

    for (int mid : cfg_.motor_ids) {
        auto it = targets_turns.find(mid);
        if (it != targets_turns.end()) {
            cmds.push_back({mid, it->second, vel_rps, max_torque});
        } else {
            auto s = ht_->get_state(mid);
            if (!s) s = ht_->read_motor_state(mid, 0.1);
            double hold_pos = s ? s->position : 0.0;
            cmds.push_back({mid, hold_pos, 0.0, max_torque});
        }
    }
    return cmds;
}

// ============================================================================
//  move_scurve_  (mirror of Python _move_scurve)
//
//  生成 cosine-envelope S 曲线轨迹, 通过 HightorqueSerial::run_control_loop
//  以 cfg.control_rate_hz 频率发送 set_many_pos_vel_tqe.
// ============================================================================
void FafuRobotController::move_scurve_(const std::map<int, double>& targets_turns,
                                       int speed_pct) {
    if (!ht_) throw std::runtime_error("ht_ is null");

    double rate_hz = std::max(10.0, cfg_.control_rate_hz > 0 ? cfg_.control_rate_hz : 100.0);
    double v_avg_target = (speed_pct / 100.0) * VEL_AVG_MAX_TPS_;

    // 1) 抓取所有电机的起始位置 (含 gripper, 否则它会被发空命令导致松开)
    std::map<int, double> start_pos;
    for (int mid : cfg_.motor_ids) {
        auto s = ht_->get_state(mid);
        if (!s) s = ht_->read_motor_state(mid, 0.1);
        if (!s) {
            std::ostringstream oss;
            oss << "无法读取电机 " << mid << " 起始位置, 拒绝执行 move_j";
            throw std::runtime_error(oss.str());
        }
        start_pos[mid] = s->position;
    }

    // 2) 自适应段时间 (依赖最大位移)
    double max_abs_dpos = 0.0;
    for (const auto& [mid, tgt] : targets_turns) {
        max_abs_dpos = std::max(max_abs_dpos, std::abs(tgt - start_pos[mid]));
    }

    double dt_s = std::max(DT_MIN_S_, cfg_.trajectory_dt_s > 0 ? cfg_.trajectory_dt_s : 1.0);
    if (max_abs_dpos > 1e-5) {
        double dt_target = max_abs_dpos / std::max(v_avg_target, 1e-3);
        dt_s = std::max(DT_MIN_S_, dt_target);
    }

    // 3) per-motor plan: (delta, peak velocity signed)
    struct Plan { double dpos; double v_peak; };
    std::map<int, Plan> plans;
    for (const auto& [mid, tgt] : targets_turns) {
        double dpos = tgt - start_pos[mid];
        if (std::abs(dpos) < 1e-5) {
            plans[mid] = {0.0, 0.0};
            continue;
        }
        double v_avg = std::abs(dpos) / dt_s;
        double v_peak = std::min(VEL_AVG_MAX_TPS_, v_avg) * (kPi / 2.0);
        plans[mid] = {dpos, std::copysign(v_peak, dpos)};
    }

    int total_ticks  = std::max(1, static_cast<int>(dt_s * rate_hz));
    int settle_ticks = std::max(1, static_cast<int>(SETTLE_MS_ * rate_hz / 1000.0));
    int last_tick    = total_ticks + settle_ticks;

    int max_mid = 0;
    for (int mid : cfg_.motor_ids) max_mid = std::max(max_mid, mid);

    int max_torque = cfg_.max_torque_raw;
    std::vector<int> all_ids = cfg_.motor_ids;

    // 4) on_tick lambda
    auto on_tick = [&](int tick, double /*period_ms*/) -> bool {
        if (tick >= last_tick) return false;

        double alpha     = std::min(1.0, static_cast<double>(tick) / total_ticks);
        double smooth    = 0.5 * (1.0 - std::cos(kPi * alpha));
        double vel_factor = std::sin(kPi * alpha);

        std::vector<hightorque::HightorqueSerial::ManyMotorCmd> cmds;
        cmds.reserve(all_ids.size());

        for (int mid : all_ids) {
            auto plan_it = plans.find(mid);
            if (plan_it != plans.end() && plan_it->second.v_peak != 0.0) {
                double dpos    = plan_it->second.dpos;
                double v_peak  = plan_it->second.v_peak;
                double desired = start_pos[mid] + smooth * dpos;
                double v_now   = vel_factor * v_peak;
                cmds.push_back({mid, desired, v_now, max_torque});
            } else {
                // 保持位置
                cmds.push_back({mid, start_pos[mid], 0.0, max_torque});
            }
        }

        try {
            ht_->set_many_pos_vel_tqe(cmds, hightorque::PosUnit::Turns, max_mid, 0.0);
        } catch (const std::exception& e) {
            log_warn(std::string("move_scurve tick send 失败: ") + e.what());
            return false;
        }
        return true;
    };

    hightorque::HightorqueSerial::ControlLoopOptions loop_opts;
    loop_opts.rate_hz             = rate_hz;
    loop_opts.stop_motor_ids      = cfg_.motor_ids;
    loop_opts.stop_on_finish      = false;   // 跟 Python 一致: 正常完成保持 mode=10
    loop_opts.stop_on_abort       = true;

    ht_->run_control_loop(loop_opts, on_tick);
}

} // namespace fafu_robot
