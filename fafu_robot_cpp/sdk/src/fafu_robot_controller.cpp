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

fafu::core::FinishMode to_finish_mode(ReleaseMode mode) {
    switch (mode) {
        case ReleaseMode::Stop:  return fafu::core::FinishMode::Stop;
        case ReleaseMode::Brake: return fafu::core::FinishMode::Brake;
        case ReleaseMode::Hold:  return fafu::core::FinishMode::Hold;
    }
    return fafu::core::FinishMode::Stop;
}

class CoreLease {
public:
    CoreLease(fafu::core::RobotCore& core, fafu::core::OperationKind kind)
        : core_(&core), token_(core.begin_operation(kind)) {}

    ~CoreLease() {
        if (core_ != nullptr) {
            try { core_->end_operation(token_); } catch (...) {}
        }
    }

    CoreLease(const CoreLease&) = delete;
    CoreLease& operator=(const CoreLease&) = delete;

private:
    fafu::core::RobotCore* core_;
    std::uint64_t token_;
};

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

    fafu::core::CoreConfig core_cfg;
    core_cfg.all_motor_ids = cfg_.motor_ids;
    core_cfg.joint_motor_ids = joint_motor_ids_;
    core_cfg.joint_motor_models.assign(joint_motor_ids_.size(), "");
    core_cfg.max_torque_raw = cfg_.max_torque_raw;
    core_cfg.stale_feedback_timeout_ms = 500.0;
    core_cfg.polling_rate_hz =
        std::max(10.0, cfg_.control_rate_hz > 0.0
            ? cfg_.control_rate_hz : 50.0);
    core_ = std::make_unique<fafu::core::RobotCore>(*ht_, core_cfg);

    // 7) enable before asynchronous receive, then start the owned transport.
    if (opts.auto_enable) enable();

    const bool use_async =
        opts.async_rx.has_value() ? *opts.async_rx : cfg_.use_async_rx;
    try {
        core_->start_transport(
            use_async, opts.auto_polling, core_cfg.polling_rate_hz);
    } catch (const std::exception& error) {
        log_warn(std::string("start_transport failed: ") + error.what());
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
    if (!ht_ || !ht_->is_open()) return false;
    for (int mid : cfg_.motor_ids) {
        auto state = ht_->get_state(mid);
        if (!state) state = ht_->read_motor_state(mid, 0.05);
        if (!state || state->mode != static_cast<int>(MODE_POSITION)) {
            return false;
        }
    }
    return true;
}

fafu::core::RobotState FafuRobotController::state() const {
    return core_ ? core_->state() : fafu::core::RobotState::Disconnected;
}

fafu::core::HealthSnapshot FafuRobotController::health() const {
    if (!core_) return {};
    return core_->health();
}

bool FafuRobotController::check_alive(bool fresh, double timeout_s) {
    return core_ && core_->check_alive(fresh, timeout_s);
}

bool FafuRobotController::recover(bool confirm, double timeout_s) {
    if (!confirm) {
        throw std::invalid_argument(
            "recover requires confirm=true after checking the workspace");
    }
    return core_ && core_->recover(true, timeout_s);
}

// ============================================================================
//  电源管理
// ============================================================================
void FafuRobotController::enable() {
    if (!core_) throw std::runtime_error("controller core is unavailable");
    const auto result = core_->enable();
    if (!result.success) {
        std::ostringstream message;
        message << "enable failed: " << result.message;
        if (!result.failed_motor_ids.empty()) {
            message << " (motors";
            for (int id : result.failed_motor_ids) message << ' ' << id;
            message << ')';
        }
        throw std::runtime_error(message.str());
    }
}

void FafuRobotController::disable() {
    if (!core_) throw std::runtime_error("controller core is unavailable");
    core_->disable();
}

void FafuRobotController::brake() {
    if (!core_) throw std::runtime_error("controller core is unavailable");
    core_->brake();
}

// ============================================================================
//  关节空间
// ============================================================================
bool FafuRobotController::move_j(const std::vector<double>& joint_angles,
                                 const MoveOpts& opts) {
    if (!core_) throw std::runtime_error("controller core is unavailable");
    CoreLease operation(*core_, fafu::core::OperationKind::JointMotion);
    if (!core_->stream_link_ok()) {
        throw fafu::core::StateError(core_->dead_reason());
    }
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
        auto command = core_->command_guard();
        ht_->set_many_pos_vel_tqe(
            cmds, hightorque::PosUnit::Turns, max_mid, 0.05);
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
    if (!core_) throw std::runtime_error("controller core is unavailable");

    if (!opts.motor_models.empty()) {
        core_->set_joint_motor_models(opts.motor_models);
    }

    fafu::core::ServoOptions native;
    native.watchdog_ms = opts.watchdog_ms;
    native.max_velocity_rad_s = opts.max_vel;
    native.max_step_rad = opts.max_step_rad;
    native.max_lag_rad = opts.max_lag_rad;
    native.nominal_rate_hz = opts.rate_hz;
    native.input_is_radians = opts.is_radians;
    native.feedforward_velocity = opts.feedforward_vel;
    native.lookahead_time_s = opts.lookahead_time;
    native.lag_abort_consecutive = opts.lag_abort_consecutive;
    native.channel = opts.use_mit
        ? fafu::core::ServoChannel::Mit
        : fafu::core::ServoChannel::Position;
    native.mit_kp = opts.mit_kp;
    native.mit_kd = opts.mit_kd;
    core_->servo_start(native);
}

bool FafuRobotController::servo_j(
        const std::vector<double>& target_angles) {
    if (!core_) return false;
    const auto result = core_->servo_tick(target_angles);
    if (!result.sent && !result.message.empty()) {
        log_warn("servo_j: " + result.message);
    }
    return result.sent && !result.aborted;
}

void FafuRobotController::servo_end(ReleaseMode finish_mode) {
    if (!core_) return;
    const auto summary = core_->servo_end(to_finish_mode(finish_mode));
    std::ostringstream message;
    message << "servo_end (" << release_mode_name_(finish_mode) << "): "
            << summary.tick_count << " ticks in " << summary.elapsed_s
            << "s (~" << summary.average_rate_hz << " Hz)";
    if (!summary.aborted_reason.empty()) {
        message << " [aborted: " << summary.aborted_reason << ']';
    }
    log_info(message.str());
}

bool FafuRobotController::is_servoing() const {
    return core_ && core_->is_servoing();
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
    if (!core_) throw std::runtime_error("controller core is unavailable");
    CoreLease operation(*core_, fafu::core::OperationKind::GripperMotion);
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");
    if (!core_->stream_link_ok()) {
        throw fafu::core::StateError(core_->dead_reason());
    }

    double pos_turns = opts.is_radians ? rad_to_turns_(angle) : (angle / 360.0);

    {
        auto command = core_->command_guard();
        if (opts.effort.has_value()) {
            ht_->set_pos_vel_tqe(gripper_motor_id_, pos_turns, opts.vel,
                                 *opts.effort, hightorque::PosUnit::Turns);
        } else {
            ht_->set_pos_vel_acc(gripper_motor_id_, pos_turns, opts.vel,
                                 opts.acc, hightorque::PosUnit::Turns);
        }
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
    if (!core_->stream_link_ok()) {
        throw fafu::core::StateError(core_->dead_reason());
    }

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
    if (!core_) throw std::runtime_error("controller core is unavailable");
    CoreLease operation(*core_, fafu::core::OperationKind::Grasp);
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

    {
        auto command = core_->command_guard();
        if (opts.effort.has_value()) {
            ht_->set_pos_vel_tqe(gripper_motor_id_, target_turns, opts.vel,
                                 *opts.effort, hightorque::PosUnit::Turns);
        } else {
            ht_->set_pos_vel_acc(gripper_motor_id_, target_turns, opts.vel,
                                 opts.acc, hightorque::PosUnit::Turns);
        }
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
    if (core_) core_->emergency_stop();
    log_warn("EMERGENCY STOP issued — all motors PWM off");
}

void FafuRobotController::resume() {
    if (!core_) throw std::runtime_error("controller core is unavailable");
    if (!core_->resume()) {
        throw std::runtime_error("resume failed to enable all motors");
    }
}

hightorque::CanStatus FafuRobotController::get_can_status() {
    return ht_->read_can_status();
}

void FafuRobotController::reset_zero(int motor_id, bool confirm) {
    if (!confirm) {
        log_warn("reset_zero: confirm=false, 跳过. 这是硬件级永久标定, 请显式 confirm=true.");
        return;
    }
    if (!core_) throw std::runtime_error("controller core is unavailable");
    if (std::find(cfg_.motor_ids.begin(), cfg_.motor_ids.end(), motor_id) ==
        cfg_.motor_ids.end()) {
        throw std::invalid_argument("reset_zero motor is not configured");
    }

    CoreLease operation(*core_, fafu::core::OperationKind::RawStream);
    auto command = core_->command_guard();
    auto msg = ht_->reset_zero(motor_id);
    std::ostringstream oss;
    oss << "reset_zero motor " << motor_id << ": " << msg;
    log_info(oss.str());
}

// ============================================================================
//  close_connection
// ============================================================================
void FafuRobotController::close_connection(
        ReleaseMode joint_release, ReleaseMode gripper_release) {
    if (!ht_ || !core_ ||
        core_->state() == fafu::core::RobotState::Disconnected) {
        return;
    }

    core_->shutdown(
        to_finish_mode(joint_release),
        to_finish_mode(gripper_release),
        5.0);
    ht_->close();

    std::ostringstream message;
    message << "connection closed (joints="
            << release_mode_name_(joint_release)
            << ", gripper=" << release_mode_name_(gripper_release) << ").";
    log_info(message.str());
}

// ============================================================================
//  内部 helpers
// ============================================================================
std::string FafuRobotController::pick_serial_port_(
        const std::string& preferred) {
    if (!preferred.empty() && preferred != "auto") {
        return preferred;
    }

    const auto candidates = hightorque::find_likely_debug_boards();
    if (candidates.empty()) {
        throw std::runtime_error(
            "auto: no USB debug board found; specify an explicit port");
    }
    if (candidates.size() != 1) {
        std::ostringstream message;
        message << "auto: multiple USB debug boards found (";
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (i != 0) message << ", ";
            message << candidates[i].port;
        }
        message << "); specify one port per SDK instance";
        throw std::runtime_error(message.str());
    }
    return candidates.front().port;
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
        if (core_ &&
            (core_->cancel_requested() || !core_->stream_link_ok())) {
            const double elapsed =
                std::chrono::duration<double>(now - t0).count();
            return make_grasp_result_(
                "cancelled", false, last_pos, start_pos,
                peak_torque, elapsed);
        }
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

        auto command = core_->command_guard();
        ht_->set_many_pos_vel_tqe(
            cmds, hightorque::PosUnit::Turns, max_mid, 0.0);
        return true;
    };

    hightorque::HightorqueSerial::ControlLoopOptions loop_opts;
    loop_opts.rate_hz             = rate_hz;
    loop_opts.stop_motor_ids      = cfg_.motor_ids;
    loop_opts.stop_on_finish      = false;   // 跟 Python 一致: 正常完成保持 mode=10
    loop_opts.stop_on_abort       = true;
    loop_opts.abort_check = [this] {
        return !core_ || core_->cancel_requested() ||
               !core_->stream_link_ok();
    };

    const int result = ht_->run_control_loop(loop_opts, on_tick);
    if (result == 1) {
        throw std::runtime_error("control loop was cancelled");
    }
    if (result == 2) {
        throw std::runtime_error("control loop exited after a send error");
    }
}

} // namespace fafu_robot
