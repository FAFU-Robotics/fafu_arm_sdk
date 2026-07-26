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
    CoreLease(fafu::core::RobotCore& core,
              fafu::core::OperationKind kind,
              bool borrow_owned_servo = false)
        : core_(nullptr), token_(0) {
        // A non-blocking gripper position command may share the current
        // thread's Servo writer. Do not acquire/end another operation: the
        // controller must remain SERVOING for the whole Servo session.
        if (borrow_owned_servo &&
            kind == fafu::core::OperationKind::GripperMotion &&
            core.active_operation() == fafu::core::OperationKind::Servo &&
            core.operation_owned_by_current_thread()) {
            return;
        }
        core_ = &core;
        token_ = core.begin_operation(kind);
    }

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
FafuRobotController::FafuRobotController(const std::string& cfg_path)
    : FafuRobotController(cfg_path, Options{}) {}

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
bool FafuRobotController::move_j(const std::vector<double>& joint_angles) {
    return move_j(joint_angles, MoveOpts{});
}

bool FafuRobotController::move_j(const std::vector<double>& joint_angles,
                                 const MoveOpts& opts) {
    if (!core_) throw std::runtime_error("controller core is unavailable");

    std::vector<double> angles_rad;
    angles_rad.reserve(joint_angles.size());
    for (double angle : joint_angles) {
        if (!std::isfinite(angle)) {
            throw std::invalid_argument("joint angles must be finite");
        }
        angles_rad.push_back(
            opts.is_radians ? angle : angle * kPi / 180.0);
    }

    const int speed = std::clamp(opts.speed, 1, 100);
    const double configured_rate =
        std::isfinite(cfg_.control_rate_hz) && cfg_.control_rate_hz > 0.0
            ? cfg_.control_rate_hz : 100.0;
    const double configured_duration =
        std::isfinite(cfg_.trajectory_dt_s) && cfg_.trajectory_dt_s > 0.0
            ? cfg_.trajectory_dt_s : DT_MIN_S_;

    fafu::core::MoveJOptions native;
    native.block = opts.block;
    native.max_velocity_rad_s =
        (speed / 100.0) * VEL_AVG_MAX_TPS_ * kTwoPi;
    native.control_rate_hz = std::clamp(configured_rate, 10.0, 1000.0);
    native.min_duration_s =
        std::clamp(configured_duration, DT_MIN_S_, 60.0);
    native.tolerance_rad =
        opts.is_radians ? opts.tolerance : opts.tolerance * kPi / 180.0;
    native.settle_timeout_s = 1.0;

    const auto result = core_->move_j(angles_rad, native);
    return result.sent && (!native.block || result.reached);
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
void FafuRobotController::servo_start() {
    servo_start(ServoOpts{});
}

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
    native.position_error_deadband_rad = opts.position_error_deadband_rad;
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
FafuRobotController::gripper_control(double angle) {
    return gripper_control(angle, GripperOpts{});
}

std::optional<GraspResult>
FafuRobotController::gripper_control(double angle, const GripperOpts& opts) {
    if (!core_) throw std::runtime_error("controller core is unavailable");
    CoreLease operation(
        *core_,
        fafu::core::OperationKind::GripperMotion,
        /*borrow_owned_servo=*/!opts.block);
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");
    if (!core_->stream_link_ok()) {
        throw fafu::core::StateError(core_->dead_reason());
    }
    if (opts.effort.has_value() &&
        (*opts.effort < 1 || *opts.effort > 32767)) {
        throw std::invalid_argument("effort must be in [1, 32767]");
    }
    if (opts.effort_threshold.has_value() &&
        (*opts.effort_threshold < 1 || *opts.effort_threshold > 32767)) {
        throw std::invalid_argument(
            "effort_threshold must be in [1, 32767]");
    }

    std::optional<int> command_effort = opts.effort;
    if (opts.effort_threshold.has_value()) {
        command_effort = command_effort.has_value()
            ? std::min(*command_effort, *opts.effort_threshold)
            : opts.effort_threshold;
    }

    double pos_turns = opts.is_radians ? rad_to_turns_(angle) : (angle / 360.0);

    {
        auto command = core_->command_guard();
        if (command_effort.has_value()) {
            ht_->set_pos_vel_tqe(gripper_motor_id_, pos_turns, opts.vel,
                                 *command_effort, hightorque::PosUnit::Turns);
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

bool FafuRobotController::open_gripper(std::optional<double> angle) {
    return open_gripper(angle, GripperOpts{});
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

bool FafuRobotController::close_gripper(std::optional<double> angle) {
    return close_gripper(angle, GripperOpts{});
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

GraspResult FafuRobotController::grasp() {
    return grasp(GraspOpts{});
}

GraspResult FafuRobotController::grasp(const GraspOpts& opts) {
    if (!core_) throw std::runtime_error("controller core is unavailable");
    CoreLease operation(*core_, fafu::core::OperationKind::Grasp);
    if (!has_gripper_)
        throw std::runtime_error("FafuRobotController was constructed without a gripper");

    if (opts.force_threshold < 1 || opts.force_threshold > 32767) {
        throw std::invalid_argument("force_threshold must be in [1, 32767]");
    }
    const int effort_limit = opts.effort.value_or(opts.force_threshold);
    if (effort_limit < 1 || effort_limit > 32767) {
        throw std::invalid_argument("effort must be in [1, 32767]");
    }

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
        ht_->set_pos_vel_tqe(gripper_motor_id_, target_turns, opts.vel,
                             effort_limit, hightorque::PosUnit::Turns);
    }

    double min_progress_turns = std::max(0.0, opts.min_close_deg) / 360.0;
    return wait_until_gripper_done_(
        target_turns,
        opts.timeout_s,
        /*tolerance_turns=*/std::nullopt,
        std::make_optional<int>(opts.force_threshold),
        min_progress_turns);
}

void FafuRobotController::release() {
    release(GripperOpts{});
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
    if (std::find(cfg_.motor_ids.begin(), cfg_.motor_ids.end(), motor_id) ==
        cfg_.motor_ids.end()) {
        throw std::invalid_argument("set_limit motor is not configured");
    }
    if (!std::isfinite(lo) || !std::isfinite(hi)) {
        throw std::invalid_argument("position limits must be finite");
    }
    if (lo > hi) {
        throw std::invalid_argument(
            "position limit lower bound must not exceed upper bound");
    }
    if (!core_) throw std::runtime_error("controller core is unavailable");

    CoreLease operation(*core_, fafu::core::OperationKind::Lifecycle);
    ht_->enable_position_limit(motor_id, lo, hi, unit);
}

std::optional<std::pair<double, double>>
FafuRobotController::get_limit(int motor_id) const {
    if (std::find(cfg_.motor_ids.begin(), cfg_.motor_ids.end(), motor_id) ==
        cfg_.motor_ids.end()) {
        throw std::invalid_argument("get_limit motor is not configured");
    }
    double lo = 0.0, hi = 0.0;
    if (!ht_->get_position_limit_turns(motor_id, lo, hi)) return std::nullopt;
    return std::make_pair(lo, hi);
}

void FafuRobotController::disable_limit(int motor_id) {
    if (std::find(cfg_.motor_ids.begin(), cfg_.motor_ids.end(), motor_id) ==
        cfg_.motor_ids.end()) {
        throw std::invalid_argument("disable_limit motor is not configured");
    }
    if (!core_) throw std::runtime_error("controller core is unavailable");

    CoreLease operation(*core_, fafu::core::OperationKind::Lifecycle);
    ht_->disable_position_limit(motor_id);
}

void FafuRobotController::clear_limits() {
    if (!core_) throw std::runtime_error("controller core is unavailable");

    CoreLease operation(*core_, fafu::core::OperationKind::Lifecycle);
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
    if (!ht_) return;

    const bool needs_shutdown =
        core_ && core_->state() != fafu::core::RobotState::Disconnected;
    if (!needs_shutdown && !ht_->is_open()) return;

    if (needs_shutdown) {
        core_->shutdown(
            to_finish_mode(joint_release),
            to_finish_mode(gripper_release),
            5.0);
    }
    if (ht_->is_open()) ht_->close();

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

} // namespace fafu_robot
