#include "fafu/core/robot_core.hpp"

#include "fafu/core/motor_calibration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace fafu::core {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

bool is_latched_or_disconnected(RobotState state) {
    return state == RobotState::Estop ||
           state == RobotState::Dead ||
           state == RobotState::Disconnected;
}

std::string stale_message(const std::vector<int>& motor_ids,
                          double timeout_ms) {
    std::ostringstream oss;
    oss << "stale motor feedback:";
    for (int motor_id : motor_ids) {
        oss << " M" << motor_id;
    }
    oss << " (limit=" << timeout_ms << "ms)";
    return oss.str();
}

}  // namespace

RobotCore::RobotCore(MotorIO& io, CoreConfig config)
    : io_(io), config_(std::move(config)),
      stale_feedback_timeout_ms_(config_.stale_feedback_timeout_ms),
      controller_state_(RobotState::Disabled),
      polling_rate_hz_(config_.polling_rate_hz) {
    validate_config(config_);
    if (config_.joint_motor_models.empty()) {
        config_.joint_motor_models.resize(config_.joint_motor_ids.size());
    }
}

RobotCore::RobotCore(hightorque::HightorqueSerial& serial, CoreConfig config)
    : owned_adapter_(std::make_unique<HightorqueMotorIO>(serial)),
      io_(*owned_adapter_), config_(std::move(config)),
      stale_feedback_timeout_ms_(config_.stale_feedback_timeout_ms),
      controller_state_(RobotState::Disabled),
      polling_rate_hz_(config_.polling_rate_hz) {
    validate_config(config_);
    if (config_.joint_motor_models.empty()) {
        config_.joint_motor_models.resize(config_.joint_motor_ids.size());
    }
}

RobotCore::~RobotCore() {
    try {
        if (state() != RobotState::Disconnected) {
            shutdown(FinishMode::Stop, FinishMode::Brake, 1.0);
        }
    } catch (...) {
        try {
            stop_transport();
        } catch (...) {
        }
    }
}

void RobotCore::validate_config(const CoreConfig& config) {
    if (config.all_motor_ids.empty()) {
        throw std::invalid_argument("all_motor_ids cannot be empty");
    }
    if (config.joint_motor_ids.empty()) {
        throw std::invalid_argument("joint_motor_ids cannot be empty");
    }

    const std::set<int> all(config.all_motor_ids.begin(),
                            config.all_motor_ids.end());
    if (all.size() != config.all_motor_ids.size() || *all.begin() <= 0) {
        throw std::invalid_argument(
            "all_motor_ids must contain unique positive IDs");
    }
    const std::set<int> joints(config.joint_motor_ids.begin(),
                               config.joint_motor_ids.end());
    if (joints.size() != config.joint_motor_ids.size() ||
        *joints.begin() <= 0) {
        throw std::invalid_argument(
            "joint_motor_ids must contain unique positive IDs");
    }
    for (int motor_id : joints) {
        if (all.count(motor_id) == 0) {
            throw std::invalid_argument(
                "joint_motor_ids must be a subset of all_motor_ids");
        }
    }
    if (!config.joint_motor_models.empty() &&
        config.joint_motor_models.size() != config.joint_motor_ids.size()) {
        throw std::invalid_argument(
            "joint_motor_models must be empty or match joint_motor_ids");
    }
    for (const std::string& model : config.joint_motor_models) {
        if (!model.empty()) {
            (void)torque_coefficient(model);
        }
    }
    if (!std::isfinite(config.stale_feedback_timeout_ms) ||
        config.stale_feedback_timeout_ms <= 0.0) {
        throw std::invalid_argument(
            "stale_feedback_timeout_ms must be positive and finite");
    }
    if (!std::isfinite(config.polling_rate_hz) ||
        config.polling_rate_hz <= 0.0) {
        throw std::invalid_argument(
            "polling_rate_hz must be positive and finite");
    }
}

RobotState RobotCore::state() const {
    return controller_state_.state();
}

OperationKind RobotCore::active_operation() const {
    return controller_state_.active_operation();
}

bool RobotCore::operation_owned_by_current_thread() const {
    return controller_state_.operation_owned_by_current_thread();
}

bool RobotCore::cancel_requested() const noexcept {
    return controller_state_.cancel_requested();
}

std::string RobotCore::dead_reason() const {
    return controller_state_.dead_reason();
}

HealthSnapshot RobotCore::health() const {
    HealthSnapshot snapshot;
    snapshot.state = state();
    snapshot.active_operation = active_operation();
    snapshot.closing = controller_state_.closing();
    snapshot.cancel_requested = cancel_requested();
    snapshot.dead_reason = dead_reason();
    snapshot.stale_motor_ids = stale_joint_ids();
    snapshot.link_ok = io_.is_open() && snapshot.stale_motor_ids.empty();
    return snapshot;
}

std::uint64_t RobotCore::begin_operation(OperationKind kind) {
    return controller_state_.begin(kind);
}

void RobotCore::end_operation(std::uint64_t token) {
    controller_state_.end(token);
}

CommandLease RobotCore::command_guard() {
    return CommandLease(*this);
}

void RobotCore::ensure_command_allowed() const {
    if (!controller_state_.operation_owned_by_current_thread()) {
        throw BusyError(
            "command send requires ownership of the active operation");
    }
    if (controller_state_.closing()) {
        throw StateError("controller is closing");
    }
    const RobotState current = state();
    if (current == RobotState::Estop) {
        throw StateError("command rejected: ESTOP is latched");
    }
    if (current == RobotState::Dead) {
        throw StateError("command rejected: controller is DEAD");
    }
    if (current == RobotState::Disconnected) {
        throw StateError("command rejected: controller is disconnected");
    }
}

CommandLease::CommandLease(RobotCore& core)
    : lock_(core.command_mutex_) {
    core.ensure_command_allowed();
}

void RobotCore::transition(RobotState next) {
    controller_state_.transition(next);
}

void RobotCore::start_transport(
        bool async_rx, bool polling, double polling_rate_hz) {
    std::lock_guard<std::mutex> lock(transport_mutex_);
    requested_async_rx_.store(async_rx, std::memory_order_release);
    requested_polling_.store(polling, std::memory_order_release);
    if (polling_rate_hz > 0.0) {
        polling_rate_hz_.store(polling_rate_hz, std::memory_order_release);
    }

    try {
        if (async_rx && !io_.is_async_rx()) {
            io_.enable_async_rx();
            owns_async_rx_ = true;
        }
        if (polling && !io_.is_polling()) {
            io_.start_polling(
                config_.all_motor_ids,
                std::max(10.0, polling_rate_hz_.load(
                    std::memory_order_acquire)));
            owns_polling_ = true;
        }
    } catch (...) {
        if (owns_polling_) {
            try {
                io_.stop_polling();
            } catch (...) {
            }
            owns_polling_ = false;
        }
        if (owns_async_rx_) {
            try {
                io_.disable_async_rx();
            } catch (...) {
            }
            owns_async_rx_ = false;
        }
        throw;
    }
}

void RobotCore::stop_transport() {
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if (owns_polling_ || io_.is_polling()) {
        io_.stop_polling();
    }
    owns_polling_ = false;
    if (owns_async_rx_ || io_.is_async_rx()) {
        io_.disable_async_rx();
    }
    owns_async_rx_ = false;
}

void RobotCore::sleep_seconds(double seconds) {
    if (seconds > 0.0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    }
}

bool RobotCore::switch_mode_all(
        int mode, int retries, const EnableOptions& options,
        std::vector<int>& failed) {
    retries = std::max(1, retries);
    for (int attempt = 0; attempt < retries; ++attempt) {
        if (cancel_requested()) {
            return false;
        }

        failed.clear();
        for (int motor_id : config_.all_motor_ids) {
            if (cancel_requested()) {
                return false;
            }
            try {
                auto command = command_guard();
                const auto motor = io_.set_mode(motor_id, mode);
                if (!motor || motor->mode != mode) {
                    failed.push_back(motor_id);
                }
            } catch (...) {
                if (cancel_requested()) {
                    return false;
                }
                failed.push_back(motor_id);
            }
        }

        if (!failed.empty()) {
            sleep_seconds(options.verify_delay_s);
            std::vector<int> verified_failed;
            for (int motor_id : failed) {
                if (cancel_requested()) {
                    return false;
                }
                try {
                    const auto motor = io_.read_state(motor_id, 0.3);
                    if (!motor || motor->mode != mode) {
                        verified_failed.push_back(motor_id);
                    }
                } catch (...) {
                    verified_failed.push_back(motor_id);
                }
            }
            failed.swap(verified_failed);
        }
        if (failed.empty()) {
            return true;
        }
        if (attempt + 1 < retries) {
            sleep_seconds(options.retry_delay_s);
        }
    }
    return false;
}

std::vector<MotorDiagnostic> RobotCore::collect_diagnostics(
        double timeout_s) {
    std::vector<MotorDiagnostic> diagnostics;
    diagnostics.reserve(config_.all_motor_ids.size());
    for (int motor_id : config_.all_motor_ids) {
        MotorDiagnostic diagnostic;
        diagnostic.motor_id = motor_id;
        try {
            const auto motor = io_.read_state(motor_id, timeout_s);
            if (motor) {
                diagnostic.responded = true;
                diagnostic.mode = motor->mode;
                diagnostic.fault = motor->fault;
                diagnostic.position_turns = motor->position;
            } else {
                diagnostic.detail = "no response";
            }
        } catch (const std::exception& error) {
            diagnostic.detail = error.what();
        } catch (...) {
            diagnostic.detail = "unknown read error";
        }
        diagnostics.push_back(std::move(diagnostic));
    }
    return diagnostics;
}

EnableResult RobotCore::switch_to_active(const EnableOptions& options) {
    EnableResult result;
    const int retries = std::max(1, options.normal_retries);

    bool all_active = true;
    bool has_mit_residue = false;
    for (int motor_id : config_.all_motor_ids) {
        try {
            const auto motor = io_.read_state(motor_id, 0.2);
            if (!motor || motor->mode != MODE_ACTIVE) {
                all_active = false;
            }
            if (motor && motor->mode == MODE_MIT) {
                has_mit_residue = true;
            }
        } catch (...) {
            all_active = false;
        }
    }
    if (all_active) {
        result.success = true;
        result.stage = ModeStage::AlreadyActive;
        result.message = "all motors are already active";
        return result;
    }

    auto reset_motors = [&](const std::vector<int>& motor_ids,
                            int repeat) {
        for (int motor_id : motor_ids) {
            if (cancel_requested()) {
                return;
            }
            try {
                auto command = command_guard();
                io_.stop(motor_id);
            } catch (...) {
                if (cancel_requested()) {
                    return;
                }
            }
            for (int count = 0; count < repeat; ++count) {
                if (cancel_requested()) {
                    return;
                }
                try {
                    auto command = command_guard();
                    io_.motor_reset(motor_id);
                } catch (...) {
                    if (cancel_requested()) {
                        return;
                    }
                }
                sleep_seconds(options.reset_spacing_s);
            }
        }
    };

    std::vector<int> failed;
    if (has_mit_residue && options.allow_motor_reset) {
        reset_motors(config_.all_motor_ids, 1);
        sleep_seconds(options.reset_wait_s);
        if (switch_mode_all(MODE_ACTIVE, retries, options, failed)) {
            result.success = true;
            result.stage = ModeStage::MitReset;
            result.message = "recovered from MIT residual mode";
            return result;
        }

        for (int round = 0;
             round < std::max(0, options.aggressive_reset_rounds);
             ++round) {
            std::vector<int> stuck;
            for (int motor_id : config_.all_motor_ids) {
                try {
                    const auto motor = io_.read_state(motor_id, 0.2);
                    if (!motor || motor->mode == MODE_MIT) {
                        stuck.push_back(motor_id);
                    }
                } catch (...) {
                    stuck.push_back(motor_id);
                }
            }
            if (stuck.empty()) {
                break;
            }
            reset_motors(
                stuck, std::max(1, options.resets_per_aggressive_round));
            sleep_seconds(options.aggressive_reset_wait_s);
            if (switch_mode_all(MODE_ACTIVE, retries, options, failed)) {
                result.success = true;
                result.stage = ModeStage::AggressiveReset;
                result.message = "recovered after aggressive MIT reset";
                return result;
            }
        }
    } else if (switch_mode_all(
                   MODE_ACTIVE, retries, options, failed)) {
        result.success = true;
        result.stage = ModeStage::NormalSwitch;
        result.message = "switched all motors to active mode";
        return result;
    }

    if (!options.allow_motor_reset) {
        result.failed_motor_ids = std::move(failed);
        result.diagnostics = collect_diagnostics(0.3);
        result.message = "mode switch failed and motor reset is disabled";
        return result;
    }

    reset_motors(config_.all_motor_ids, 1);
    sleep_seconds(options.reset_wait_s);
    if (switch_mode_all(MODE_ACTIVE, retries, options, failed)) {
        result.success = true;
        result.stage = has_mit_residue
            ? ModeStage::MitReset : ModeStage::SoftReset;
        result.message = "recovered after motor reset";
        return result;
    }

    result.failed_motor_ids = std::move(failed);
    result.diagnostics = collect_diagnostics(0.3);
    result.message = "motors did not enter active mode after recovery";
    return result;
}

EnableResult RobotCore::enable(const EnableOptions& options) {
    OperationLease operation(controller_state_, OperationKind::Lifecycle);
    EnableResult result = switch_to_active(options);
    if (result.success) {
        controller_state_.transition(RobotState::Idle);
        return result;
    }

    // A partial enable is not a usable IDLE state. Serialize the fallback
    // STOP with all other writers and leave the controller DISABLED.
    std::lock_guard<std::mutex> command(command_mutex_);
    const RobotState current = state();
    if (!is_latched_or_disconnected(current)) {
        release_motors_unlocked(config_.all_motor_ids, FinishMode::Stop);
        controller_state_.transition(RobotState::Disabled);
    }
    return result;
}

void RobotCore::release_motors_unlocked(
        const std::vector<int>& motor_ids, FinishMode mode) {
    for (int motor_id : motor_ids) {
        try {
            switch (mode) {
                case FinishMode::Stop:
                    io_.stop(motor_id);
                    break;
                case FinishMode::Brake:
                    io_.brake(motor_id);
                    break;
                case FinishMode::Hold:
                    (void)io_.set_mode(motor_id, MODE_ACTIVE);
                    break;
            }
        } catch (...) {
            // Best effort: one failed motor must not prevent the remaining
            // motors from reaching the requested safe state.
        }
    }
}

void RobotCore::disable() {
    OperationLease operation(controller_state_, OperationKind::Lifecycle);
    auto command = command_guard();
    release_motors_unlocked(config_.all_motor_ids, FinishMode::Stop);
    controller_state_.transition(RobotState::Disabled);
}

void RobotCore::brake() {
    OperationLease operation(controller_state_, OperationKind::Lifecycle);
    auto command = command_guard();
    release_motors_unlocked(config_.all_motor_ids, FinishMode::Brake);
    controller_state_.transition(RobotState::Braked);
}

void RobotCore::emergency_stop() {
    std::lock_guard<std::mutex> safety(safety_mutex_);
    if (state() == RobotState::Disconnected) {
        return;
    }
    safety_generation_.fetch_add(1, std::memory_order_acq_rel);
    controller_state_.latch_estop();
    std::lock_guard<std::mutex> command(command_mutex_);
    release_motors_unlocked(config_.all_motor_ids, FinishMode::Stop);
}

bool RobotCore::resume(const EnableOptions& options) {
    const std::uint64_t generation =
        safety_generation_.load(std::memory_order_acquire);
    std::unique_lock<std::mutex> safety(safety_mutex_);
    if (generation !=
        safety_generation_.load(std::memory_order_acquire)) {
        return false;
    }
    if (state() != RobotState::Estop) {
        return state() == RobotState::Idle;
    }
    if (active_operation() != OperationKind::None) {
        throw BusyError("cannot resume until the interrupted operation exits");
    }
    controller_state_.clear_latched(RobotState::Disabled);
    safety.unlock();
    return enable(options).success;
}

std::vector<int> RobotCore::stale_joint_ids() const {
    std::vector<int> stale;
    bool streaming = false;
    try {
        streaming = io_.is_async_rx() || io_.is_polling();
    } catch (...) {
        return config_.joint_motor_ids;
    }
    if (!streaming) {
        return stale;
    }
    for (int motor_id : config_.joint_motor_ids) {
        try {
            const double age_ms = io_.state_age_ms(motor_id);
            if (!std::isfinite(age_ms) ||
                age_ms > stale_feedback_timeout_ms_.load(
                    std::memory_order_acquire)) {
                stale.push_back(motor_id);
            }
        } catch (...) {
            stale.push_back(motor_id);
        }
    }
    return stale;
}

void RobotCore::latch_dead(const std::vector<int>& stale) {
    std::lock_guard<std::mutex> safety(safety_mutex_);
    if (state() == RobotState::Disconnected) {
        return;
    }
    safety_generation_.fetch_add(1, std::memory_order_acq_rel);
    controller_state_.latch_dead(
        stale_message(stale, stale_feedback_timeout_ms_.load(
            std::memory_order_acquire)));
    std::lock_guard<std::mutex> command(command_mutex_);
    release_motors_unlocked(config_.all_motor_ids, FinishMode::Stop);
}

bool RobotCore::stream_link_ok() {
    if (state() == RobotState::Dead ||
        state() == RobotState::Disconnected) {
        return false;
    }
    const std::vector<int> stale = stale_joint_ids();
    if (stale.empty()) {
        return true;
    }
    latch_dead(stale);
    return false;
}

bool RobotCore::check_alive(bool fresh, double timeout_s) {
    std::vector<int> missing;
    for (int motor_id : config_.joint_motor_ids) {
        try {
            const auto motor = fresh
                ? io_.read_state(motor_id, timeout_s)
                : io_.cached_state(motor_id);
            if (!motor) {
                missing.push_back(motor_id);
            }
        } catch (...) {
            missing.push_back(motor_id);
        }
    }
    if (missing.empty()) {
        return true;
    }
    latch_dead(missing);
    return false;
}

bool RobotCore::recover(bool confirm, double timeout_s) {
    if (!confirm) {
        return false;
    }
    const std::uint64_t generation =
        safety_generation_.load(std::memory_order_acquire);
    const RobotState current = state();
    if (current != RobotState::Dead) {
        // DEAD recovery must never masquerade as ESTOP recovery.
        return current != RobotState::Estop &&
               current != RobotState::Disconnected;
    }
    if (active_operation() != OperationKind::None) {
        throw BusyError("cannot recover until the interrupted operation exits");
    }

    stop_transport();
    start_transport(
        requested_async_rx_.load(std::memory_order_acquire),
        requested_polling_.load(std::memory_order_acquire),
        polling_rate_hz_.load(std::memory_order_acquire));

    for (int motor_id : config_.all_motor_ids) {
        try {
            if (!io_.read_state(motor_id, timeout_s)) {
                return false;
            }
        } catch (...) {
            return false;
        }
    }
    std::lock_guard<std::mutex> safety(safety_mutex_);
    if (generation !=
        safety_generation_.load(std::memory_order_acquire)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> command(command_mutex_);
        release_motors_unlocked(config_.all_motor_ids, FinishMode::Stop);
        controller_state_.clear_latched(RobotState::Disabled);
    }
    return true;
}

void RobotCore::set_joint_motor_models(
        std::vector<std::string> motor_models) {
    if (motor_models.size() != config_.joint_motor_ids.size()) {
        throw std::invalid_argument(
            "motor model count must match joint motor count");
    }
    for (const std::string& model : motor_models) {
        if (!model.empty()) {
            (void)torque_coefficient(model);
        }
    }
    std::lock_guard<std::mutex> lock(servo_mutex_);
    if (active_operation() != OperationKind::None) {
        throw BusyError("cannot change motor models during an operation");
    }
    config_.joint_motor_models = std::move(motor_models);
}

void RobotCore::set_stale_feedback_timeout_ms(double timeout_ms) {
    if (!std::isfinite(timeout_ms) || timeout_ms <= 0.0) {
        throw std::invalid_argument(
            "feedback timeout must be positive and finite");
    }
    stale_feedback_timeout_ms_.store(timeout_ms, std::memory_order_release);
}

ServoOptions RobotCore::normalize_servo_options(ServoOptions options) {
    if (options.watchdog_ms < 0) {
        throw std::invalid_argument("watchdog_ms cannot be negative");
    }
    options.watchdog_ms = std::min(options.watchdog_ms, 32767);
    if (!std::isfinite(options.max_velocity_rad_s) ||
        options.max_velocity_rad_s <= 0.0) {
        options.max_velocity_rad_s = 1.0;
    }
    if (!std::isfinite(options.max_step_rad) ||
        options.max_step_rad <= 0.0) {
        options.max_step_rad = 0.05;
    }
    if (!std::isfinite(options.max_lag_rad)) {
        throw std::invalid_argument("max_lag_rad must be finite");
    }
    if (!std::isfinite(options.nominal_rate_hz) ||
        options.nominal_rate_hz <= 0.0) {
        options.nominal_rate_hz = 100.0;
    }
    if (!std::isfinite(options.lookahead_time_s) ||
        options.lookahead_time_s < 0.0) {
        options.lookahead_time_s = 0.0;
    }
    options.lag_abort_consecutive =
        std::max(0, options.lag_abort_consecutive);
    return options;
}

std::vector<double> RobotCore::expand_gains(
        const std::vector<double>& values, std::size_t count, bool kp) {
    if (values.empty()) {
        if (count == 6) {
            return kp
                ? std::vector<double>{30.0, 40.0, 55.0, 15.0, 7.0, 5.0}
                : std::vector<double>{3.0, 4.0, 5.5, 1.5, 0.7, 0.5};
        }
        return std::vector<double>(count, kp ? 20.0 : 2.0);
    }
    if (values.size() == 1) {
        return std::vector<double>(count, values.front());
    }
    if (values.size() != count) {
        throw std::invalid_argument(
            "MIT gain vector must be scalar or match joint count");
    }
    return values;
}

void RobotCore::servo_start(const ServoOptions& options) {
    if (state() == RobotState::Disabled || state() == RobotState::Braked) {
        const EnableResult enabled = enable();
        if (!enabled.success) {
            throw StateError("servo_start could not enable all motors: " +
                             enabled.message);
        }
    }

    std::lock_guard<std::mutex> lock(servo_mutex_);
    if (servo_.active) {
        if (controller_state_.operation_owned_by_current_thread()) {
            return;
        }
        throw BusyError("another thread owns the active servo session");
    }

    ServoOptions normalized = normalize_servo_options(options);
    if (normalized.channel == ServoChannel::Mit &&
        (*std::max_element(config_.joint_motor_ids.begin(),
                           config_.joint_motor_ids.end()) > 6 ||
         config_.joint_motor_ids.size() > 6)) {
        throw std::invalid_argument(
            "MIT broadcast supports at most six joint slots");
    }

    const std::uint64_t token =
        controller_state_.begin(OperationKind::Servo);
    try {
        if (!io_.is_async_rx()) {
            start_transport(
                true,
                requested_polling_.load(std::memory_order_acquire),
                polling_rate_hz_.load(std::memory_order_acquire));
        }

        std::vector<double> initial;
        initial.reserve(config_.joint_motor_ids.size());
        for (int motor_id : config_.joint_motor_ids) {
            auto motor = io_.cached_state(motor_id);
            if (!motor) {
                motor = io_.read_state(motor_id, 0.1);
            }
            if (!motor) {
                throw StateError(
                    "servo_start cannot read motor " +
                    std::to_string(motor_id));
            }
            initial.push_back(motor->position);
        }

        if (normalized.watchdog_ms > 0) {
            auto command = command_guard();
            for (int motor_id : config_.joint_motor_ids) {
                io_.set_watchdog(motor_id, normalized.watchdog_ms);
            }
        }

        std::vector<int> kp_raw;
        std::vector<int> kd_raw;
        if (normalized.channel == ServoChannel::Mit) {
            const std::vector<double> kp = expand_gains(
                normalized.mit_kp, config_.joint_motor_ids.size(), true);
            const std::vector<double> kd = expand_gains(
                normalized.mit_kd, config_.joint_motor_ids.size(), false);
            kp_raw.reserve(kp.size());
            kd_raw.reserve(kd.size());
            for (std::size_t i = 0; i < kp.size(); ++i) {
                kp_raw.push_back(gain_to_raw(
                    kp[i], config_.joint_motor_models[i]));
                kd_raw.push_back(gain_to_raw(
                    kd[i], config_.joint_motor_models[i]));
            }
        }

        servo_ = {};
        servo_.active = true;
        servo_.operation_token = token;
        servo_.options = std::move(normalized);
        servo_.last_target_turns = initial;
        servo_.filtered_target_turns = std::move(initial);
        servo_.kp_raw = std::move(kp_raw);
        servo_.kd_raw = std::move(kd_raw);
        servo_.started_at = std::chrono::steady_clock::now();
    } catch (...) {
        std::lock_guard<std::mutex> command(command_mutex_);
        for (int motor_id : config_.joint_motor_ids) {
            try {
                io_.set_watchdog(motor_id, 0);
            } catch (...) {
            }
        }
        controller_state_.end(token);
        throw;
    }
}

ServoTickResult RobotCore::servo_tick(
        const std::vector<double>& target_angles,
        const std::vector<double>& torque_ff_nm) {
    std::lock_guard<std::mutex> lock(servo_mutex_);
    ServoTickResult result;
    if (!servo_.active) {
        result.message = "servo session is not active";
        return result;
    }
    if (!controller_state_.operation_owned_by_current_thread()) {
        throw BusyError("servo tick must run on the servo owner thread");
    }
    if (cancel_requested() || state() != RobotState::Servoing) {
        result.message = "servo session was cancelled";
        result.aborted = true;
        end_servo_locked(FinishMode::Stop, true);
        return result;
    }
    if (!stream_link_ok()) {
        result.message = dead_reason();
        result.aborted = true;
        end_servo_locked(FinishMode::Stop, true);
        return result;
    }

    const std::size_t count = config_.joint_motor_ids.size();
    if (target_angles.size() != count) {
        result.message = "target length does not match joint count";
        return result;
    }
    if (!torque_ff_nm.empty() && torque_ff_nm.size() != count) {
        result.message = "torque feed-forward length does not match joint count";
        return result;
    }
    for (double value : target_angles) {
        if (!std::isfinite(value)) {
            result.message = "target contains NaN or infinity";
            return result;
        }
    }
    for (double value : torque_ff_nm) {
        if (!std::isfinite(value)) {
            result.message = "torque feed-forward contains NaN or infinity";
            return result;
        }
    }

    const ServoOptions& options = servo_.options;
    const double dt_s = 1.0 / options.nominal_rate_hz;
    const double max_velocity_turns_s =
        options.max_velocity_rad_s / kTwoPi;
    const double max_step_turns = options.max_step_rad / kTwoPi;

    std::vector<double> target_turns(count);
    for (std::size_t i = 0; i < count; ++i) {
        target_turns[i] = options.input_is_radians
            ? target_angles[i] / kTwoPi
            : target_angles[i] / 360.0;
        const double delta =
            target_turns[i] - servo_.last_target_turns[i];
        if (std::abs(delta) > max_step_turns) {
            target_turns[i] = servo_.last_target_turns[i] +
                std::copysign(max_step_turns, delta);
            result.clamped = true;
        }
    }

    std::vector<double> positions(count);
    if (options.lookahead_time_s > 0.0) {
        const double alpha =
            dt_s / (options.lookahead_time_s + dt_s);
        for (std::size_t i = 0; i < count; ++i) {
            positions[i] = servo_.filtered_target_turns[i] +
                alpha * (target_turns[i] -
                         servo_.filtered_target_turns[i]);
        }
    } else {
        positions = target_turns;
    }

    // Position frames use this as a positive velocity limit. MIT frames use
    // it as a signed desired velocity and therefore must default to zero.
    std::vector<double> velocities(
        count,
        options.channel == ServoChannel::Position
            ? max_velocity_turns_s : 0.0);
    if (options.feedforward_velocity) {
        for (std::size_t i = 0; i < count; ++i) {
            const double velocity =
                (positions[i] - servo_.filtered_target_turns[i]) / dt_s;
            velocities[i] = std::clamp(
                velocity, -max_velocity_turns_s, max_velocity_turns_s);
        }
    }

    if (options.max_lag_rad > 0.0) {
        const double max_lag_turns = options.max_lag_rad / kTwoPi;
        for (std::size_t i = 0; i < count; ++i) {
            const auto motor = io_.cached_state(config_.joint_motor_ids[i]);
            if (motor &&
                std::abs(motor->position - positions[i]) > max_lag_turns) {
                result.lag_tripped = true;
                break;
            }
        }
    }
    if (result.lag_tripped) {
        ++servo_.lag_streak;
        ++servo_.summary.lag_count;
    } else {
        servo_.lag_streak = 0;
    }

    try {
        auto command = command_guard();
        if (options.channel == ServoChannel::Mit) {
            const std::vector<int> torque_raw = torque_ff_nm.empty()
                ? std::vector<int>(count, 0)
                : torques_to_raw(
                    torque_ff_nm, config_.joint_motor_models, 1.0);
            const int max_joint_id = *std::max_element(
                config_.joint_motor_ids.begin(),
                config_.joint_motor_ids.end());
            io_.send_mit(
                config_.joint_motor_ids, positions, velocities,
                torque_raw, servo_.kp_raw, servo_.kd_raw, max_joint_id);
        } else {
            const int max_motor_id = *std::max_element(
                config_.all_motor_ids.begin(),
                config_.all_motor_ids.end());
            io_.send_position(
                config_.joint_motor_ids, positions, velocities,
                config_.max_torque_raw, max_motor_id);
        }
    } catch (const std::exception& error) {
        result.message = error.what();
        return result;
    } catch (...) {
        result.message = "unknown send error";
        return result;
    }

    servo_.last_target_turns = std::move(target_turns);
    servo_.filtered_target_turns = std::move(positions);
    ++servo_.summary.tick_count;
    if (result.clamped) {
        ++servo_.summary.clamp_count;
    }
    result.sent = true;

    if (options.lag_abort_consecutive > 0 &&
        servo_.lag_streak >= options.lag_abort_consecutive) {
        servo_.summary.aborted_reason =
            "persistent tracking lag for " +
            std::to_string(servo_.lag_streak) + " ticks";
        result.aborted = true;
        result.message = servo_.summary.aborted_reason;
        end_servo_locked(FinishMode::Brake, true);
    }
    return result;
}

ServoSummary RobotCore::end_servo_locked(
        FinishMode finish_mode, bool require_owner) {
    if (!servo_.active) {
        return servo_.summary;
    }
    if (require_owner &&
        !controller_state_.operation_owned_by_current_thread()) {
        throw BusyError("servo session must be ended by its owner thread");
    }

    {
        std::lock_guard<std::mutex> command(command_mutex_);
        for (int motor_id : config_.joint_motor_ids) {
            try {
                io_.set_watchdog(motor_id, 0);
            } catch (...) {
            }
        }

        const RobotState current = state();
        if (current == RobotState::Estop ||
            current == RobotState::Dead ||
            current == RobotState::Disconnected) {
            finish_mode = FinishMode::Stop;
        }
        if (finish_mode == FinishMode::Hold) {
            if (servo_.options.channel == ServoChannel::Mit) {
                release_motors_unlocked(
                    config_.joint_motor_ids, FinishMode::Hold);
            }
        } else {
            release_motors_unlocked(config_.joint_motor_ids, finish_mode);
        }
    }

    const double elapsed_s = std::max(
        1e-9,
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - servo_.started_at).count());
    servo_.summary.elapsed_s = elapsed_s;
    servo_.summary.average_rate_hz =
        static_cast<double>(servo_.summary.tick_count) / elapsed_s;

    const std::uint64_t token = servo_.operation_token;
    servo_.active = false;
    servo_.operation_token = 0;
    servo_.lag_streak = 0;
    servo_.last_target_turns.clear();
    servo_.filtered_target_turns.clear();
    servo_.kp_raw.clear();
    servo_.kd_raw.clear();

    try {
        if (state() == RobotState::Servoing) {
            controller_state_.transition(
                finish_mode == FinishMode::Hold
                    ? RobotState::Idle
                    : finish_mode == FinishMode::Brake
                        ? RobotState::Braked
                        : RobotState::Disabled);
        }
    } catch (...) {
        // An ESTOP/DEAD latch may win between the state read and transition.
        // Always release the servo operation token so recovery cannot wedge.
        controller_state_.end(token);
        if (is_latched_or_disconnected(state())) {
            return servo_.summary;
        }
        throw;
    }
    controller_state_.end(token);
    return servo_.summary;
}

ServoSummary RobotCore::servo_end(FinishMode finish_mode) {
    std::lock_guard<std::mutex> lock(servo_mutex_);
    return end_servo_locked(finish_mode, true);
}

bool RobotCore::is_servoing() const {
    std::lock_guard<std::mutex> lock(servo_mutex_);
    return servo_.active;
}

ServoSummary RobotCore::servo_summary() const {
    std::lock_guard<std::mutex> lock(servo_mutex_);
    return servo_.summary;
}

void RobotCore::shutdown(
        FinishMode joint_release,
        FinishMode auxiliary_release,
        double wait_timeout_s) {
    {
        std::lock_guard<std::mutex> lock(servo_mutex_);
        controller_state_.begin_closing();
        if (servo_.active) {
            // begin_closing cancels the session. The servo mutex guarantees
            // that no tick is currently using the transport.
            end_servo_locked(joint_release, false);
        }
    }

    if (!controller_state_.wait_until_idle(wait_timeout_s)) {
        throw BusyError(
            "timed out waiting for the active operation to stop");
    }

    stop_transport();
    std::lock_guard<std::mutex> command(command_mutex_);
    if (state() == RobotState::Estop || state() == RobotState::Dead) {
        joint_release = FinishMode::Stop;
        auxiliary_release = FinishMode::Stop;
    }
    release_motors_unlocked(config_.joint_motor_ids, joint_release);

    std::set<int> joints(
        config_.joint_motor_ids.begin(), config_.joint_motor_ids.end());
    std::vector<int> auxiliary;
    for (int motor_id : config_.all_motor_ids) {
        if (joints.count(motor_id) == 0) {
            auxiliary.push_back(motor_id);
        }
    }
    release_motors_unlocked(auxiliary, auxiliary_release);
    controller_state_.mark_disconnected();
}

}  // namespace fafu::core
