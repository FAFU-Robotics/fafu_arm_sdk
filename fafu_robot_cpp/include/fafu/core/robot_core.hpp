#pragma once

#include "fafu/core/controller_state.hpp"
#include "fafu/core/core_types.hpp"
#include "fafu/core/motor_io.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace fafu::core {

class CommandLease;

struct CoreConfig {
    std::vector<int> all_motor_ids;
    std::vector<int> joint_motor_ids;
    std::vector<std::string> joint_motor_models;
    int max_torque_raw = 0;
    double stale_feedback_timeout_ms = 500.0;
    double polling_rate_hz = 50.0;
};

class RobotCore {
public:
    RobotCore(MotorIO& io, CoreConfig config);
    RobotCore(hightorque::HightorqueSerial& serial, CoreConfig config);
    ~RobotCore();

    RobotCore(const RobotCore&) = delete;
    RobotCore& operator=(const RobotCore&) = delete;

    RobotState state() const;
    OperationKind active_operation() const;
    bool operation_owned_by_current_thread() const;
    bool cancel_requested() const noexcept;
    std::string dead_reason() const;
    HealthSnapshot health() const;

    std::uint64_t begin_operation(OperationKind kind);
    void end_operation(std::uint64_t token);
    CommandLease command_guard();
    void transition(RobotState next);

    void start_transport(bool async_rx, bool polling,
                         double polling_rate_hz = 0.0);
    void stop_transport();

    EnableResult enable();
    EnableResult enable(const EnableOptions& options);
    void disable();
    void brake();
    // Owner-only abort cleanup for an in-flight joint/raw motion. It brakes
    // every motor without ending the operation token; the caller's lease
    // remains responsible for unwinding. Safety latches are preserved.
    void brake_active_operation();
    void emergency_stop();
    bool resume();
    bool resume(const EnableOptions& options);
    bool check_alive(bool fresh = true, double timeout_s = 0.1);
    bool recover(bool confirm, double timeout_s = 0.2);

    MoveJResult move_j(const std::vector<double>& joint_angles_rad);
    MoveJResult move_j(const std::vector<double>& joint_angles_rad,
                       const MoveJOptions& options);

    void set_joint_motor_models(std::vector<std::string> motor_models);
    void set_stale_feedback_timeout_ms(double timeout_ms);
    bool stream_link_ok();

    void servo_start();
    void servo_start(const ServoOptions& options);
    ServoTickResult servo_tick(const std::vector<double>& target_angles);
    ServoTickResult servo_tick(
        const std::vector<double>& target_angles,
        const std::vector<double>& torque_ff_nm);
    ServoSummary servo_end(FinishMode finish_mode = FinishMode::Hold);
    bool is_servoing() const;
    ServoSummary servo_summary() const;

    void shutdown(FinishMode joint_release = FinishMode::Brake,
                  FinishMode auxiliary_release = FinishMode::Brake,
                  double wait_timeout_s = 5.0);

private:
    friend class CommandLease;

    struct ServoSession {
        bool active = false;
        std::uint64_t operation_token = 0;
        ServoOptions options;
        std::vector<double> last_target_turns;
        std::vector<double> filtered_target_turns;
        std::vector<int> kp_raw;
        std::vector<int> kd_raw;
        int lag_streak = 0;
        ServoSummary summary;
        std::chrono::steady_clock::time_point started_at{};
    };

    static void validate_config(const CoreConfig& config);
    static ServoOptions normalize_servo_options(ServoOptions options);
    static std::vector<double> expand_gains(
        const std::vector<double>& values, std::size_t count, bool kp);
    static void sleep_seconds(double seconds);

    EnableResult switch_to_active(const EnableOptions& options);
    bool switch_mode_all(int mode, int retries, const EnableOptions& options,
                         std::vector<int>& failed);
    std::vector<MotorDiagnostic> collect_diagnostics(double timeout_s);
    std::vector<int> stale_joint_ids() const;
    void latch_dead(const std::vector<int>& stale);
    void latch_dead(std::string reason);
    double validated_position_target_turns(
        int motor_id, double position_turns) const;
    void ensure_command_allowed() const;
    void release_motors_unlocked(
        const std::vector<int>& motor_ids, FinishMode mode);
    ServoSummary end_servo_locked(FinishMode finish_mode, bool require_owner);

    std::unique_ptr<HightorqueMotorIO> owned_adapter_;
    MotorIO& io_;
    CoreConfig config_;
    std::atomic<double> stale_feedback_timeout_ms_;
    ControllerState controller_state_;

    // Serializes all writes for this controller instance. Safety paths latch
    // state first, then acquire this mutex and send the state-specific release:
    // ESTOP uses STOP; DEAD/feedback timeout uses BRAKE.
    mutable std::mutex command_mutex_;
    mutable std::mutex safety_mutex_;
    std::atomic<std::uint64_t> safety_generation_{0};
    mutable std::mutex transport_mutex_;
    bool owns_async_rx_ = false;
    bool owns_polling_ = false;
    std::atomic<bool> requested_async_rx_{false};
    std::atomic<bool> requested_polling_{false};
    std::atomic<double> polling_rate_hz_;

    mutable std::mutex servo_mutex_;
    ServoSession servo_;
};

class CommandLease {
public:
    ~CommandLease() = default;

    CommandLease(const CommandLease&) = delete;
    CommandLease& operator=(const CommandLease&) = delete;
    CommandLease(CommandLease&&) noexcept = default;
    CommandLease& operator=(CommandLease&&) noexcept = default;

    void release() noexcept {
        if (lock_.owns_lock()) {
            lock_.unlock();
        }
    }

private:
    friend class RobotCore;
    explicit CommandLease(RobotCore& core);

    std::unique_lock<std::mutex> lock_;
};

}  // namespace fafu::core
