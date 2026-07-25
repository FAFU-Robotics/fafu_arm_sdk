#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fafu::core {

inline constexpr int CORE_ABI_VERSION = 2;
inline constexpr int MODE_STOP = 0x00;
inline constexpr int MODE_ACTIVE = 0x0A;
inline constexpr int MODE_MIT = 0x0B;
inline constexpr int MODE_BRAKE = 0x0F;

enum class RobotState {
    Disconnected,
    Disabled,
    Braked,
    Idle,
    Moving,
    Servoing,
    Grasping,
    GravityComp,
    Estop,
    Dead,
};

enum class OperationKind {
    None,
    Lifecycle,
    JointMotion,
    Servo,
    GripperMotion,
    Grasp,
    GravityComp,
    RawStream,
};

enum class ServoChannel {
    Position,
    Mit,
};

enum class FinishMode {
    Stop,
    Brake,
    Hold,
};

enum class ModeStage {
    AlreadyActive,
    NormalSwitch,
    MitReset,
    SoftReset,
    AggressiveReset,
    Failed,
};

struct MotorDiagnostic {
    int motor_id = 0;
    bool responded = false;
    int mode = 0;
    int fault = 0;
    double position_turns = 0.0;
    std::string detail;
};

struct EnableOptions {
    bool allow_motor_reset = true;
    int normal_retries = 3;
    int aggressive_reset_rounds = 3;
    int resets_per_aggressive_round = 3;
    double verify_delay_s = 0.03;
    double retry_delay_s = 0.10;
    double reset_spacing_s = 0.05;
    double reset_wait_s = 1.0;
    double aggressive_reset_wait_s = 1.2;
};

struct EnableResult {
    bool success = false;
    ModeStage stage = ModeStage::Failed;
    std::vector<int> failed_motor_ids;
    std::vector<MotorDiagnostic> diagnostics;
    std::string message;
};

struct ServoOptions {
    int watchdog_ms = 100;
    double max_velocity_rad_s = 1.0;
    double max_step_rad = 0.05;
    double max_lag_rad = 0.2;
    double nominal_rate_hz = 100.0;
    bool input_is_radians = true;
    bool feedforward_velocity = true;
    double lookahead_time_s = 0.0;
    int lag_abort_consecutive = 0;
    ServoChannel channel = ServoChannel::Mit;
    std::vector<double> mit_kp;
    std::vector<double> mit_kd;
};

struct ServoTickResult {
    bool sent = false;
    bool clamped = false;
    bool lag_tripped = false;
    bool aborted = false;
    std::string message;
};

struct ServoSummary {
    std::uint64_t tick_count = 0;
    std::uint64_t clamp_count = 0;
    std::uint64_t lag_count = 0;
    double elapsed_s = 0.0;
    double average_rate_hz = 0.0;
    std::string aborted_reason;
};

struct HealthSnapshot {
    RobotState state = RobotState::Disconnected;
    OperationKind active_operation = OperationKind::None;
    bool closing = false;
    bool cancel_requested = false;
    bool link_ok = false;
    std::string dead_reason;
    std::vector<int> stale_motor_ids;
};

}  // namespace fafu::core
