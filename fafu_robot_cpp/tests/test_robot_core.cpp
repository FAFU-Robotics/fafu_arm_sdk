#include "fake_motor_io.hpp"

#include "fafu/core/motor_calibration.hpp"
#include "fafu/core/robot_core.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using fafu::core::BusyError;
using fafu::core::CoreConfig;
using fafu::core::EnableOptions;
using fafu::core::FinishMode;
using fafu::core::MODE_BRAKE;
using fafu::core::MODE_ACTIVE;
using fafu::core::MODE_MIT;
using fafu::core::ModeStage;
using fafu::core::MoveJOptions;
using fafu::core::OperationKind;
using fafu::core::RobotCore;
using fafu::core::RobotState;
using fafu::core::ServoChannel;
using fafu::core::ServoOptions;
using fafu::core::StateError;
using fafu::core::gain_to_raw;
using fafu::core::test::FakeMotorIO;
using fafu::core::torque_to_raw;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

CoreConfig config(std::vector<int> all, std::vector<int> joints) {
    CoreConfig value;
    value.all_motor_ids = std::move(all);
    value.joint_motor_ids = std::move(joints);
    value.joint_motor_models =
        std::vector<std::string>(value.joint_motor_ids.size(), "M5036_02");
    value.stale_feedback_timeout_ms = 500.0;
    return value;
}

EnableOptions fast_enable_options() {
    EnableOptions options;
    options.verify_delay_s = 0.0;
    options.retry_delay_s = 0.0;
    options.reset_spacing_s = 0.0;
    options.reset_wait_s = 0.0;
    options.aggressive_reset_wait_s = 0.0;
    return options;
}

void test_calibration() {
    require(torque_to_raw(0.67, "M5036_02") == 100,
            "torque conversion must include firmware 0.01 scale");
    require(gain_to_raw(1.0, "M5036_02") ==
                static_cast<int>(std::llround(
                    (1.0 / 0.67) * 10.0 * 6.28318530717958647692)),
            "gain conversion mismatch");
    require(torque_to_raw(1e9, "M5036_02") == 32767,
            "positive torque must saturate");
    require(torque_to_raw(-1e9, "M5036_02") == -32768,
            "negative torque must saturate");
}

void test_operation_arbitration() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) {
        io.set_mode_direct(id, MODE_ACTIVE);
    }
    require(core.enable(fast_enable_options()).success,
            "enable should accept already-active motors");

    const auto token = core.begin_operation(OperationKind::JointMotion);
    std::atomic<bool> got_busy{false};
    std::thread contender([&] {
        try {
            (void)core.begin_operation(OperationKind::JointMotion);
        } catch (const BusyError&) {
            got_busy.store(true);
        }
    });
    contender.join();
    require(got_busy.load(), "a second writer must fail with BusyError");
    require(core.state() == RobotState::Moving,
            "joint operation must expose MOVING");
    core.end_operation(token);
    require(core.state() == RobotState::Idle,
            "ending a joint operation must restore IDLE");
}

void test_enable_recovery() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    io.set_false_negative_mode_switches(2);
    auto result = core.enable(fast_enable_options());
    require(result.success, "fresh read must filter mode-switch false negatives");
    require(result.stage == ModeStage::NormalSwitch,
            "false-negative recovery should still be a normal switch");

    core.disable();
    io.set_mode_direct(1, MODE_MIT);
    result = core.enable(fast_enable_options());
    require(result.success, "MIT residue should recover through motor reset");
    require(io.reset_count() > 0, "MIT recovery must reset motors");
}

void test_servo_safety() {
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) {
        io.set_mode_direct(id, MODE_ACTIVE);
    }
    require(core.enable(fast_enable_options()).success,
            "servo fixture enable failed");
    core.start_transport(true, false);

    ServoOptions options;
    options.channel = ServoChannel::Position;
    options.max_step_rad = 0.05;
    options.max_lag_rad = 0.01;
    options.lag_abort_consecutive = 0;
    core.servo_start(options);

    io.set_position(1, -1.0);
    const auto tick = core.servo_tick({1.0, 0.0});
    require(tick.sent, "lagging servo tick must still send");
    require(tick.clamped, "large target jump must be clamped");
    require(tick.lag_tripped, "large measured lag must be counted");
    require(io.frames().size() == 1,
            "lag policy must not suppress the hardware frame");
    const auto summary = core.servo_end(FinishMode::Hold);
    require(summary.tick_count == 1, "servo tick count mismatch");
    require(summary.clamp_count == 1, "servo clamp count mismatch");
    require(summary.lag_count == 1, "servo lag count mismatch");
}

void test_position_servo_velocity_policy() {
    constexpr double kTwoPi = 6.28318530717958647692;
    FakeMotorIO io({1});
    RobotCore core(io, config({1}, {1}));
    io.set_mode_direct(1, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    core.start_transport(true, false);

    ServoOptions options;
    options.channel = ServoChannel::Position;
    options.max_velocity_rad_s = 0.5;
    options.max_step_rad = 1.0;
    options.max_lag_rad = 0.0;
    options.nominal_rate_hz = 100.0;
    options.feedforward_velocity = true;
    options.position_error_deadband_rad = 0.001;
    core.servo_start(options);

    require(core.servo_tick({-0.01}).sent, "negative Position tick failed");
    auto frames = io.frames();
    const double cap_turns_s = options.max_velocity_rad_s / kTwoPi;
    require(frames.size() == 1 && !frames.back().mit,
            "expected one Position frame");
    require(frames.back().velocities[0] > 0.0 &&
            frames.back().velocities[0] <= cap_turns_s + 1e-12,
            "Position velocity must be a positive capped limit");

    // The target no longer changes, but measured position still lags it.
    io.set_position(1, -0.005 / kTwoPi);
    require(core.servo_tick({-0.01}).sent, "Position catch-up tick failed");
    frames = io.frames();
    require(frames.back().velocities[0] > 0.0,
            "residual Position error must retain catch-up velocity");

    // Once measured error is inside the deadband, a stationary target settles.
    io.set_position(1, -0.0095 / kTwoPi);
    require(core.servo_tick({-0.01}).sent, "Position settle tick failed");
    frames = io.frames();
    require(std::abs(frames.back().velocities[0]) < 1e-12,
            "Position velocity must settle to zero inside the deadband");
    core.servo_end(FinishMode::Hold);

    options.feedforward_velocity = false;
    core.servo_start(options);
    require(core.servo_tick({-0.01}).sent, "fixed-limit Position tick failed");
    frames = io.frames();
    require(std::abs(frames.back().velocities[0] - cap_turns_s) < 1e-12,
            "disabled Position feed-forward must keep the fixed positive cap");
    core.servo_end(FinishMode::Hold);
}

void test_move_j_uses_radians_and_holds_auxiliary_motors() {
    constexpr double kPi = 3.14159265358979323846;
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
    io.set_position(1, 0.1);
    io.set_position(2, -0.2);
    io.set_position(7, 0.33);
    require(core.enable(fast_enable_options()).success, "enable failed");

    MoveJOptions options;
    options.block = false;
    options.max_velocity_rad_s = 1.0;
    const auto result = core.move_j({kPi, -kPi / 2.0}, options);

    require(result.sent && !result.reached,
            "non-blocking move_j must report one unconfirmed send");
    require(core.state() == RobotState::Idle,
            "move_j must release its JointMotion lease");
    const auto frames = io.frames();
    require(frames.size() == 1 && !frames.front().mit,
            "non-blocking move_j must send exactly one Position frame");
    require(frames.front().motor_ids == std::vector<int>({1, 2, 7}),
            "move_j frame must cover every motor in instance order");
    require(std::abs(frames.front().positions[0] - 0.5) < 1e-12 &&
            std::abs(frames.front().positions[1] + 0.25) < 1e-12,
            "radian targets must be converted to turns exactly once");
    require(std::abs(frames.front().positions[2] - 0.33) < 1e-12 &&
            frames.front().velocities[2] == 0.0,
            "auxiliary motors must hold their fresh measured position");
    require(frames.front().velocities[0] > 0.0 &&
            frames.front().velocities[1] > 0.0,
            "Position velocity is a non-negative speed cap");
}

void test_move_j_validates_before_state_or_hardware_side_effects() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));

    MoveJOptions invalid;
    invalid.max_velocity_rad_s =
        std::numeric_limits<double>::quiet_NaN();
    bool invalid_rejected = false;
    try {
        (void)core.move_j({0.0, 0.0}, invalid);
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    require(invalid_rejected, "non-finite move options must be rejected");
    require(core.state() == RobotState::Disabled && io.frames().empty(),
            "invalid move options must not acquire a lease or write");

    MoveJOptions valid;
    valid.block = false;
    bool disabled_rejected = false;
    try {
        (void)core.move_j({0.0, 0.0}, valid);
    } catch (const StateError&) {
        disabled_rejected = true;
    }
    require(disabled_rejected,
            "move_j must not implicitly enable a disabled controller");
    require(core.state() == RobotState::Disabled && io.frames().empty(),
            "rejected disabled move_j must leave hardware untouched");
    for (int id : {1, 2}) {
        const auto motor = io.read_state(id, 0.0);
        require(motor && motor->mode == fafu::core::MODE_STOP,
                "validation/state rejection must not energize motors");
    }
}

void test_blocking_move_j_confirms_feedback_and_keeps_velocity_unsigned() {
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
    io.set_position(7, 0.4);
    require(core.enable(fast_enable_options()).success, "enable failed");

    MoveJOptions options;
    options.block = true;
    options.max_velocity_rad_s = 100.0;
    options.control_rate_hz = 1000.0;
    options.min_duration_s = 0.0;
    options.tolerance_rad = 1e-4;
    options.settle_timeout_s = 0.0;
    const auto result = core.move_j({0.02, -0.02}, options);

    require(result.sent && result.reached,
            "blocking move_j must confirm all joint feedback");
    require(result.max_error_rad <= options.tolerance_rad,
            "blocking move_j returned outside its tolerance");
    const auto frames = io.frames();
    require(!frames.empty(), "blocking move_j sent no frames");
    require(std::abs(frames.back().positions[0] - 0.0031) < 1e-12 &&
            std::abs(frames.back().positions[1] + 0.0031) < 1e-12,
            "move_j must send the protocol-quantized target");
    require(frames.back().velocities[0] > 0.0 &&
            frames.back().velocities[1] > 0.0,
            "terminal Position frame must retain a positive speed cap");
    for (const auto& frame : frames) {
        require(std::all_of(
                    frame.velocities.begin(), frame.velocities.end(),
                    [](double value) { return value >= 0.0; }),
                "Position trajectory must never encode signed velocity");
        require(std::abs(frame.positions[2] - 0.4) < 1e-12,
                "blocking move_j must hold the auxiliary motor");
    }
}

void test_move_j_timeout_brakes_and_uses_positive_catchup_cap() {
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    io.set_follow_position_commands(false);

    MoveJOptions options;
    options.max_velocity_rad_s = 100.0;
    options.control_rate_hz = 1000.0;
    options.min_duration_s = 0.0;
    options.tolerance_rad = 1e-4;
    options.settle_timeout_s = 0.005;

    bool timed_out = false;
    try {
        (void)core.move_j({0.02, -0.02}, options);
    } catch (const std::runtime_error& error) {
        timed_out = std::string(error.what()).find("timed out") !=
            std::string::npos;
    }
    require(timed_out, "unreached blocking move_j must time out");
    require(core.state() == RobotState::Braked,
            "generic move_j failure must brake the instance");
    const auto frames = io.frames();
    require(std::any_of(
                frames.begin(), frames.end(),
                [](const auto& frame) {
                    return frame.velocities[0] > 1e-6 &&
                           frame.velocities[1] > 1e-6;
                }),
            "settle phase must keep a positive catch-up velocity cap");
    for (int id : {1, 2, 7}) {
        const auto motor = io.read_state(id, 0.0);
        require(motor && motor->mode == MODE_BRAKE,
                "move_j timeout must brake joints and auxiliaries");
    }
}

void test_invalid_servo_options_do_not_enable() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));

    ServoOptions options;
    options.channel = ServoChannel::Position;
    options.watchdog_ms = -1;
    bool rejected = false;
    try {
        core.servo_start(options);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid Servo options must be rejected");
    require(core.state() == RobotState::Disabled,
            "invalid Servo options must not enable a disabled controller");
    require(!io.is_async_rx() && io.frames().empty(),
            "invalid Servo options must not start transport or write");
    for (int id : {1, 2}) {
        const auto motor = io.read_state(id, 0.0);
        require(motor && motor->mode == fafu::core::MODE_STOP,
                "invalid Servo options must leave motors stopped");
    }
}

void test_motion_targets_respect_protocol_and_soft_limits() {
    FakeMotorIO io({1, 2});
    io.set_position_limit(1, -0.1, 0.1);
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    bool limit_read_under_lease = false;
    io.set_position_limit_observer([&] {
        limit_read_under_lease =
            core.active_operation() == OperationKind::JointMotion &&
            core.operation_owned_by_current_thread();
    });

    bool soft_limit_rejected = false;
    try {
        (void)core.move_j({1.0, 0.0});
    } catch (const std::invalid_argument&) {
        soft_limit_rejected = true;
    }
    require(soft_limit_rejected && io.frames().empty() &&
            limit_read_under_lease &&
            core.state() == RobotState::Idle,
            "move_j must validate soft limits under its motion lease");

    bool range_rejected = false;
    try {
        (void)core.move_j({25.0, 0.0});
    } catch (const std::invalid_argument&) {
        range_rejected = true;
    }
    require(range_rejected && io.frames().empty() &&
            core.state() == RobotState::Idle,
            "move_j must reject an unencodable Position target");

    io.set_position_limit(1, -0.00016, -0.00014);
    limit_read_under_lease = false;
    bool quantized_limit_rejected = false;
    try {
        MoveJOptions move;
        move.block = false;
        (void)core.move_j(
            {-0.00015 * 6.28318530717958647692, 0.0}, move);
    } catch (const std::invalid_argument&) {
        quantized_limit_rejected = true;
    }
    require(quantized_limit_rejected && io.frames().empty() &&
            limit_read_under_lease && core.state() == RobotState::Idle,
            "protocol quantization must not cross a soft limit");

    bool slot_rejected = false;
    try {
        FakeMotorIO high_id_io({1, 11});
        RobotCore high_id(
            high_id_io, config({1, 11}, {1}));
    } catch (const std::invalid_argument&) {
        slot_rejected = true;
    }
    require(slot_rejected,
            "CoreConfig must reject Position broadcast slots above 10");

    FakeMotorIO closed_io({1, 2});
    RobotCore closed(closed_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) {
        closed_io.set_mode_direct(id, MODE_ACTIVE);
    }
    require(closed.enable(fast_enable_options()).success, "enable failed");
    closed_io.set_open(false);
    bool closed_rejected = false;
    try {
        MoveJOptions move;
        move.block = false;
        (void)closed.move_j({0.0, 0.0}, move);
    } catch (const StateError& error) {
        closed_rejected =
            std::string(error.what()).find("transport is closed") !=
            std::string::npos;
    }
    require(closed_rejected && closed.state() == RobotState::Dead,
            "move_j closed transport must latch the transport reason");

    FakeMotorIO throwing_io({1, 2});
    RobotCore throwing(throwing_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) throwing_io.set_mode_direct(id, MODE_ACTIVE);
    require(throwing.enable(fast_enable_options()).success, "enable failed");
    throwing_io.set_throw_on_is_open(true);
    bool throwing_rejected = false;
    try {
        MoveJOptions move;
        move.block = false;
        (void)throwing.move_j({0.0, 0.0}, move);
    } catch (const StateError& error) {
        throwing_rejected =
            std::string(error.what()).find("transport is closed") !=
            std::string::npos;
    }
    require(throwing_rejected && throwing.state() == RobotState::Dead,
            "is_open failure must latch DEAD instead of escaping");
}

void test_servo_start_uses_fresh_feedback_and_fails_braked() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    io.clear_read_counts();

    ServoOptions options;
    options.channel = ServoChannel::Position;
    core.servo_start(options);
    require(io.read_count(1) > 0 && io.read_count(2) > 0,
            "servo_start must fresh-read every joint");
    core.servo_end(FinishMode::Brake);
    require(core.state() == RobotState::Braked,
            "Servo Brake finish must leave BRAKED state");

    require(core.enable(fast_enable_options()).success, "re-enable failed");
    core.servo_start(options);
    core.servo_end(FinishMode::Stop);
    require(core.state() == RobotState::Disabled,
            "Servo Stop finish must leave DISABLED state");

    FakeMotorIO failing_io({1, 2});
    RobotCore failing(failing_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) failing_io.set_mode_direct(id, MODE_ACTIVE);
    require(failing.enable(fast_enable_options()).success, "enable failed");
    failing_io.fail_next_read(1);
    bool start_failed = false;
    try {
        failing.servo_start(options);
    } catch (const StateError&) {
        start_failed = true;
    }
    require(start_failed && failing.state() == RobotState::Braked &&
            !failing.is_servoing(),
            "failed servo_start must brake and release its operation");
}

void test_servo_invalid_target_and_send_failure_abort_safely() {
    ServoOptions options;
    options.channel = ServoChannel::Position;

    FakeMotorIO limited_io({1, 2});
    limited_io.set_position_limit(1, -0.1, 0.1);
    RobotCore limited(limited_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) limited_io.set_mode_direct(id, MODE_ACTIVE);
    require(limited.enable(fast_enable_options()).success, "enable failed");
    limited.servo_start(options);
    const auto invalid = limited.servo_tick({1.0, 0.0});
    require(invalid.aborted && !invalid.sent &&
            limited.state() == RobotState::Braked &&
            limited_io.watchdog(1) == 0,
            "invalid Servo target must abort, clear watchdog, and brake");

    FakeMotorIO send_io({1, 2});
    RobotCore send_core(send_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) send_io.set_mode_direct(id, MODE_ACTIVE);
    require(send_core.enable(fast_enable_options()).success, "enable failed");
    send_core.servo_start(options);
    send_io.fail_next_send();
    const auto failed_send = send_core.servo_tick({0.01, 0.0});
    require(failed_send.aborted && !failed_send.sent &&
            send_core.state() == RobotState::Braked &&
            send_io.watchdog(1) == 0,
            "Servo send failure must abort, clear watchdog, and brake");

    FakeMotorIO closed_io({1, 2});
    RobotCore closed(closed_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) closed_io.set_mode_direct(id, MODE_ACTIVE);
    require(closed.enable(fast_enable_options()).success, "enable failed");
    closed.servo_start(options);
    closed_io.set_open(false);
    const auto closed_tick = closed.servo_tick({0.0, 0.0});
    require(closed_tick.aborted && closed.state() == RobotState::Dead &&
            !closed.is_servoing(),
            "closed Servo transport must latch DEAD and release its token");
}

void test_dead_is_per_instance() {
    FakeMotorIO left_io({1, 2});
    FakeMotorIO right_io({1, 2});
    RobotCore left(left_io, config({1, 2}, {1, 2}));
    RobotCore right(right_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) {
        left_io.set_mode_direct(id, MODE_ACTIVE);
        right_io.set_mode_direct(id, MODE_ACTIVE);
    }
    require(left.enable(fast_enable_options()).success,
            "left enable failed");
    require(right.enable(fast_enable_options()).success,
            "right enable failed");
    left.start_transport(true, false);
    right.start_transport(true, false);

    left_io.set_age_ms(2, 1000.0);
    require(!left.stream_link_ok(), "left stale motor must latch DEAD");
    require(left.state() == RobotState::Dead,
            "left controller did not enter DEAD");
    for (int id : {1, 2}) {
        const auto motor = left_io.read_state(id, 0.0);
        require(motor && motor->mode == MODE_BRAKE,
                "DEAD must brake every motor in the affected instance");
    }
    require(right.stream_link_ok(), "right controller must remain healthy");
    require(right.state() == RobotState::Idle,
            "right state leaked from left controller");
}

void test_servo_feedback_timeout_stays_braked() {
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    core.start_transport(true, false);

    ServoOptions options;
    options.channel = ServoChannel::Position;
    options.max_lag_rad = 0.0;
    core.servo_start(options);
    io.set_age_ms(2, 1000.0);

    const auto tick = core.servo_tick({0.0, 0.0});
    require(tick.aborted && !tick.sent,
            "stale feedback must abort before another Servo frame");
    require(core.state() == RobotState::Dead,
            "Servo feedback timeout must latch DEAD");
    for (int id : {1, 2, 7}) {
        const auto motor = io.read_state(id, 0.0);
        require(motor && motor->mode == MODE_BRAKE,
                "Servo DEAD cleanup must not overwrite BRAKE with STOP");
    }
}

void test_dead_and_default_shutdown_use_brake() {
    {
        FakeMotorIO io({1, 2, 7});
        RobotCore core(io, config({1, 2, 7}, {1, 2}));
        for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
        require(core.enable(fast_enable_options()).success, "enable failed");
        core.start_transport(true, false);
        io.set_age_ms(1, 1000.0);
        require(!core.stream_link_ok(), "fixture must enter DEAD");

        core.shutdown(FinishMode::Hold, FinishMode::Stop, 1.0);
        for (int id : {1, 2, 7}) {
            const auto motor = io.read_state(id, 0.0);
            require(motor && motor->mode == MODE_BRAKE,
                    "DEAD shutdown must preserve BRAKE for joints and auxiliary motors");
        }
    }

    {
        FakeMotorIO io({1, 2, 7});
        RobotCore core(io, config({1, 2, 7}, {1, 2}));
        for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
        require(core.enable(fast_enable_options()).success, "enable failed");
        core.start_transport(true, false);
        io.set_age_ms(2, 1000.0);
        require(!core.stream_link_ok(), "fixture must enter DEAD");

        require(core.recover(true, 0.0), "DEAD recovery failed");
        require(core.state() == RobotState::Braked,
                "DEAD recovery must leave the controller BRAKED");
        for (int id : {1, 2, 7}) {
            const auto motor = io.read_state(id, 0.0);
            require(motor && motor->mode == MODE_BRAKE,
                    "DEAD recovery must leave every motor braked");
        }
    }

    {
        FakeMotorIO io({1, 2, 7});
        RobotCore core(io, config({1, 2, 7}, {1, 2}));
        for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
        require(core.enable(fast_enable_options()).success, "enable failed");

        core.shutdown();
        for (int id : {1, 2, 7}) {
            const auto motor = io.read_state(id, 0.0);
            require(motor && motor->mode == MODE_BRAKE,
                    "default shutdown must brake joints and auxiliary motors");
        }
    }
}

void test_generic_abort_brakes_and_estop_dominates_dead() {
    {
        FakeMotorIO io({1, 2, 7});
        RobotCore core(io, config({1, 2, 7}, {1, 2}));
        for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
        require(core.enable(fast_enable_options()).success, "enable failed");

        const auto token =
            core.begin_operation(OperationKind::JointMotion);
        core.brake_active_operation();
        require(core.state() == RobotState::Braked,
                "generic active-operation abort must transition to BRAKED");
        require(core.active_operation() == OperationKind::JointMotion,
                "abort cleanup must leave token unwinding to its owner");
        for (int id : {1, 2, 7}) {
            const auto motor = io.read_state(id, 0.0);
            require(motor && motor->mode == MODE_BRAKE,
                    "generic active-operation abort must brake every motor");
        }
        core.end_operation(token);
        require(core.state() == RobotState::Braked,
                "operation unwind must preserve BRAKED after abort");
    }

    {
        FakeMotorIO io({1, 2});
        RobotCore core(io, config({1, 2}, {1, 2}));
        for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
        require(core.enable(fast_enable_options()).success, "enable failed");
        core.start_transport(true, false);

        ServoOptions options;
        core.servo_start(options);
        bool rejected = false;
        try {
            core.brake_active_operation();
        } catch (const StateError&) {
            rejected = true;
        }
        require(rejected,
                "active-operation brake must reject a Servo session");
        require(core.state() == RobotState::Servoing && core.is_servoing(),
                "rejected abort cleanup must leave Servo state consistent");
        core.servo_end(FinishMode::Brake);
    }

    {
        FakeMotorIO io({1, 2, 7});
        RobotCore core(io, config({1, 2, 7}, {1, 2}));
        for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
        require(core.enable(fast_enable_options()).success, "enable failed");
        core.start_transport(true, false);
        io.set_age_ms(1, 1000.0);
        require(!core.stream_link_ok(), "fixture must enter DEAD");

        core.emergency_stop();
        require(core.state() == RobotState::Estop,
                "explicit ESTOP must dominate an earlier DEAD latch");
        for (int id : {1, 2, 7}) {
            const auto motor = io.read_state(id, 0.0);
            require(motor && motor->mode == fafu::core::MODE_STOP,
                    "DEAD -> ESTOP must stop every motor");
        }

        core.shutdown(FinishMode::Hold, FinishMode::Brake, 1.0);
        for (int id : {1, 2, 7}) {
            const auto motor = io.read_state(id, 0.0);
            require(motor && motor->mode == fafu::core::MODE_STOP,
                    "shutdown must preserve STOP after DEAD -> ESTOP");
        }
    }
}

void test_concurrent_readers() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) {
        io.set_mode_direct(id, MODE_ACTIVE);
    }
    require(core.enable(fast_enable_options()).success,
            "reader fixture enable failed");

    std::vector<std::thread> readers;
    for (int index = 0; index < 8; ++index) {
        readers.emplace_back([&] {
            for (int iteration = 0; iteration < 1000; ++iteration) {
                const auto health = core.health();
                require(health.state == RobotState::Idle,
                        "concurrent health read observed invalid state");
            }
        });
    }
    for (auto& reader : readers) {
        reader.join();
    }
}


void test_instances_are_independent() {
    FakeMotorIO left_io({1, 2});
    FakeMotorIO right_io({1, 2});
    RobotCore left(left_io, config({1, 2}, {1, 2}));
    RobotCore right(right_io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) {
        left_io.set_mode_direct(id, MODE_ACTIVE);
        right_io.set_mode_direct(id, MODE_ACTIVE);
    }
    require(left.enable(fast_enable_options()).success, "left enable failed");
    require(right.enable(fast_enable_options()).success, "right enable failed");

    const auto left_token =
        left.begin_operation(OperationKind::JointMotion);
    const auto right_token =
        right.begin_operation(OperationKind::JointMotion);
    require(left.state() == RobotState::Moving,
            "left operation did not start");
    require(right.state() == RobotState::Moving,
            "right operation was incorrectly blocked by left");
    {
        auto left_command = left.command_guard();
        auto right_command = right.command_guard();
        (void)left_command;
        (void)right_command;
    }
    right.end_operation(right_token);
    left.end_operation(left_token);
}

void test_estop_from_another_thread() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");

    const auto token =
        core.begin_operation(OperationKind::JointMotion);
    std::thread safety_thread([&] { core.emergency_stop(); });
    safety_thread.join();

    require(core.cancel_requested(), "ESTOP must request cancellation");
    require(core.state() == RobotState::Estop, "ESTOP must latch");
    core.end_operation(token);
    require(core.state() == RobotState::Estop,
            "operation exit must not clear ESTOP");
    require(!core.recover(true, 0.0),
            "DEAD recovery must not claim to recover ESTOP");
    require(core.state() == RobotState::Estop,
            "DEAD recovery must preserve the ESTOP latch");
    require(core.resume(fast_enable_options()),
            "resume should re-enable after operation exits");
}

void test_shutdown_waits_for_writer() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");

    std::atomic<bool> entered{false};
    std::thread writer([&] {
        const auto token =
            core.begin_operation(OperationKind::JointMotion);
        entered.store(true);
        while (!core.cancel_requested()) {
            std::this_thread::yield();
        }
        core.end_operation(token);
    });
    while (!entered.load()) std::this_thread::yield();

    core.shutdown(FinishMode::Stop, FinishMode::Brake, 1.0);
    writer.join();
    require(core.state() == RobotState::Disconnected,
            "shutdown must wait and mark DISCONNECTED");
}

void test_shutdown_from_writer_does_not_poison_core() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");

    const auto token =
        core.begin_operation(OperationKind::JointMotion);
    bool rejected = false;
    try {
        core.shutdown(FinishMode::Brake, FinishMode::Brake, 0.0);
    } catch (const BusyError&) {
        rejected = true;
    }
    require(rejected,
            "shutdown by writer owner must fail before closing starts");

    const auto snapshot = core.health();
    require(!snapshot.closing, "rejected shutdown must not poison closing");
    require(!snapshot.cancel_requested,
            "rejected shutdown must not request cancellation");
    require(core.state() == RobotState::Moving,
            "rejected shutdown must preserve the active operation");

    core.end_operation(token);
    require(core.state() == RobotState::Idle,
            "writer must unwind normally after rejected shutdown");
    core.shutdown();
    require(core.state() == RobotState::Disconnected,
            "a later shutdown must still succeed");
}


void test_enable_failure_leaves_disabled() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "initial enable failed");

    for (int id : {1, 2}) {
        io.set_mode_direct(id, fafu::core::MODE_STOP);
        io.set_missing(id, true);
    }
    auto options = fast_enable_options();
    options.allow_motor_reset = false;
    const auto result = core.enable(options);
    require(!result.success, "enable should fail with all motors missing");
    require(core.state() == RobotState::Disabled,
            "partial enable failure must leave the controller DISABLED");
}

void test_estop_orders_after_inflight_command() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");

    const auto token = core.begin_operation(OperationKind::JointMotion);
    auto command = core.command_guard();
    std::thread safety([&] { core.emergency_stop(); });

    while (core.state() != RobotState::Estop) {
        std::this_thread::yield();
    }
    command.release();
    safety.join();

    bool rejected = false;
    try {
        (void)core.command_guard();
    } catch (const StateError&) {
        rejected = true;
    }
    require(rejected, "normal commands must be rejected after ESTOP");
    core.end_operation(token);
    require(core.state() == RobotState::Estop,
            "operation exit must preserve the ESTOP latch");
}

void test_mit_without_velocity_feedforward_sends_zero_velocity() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    core.start_transport(true, false);

    ServoOptions options;
    options.channel = ServoChannel::Mit;
    options.feedforward_velocity = false;
    options.max_lag_rad = 0.0;
    core.servo_start(options);
    const auto tick = core.servo_tick({0.01, -0.01});
    require(tick.sent, "MIT servo tick was not sent");
    const auto frames = io.frames();
    require(frames.size() == 1 && frames.front().mit,
            "expected one MIT frame");
    require(std::all_of(
                frames.front().velocities.begin(),
                frames.front().velocities.end(),
                [](double value) { return value == 0.0; }),
            "MIT desired velocity must be zero when feed-forward is disabled");
    core.servo_end(FinishMode::Stop);
}

void test_mit_velocity_feedforward_remains_signed() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    core.start_transport(true, false);

    ServoOptions options;
    options.channel = ServoChannel::Mit;
    options.feedforward_velocity = true;
    options.max_step_rad = 1.0;
    options.max_lag_rad = 0.0;
    core.servo_start(options);
    const auto tick = core.servo_tick({-0.01, 0.01});
    require(tick.sent, "MIT signed-velocity tick was not sent");
    const auto frames = io.frames();
    require(frames.size() == 1 && frames.front().mit,
            "expected one MIT frame");
    require(frames.front().velocities[0] < 0.0 &&
            frames.front().velocities[1] > 0.0,
            "MIT feed-forward velocity must retain its sign");
    core.servo_end(FinishMode::Stop);
}

void test_unknown_motor_model_is_rejected_for_nonzero_output() {
    bool torque_rejected = false;
    try {
        (void)torque_to_raw(1.0, "");
    } catch (const std::invalid_argument&) {
        torque_rejected = true;
    }
    require(torque_rejected,
            "non-zero torque must reject an unknown motor model");
    require(torque_to_raw(0.0, "") == 0,
            "zero torque is safe without a motor model");

    FakeMotorIO io({1, 2});
    CoreConfig unknown = config({1, 2}, {1, 2});
    unknown.joint_motor_models = {"", ""};
    RobotCore core(io, unknown);
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    core.start_transport(true, false);

    bool servo_rejected = false;
    try {
        core.servo_start(ServoOptions{});
    } catch (const std::invalid_argument&) {
        servo_rejected = true;
    }
    require(servo_rejected,
            "MIT servo gains must reject unknown motor models");
    require(core.state() == RobotState::Idle,
            "failed servo_start must release its operation lease");

    ServoOptions invalid_watchdog;
    invalid_watchdog.channel = ServoChannel::Position;
    invalid_watchdog.watchdog_ms = -1;
    bool watchdog_rejected = false;
    try {
        core.servo_start(invalid_watchdog);
    } catch (const std::invalid_argument&) {
        watchdog_rejected = true;
    }
    require(watchdog_rejected,
            "negative watchdog values must be rejected, not disabled");
    require(core.state() == RobotState::Idle,
            "invalid watchdog must not leave an active servo operation");
}

void test_shutdown_cannot_override_latched_stop() {
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");

    core.emergency_stop();
    core.shutdown(FinishMode::Hold, FinishMode::Brake, 1.0);
    core.shutdown(FinishMode::Hold, FinishMode::Hold, 1.0);
    for (int id : {1, 2, 7}) {
        const auto motor = io.read_state(id, 0.0);
        require(motor && motor->mode == fafu::core::MODE_STOP,
                "repeated shutdown must not override a safety STOP");
    }
}

void test_shutdown_is_idempotent() {
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");

    core.shutdown(FinishMode::Brake, FinishMode::Brake, 1.0);
    core.shutdown(FinishMode::Hold, FinishMode::Hold, 1.0);
    for (int id : {1, 2, 7}) {
        const auto motor = io.read_state(id, 0.0);
        require(motor && motor->mode == MODE_BRAKE,
                "repeated shutdown must preserve the first release policy");
    }
}

void test_servo_has_an_owner_thread() {
    FakeMotorIO io({1, 2});
    RobotCore core(io, config({1, 2}, {1, 2}));
    for (int id : {1, 2}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    core.start_transport(true, false);

    ServoOptions options;
    options.channel = ServoChannel::Position;
    core.servo_start(options);

    std::atomic<bool> start_got_busy{false};
    std::atomic<bool> tick_got_busy{false};
    std::thread other([&] {
        try {
            core.servo_start(options);
        } catch (const BusyError&) {
            start_got_busy.store(true);
        }
        try {
            (void)core.servo_tick({0.0, 0.0});
        } catch (const BusyError&) {
            tick_got_busy.store(true);
        }
    });
    other.join();
    require(start_got_busy.load(),
            "restarting an owned servo from another thread must fail");
    require(tick_got_busy.load(),
            "servo ticks from a non-owner thread must fail");
    core.servo_end(FinishMode::Hold);
}

void test_servo_keeps_generic_arbitration_strict() {
    FakeMotorIO io({1, 2, 7});
    RobotCore core(io, config({1, 2, 7}, {1, 2}));
    for (int id : {1, 2, 7}) io.set_mode_direct(id, MODE_ACTIVE);
    require(core.enable(fast_enable_options()).success, "enable failed");
    core.start_transport(true, false);

    ServoOptions options;
    options.channel = ServoChannel::Position;
    core.servo_start(options);

    const auto require_same_thread_busy =
        [&](OperationKind kind, const std::string& message) {
            bool got_busy = false;
            try {
                (void)core.begin_operation(kind);
            } catch (const BusyError&) {
                got_busy = true;
            }
            require(got_busy, message);
        };
    require_same_thread_busy(
        OperationKind::GripperMotion,
        "generic Servo -> GripperMotion nesting must remain rejected");
    require_same_thread_busy(
        OperationKind::Grasp,
        "Servo -> Grasp nesting must remain rejected");
    require_same_thread_busy(
        OperationKind::JointMotion,
        "Servo -> JointMotion nesting must remain rejected");

    std::atomic<bool> other_thread_busy{false};
    std::thread other([&] {
        try {
            (void)core.begin_operation(OperationKind::GripperMotion);
        } catch (const BusyError&) {
            other_thread_busy.store(true);
        }
    });
    other.join();

    require(other_thread_busy.load(),
            "another thread must not borrow the Servo writer");
    require(core.active_operation() == OperationKind::Servo,
            "rejected operations must leave Servo active");
    require(core.state() == RobotState::Servoing,
            "rejected operations must leave the controller SERVOING");

    const auto tick = core.servo_tick({0.0, 0.0});
    require(tick.sent, "Servo must remain usable after rejected operations");
    core.servo_end(FinishMode::Hold);
}


}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"calibration", test_calibration},
        {"operation arbitration", test_operation_arbitration},
        {"enable recovery", test_enable_recovery},
        {"servo safety", test_servo_safety},
        {"Position servo velocity policy",
         test_position_servo_velocity_policy},
        {"move_j radians and auxiliary hold",
         test_move_j_uses_radians_and_holds_auxiliary_motors},
        {"move_j validates before side effects",
         test_move_j_validates_before_state_or_hardware_side_effects},
        {"blocking move_j feedback confirmation",
         test_blocking_move_j_confirms_feedback_and_keeps_velocity_unsigned},
        {"move_j timeout safety cleanup",
         test_move_j_timeout_brakes_and_uses_positive_catchup_cap},
        {"Servo validation before enable",
         test_invalid_servo_options_do_not_enable},
        {"motion target bounds",
         test_motion_targets_respect_protocol_and_soft_limits},
        {"Servo fresh start and failure cleanup",
         test_servo_start_uses_fresh_feedback_and_fails_braked},
        {"Servo invalid/send cleanup",
         test_servo_invalid_target_and_send_failure_abort_safely},
        {"per-instance DEAD isolation", test_dead_is_per_instance},
        {"Servo feedback timeout stays braked",
         test_servo_feedback_timeout_stays_braked},
        {"DEAD/default shutdown use brake",
         test_dead_and_default_shutdown_use_brake},
        {"generic abort brake and ESTOP priority",
         test_generic_abort_brakes_and_estop_dominates_dead},
        {"concurrent readers", test_concurrent_readers},
        {"independent SDK instances", test_instances_are_independent},
        {"cross-thread ESTOP", test_estop_from_another_thread},
        {"enable failure leaves disabled", test_enable_failure_leaves_disabled},
        {"ESTOP command ordering", test_estop_orders_after_inflight_command},
        {"MIT zero desired velocity",
         test_mit_without_velocity_feedforward_sends_zero_velocity},
        {"MIT signed desired velocity",
         test_mit_velocity_feedforward_remains_signed},
        {"unknown motor model rejection",
         test_unknown_motor_model_is_rejected_for_nonzero_output},
        {"shutdown waits for writer", test_shutdown_waits_for_writer},
        {"same-thread shutdown rejection stays recoverable",
         test_shutdown_from_writer_does_not_poison_core},
        {"shutdown preserves safety STOP",
         test_shutdown_cannot_override_latched_stop},
        {"shutdown is idempotent", test_shutdown_is_idempotent},
        {"servo owner thread", test_servo_has_an_owner_thread},
        {"Servo wrapper exception remains narrow",
         test_servo_keeps_generic_arbitration_strict},
    };

    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All core tests passed\n";
    return EXIT_SUCCESS;
}
