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
