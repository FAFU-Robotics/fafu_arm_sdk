#include "fafu/core/motor_io.hpp"

#include <algorithm>
#include <stdexcept>

namespace fafu::core {

bool HightorqueMotorIO::is_open() const {
    return serial_.is_open();
}

std::optional<hightorque::MotorState> HightorqueMotorIO::read_state(
        int motor_id, double timeout_s) {
    return serial_.read_motor_state(motor_id, timeout_s);
}

std::optional<hightorque::MotorState> HightorqueMotorIO::cached_state(
        int motor_id) const {
    return serial_.get_cached_state(motor_id);
}

double HightorqueMotorIO::state_age_ms(int motor_id) const {
    return serial_.get_state_age_ms(motor_id);
}

std::optional<std::pair<double, double>>
HightorqueMotorIO::position_limit_turns(int motor_id) const {
    double lo = 0.0;
    double hi = 0.0;
    if (!serial_.get_position_limit_turns(motor_id, lo, hi)) {
        return std::nullopt;
    }
    return std::make_pair(lo, hi);
}

std::optional<hightorque::MotorState> HightorqueMotorIO::set_mode(
        int motor_id, int mode) {
    if (mode < 0 || mode > 0xFF) {
        throw std::invalid_argument("motor mode must fit in uint8");
    }
    return serial_.set_motor_mode(
        motor_id, static_cast<std::uint8_t>(mode));
}

void HightorqueMotorIO::motor_reset(int motor_id) {
    (void)serial_.motor_reset(motor_id);
}

void HightorqueMotorIO::set_watchdog(int motor_id, int timeout_ms) {
    if (timeout_ms < 0 || timeout_ms > 32767) {
        throw std::invalid_argument("watchdog must be in [0, 32767] ms");
    }
    (void)serial_.set_timeout(
        motor_id, static_cast<std::int16_t>(timeout_ms));
}

void HightorqueMotorIO::stop(int motor_id) {
    (void)serial_.stop(motor_id);
}

void HightorqueMotorIO::brake(int motor_id) {
    (void)serial_.brake(motor_id);
}

void HightorqueMotorIO::send_position(
        const std::vector<int>& motor_ids,
        const std::vector<double>& positions_turns,
        const std::vector<double>& velocities_turns_s,
        int max_torque_raw,
        int max_motor_id) {
    if (motor_ids.size() != positions_turns.size() ||
        motor_ids.size() != velocities_turns_s.size()) {
        throw std::invalid_argument(
            "position command vectors must have the same length");
    }
    serial_.set_many_pos_vel_tqe_partial(
        motor_ids, positions_turns, velocities_turns_s,
        static_cast<std::int16_t>(
            std::clamp(max_torque_raw, -32768, 32767)),
        {}, hightorque::PosUnit::Turns, max_motor_id, 0.0);
}

void HightorqueMotorIO::send_mit(
        const std::vector<int>& motor_ids,
        const std::vector<double>& positions_turns,
        const std::vector<double>& velocities_turns_s,
        const std::vector<int>& torques_raw,
        const std::vector<int>& kp_raw,
        const std::vector<int>& kd_raw,
        int max_motor_id) {
    (void)serial_.set_many_mit(
        motor_ids, positions_turns, velocities_turns_s,
        torques_raw, kp_raw, kd_raw,
        hightorque::PosUnit::Turns, max_motor_id, 0.0);
}

bool HightorqueMotorIO::is_async_rx() const {
    return serial_.is_async_rx();
}

bool HightorqueMotorIO::is_polling() const {
    return serial_.is_polling();
}

void HightorqueMotorIO::enable_async_rx() {
    serial_.enable_async_rx();
}

void HightorqueMotorIO::disable_async_rx() {
    serial_.disable_async_rx();
}

void HightorqueMotorIO::start_polling(
        const std::vector<int>& motor_ids, double rate_hz) {
    serial_.start_state_polling(motor_ids, rate_hz);
}

void HightorqueMotorIO::stop_polling() {
    serial_.stop_state_polling();
}

}  // namespace fafu::core
