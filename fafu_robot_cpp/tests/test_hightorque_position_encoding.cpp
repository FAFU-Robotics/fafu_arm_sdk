#include "hightorque_serial.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int16_t read_i16(const std::vector<uint8_t>& payload, std::size_t offset) {
    require(offset + 1 < payload.size(), "payload field is out of range");
    const uint16_t raw = static_cast<uint16_t>(payload[offset]) |
                         (static_cast<uint16_t>(payload[offset + 1]) << 8);
    return static_cast<int16_t>(raw);
}

void test_position_limits_are_non_negative() {
    const int16_t positive_speed =
        hightorque::position_speed_limit_to_int16(0.5);
    const int16_t negative_speed =
        hightorque::position_speed_limit_to_int16(-0.5);
    require(positive_speed > 0 && negative_speed == positive_speed,
            "Position speed must be encoded as a non-negative magnitude");

    const int16_t positive_acceleration =
        hightorque::position_acceleration_limit_to_int16(2.0);
    const int16_t negative_acceleration =
        hightorque::position_acceleration_limit_to_int16(-2.0);
    require(positive_acceleration > 0 &&
                negative_acceleration == positive_acceleration,
            "Position acceleration must be encoded as a non-negative magnitude");
}

void test_negative_target_uses_positive_position_limits() {
    const int16_t signed_velocity = hightorque::rps_to_int16(-0.5);
    const auto pos_vel_tqe = hightorque::build_pos_vel_tqe_int16(
        hightorque::turns_to_int16(-0.25),
        signed_velocity, hightorque::NAN_INT16);
    require(read_i16(pos_vel_tqe, 13) < 0,
            "test setup must encode a negative Position target");
    require(read_i16(pos_vel_tqe, 7) > 0,
            "negative Position target must still carry a positive speed limit");

    const auto pos_vel_acc = hightorque::build_pos_velmax_acc_int16(
        hightorque::turns_to_int16(-0.25),
        hightorque::position_speed_limit_to_int16(-0.5),
        hightorque::position_acceleration_limit_to_int16(-2.0));
    require(read_i16(pos_vel_acc, 5) < 0,
            "test setup must encode a negative Position target");
    require(read_i16(pos_vel_acc, 9) > 0 && read_i16(pos_vel_acc, 11) > 0,
            "Position speed and acceleration limits must stay positive");

    const auto broadcast = hightorque::build_many_pos_vel_tqe_int16(
        {hightorque::turns_to_int16(-0.25)},
        {signed_velocity},
        {hightorque::NAN_INT16});
    require(read_i16(broadcast, 0) < 0 && read_i16(broadcast, 2) > 0,
            "0x8090 Position broadcast must use a positive speed limit");

    const auto inactive = hightorque::build_many_pos_vel_tqe_int16(
        {hightorque::NAN_INT16},
        {hightorque::NAN_INT16},
        {hightorque::NAN_INT16});
    require(read_i16(inactive, 2) == hightorque::NAN_INT16,
            "inactive Position slot must preserve the NAN velocity sentinel");
}

void test_mit_velocity_remains_signed() {
    const int16_t signed_velocity = hightorque::rps_to_int16(-0.5);
    const auto single = hightorque::build_pos_vel_tqe_kp_kd_int16(
        hightorque::turns_to_int16(-0.25), signed_velocity, 0, 1, 1);
    require(read_i16(single, 7) < 0,
            "MIT single-motor velocity must remain signed");
}

void test_non_finite_limits_are_rejected() {
    bool threw = false;
    try {
        (void)hightorque::position_speed_limit_to_int16(
            std::numeric_limits<double>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "non-finite Position speed limit must be rejected");
}

void test_position_broadcast_rejects_more_than_ten_slots() {
    const std::vector<int16_t> values(11, hightorque::NAN_INT16);
    bool threw = false;
    try {
        (void)hightorque::build_many_pos_vel_tqe_int16(
            values, values, values);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw,
            "0x8090 Position broadcast must reject payloads over 64 bytes");
}

}  // namespace

int main() {
    try {
        test_position_limits_are_non_negative();
        test_negative_target_uses_positive_position_limits();
        test_mit_velocity_remains_signed();
        test_non_finite_limits_are_rejected();
        test_position_broadcast_rejects_more_than_ten_slots();
        std::cout << "Hightorque Position encoding tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
