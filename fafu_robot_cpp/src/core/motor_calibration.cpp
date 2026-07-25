#include "fafu/core/motor_calibration.hpp"

#include "hightorque_serial.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fafu::core {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

std::int16_t checked_i16(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
    const auto rounded = static_cast<long long>(std::llround(value));
    return static_cast<std::int16_t>(
        std::clamp<long long>(rounded,
                              std::numeric_limits<std::int16_t>::min(),
                              std::numeric_limits<std::int16_t>::max()));
}

}  // namespace

double torque_coefficient(const std::string& motor_model) {
    const auto it = hightorque::TORQUE_COEFF.find(motor_model);
    if (it == hightorque::TORQUE_COEFF.end() || it->second == 0.0) {
        throw std::invalid_argument(
            "unknown motor model '" + motor_model +
            "'; configure the exact model before sending torque or MIT gains");
    }
    return it->second;
}

std::int16_t torque_to_raw(double torque_nm,
                           const std::string& motor_model,
                           double torque_scale) {
    if (!std::isfinite(torque_nm)) {
        throw std::invalid_argument("torque must be finite");
    }
    if (!std::isfinite(torque_scale) || torque_scale < 0.0) {
        throw std::invalid_argument(
            "torque_scale must be finite and non-negative");
    }
    if (torque_nm == 0.0 || torque_scale == 0.0) {
        return 0;
    }
    const double coeff = torque_coefficient(motor_model);
    return checked_i16(
        torque_nm * torque_scale / (coeff * 0.01), "torque");
}

std::int16_t gain_to_raw(double gain, const std::string& motor_model) {
    if (!std::isfinite(gain) || gain < 0.0) {
        throw std::invalid_argument(
            "gain must be finite and non-negative");
    }
    if (gain == 0.0) {
        return 0;
    }
    const double coeff = torque_coefficient(motor_model);
    return checked_i16((gain / coeff) * 10.0 * kTwoPi, "gain");
}

std::vector<int> torques_to_raw(const std::vector<double>& torques_nm,
                                const std::vector<std::string>& motor_models,
                                double torque_scale) {
    if (!motor_models.empty() && motor_models.size() != torques_nm.size()) {
        throw std::invalid_argument(
            "motor_models and torques_nm must have the same length");
    }
    std::vector<int> result;
    result.reserve(torques_nm.size());
    for (std::size_t i = 0; i < torques_nm.size(); ++i) {
        const std::string model =
            motor_models.empty() ? std::string{} : motor_models[i];
        result.push_back(torque_to_raw(torques_nm[i], model, torque_scale));
    }
    return result;
}

}  // namespace fafu::core
