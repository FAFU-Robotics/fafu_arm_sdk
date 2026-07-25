#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fafu::core {

double torque_coefficient(const std::string& motor_model);

std::int16_t torque_to_raw(double torque_nm,
                           const std::string& motor_model,
                           double torque_scale = 1.0);

std::int16_t gain_to_raw(double gain,
                         const std::string& motor_model);

std::vector<int> torques_to_raw(const std::vector<double>& torques_nm,
                                const std::vector<std::string>& motor_models,
                                double torque_scale = 1.0);

}  // namespace fafu::core
