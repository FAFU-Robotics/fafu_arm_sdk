#pragma once

#include "hightorque_serial.hpp"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fafu::core {

class MotorIO {
public:
    virtual ~MotorIO() = default;

    virtual bool is_open() const = 0;
    virtual std::optional<hightorque::MotorState> read_state(
        int motor_id, double timeout_s) = 0;
    virtual std::optional<hightorque::MotorState> cached_state(
        int motor_id) const = 0;
    virtual double state_age_ms(int motor_id) const = 0;
    virtual std::optional<std::pair<double, double>> position_limit_turns(
        int motor_id) const = 0;

    virtual std::optional<hightorque::MotorState> set_mode(
        int motor_id, int mode) = 0;
    virtual void motor_reset(int motor_id) = 0;
    virtual void set_watchdog(int motor_id, int timeout_ms) = 0;
    virtual void stop(int motor_id) = 0;
    virtual void brake(int motor_id) = 0;

    virtual void send_position(const std::vector<int>& motor_ids,
                               const std::vector<double>& positions_turns,
                               const std::vector<double>& velocities_turns_s,
                               int max_torque_raw,
                               int max_motor_id) = 0;

    virtual void send_mit(const std::vector<int>& motor_ids,
                          const std::vector<double>& positions_turns,
                          const std::vector<double>& velocities_turns_s,
                          const std::vector<int>& torques_raw,
                          const std::vector<int>& kp_raw,
                          const std::vector<int>& kd_raw,
                          int max_motor_id) = 0;

    virtual bool is_async_rx() const = 0;
    virtual bool is_polling() const = 0;
    virtual void enable_async_rx() = 0;
    virtual void disable_async_rx() = 0;
    virtual void start_polling(const std::vector<int>& motor_ids,
                               double rate_hz) = 0;
    virtual void stop_polling() = 0;
};

class HightorqueMotorIO final : public MotorIO {
public:
    explicit HightorqueMotorIO(hightorque::HightorqueSerial& serial)
        : serial_(serial) {}

    bool is_open() const override;
    std::optional<hightorque::MotorState> read_state(
        int motor_id, double timeout_s) override;
    std::optional<hightorque::MotorState> cached_state(
        int motor_id) const override;
    double state_age_ms(int motor_id) const override;
    std::optional<std::pair<double, double>> position_limit_turns(
        int motor_id) const override;

    std::optional<hightorque::MotorState> set_mode(
        int motor_id, int mode) override;
    void motor_reset(int motor_id) override;
    void set_watchdog(int motor_id, int timeout_ms) override;
    void stop(int motor_id) override;
    void brake(int motor_id) override;

    void send_position(const std::vector<int>& motor_ids,
                       const std::vector<double>& positions_turns,
                       const std::vector<double>& velocities_turns_s,
                       int max_torque_raw,
                       int max_motor_id) override;

    void send_mit(const std::vector<int>& motor_ids,
                  const std::vector<double>& positions_turns,
                  const std::vector<double>& velocities_turns_s,
                  const std::vector<int>& torques_raw,
                  const std::vector<int>& kp_raw,
                  const std::vector<int>& kd_raw,
                  int max_motor_id) override;

    bool is_async_rx() const override;
    bool is_polling() const override;
    void enable_async_rx() override;
    void disable_async_rx() override;
    void start_polling(const std::vector<int>& motor_ids,
                       double rate_hz) override;
    void stop_polling() override;

private:
    hightorque::HightorqueSerial& serial_;
};

}  // namespace fafu::core
