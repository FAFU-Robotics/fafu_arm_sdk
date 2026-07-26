#pragma once

#include "fafu/core/core_types.hpp"
#include "fafu/core/motor_io.hpp"

#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fafu::core::test {

struct SentFrame {
    bool mit = false;
    std::vector<int> motor_ids;
    std::vector<double> positions;
    std::vector<double> velocities;
    std::vector<int> torques;
};

class FakeMotorIO final : public MotorIO {
public:
    explicit FakeMotorIO(std::vector<int> motor_ids) {
        for (int motor_id : motor_ids) {
            hightorque::MotorState state;
            state.id = motor_id;
            state.mode = MODE_STOP;
            states_[motor_id] = state;
            ages_ms_[motor_id] = 0.0;
        }
    }

    bool is_open() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (throw_on_is_open_) {
            throw std::runtime_error("injected is_open failure");
        }
        return open_;
    }

    std::optional<hightorque::MotorState> read_state(
            int motor_id, double) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++read_counts_[motor_id];
        if (fail_next_reads_.erase(motor_id) != 0) return std::nullopt;
        if (!open_ || missing_.count(motor_id) != 0) {
            return std::nullopt;
        }
        const auto it = states_.find(motor_id);
        return it == states_.end()
            ? std::nullopt
            : std::optional<hightorque::MotorState>(it->second);
    }

    std::optional<hightorque::MotorState> cached_state(
            int motor_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = states_.find(motor_id);
        return it == states_.end()
            ? std::nullopt
            : std::optional<hightorque::MotorState>(it->second);
    }

    double state_age_ms(int motor_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = ages_ms_.find(motor_id);
        return it == ages_ms_.end()
            ? std::numeric_limits<double>::infinity()
            : it->second;
    }

    std::optional<std::pair<double, double>> position_limit_turns(
            int motor_id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (position_limit_observer_) position_limit_observer_();
        const auto it = limits_.find(motor_id);
        if (it == limits_.end()) return std::nullopt;
        return it->second;
    }

    std::optional<hightorque::MotorState> set_mode(
            int motor_id, int mode) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = states_.find(motor_id);
        if (it == states_.end() || missing_.count(motor_id) != 0) {
            return std::nullopt;
        }
        if (it->second.mode != MODE_MIT || reset_clears_mit_) {
            it->second.mode = mode;
        }
        if (false_negative_mode_switches_ > 0) {
            --false_negative_mode_switches_;
            return std::nullopt;
        }
        return it->second;
    }

    void motor_reset(int motor_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        ++reset_count_;
        auto it = states_.find(motor_id);
        if (it != states_.end() && reset_clears_mit_) {
            it->second.mode = MODE_STOP;
        }
    }

    void set_watchdog(int motor_id, int timeout_ms) override {
        std::lock_guard<std::mutex> lock(mutex_);
        watchdogs_[motor_id] = timeout_ms;
    }

    void stop(int motor_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.at(motor_id).mode = MODE_STOP;
    }

    void brake(int motor_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.at(motor_id).mode = MODE_BRAKE;
    }

    void send_position(
            const std::vector<int>& motor_ids,
            const std::vector<double>& positions_turns,
            const std::vector<double>& velocities_turns_s,
            int, int) override {
        record_frame(
            false, motor_ids, positions_turns, velocities_turns_s, {});
    }

    void send_mit(
            const std::vector<int>& motor_ids,
            const std::vector<double>& positions_turns,
            const std::vector<double>& velocities_turns_s,
            const std::vector<int>& torques_raw,
            const std::vector<int>&,
            const std::vector<int>&,
            int) override {
        record_frame(
            true, motor_ids, positions_turns, velocities_turns_s,
            torques_raw);
    }

    bool is_async_rx() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return async_rx_;
    }

    bool is_polling() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return polling_;
    }

    void enable_async_rx() override {
        std::lock_guard<std::mutex> lock(mutex_);
        async_rx_ = true;
    }

    void disable_async_rx() override {
        std::lock_guard<std::mutex> lock(mutex_);
        async_rx_ = false;
    }

    void start_polling(const std::vector<int>&, double) override {
        std::lock_guard<std::mutex> lock(mutex_);
        polling_ = true;
    }

    void stop_polling() override {
        std::lock_guard<std::mutex> lock(mutex_);
        polling_ = false;
    }

    void set_position(int motor_id, double turns) {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.at(motor_id).position = turns;
    }

    void set_position_limit(int motor_id, double lo, double hi) {
        std::lock_guard<std::mutex> lock(mutex_);
        limits_[motor_id] = {lo, hi};
    }

    void set_position_limit_observer(std::function<void()> observer) {
        std::lock_guard<std::mutex> lock(mutex_);
        position_limit_observer_ = std::move(observer);
    }

    void fail_next_read(int motor_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_next_reads_.insert(motor_id);
    }

    void fail_next_send() {
        std::lock_guard<std::mutex> lock(mutex_);
        fail_next_send_ = true;
    }

    void set_open(bool open) {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = open;
    }

    void set_throw_on_is_open(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        throw_on_is_open_ = enabled;
    }

    int read_count(int motor_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = read_counts_.find(motor_id);
        return it == read_counts_.end() ? 0 : it->second;
    }

    void clear_read_counts() {
        std::lock_guard<std::mutex> lock(mutex_);
        read_counts_.clear();
    }

    int watchdog(int motor_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = watchdogs_.find(motor_id);
        return it == watchdogs_.end() ? 0 : it->second;
    }

    void set_mode_direct(int motor_id, int mode) {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.at(motor_id).mode = mode;
    }

    void set_age_ms(int motor_id, double age_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        ages_ms_[motor_id] = age_ms;
    }

    void set_missing(int motor_id, bool missing) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (missing) {
            missing_.insert(motor_id);
        } else {
            missing_.erase(motor_id);
        }
    }

    void set_false_negative_mode_switches(int count) {
        std::lock_guard<std::mutex> lock(mutex_);
        false_negative_mode_switches_ = count;
    }

    void set_follow_position_commands(bool follow) {
        std::lock_guard<std::mutex> lock(mutex_);
        follow_position_commands_ = follow;
    }

    int reset_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return reset_count_;
    }

    std::vector<SentFrame> frames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return frames_;
    }

private:
    void record_frame(bool mit,
                      const std::vector<int>& motor_ids,
                      const std::vector<double>& positions,
                      const std::vector<double>& velocities,
                      const std::vector<int>& torques) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_) {
            throw std::runtime_error("fake transport is closed");
        }
        if (fail_next_send_) {
            fail_next_send_ = false;
            throw std::runtime_error("injected send failure");
        }
        frames_.push_back(
            SentFrame{mit, motor_ids, positions, velocities, torques});
        for (std::size_t i = 0; i < motor_ids.size(); ++i) {
            if (mit || follow_position_commands_) {
                states_.at(motor_ids[i]).position = positions[i];
            }
            states_.at(motor_ids[i]).velocity = velocities[i];
            states_.at(motor_ids[i]).mode =
                mit ? MODE_MIT : MODE_ACTIVE;
            ages_ms_[motor_ids[i]] = 0.0;
        }
    }

    mutable std::mutex mutex_;
    bool open_ = true;
    bool throw_on_is_open_ = false;
    bool async_rx_ = false;
    bool polling_ = false;
    bool reset_clears_mit_ = true;
    bool follow_position_commands_ = true;
    bool fail_next_send_ = false;
    int false_negative_mode_switches_ = 0;
    int reset_count_ = 0;
    std::map<int, hightorque::MotorState> states_;
    std::map<int, double> ages_ms_;
    std::map<int, int> watchdogs_;
    std::map<int, int> read_counts_;
    std::map<int, std::pair<double, double>> limits_;
    std::function<void()> position_limit_observer_;
    std::set<int> missing_;
    std::set<int> fail_next_reads_;
    std::vector<SentFrame> frames_;
};

}  // namespace fafu::core::test
