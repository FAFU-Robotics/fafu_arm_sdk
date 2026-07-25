#pragma once

#include "fafu/core/core_types.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace fafu::core {

class StateError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class BusyError : public StateError {
public:
    using StateError::StateError;
};

class ControllerState {
public:
    explicit ControllerState(RobotState initial = RobotState::Disabled);

    RobotState state() const;
    OperationKind active_operation() const;
    bool operation_owned_by_current_thread() const;
    bool closing() const;
    bool cancel_requested() const noexcept;
    std::string dead_reason() const;

    std::uint64_t begin(OperationKind kind);
    void end(std::uint64_t token);

    void transition(RobotState next);
    void latch_dead(std::string reason);
    void latch_estop();
    void clear_latched(RobotState next);

    void begin_closing();
    bool wait_until_idle(double timeout_s);
    void mark_disconnected();

private:
    static RobotState busy_state_for(OperationKind kind);
    static bool is_busy_state(RobotState state);
    void ensure_operation_allowed_locked(OperationKind kind) const;

    mutable std::mutex mutex_;
    std::condition_variable idle_cv_;
    RobotState state_;
    OperationKind active_operation_ = OperationKind::None;
    std::thread::id owner_thread_;
    std::uint64_t generation_ = 0;
    unsigned depth_ = 0;
    bool closing_ = false;
    std::string dead_reason_;
    std::atomic<bool> cancel_requested_{false};
};

class OperationLease {
public:
    OperationLease() = default;
    OperationLease(ControllerState& state, OperationKind kind);
    ~OperationLease();

    OperationLease(const OperationLease&) = delete;
    OperationLease& operator=(const OperationLease&) = delete;
    OperationLease(OperationLease&& other) noexcept;
    OperationLease& operator=(OperationLease&& other) noexcept;

    std::uint64_t token() const noexcept { return token_; }
    void release();

private:
    ControllerState* state_ = nullptr;
    std::uint64_t token_ = 0;
};

}  // namespace fafu::core
