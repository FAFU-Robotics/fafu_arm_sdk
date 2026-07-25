#include "fafu/core/controller_state.hpp"

#include <chrono>
#include <sstream>
#include <utility>

namespace fafu::core {

namespace {

const char* state_name(RobotState state) {
    switch (state) {
        case RobotState::Disconnected: return "DISCONNECTED";
        case RobotState::Disabled: return "DISABLED";
        case RobotState::Braked: return "BRAKED";
        case RobotState::Idle: return "IDLE";
        case RobotState::Moving: return "MOVING";
        case RobotState::Servoing: return "SERVOING";
        case RobotState::Grasping: return "GRASPING";
        case RobotState::GravityComp: return "GRAVITY_COMP";
        case RobotState::Estop: return "ESTOP";
        case RobotState::Dead: return "DEAD";
    }
    return "UNKNOWN";
}

}  // namespace

ControllerState::ControllerState(RobotState initial) : state_(initial) {}

RobotState ControllerState::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

OperationKind ControllerState::active_operation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_operation_;
}

bool ControllerState::operation_owned_by_current_thread() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return depth_ > 0 && owner_thread_ == std::this_thread::get_id();
}

bool ControllerState::closing() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closing_;
}

bool ControllerState::cancel_requested() const noexcept {
    return cancel_requested_.load(std::memory_order_acquire);
}

std::string ControllerState::dead_reason() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dead_reason_;
}

RobotState ControllerState::busy_state_for(OperationKind kind) {
    switch (kind) {
        case OperationKind::JointMotion:
        case OperationKind::RawStream:
            return RobotState::Moving;
        case OperationKind::Servo:
            return RobotState::Servoing;
        case OperationKind::GripperMotion:
        case OperationKind::Grasp:
            return RobotState::Grasping;
        case OperationKind::GravityComp:
            return RobotState::GravityComp;
        case OperationKind::None:
        case OperationKind::Lifecycle:
            return RobotState::Idle;
    }
    return RobotState::Idle;
}

bool ControllerState::is_busy_state(RobotState state) {
    return state == RobotState::Moving ||
           state == RobotState::Servoing ||
           state == RobotState::Grasping ||
           state == RobotState::GravityComp;
}

void ControllerState::ensure_operation_allowed_locked(OperationKind kind) const {
    if (kind == OperationKind::None) {
        throw StateError("cannot acquire OperationKind::None");
    }
    if (closing_) {
        throw StateError("controller is closing");
    }
    if (state_ == RobotState::Disconnected) {
        throw StateError("controller is disconnected");
    }
    if (state_ == RobotState::Dead) {
        throw StateError("controller is DEAD: " + dead_reason_);
    }
    if (state_ == RobotState::Estop) {
        throw StateError("controller is in ESTOP");
    }
    if (kind != OperationKind::Lifecycle && state_ != RobotState::Idle) {
        std::ostringstream oss;
        oss << "control operation requires IDLE; current state="
            << state_name(state_);
        throw StateError(oss.str());
    }
}

std::uint64_t ControllerState::begin(OperationKind kind) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto caller = std::this_thread::get_id();

    if (depth_ > 0) {
        if (owner_thread_ != caller) {
            throw BusyError("another thread owns the active control operation");
        }
        if (active_operation_ != kind) {
            throw BusyError("nested control operations must have the same kind");
        }
        ++depth_;
        return generation_;
    }

    ensure_operation_allowed_locked(kind);
    active_operation_ = kind;
    owner_thread_ = caller;
    depth_ = 1;
    ++generation_;
    cancel_requested_.store(false, std::memory_order_release);
    if (kind != OperationKind::Lifecycle) {
        state_ = busy_state_for(kind);
    }
    return generation_;
}

void ControllerState::end(std::uint64_t token) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (depth_ == 0) {
        return;  // A latched safety transition may already have cancelled it.
    }
    if (token != generation_) {
        throw StateError("operation token is stale");
    }
    if (owner_thread_ != std::this_thread::get_id()) {
        throw StateError("operation must be ended by its owner thread");
    }

    --depth_;
    if (depth_ != 0) {
        return;
    }

    const RobotState operation_state = busy_state_for(active_operation_);
    active_operation_ = OperationKind::None;
    owner_thread_ = {};
    if (!closing_ && state_ == operation_state && is_busy_state(state_)) {
        state_ = RobotState::Idle;
    }
    idle_cv_.notify_all();
}

void ControllerState::transition(RobotState next) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == RobotState::Disconnected && next != RobotState::Disconnected) {
        throw StateError("cannot leave DISCONNECTED");
    }
    if ((state_ == RobotState::Dead || state_ == RobotState::Estop) &&
        next != state_) {
        throw StateError("latched safety state requires explicit recovery");
    }
    state_ = next;
}

void ControllerState::latch_dead(std::string reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == RobotState::Disconnected) {
        return;
    }
    dead_reason_ = std::move(reason);
    state_ = RobotState::Dead;
    cancel_requested_.store(true, std::memory_order_release);
}

void ControllerState::latch_estop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == RobotState::Disconnected || state_ == RobotState::Dead) {
        return;
    }
    state_ = RobotState::Estop;
    cancel_requested_.store(true, std::memory_order_release);
}

void ControllerState::clear_latched(RobotState next) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != RobotState::Dead && state_ != RobotState::Estop) {
        throw StateError("controller is not in a recoverable latched state");
    }
    if (depth_ != 0) {
        throw BusyError("active operation has not stopped yet");
    }
    dead_reason_.clear();
    state_ = next;
    cancel_requested_.store(false, std::memory_order_release);
}

void ControllerState::begin_closing() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == RobotState::Disconnected) {
        return;
    }
    closing_ = true;
    cancel_requested_.store(true, std::memory_order_release);
    // RobotCore serializes servo ticks with its servo mutex. A dormant
    // cross-call servo session can therefore be completed by shutdown.
    if (active_operation_ == OperationKind::Servo) {
        active_operation_ = OperationKind::None;
        owner_thread_ = {};
        depth_ = 0;
        idle_cv_.notify_all();
    }
}

bool ControllerState::wait_until_idle(double timeout_s) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (depth_ == 0) {
        return true;
    }
    if (owner_thread_ == std::this_thread::get_id()) {
        return false;
    }
    const auto timeout = std::chrono::duration<double>(
        timeout_s > 0.0 ? timeout_s : 0.0);
    return idle_cv_.wait_for(lock, timeout, [&] { return depth_ == 0; });
}

void ControllerState::mark_disconnected() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = RobotState::Disconnected;
    active_operation_ = OperationKind::None;
    owner_thread_ = {};
    depth_ = 0;
    closing_ = true;
    cancel_requested_.store(true, std::memory_order_release);
    idle_cv_.notify_all();
}

OperationLease::OperationLease(ControllerState& state, OperationKind kind)
    : state_(&state), token_(state.begin(kind)) {}

OperationLease::~OperationLease() {
    try {
        release();
    } catch (...) {
        // Destructors must not throw. Explicit release() still reports misuse.
    }
}

OperationLease::OperationLease(OperationLease&& other) noexcept
    : state_(other.state_), token_(other.token_) {
    other.state_ = nullptr;
    other.token_ = 0;
}

OperationLease& OperationLease::operator=(OperationLease&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    try {
        release();
    } catch (...) {
    }
    state_ = other.state_;
    token_ = other.token_;
    other.state_ = nullptr;
    other.token_ = 0;
    return *this;
}

void OperationLease::release() {
    if (state_ == nullptr) {
        return;
    }
    ControllerState* state = state_;
    const std::uint64_t token = token_;
    state_ = nullptr;
    token_ = 0;
    state->end(token);
}

}  // namespace fafu::core
