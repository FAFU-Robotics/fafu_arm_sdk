#include "fafu/core/bindings.hpp"
#include "fafu/core/controller_state.hpp"
#include "fafu/core/core_types.hpp"
#include "fafu/core/motor_calibration.hpp"
#include "fafu/core/robot_core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace fafu::core;

namespace fafu::core {

void bind_core(py::module_& module) {
    module.attr("CORE_ABI_VERSION") = CORE_ABI_VERSION;

    const auto state_error =
        py::register_exception<StateError>(module, "StateError");
    py::register_exception<BusyError>(
        module, "BusyError", state_error.ptr());

    py::enum_<RobotState>(module, "RobotState")
        .value("DISCONNECTED", RobotState::Disconnected)
        .value("DISABLED", RobotState::Disabled)
        .value("BRAKED", RobotState::Braked)
        .value("IDLE", RobotState::Idle)
        .value("MOVING", RobotState::Moving)
        .value("SERVOING", RobotState::Servoing)
        .value("GRASPING", RobotState::Grasping)
        .value("GRAVITY_COMP", RobotState::GravityComp)
        .value("ESTOP", RobotState::Estop)
        .value("DEAD", RobotState::Dead);

    py::enum_<OperationKind>(module, "OperationKind")
        .value("NONE", OperationKind::None)
        .value("LIFECYCLE", OperationKind::Lifecycle)
        .value("JOINT_MOTION", OperationKind::JointMotion)
        .value("SERVO", OperationKind::Servo)
        .value("GRIPPER_MOTION", OperationKind::GripperMotion)
        .value("GRASP", OperationKind::Grasp)
        .value("GRAVITY_COMP", OperationKind::GravityComp)
        .value("RAW_STREAM", OperationKind::RawStream);

    py::enum_<ServoChannel>(module, "ServoChannel")
        .value("POSITION", ServoChannel::Position)
        .value("MIT", ServoChannel::Mit);

    py::enum_<FinishMode>(module, "FinishMode")
        .value("STOP", FinishMode::Stop)
        .value("BRAKE", FinishMode::Brake)
        .value("HOLD", FinishMode::Hold);

    py::enum_<ModeStage>(module, "ModeStage")
        .value("ALREADY_ACTIVE", ModeStage::AlreadyActive)
        .value("NORMAL_SWITCH", ModeStage::NormalSwitch)
        .value("MIT_RESET", ModeStage::MitReset)
        .value("SOFT_RESET", ModeStage::SoftReset)
        .value("AGGRESSIVE_RESET", ModeStage::AggressiveReset)
        .value("FAILED", ModeStage::Failed);

    py::class_<MotorDiagnostic>(module, "MotorDiagnostic")
        .def_readonly("motor_id", &MotorDiagnostic::motor_id)
        .def_readonly("responded", &MotorDiagnostic::responded)
        .def_readonly("mode", &MotorDiagnostic::mode)
        .def_readonly("fault", &MotorDiagnostic::fault)
        .def_readonly("position_turns",
                      &MotorDiagnostic::position_turns)
        .def_readonly("detail", &MotorDiagnostic::detail);

    py::class_<EnableOptions>(module, "EnableOptions")
        .def(py::init<>())
        .def_readwrite("allow_motor_reset",
                       &EnableOptions::allow_motor_reset)
        .def_readwrite("normal_retries",
                       &EnableOptions::normal_retries)
        .def_readwrite("aggressive_reset_rounds",
                       &EnableOptions::aggressive_reset_rounds)
        .def_readwrite("resets_per_aggressive_round",
                       &EnableOptions::resets_per_aggressive_round)
        .def_readwrite("verify_delay_s",
                       &EnableOptions::verify_delay_s)
        .def_readwrite("retry_delay_s",
                       &EnableOptions::retry_delay_s)
        .def_readwrite("reset_spacing_s",
                       &EnableOptions::reset_spacing_s)
        .def_readwrite("reset_wait_s",
                       &EnableOptions::reset_wait_s)
        .def_readwrite("aggressive_reset_wait_s",
                       &EnableOptions::aggressive_reset_wait_s);

    py::class_<EnableResult>(module, "EnableResult")
        .def_readonly("success", &EnableResult::success)
        .def_readonly("stage", &EnableResult::stage)
        .def_readonly("failed_motor_ids",
                      &EnableResult::failed_motor_ids)
        .def_readonly("diagnostics", &EnableResult::diagnostics)
        .def_readonly("message", &EnableResult::message);

    py::class_<ServoOptions>(module, "ServoOptions")
        .def(py::init<>())
        .def_readwrite("watchdog_ms", &ServoOptions::watchdog_ms)
        .def_readwrite("max_velocity_rad_s",
                       &ServoOptions::max_velocity_rad_s)
        .def_readwrite("max_step_rad", &ServoOptions::max_step_rad)
        .def_readwrite("max_lag_rad", &ServoOptions::max_lag_rad)
        .def_readwrite("nominal_rate_hz",
                       &ServoOptions::nominal_rate_hz)
        .def_readwrite("input_is_radians",
                       &ServoOptions::input_is_radians)
        .def_readwrite("feedforward_velocity",
                       &ServoOptions::feedforward_velocity)
        .def_readwrite("position_error_deadband_rad",
                       &ServoOptions::position_error_deadband_rad)
        .def_readwrite("lookahead_time_s",
                       &ServoOptions::lookahead_time_s)
        .def_readwrite("lag_abort_consecutive",
                       &ServoOptions::lag_abort_consecutive)
        .def_readwrite("channel", &ServoOptions::channel)
        .def_readwrite("mit_kp", &ServoOptions::mit_kp)
        .def_readwrite("mit_kd", &ServoOptions::mit_kd);

    py::class_<ServoTickResult>(module, "ServoTickResult")
        .def_readonly("sent", &ServoTickResult::sent)
        .def_readonly("clamped", &ServoTickResult::clamped)
        .def_readonly("lag_tripped",
                      &ServoTickResult::lag_tripped)
        .def_readonly("aborted", &ServoTickResult::aborted)
        .def_readonly("message", &ServoTickResult::message);

    py::class_<ServoSummary>(module, "ServoSummary")
        .def_readonly("tick_count", &ServoSummary::tick_count)
        .def_readonly("clamp_count", &ServoSummary::clamp_count)
        .def_readonly("lag_count", &ServoSummary::lag_count)
        .def_readonly("elapsed_s", &ServoSummary::elapsed_s)
        .def_readonly("average_rate_hz",
                      &ServoSummary::average_rate_hz)
        .def_readonly("aborted_reason",
                      &ServoSummary::aborted_reason);

    py::class_<HealthSnapshot>(module, "HealthSnapshot")
        .def_readonly("state", &HealthSnapshot::state)
        .def_readonly("active_operation",
                      &HealthSnapshot::active_operation)
        .def_readonly("closing", &HealthSnapshot::closing)
        .def_readonly("cancel_requested",
                      &HealthSnapshot::cancel_requested)
        .def_readonly("link_ok", &HealthSnapshot::link_ok)
        .def_readonly("dead_reason", &HealthSnapshot::dead_reason)
        .def_readonly("stale_motor_ids",
                      &HealthSnapshot::stale_motor_ids);

    py::class_<CoreConfig>(module, "CoreConfig")
        .def(py::init<>())
        .def_readwrite("all_motor_ids", &CoreConfig::all_motor_ids)
        .def_readwrite("joint_motor_ids",
                       &CoreConfig::joint_motor_ids)
        .def_readwrite("joint_motor_models",
                       &CoreConfig::joint_motor_models)
        .def_readwrite("max_torque_raw",
                       &CoreConfig::max_torque_raw)
        .def_readwrite("stale_feedback_timeout_ms",
                       &CoreConfig::stale_feedback_timeout_ms)
        .def_readwrite("polling_rate_hz",
                       &CoreConfig::polling_rate_hz);

    py::class_<CommandLease>(module, "CommandLease")
        .def("__enter__",
             [](CommandLease& lease) -> CommandLease& { return lease; },
             py::return_value_policy::reference_internal)
        .def("__exit__",
             [](CommandLease& lease, py::object, py::object, py::object) {
                 lease.release();
                 return false;
             })
        .def("release", &CommandLease::release);

    py::class_<RobotCore>(module, "RobotCore")
        .def(py::init<hightorque::HightorqueSerial&, CoreConfig>(),
             py::arg("driver"), py::arg("config"),
             py::keep_alive<1, 2>())
        .def_property_readonly("state", &RobotCore::state)
        .def_property_readonly("active_operation",
                               &RobotCore::active_operation)
        .def_property_readonly("cancel_requested",
                               &RobotCore::cancel_requested)
        .def_property_readonly("dead_reason",
                               &RobotCore::dead_reason)
        .def_property_readonly("is_servoing",
                               &RobotCore::is_servoing)
        .def("health", &RobotCore::health)
        .def("operation_owned_by_current_thread",
             &RobotCore::operation_owned_by_current_thread)
        .def("begin_operation", &RobotCore::begin_operation,
             py::arg("kind"))
        .def("end_operation", &RobotCore::end_operation,
             py::arg("token"))
        .def("command_guard", &RobotCore::command_guard,
             py::call_guard<py::gil_scoped_release>(),
             py::keep_alive<0, 1>())
        .def("transition", &RobotCore::transition, py::arg("state"))
        .def("start_transport", &RobotCore::start_transport,
             py::arg("async_rx"), py::arg("polling"),
             py::arg("polling_rate_hz") = 0.0,
             py::call_guard<py::gil_scoped_release>())
        .def("stop_transport", &RobotCore::stop_transport,
             py::call_guard<py::gil_scoped_release>())
        .def("enable",
             py::overload_cast<const EnableOptions&>(&RobotCore::enable),
             py::arg("options") = EnableOptions{},
             py::call_guard<py::gil_scoped_release>())
        .def("disable", &RobotCore::disable,
             py::call_guard<py::gil_scoped_release>())
        .def("brake", &RobotCore::brake,
             py::call_guard<py::gil_scoped_release>())
        .def("brake_active_operation", &RobotCore::brake_active_operation,
             py::call_guard<py::gil_scoped_release>())
        .def("emergency_stop", &RobotCore::emergency_stop,
             py::call_guard<py::gil_scoped_release>())
        .def("resume",
             py::overload_cast<const EnableOptions&>(&RobotCore::resume),
             py::arg("options") = EnableOptions{},
             py::call_guard<py::gil_scoped_release>())
        .def("check_alive", &RobotCore::check_alive,
             py::arg("fresh") = true, py::arg("timeout_s") = 0.1,
             py::call_guard<py::gil_scoped_release>())
        .def("recover", &RobotCore::recover,
             py::arg("confirm"), py::arg("timeout_s") = 0.2,
             py::call_guard<py::gil_scoped_release>())
        .def("set_joint_motor_models",
             &RobotCore::set_joint_motor_models,
             py::arg("motor_models"))
        .def("set_stale_feedback_timeout_ms",
             &RobotCore::set_stale_feedback_timeout_ms,
             py::arg("timeout_ms"))
        .def("stream_link_ok", &RobotCore::stream_link_ok,
             py::call_guard<py::gil_scoped_release>())
        .def("servo_start",
             py::overload_cast<const ServoOptions&>(&RobotCore::servo_start),
             py::arg("options") = ServoOptions{},
             py::call_guard<py::gil_scoped_release>())
        .def("servo_tick",
             py::overload_cast<const std::vector<double>&,
                               const std::vector<double>&>(&RobotCore::servo_tick),
             py::arg("target_angles"),
             py::arg("torque_ff_nm") = std::vector<double>{},
             py::call_guard<py::gil_scoped_release>())
        .def("servo_end", &RobotCore::servo_end,
             py::arg("finish_mode") = FinishMode::Hold,
             py::call_guard<py::gil_scoped_release>())
        .def("servo_summary", &RobotCore::servo_summary)
        .def("shutdown", &RobotCore::shutdown,
             py::arg("joint_release") = FinishMode::Brake,
             py::arg("auxiliary_release") = FinishMode::Brake,
             py::arg("wait_timeout_s") = 5.0,
             py::call_guard<py::gil_scoped_release>());

    module.def("torque_coefficient", &torque_coefficient,
               py::arg("motor_model"));
    module.def("torque_to_raw", &torque_to_raw,
               py::arg("torque_nm"), py::arg("motor_model"),
               py::arg("torque_scale") = 1.0);
    module.def("gain_to_raw", &gain_to_raw,
               py::arg("gain"), py::arg("motor_model"));
    module.def("torques_to_raw", &torques_to_raw,
               py::arg("torques_nm"), py::arg("motor_models"),
               py::arg("torque_scale") = 1.0);
}

}  // namespace fafu::core
