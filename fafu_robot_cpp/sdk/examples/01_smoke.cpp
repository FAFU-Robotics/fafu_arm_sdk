// =============================================================================
//  01_smoke.cpp
//
//  最小连通性测试: 连上 Fafu 机械臂 → 列电机 → 读一次状态 → 干净断开.
//  不发任何位置 / 力矩命令, 安全运行 (不需要给电机上电).
//
//  用法 (Windows):
//      cd fafu_robot_sdk\fafu_robot_cpp\build\bin\Release
//      01_smoke.exe                          # 用同目录的 robot.cfg
//      01_smoke.exe path\to\robot.cfg
// =============================================================================
#include "fafu/fafu_robot_controller.hpp"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    using namespace fafu_robot;

    std::string cfg = (argc >= 2) ? argv[1] : "robot.cfg";

    std::cout << "============================================================\n";
    std::cout << " Fafu Robot SDK — 01_smoke\n";
    std::cout << " cfg = " << cfg << "\n";
    std::cout << "============================================================\n";

    try {
        // 不上电只读状态: auto_enable=false, auto_polling=true (拿 cache 更省事)
        FafuRobotController::Options opts;
        opts.auto_enable  = false;
        opts.auto_polling = true;

        FafuRobotController arm(cfg, opts);

        std::cout << "\n[CFG] " << arm.cfg().to_string() << "\n";
        std::cout << "\n[INFO] joint motor ids =";
        for (int m : arm.joint_motor_ids()) std::cout << " " << m;
        std::cout << "\n[INFO] num_joints       = " << arm.num_joints() << "\n";
        std::cout << "[INFO] has_gripper      = " << (arm.has_gripper() ? "true" : "false") << "\n";
        std::cout << "[INFO] gripper_motor_id = " << arm.gripper_motor_id() << "\n";
        std::cout << "[INFO] port             = " << arm.port() << "\n";
        std::cout << "[INFO] baudrate         = " << arm.baudrate() << "\n";

        std::cout << "\n[STATE] reading once...\n";
        auto states = arm.get_motor_states(/*prefer_cache=*/false);
        for (const auto& [mid, st] : states) {
            std::cout << "  motor " << mid << ": " << st.to_string() << "\n";
        }

        std::cout << "\n[CAN] " << arm.get_can_status().to_string() << "\n";

        // 析构会自动 close_connection(joints=stop, gripper=brake)
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nOK.\n";
    return 0;
}
