// =============================================================================
//  03_gripper.cpp
//
//  夹爪 demo: open → close → grasp(force_threshold=500) → release.
//  最后会打印 GraspResult, 方便人工核对 reason / peak_torque_raw.
//
//  使用前提:
//    - robot.cfg 里的 motor_ids 包含 7
//    - 给 has_gripper=true, gripper_motor_id=7
//    - 夹爪机械上能动 (单独闭合也不会卡死)
//
//  ★ 会让夹爪闭合, 请把手指拿开 ★
//
//  用法:
//      03_gripper.exe                          # 用同目录 robot.cfg, gripper_id=7
//      03_gripper.exe path\to\robot.cfg 7
// =============================================================================
#include "fafu/fafu_robot_controller.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main(int argc, char* argv[]) {
    using namespace fafu_robot;

    std::string cfg = (argc >= 2) ? argv[1] : "robot.cfg";
    int gripper_id  = (argc >= 3) ? std::stoi(argv[2]) : 7;

    std::cout << "============================================================\n";
    std::cout << " Fafu Robot SDK — 03_gripper\n";
    std::cout << " cfg = " << cfg << "    gripper_motor_id = " << gripper_id << "\n";
    std::cout << " ★ 夹爪会动, 注意手指 ★\n";
    std::cout << "============================================================\n";
    std::cout << "Press ENTER to continue, Ctrl+C to abort..." << std::flush;
    std::cin.get();

    try {
        FafuRobotController::Options opts;
        opts.has_gripper      = true;
        opts.gripper_motor_id = gripper_id;
        FafuRobotController arm(cfg, opts);

        // ---------- 1) open ----------
        std::cout << "\n[1/4] open_gripper() ...\n";
        arm.open_gripper();
        auto g = arm.get_gripper_state();
        std::cout << "       after open: pos=" << g.position
                  << " turns, torque(raw)=" << static_cast<int>(g.torque) << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // ---------- 2) close to soft-limit ----------
        std::cout << "\n[2/4] close_gripper() (to lower soft limit) ...\n";
        arm.close_gripper();
        g = arm.get_gripper_state();
        std::cout << "       after close: pos=" << g.position
                  << " turns, torque(raw)=" << static_cast<int>(g.torque) << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // ---------- 3) open again, then grasp ----------
        std::cout << "\n[3/4] open then grasp(force_threshold=500) ...\n";
        arm.open_gripper();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        FafuRobotController::GraspOpts go;
        go.force_threshold = 500;       // 调成你硬件标定后的值
        go.vel             = 0.15;
        go.timeout_s       = 5.0;
        auto r = arm.grasp(go);
        std::cout << "       " << r.to_string() << "\n";
        std::cout << "       grasped?      " << (r.grasped ? "YES" : "NO") << "\n";
        std::cout << "       reason         " << r.reason << "\n";
        std::cout << "       closed_deg     " << r.closed_deg << "\n";
        std::cout << "       peak_torque    " << r.peak_torque_raw << "\n";
        std::cout << "       duration_s     " << r.duration_s << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // ---------- 4) release ----------
        std::cout << "\n[4/4] release() ...\n";
        arm.release();

        // 析构会自动 close (gripper=brake, 不会突然弹开)
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nOK.\n";
    return 0;
}
