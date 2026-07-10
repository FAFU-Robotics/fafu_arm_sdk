// =============================================================================
//  02_move_j.cpp
//
//  关节空间运动 demo: 上电 → 读起始 → go_home() → 各关节走一个小幅 sin 形状 → 归零.
//
//  ★ 会让机械臂实际运动 ★ 运行前确认:
//    - 手已经离开机械臂工作空间
//    - 电源就位, 急停按钮在手边
//    - 速度参数已经按你硬件适当调小
//
//  用法:
//      02_move_j.exe                          # 用同目录 robot.cfg, 默认 6 关节
//      02_move_j.exe path\to\robot.cfg
// =============================================================================
#include "fafu/fafu_robot_controller.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

void print_joints(const std::vector<double>& q, const char* label) {
    std::cout << label << " [";
    for (size_t i = 0; i < q.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3)
                  << (q[i] * 180.0 / kPi) << "deg";
    }
    std::cout << "]\n";
}

}  // anonymous

int main(int argc, char* argv[]) {
    using namespace fafu_robot;

    std::string cfg = (argc >= 2) ? argv[1] : "robot.cfg";

    std::cout << "============================================================\n";
    std::cout << " Fafu Robot SDK — 02_move_j\n";
    std::cout << " cfg = " << cfg << "\n";
    std::cout << " ★ 机械臂会运动, 请确认安全 ★\n";
    std::cout << "============================================================\n";
    std::cout << "Press ENTER to continue, Ctrl+C to abort..." << std::flush;
    std::cin.get();

    try {
        FafuRobotController arm(cfg);   // 默认: auto_enable + auto_polling
        const int N = arm.num_joints();

        std::cout << "\n[1/4] read start state\n";
        auto q0 = arm.get_joint_values(/*prefer_cache=*/false);
        print_joints(q0, "       start =");

        std::cout << "\n[2/4] go_home (speed=15) ...\n";
        arm.go_home(/*speed=*/15, /*block=*/true);
        print_joints(arm.get_joint_values(false), "       after =");

        std::cout << "\n[3/4] small sinusoidal move (~+/-10 deg, speed=20) ...\n";
        std::vector<double> tgt(N, 0.0);
        for (int joint = 0; joint < N; ++joint) {
            double sign = (joint % 2 == 0) ? +1.0 : -1.0;
            tgt[joint] = sign * (10.0 * kPi / 180.0);   // +/- 10 deg
        }
        FafuRobotController::MoveOpts mo;
        mo.is_radians = true;
        mo.speed      = 20;
        mo.block      = true;
        arm.move_j(tgt, mo);
        print_joints(arm.get_joint_values(false), "       after =");

        std::cout << "\n[4/4] return home (speed=15) ...\n";
        arm.go_home(15, true);
        print_joints(arm.get_joint_values(false), "       after =");

        std::cout << "\nclosing connection (joints will free-spin, gripper braked) ...\n";
        // 显式选保留方式; 也可以直接靠析构默认 = {Stop, Brake}
        arm.close_connection(ReleaseMode::Stop, ReleaseMode::Brake);

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nOK.\n";
    return 0;
}
