// =============================================================================
//  04_servo_j.cpp
//
//  servoJ (online streaming) 演示: 100 Hz 给每个关节流一段 sin 轨迹, 持续若干秒,
//  最后 servo_end(Brake) 干净退出.
//
//  ★ 这是 *会让所有关节实际运动* 的 demo. ★
//  运行前确认:
//    - 手已经离开机械臂工作空间
//    - 关节软限位已经在 robot.cfg 里配好 (限位会被 servoJ 自动应用)
//    - 急停按钮在手边
//
//  安全特性演示:
//    - 固件 watchdog (默认 100ms): 如果你 Ctrl-C 这个 exe, 电机会在 100ms 内
//      自动 brake, 而不是保持最后一帧目标继续冲. 自己拔 USB 也是.
//    - 单步限幅: 这里把 max_step_rad 设到 0.02 rad (~1.15deg), 100Hz 周期内
//      最大允许 2 rad/s 的目标速度, 比下面的 sin 轨迹 (峰值 ~0.6 rad/s) 留有
//      3x 余量, 防御上层规划 bug.
//    - 跟踪误差: max_lag_rad = 0.15 rad (~8.6deg), 检测机械负载过大 / 速度配错.
//
//  用法:
//      04_servo_j.exe                          # robot.cfg, 默认参数
//      04_servo_j.exe path\to\robot.cfg
//      04_servo_j.exe robot.cfg 10             # 跑 10 秒 (默认 6 秒)
// =============================================================================
#include "fafu/fafu_robot_controller.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// Ctrl-C handler: 设个 flag, 主循环看到就退出.
std::atomic<bool> g_quit{false};
void on_sigint(int) { g_quit.store(true); }

}  // anonymous

int main(int argc, char* argv[]) {
    using namespace fafu_robot;
    using clk = std::chrono::steady_clock;

    std::string cfg     = (argc >= 2) ? argv[1] : "robot.cfg";
    double      duration = (argc >= 3) ? std::stod(argv[2]) : 6.0;

    std::cout << "============================================================\n";
    std::cout << " Fafu Robot SDK — 04_servo_j (100Hz streaming sin wave)\n";
    std::cout << " cfg = " << cfg << "    duration = " << duration << "s\n";
    std::cout << " ★ 机械臂会运动, 请确认安全 ★\n";
    std::cout << "============================================================\n";
    std::cout << "Press ENTER to continue, Ctrl+C to abort (will brake safely)..."
              << std::flush;
    std::cin.get();

    std::signal(SIGINT, on_sigint);

    try {
        FafuRobotController arm(cfg);
        const int N = arm.num_joints();

        // 1) 先 go_home, 确保起点是 0
        std::cout << "\n[1/3] go_home (offline S-curve, speed=15) ...\n";
        arm.go_home(/*speed=*/15, /*block=*/true);

        auto q0 = arm.get_joint_values(/*prefer_cache=*/false);
        std::cout << "       home reached. start servo session.\n";

        // 2) servo_start: 守护策略
        FafuRobotController::ServoOpts so;
        so.watchdog_ms  = 100;     // 100ms 固件看门狗
        so.max_vel      = 1.5;     // rad/s, 单关节最大速度 (~86 deg/s)
        so.max_step_rad = 0.02;    // 单步最大 1.15deg, 防上层 bug
        so.max_lag_rad  = 0.15;    // 跟踪误差超过 8.6deg 就停
        so.is_radians   = true;
        arm.servo_start(so);

        // 3) 100Hz 跑 sin 轨迹
        std::cout << "\n[2/3] streaming sin wave at 100 Hz for " << duration << "s ...\n";
        std::cout << "      (Ctrl+C 会触发 servo_end + brake; 也可以暴力终止进程, "
                     "固件 100ms 后自动 brake)\n";

        const double freq_hz   = 0.5;                  // 0.5 Hz sin 周期 = 2s
        const double amp_rad   = 10.0 * kPi / 180.0;   // ±10 deg
        const auto   period    = std::chrono::milliseconds(10);   // 100Hz
        const auto   t_start   = clk::now();
        const auto   t_deadline = t_start + std::chrono::duration_cast<clk::duration>(
            std::chrono::duration<double>(duration));

        auto next_tick = t_start;
        int  send_ok = 0, send_fail = 0;

        while (clk::now() < t_deadline && !g_quit.load()) {
            double t = std::chrono::duration<double>(clk::now() - t_start).count();
            double s = std::sin(2.0 * kPi * freq_hz * t);

            // 让相邻关节反相, 视觉上更明显
            std::vector<double> target(N);
            for (int j = 0; j < N; ++j) {
                double sign = (j % 2 == 0) ? +1.0 : -1.0;
                target[j] = q0[j] + sign * amp_rad * s;
            }

            if (arm.servo_j(target)) ++send_ok;
            else                     ++send_fail;

            next_tick += period;
            std::this_thread::sleep_until(next_tick);
        }

        std::cout << "\n[3/3] servo_end(Brake) ...\n";
        std::cout << "      sent OK=" << send_ok << "  fail=" << send_fail
                  << (g_quit.load() ? "  (Ctrl+C 触发)" : "")
                  << "\n";

        arm.servo_end(ReleaseMode::Brake);

        // 析构会清理 (close_connection 自带 servo_end fallback)
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nOK.\n";
    return 0;
}
