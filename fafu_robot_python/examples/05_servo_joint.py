"""用 servo_j 连续发送目标，让 J1 做一次小幅正弦运动。"""

from pathlib import Path
import math
import sys
import time


SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

from fafu_robot_controller import FafuRobotController, ServoOpts  # noqa: E402


CONFIG_PATH = SDK_DIR / "robot.cfg"
GRIPPER_ID = 7
RATE_HZ = 100.0
DURATION_SECONDS = 2.0
AMPLITUDE_DEGREES = 3.0


def main() -> None:
    with FafuRobotController(
        cfg_path=str(CONFIG_PATH),
        has_gripper=True,
        gripper_motor_id=GRIPPER_ID,
    ) as arm:
        start = arm.get_joint_values()
        input("确认机械臂周围没有障碍物后，按 Enter 开始 Servo 运动...")

        options = ServoOpts(
            rate_hz=RATE_HZ,
            max_vel=0.5,
            max_step_rad=math.radians(1.0),
            watchdog_ms=100,
        )
        arm.servo_start(options)

        tick_count = int(RATE_HZ * DURATION_SECONDS)
        period = 1.0 / RATE_HZ
        amplitude = math.radians(AMPLITUDE_DEGREES)

        try:
            for tick in range(tick_count):
                phase = 2.0 * math.pi * tick / max(1, tick_count - 1)
                target = start.copy()
                target[0] = start[0] + amplitude * math.sin(phase)

                if not arm.servo_j(target):
                    print("Servo 因安全检查停止。")
                    break
                time.sleep(period)
        finally:
            arm.servo_end("hold")

        arm.move_j(start, speed=10, block=True)
        print("Servo 示例完成，已回到起点。")


if __name__ == "__main__":
    main()
