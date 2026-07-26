"""用 servo_j 安全地让 J1 做一次小幅、限位内的往返运动。"""

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
DURATION_SECONDS = 6.0
SWING_RAD = math.radians(5.0)
LIMIT_MARGIN_RAD = math.radians(2.0)


def _safe_j1_offset(start_rad: float, limit) -> float:
    if limit is None:
        raise RuntimeError("Servo 示例要求 robot.cfg 为 J1 配置软限位")

    lo_rad, hi_rad = limit
    safe_lo = lo_rad + LIMIT_MARGIN_RAD
    safe_hi = hi_rad - LIMIT_MARGIN_RAD
    if not safe_lo <= start_rad <= safe_hi:
        raise RuntimeError(
            "J1 start position is outside the soft-limit safety margin")

    room_up = hi_rad - LIMIT_MARGIN_RAD - start_rad
    room_down = start_rad - (lo_rad + LIMIT_MARGIN_RAD)
    room = max(room_up, room_down)
    if room < math.radians(0.5):
        raise RuntimeError("J1 距离软限位太近，拒绝运行 Servo 示例")

    direction = 1.0 if room_up >= room_down else -1.0
    return direction * min(SWING_RAD, 0.8 * room)


def main() -> None:
    with FafuRobotController(
        cfg_path=str(CONFIG_PATH),
        has_gripper=True,
        gripper_motor_id=GRIPPER_ID,
    ) as arm:
        start = arm.get_joint_values()
        j1_motor_id = arm.joint_motor_ids[0]
        offset_rad = _safe_j1_offset(
            float(start[0]), arm.get_limit(j1_motor_id)
        )

        print(
            "计划: J1 从当前位置向安全方向移动 "
            f"{math.degrees(offset_rad):+.2f}° 后平滑返回"
        )
        input("确认机械臂周围没有障碍物后，按 Enter 开始 Servo 运动...")

        options = ServoOpts(
            rate_hz=RATE_HZ,
            max_vel=0.25,
            max_step_rad=math.radians(0.6),
            max_lag_rad=math.radians(8.0),
            lag_abort_consecutive=5,
            watchdog_ms=200,
        )
        arm.servo_start(options)

        tick_count = max(2, int(RATE_HZ * DURATION_SECONDS))
        period = 1.0 / RATE_HZ
        next_tick = time.monotonic()
        completed = False

        try:
            for tick in range(tick_count):
                phase = 2.0 * math.pi * tick / (tick_count - 1)
                wave = 0.5 * (1.0 - math.cos(phase))
                target = start.copy()
                target[0] = start[0] + offset_rad * wave

                if not arm.servo_j(target):
                    raise RuntimeError("Servo 因原生安全检查停止")

                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))
            completed = True
        finally:
            arm.servo_end("hold" if completed else "brake")

        arm.move_j(start, speed=10, block=True)
        print("Servo 示例完成，已回到起点。")


if __name__ == "__main__":
    main()
