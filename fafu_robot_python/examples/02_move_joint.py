"""让 J1 在软限位内缓慢移动 5 度，然后回到起点。"""

from pathlib import Path
import math
import sys


SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

from fafu_robot_controller import FafuRobotController  # noqa: E402


CONFIG_PATH = SDK_DIR / "robot.cfg"
GRIPPER_ID = 7
MOVE_RAD = math.radians(5.0)
LIMIT_MARGIN_RAD = math.radians(2.0)
SPEED_PERCENT = 10


def _safe_offset(start_rad: float, limit) -> float:
    if limit is None:
        raise RuntimeError("move_j 示例要求 robot.cfg 为 J1 配置软限位")

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
        raise RuntimeError("J1 距离软限位太近，拒绝运行 move_j 示例")

    direction = 1.0 if room_up >= room_down else -1.0
    return direction * min(MOVE_RAD, 0.8 * room)


def main() -> None:
    with FafuRobotController(
        cfg_path=str(CONFIG_PATH),
        has_gripper=True,
        gripper_motor_id=GRIPPER_ID,
    ) as arm:
        start = arm.get_joint_values()
        j1_motor_id = arm.joint_motor_ids[0]
        offset_rad = _safe_offset(
            float(start[0]), arm.get_limit(j1_motor_id)
        )
        target = start.copy()
        target[0] += offset_rad

        print("start (deg):", [round(math.degrees(q), 2) for q in start])
        print("target J1 offset (deg):", round(math.degrees(offset_rad), 2))
        input("确认机械臂周围没有障碍物后，按 Enter 开始...")

        # 高层运动 API 只接收 radians；轨迹、单位转换和到位确认都在 C++。
        arm.move_j(target, speed=SPEED_PERCENT, block=True)
        arm.move_j(start, speed=SPEED_PERCENT, block=True)
        print("运动完成，已回到起点。")


if __name__ == "__main__":
    main()
