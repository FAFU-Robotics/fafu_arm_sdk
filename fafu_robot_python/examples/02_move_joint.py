"""让 J1 从当前位置移动 5 度，然后回到起点。"""

from pathlib import Path
import math
import sys


SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

from fafu_robot_controller import FafuRobotController  # noqa: E402


CONFIG_PATH = SDK_DIR / "robot.cfg"
GRIPPER_ID = 7
MOVE_DEGREES = 5.0
SPEED_PERCENT = 10


def main() -> None:
    with FafuRobotController(
        cfg_path=str(CONFIG_PATH),
        has_gripper=True,
        gripper_motor_id=GRIPPER_ID,
    ) as arm:
        start = arm.get_joint_values()
        target = start.copy()
        target[0] += math.radians(MOVE_DEGREES)

        print("start (deg):", [round(math.degrees(q), 2) for q in start])
        print("target J1 offset (deg):", MOVE_DEGREES)
        input("确认机械臂周围没有障碍物后，按 Enter 开始...")

        # move_j 接收全部关节的目标角度，默认单位为弧度。
        arm.move_j(
            target,
            speed=SPEED_PERCENT,
            block=True,
        )

        # 回到运行示例前记录的姿态。
        arm.move_j(
            start,
            speed=SPEED_PERCENT,
            block=True,
        )
        print("运动完成，已回到起点。")


if __name__ == "__main__":
    main()
