"""展示夹爪最基础的打开和关闭操作。"""

from pathlib import Path
import sys


SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

from fafu_robot_controller import FafuRobotController  # noqa: E402


CONFIG_PATH = SDK_DIR / "robot.cfg"
GRIPPER_ID = 7


def main() -> None:
    with FafuRobotController(
        cfg_path=str(CONFIG_PATH),
        has_gripper=True,
        gripper_motor_id=GRIPPER_ID,
    ) as arm:
        input("确认夹爪内没有手指或障碍物后，按 Enter 开始...")

        arm.open_gripper()
        print("夹爪已打开。")

        arm.close_gripper()
        print("夹爪已关闭。")

        arm.open_gripper()
        state = arm.get_gripper_state()
        print(f"夹爪再次打开，当前位置为 {state.position:.4f} turns。")


if __name__ == "__main__":
    main()
