"""展示使能、短路制动和自由转动三种基础状态。"""

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
        auto_enable=False,
    ) as arm:
        print("初始状态:", arm.state.value)
        input("扶稳机械臂并确认工作空间安全，然后按 Enter...")

        arm.enable()
        print("enable 后:", arm.state.value)

        arm.brake()
        print("brake 后:", arm.state.value)

        arm.enable()
        print("再次 enable 后:", arm.state.value)

        # disable 会关闭电机输出，关节可以被手动移动。
        arm.disable()
        print("disable 后:", arm.state.value)


if __name__ == "__main__":
    main()
