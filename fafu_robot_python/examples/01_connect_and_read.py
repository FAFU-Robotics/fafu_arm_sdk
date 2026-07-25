"""连接机械臂并读取状态；此示例不会使能或移动电机。"""

from pathlib import Path
import math
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
        print("controller state:", arm.state.value)
        print("joint motor ids:", arm.joint_motor_ids)

        joint_degrees = [
            round(math.degrees(value), 2) for value in arm.get_joint_values()
        ]
        print("joint positions (deg):", joint_degrees)

        for motor_id, state in arm.get_motor_states(prefer_cache=False).items():
            print(
                f"M{motor_id}: mode={state.mode}, fault={state.fault}, "
                f"position={state.position:.4f} turns, "
                f"velocity={state.velocity:.4f} turns/s"
            )


if __name__ == "__main__":
    main()
