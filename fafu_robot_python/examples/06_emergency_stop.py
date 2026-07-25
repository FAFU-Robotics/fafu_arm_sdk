"""展示急停锁定以及人工确认后的恢复。"""

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
        print("当前状态:", arm.state.value)
        input("请扶稳机械臂，按 Enter 触发急停...")

        # 急停会立即停止所有电机，并锁定 ESTOP 状态。
        arm.emergency_stop()
        print("急停后状态:", arm.state.value)

        input("确认工作空间安全、机械臂可以重新使能后，按 Enter 恢复...")

        # resume 只用于 ESTOP；掉电或通信故障应使用 recover(confirm=True)。
        arm.resume()
        print("恢复后状态:", arm.state.value)


if __name__ == "__main__":
    main()
