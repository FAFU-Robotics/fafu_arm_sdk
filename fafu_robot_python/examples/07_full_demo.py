"""Interactive read/move/gripper demonstration for the high-level SDK."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np


SDK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SDK_DIR))

from fafu_robot_controller import FafuRobotController  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FafuRobotController high-level API demo"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=str(SDK_DIR / "robot.cfg"),
        help="path to robot.cfg",
    )
    parser.add_argument(
        "--gripper-id",
        type=int,
        default=None,
        help="motor id of the gripper; omit when no gripper is installed",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=15,
        help="motion speed percentage",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="read state without issuing motion commands",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    has_gripper = args.gripper_id is not None

    if not args.no_move:
        answer = input(
            "Clear the robot workspace, support heavy links, then type "
            "'yes' to run the motion demo: "
        )
        if answer.strip().lower() != "yes":
            print("cancelled before connecting")
            return

    with FafuRobotController(
        cfg_path=args.config,
        has_gripper=has_gripper,
        gripper_motor_id=args.gripper_id,
        auto_enable=not args.no_move,
    ) as arm:
        joints = arm.get_joint_values()
        print("initial joints (deg):", np.degrees(joints).round(2).tolist())
        print("enabled:", arm.is_enabled)

        if args.no_move:
            return

        print("going home")
        arm.go_home(speed=args.speed, block=True)

        # Select a joint whose small oscillation stays inside configured limits.
        base = arm.get_joint_values()
        start = base.copy()
        amplitude = math.radians(10.0)
        margin = math.radians(2.0)
        limits = arm._joint_limits_rad()
        demo_index = None
        for index in range(arm.num_joints - 1, -1, -1):
            if limits is None:
                demo_index = index
                break
            lower, upper = limits[0][index], limits[1][index]
            if (
                start[index] - amplitude >= lower + margin
                and start[index] + amplitude <= upper - margin
            ):
                demo_index = index
                break

        if demo_index is None:
            print("no joint has enough room for the oscillation; skipping")
        else:
            print(f"oscillating J{demo_index + 1} by ±10 degrees")
            for tick in range(40):
                base[demo_index] = (
                    start[demo_index]
                    + amplitude * math.sin(tick * math.pi / 20.0)
                )
                arm.move_j(base, speed=args.speed, block=False)
                time.sleep(0.05)
            base[demo_index] = start[demo_index]
            arm.move_j(base, speed=args.speed, block=True)

        if has_gripper:
            print("opening and closing gripper")
            arm.open_gripper()
            time.sleep(0.3)
            arm.close_gripper()

        joints = arm.get_joint_values()
        print("final joints (deg):", np.degrees(joints).round(2).tolist())


if __name__ == "__main__":
    main()
