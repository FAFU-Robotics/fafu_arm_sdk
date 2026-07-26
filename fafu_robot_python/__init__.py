# -*- coding: utf-8 -*-
"""
High-level Python SDK for the Fafu robot arm.

Package imports are preferred::

    from fafu_robot_python import FafuRobotController, ServoOpts

Directly importing ``fafu_robot_controller`` remains supported for legacy
scripts that put this directory on ``sys.path``.
"""
from __future__ import annotations

from ._api_types import (
    FrictionParams,
    GraspResult,
    RobotState,
    RobotStateError,
    ServoOpts,
)
from .fafu_robot_controller import FafuRobotController

__all__ = [
    "FafuRobotController",
    "FrictionParams",
    "GraspResult",
    "RobotState",
    "RobotStateError",
    "ServoOpts",
]

__version__ = "0.1.0"
