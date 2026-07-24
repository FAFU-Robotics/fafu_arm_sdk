# -*- coding: utf-8 -*-
"""
fafu_robot_python
=================

High-level Python SDK for the **Fafu robot arm**
(built on the Hightorque debug board).

This is the Python side of ``fafu_robot_sdk``; the matching C++ binding
source lives next door in ``../fafu_robot_cpp/`` and produces the
``fafu_motor.cpXY-win_amd64.pyd`` that this package loads.

Quick start
-----------

If the parent of ``fafu_robot_sdk/`` is on ``sys.path``::

    from fafu_robot_python import FafuRobotController, GraspResult

Or if this directory itself is on ``sys.path`` (typical for scripts in
``tests/`` and ``examples/``)::

    from fafu_robot_controller import FafuRobotController, GraspResult

>>> import math
>>> from fafu_robot_python import FafuRobotController
>>>
>>> arm = FafuRobotController(
...     cfg_path="robot.cfg",
...     has_gripper=True,
...     gripper_motor_id=7,
... )
>>> arm.move_j([0, math.radians(20), math.radians(40), 0, 0, 0], speed=15)
>>> arm.open_gripper()
>>> r = arm.grasp(force_threshold=500)
>>> if r.grasped:
...     print(f"got it: closed {r.closed_deg:.1f} deg, peak torque {r.peak_torque_raw}")
>>> arm.close_connection()

What gets exported
------------------

* :class:`FafuRobotController` — the main wrapper class.
* :class:`GraspResult`         — the dataclass returned by ``grasp()`` and
                                 ``gripper_control(..., effort_threshold=...)``.

The underlying ``fafu_motor`` C++ binding is *not* re-exported by
this package — if you need raw access, do ``import fafu_motor``
after importing this package (this directory is added to
``sys.path`` below so the ``.pyd`` is findable).
"""
from __future__ import annotations

import os as _os
import sys as _sys

# Make sure the SDK directory is on sys.path so that
# `import fafu_motor` works regardless of how the user installed
# or referenced this package.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from fafu_robot_controller import (  # noqa: E402
    FafuRobotController,
    GraspResult,
    RobotState,
    RobotStateError,
)

__all__ = [
    "FafuRobotController",
    "GraspResult",
    "RobotState",
    "RobotStateError",
]

__version__ = "0.1.0"
