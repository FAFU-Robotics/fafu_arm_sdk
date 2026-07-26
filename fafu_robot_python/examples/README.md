# 基础示例

01–06 示例刻意保持简单：不使用 `argparse`，不需要传命令行参数。
串口、电机 ID 和软限位统一从上一级的 `robot.cfg` 读取；需要修改的少量参数都写在脚本顶部。

| 文件 | 展示内容 |
|---|---|
| `01_connect_and_read.py` | 连接机械臂并读取关节、电机状态，不使能电机 |
| `02_move_joint.py` | 使用 `move_j` 让 J1 小幅移动并回到起点 |
| `03_gripper.py` | 打开、关闭并再次打开夹爪 |
| `04_enable_disable.py` | `enable`、`brake`、`disable` 生命周期 |
| `05_servo_joint.py` | 使用 `servo_start/servo_j/servo_end` 流式发送目标 |
| `06_emergency_stop.py` | 急停状态和人工确认后的恢复 |
| `07_full_demo.py` | 原 Controller 内置的完整命令行演示 |
| `visible_motion.py` | 幅度更明显的综合真机演示 |

在 SDK 根目录运行，例如：

```powershell
python fafu_robot_python/examples/01_connect_and_read.py
python fafu_robot_python/examples/02_move_joint.py
```

运动类示例会先等待用户按 Enter，以便确认工作空间已经清空。

最常用的控制器创建方式如下：

```python
from fafu_robot_controller import FafuRobotController

with FafuRobotController(
    cfg_path="robot.cfg",
    has_gripper=True,
    gripper_motor_id=7,
) as arm:
    print(arm.get_joint_values())
```

如果不使用夹爪，删除 `has_gripper` 和 `gripper_motor_id` 两项即可。
请确保当前 Python 版本与 `fafu_motor.cpXX-win_amd64.pyd` 的 ABI 一致。
