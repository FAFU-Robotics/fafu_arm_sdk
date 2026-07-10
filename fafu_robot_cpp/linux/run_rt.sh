#!/usr/bin/env bash
# =============================================================================
#  run_rt.sh — 给例程套上实时优先级 + CPU 亲和
#
#  强烈推荐给 04_servo_j 用: 100Hz 上层轨迹, 抖动 > watchdog_ms 会触发固件
#  brake, 上层直接崩. RT 调度能把 max_jitter 稳定在 < 1ms.
#
#  权限方案 (三选一):
#      A. sudo 每次 :        sudo bash linux/run_rt.sh 04_servo_j
#      B. setcap 一次性 (推荐):
#             sudo setcap cap_sys_nice=eip build_linux/bin/04_servo_j
#             bash linux/run_rt.sh 04_servo_j
#      C. 改 /etc/security/limits.conf 加:  <user> hard rtprio 99
#
#  用法:
#      bash linux/run_rt.sh                              # 列出可用例程
#      bash linux/run_rt.sh 04_servo_j
#      bash linux/run_rt.sh 04_servo_j --cpu 3
#      bash linux/run_rt.sh 04_servo_j --no-rt           # 只 affinity, 不 FIFO
#      bash linux/run_rt.sh 04_servo_j -- robot.cfg 6    # 透传给例程
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
BIN_DIR="$PROJECT_DIR/build_linux/bin"

if [ $# -lt 1 ]; then
    echo "用法: bash linux/run_rt.sh <exe-name> [--cpu N] [--no-rt] [-- args...]"
    echo ""
    echo "可用例程:"
    if [ -d "$BIN_DIR" ]; then
        find "$BIN_DIR" -maxdepth 1 -type f -executable -printf "  %f\n" 2>/dev/null \
            | grep -v '\.so$' \
            | grep -v '^libserial_cmake' \
            | sort
    else
        echo "  (build_linux/bin/ 不存在, 先跑 bash linux/build.sh)"
    fi
    echo ""
    echo "推荐:"
    echo "  04_servo_j — 100Hz servoJ 跟踪, RT 优先级下 jitter 稳定"
    echo "  02_move_j  — 离线 S-curve, RT 不是必需但更平滑"
    exit 1
fi

EXE="$1"; shift
CPU=0
USE_RT=1
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)    shift; CPU="$1" ;;
        --no-rt)  USE_RT=0 ;;
        --)       shift; EXTRA_ARGS+=("$@"); break ;;
        *)        EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

EXE_PATH="$BIN_DIR/$EXE"
if [ ! -x "$EXE_PATH" ]; then
    echo "[run_rt] 找不到可执行文件: $EXE_PATH" >&2
    echo "         先跑 bash linux/build.sh 编译" >&2
    exit 1
fi

for tool in taskset chrt; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[run_rt] 错误: $tool 没装 (sudo apt install util-linux)" >&2
        exit 1
    fi
done

# 检查 RT 上限
RT_PRIO_LIMIT=$(ulimit -r 2>/dev/null || echo 0)
if [ "$USE_RT" -eq 1 ] && [ "$RT_PRIO_LIMIT" -lt 50 ] && [ "$(id -u)" -ne 0 ]; then
    echo "[run_rt] 警告: ulimit -r = $RT_PRIO_LIMIT, RT 优先级受限"
    echo "         解决方案:"
    echo "           A. 给可执行文件加 cap_sys_nice (推荐, 一次性):"
    echo "              sudo setcap cap_sys_nice=eip $EXE_PATH"
    echo "           B. 加 sudo: sudo bash linux/run_rt.sh $EXE ..."
    echo ""
    echo "         这次尝试继续 (可能会降级为 SCHED_OTHER):"
fi

echo "============================================================"
echo "  fafu_robot 实时运行"
echo "  程序:      $EXE_PATH"
echo "  CPU 亲和:  $CPU"
echo "  调度:      $([ "$USE_RT" -eq 1 ] && echo 'SCHED_FIFO prio=80' || echo '默认 SCHED_OTHER')"
echo "  额外参数:  ${EXTRA_ARGS[*]:-(无)}"
echo "============================================================"

cd "$BIN_DIR"

# prio=80 (不是 99, 留几档给内核 RT 线程, 避免 RCU 饥饿导致系统卡死)
if [ "$USE_RT" -eq 1 ]; then
    exec taskset -c "$CPU" chrt -f 80 "$EXE_PATH" "${EXTRA_ARGS[@]}"
else
    exec taskset -c "$CPU" "$EXE_PATH" "${EXTRA_ARGS[@]}"
fi
