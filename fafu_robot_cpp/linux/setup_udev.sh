#!/usr/bin/env bash
# =============================================================================
#  setup_udev.sh — 把 99-fafu-debug-board.rules 装到 /etc/udev/rules.d/
#
#  装完后:
#    - 调试板插上自动出现 /dev/fafu_debug_board (软链 → /dev/ttyUSB0)
#    - 任何用户能打开 (MODE=0666), 不用 sudo chmod
#    - 拔插换 USB 口路径都不变
#
#  用法:
#      bash linux/setup_udev.sh                # 安装
#      bash linux/setup_udev.sh --uninstall    # 卸载
#      bash linux/setup_udev.sh --verify       # 只检查不改
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RULE_SRC="$SCRIPT_DIR/99-fafu-debug-board.rules"
RULE_DST="/etc/udev/rules.d/99-fafu-debug-board.rules"
SYMLINK="/dev/fafu_debug_board"

ACTION=install
for arg in "$@"; do
    case "$arg" in
        --uninstall) ACTION=uninstall ;;
        --verify)    ACTION=verify ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "[setup_udev] 未知参数: $arg" >&2
            exit 1
            ;;
    esac
done

if [ ! -f "$RULE_SRC" ]; then
    echo "[setup_udev] 找不到规则文件 $RULE_SRC" >&2
    exit 1
fi

verify() {
    echo ""
    echo "--- 系统串口枚举 ---"
    local found=0
    for dev in /dev/ttyUSB* /dev/ttyACM*; do
        if [ -e "$dev" ]; then
            printf "  %-25s" "$dev"
            if [ -r "$dev" ] && [ -w "$dev" ]; then
                echo "可读写 OK"
            else
                echo "权限不够 (ls -l $dev)"
            fi
            found=1
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "  (没插任何串口设备)"
    fi

    echo ""
    echo "--- 固定路径软链 ---"
    if [ -e "$SYMLINK" ]; then
        echo "  $SYMLINK → $(readlink -f "$SYMLINK") (OK)"
    else
        echo "  $SYMLINK 不存在 (调试板没插 / 规则没生效 / VID 不在表里)"
    fi

    echo ""
    echo "--- USB 设备总览 (前 5 个) ---"
    if command -v lsusb >/dev/null 2>&1; then
        lsusb | head -n 5 | sed 's/^/  /'
    else
        echo "  lsusb 没装 (sudo apt install usbutils)"
    fi
}

case "$ACTION" in
    install)
        echo "============================================================"
        echo "  安装 udev 规则: $RULE_DST"
        echo "============================================================"
        sudo cp "$RULE_SRC" "$RULE_DST"
        sudo chmod 0644 "$RULE_DST"
        sudo udevadm control --reload
        sudo udevadm trigger
        sleep 1
        echo "[setup_udev] 安装完成. 如果调试板已经插着, 拔插一次让规则生效."
        verify
        ;;
    uninstall)
        echo "============================================================"
        echo "  卸载 udev 规则"
        echo "============================================================"
        if [ -f "$RULE_DST" ]; then
            sudo rm -f "$RULE_DST"
            sudo udevadm control --reload
            sudo udevadm trigger
            echo "[setup_udev] 已删除 $RULE_DST"
        else
            echo "[setup_udev] $RULE_DST 不存在, 无需卸载"
        fi
        ;;
    verify)
        echo "============================================================"
        echo "  检查模式 (不改任何东西)"
        echo "============================================================"
        if [ -f "$RULE_DST" ]; then
            echo "  规则文件:  $RULE_DST  (已安装)"
            if diff -q "$RULE_SRC" "$RULE_DST" >/dev/null 2>&1; then
                echo "  内容:      与项目一致"
            else
                echo "  内容:      与项目源码不一致 (本地有改动)"
            fi
        else
            echo "  规则文件:  未安装 (跑 bash linux/setup_udev.sh 安装)"
        fi
        verify
        ;;
esac
