#!/usr/bin/env bash
# =============================================================================
#  install_deps.sh — fafu_robot_cpp Ubuntu 一键装依赖
#
#  依赖要点:
#    - cmake >= 3.18
#    - pybind11 + python3-dev (编 fafu_motor.so 需要)
#    - 自动检测 pybind11 cmake dir, 写到 ~/.fafu_robot_cpp.env 方便 build.sh 复用
#
#  做的事:
#    1. apt 装编译工具链 + cmake + git + python3-dev + pkg-config
#    2. 校验 gcc >= 7 / cmake >= 3.18
#    3. pip install pybind11 (给当前 python3)
#    4. 把当前用户加入 dialout 组
#
#  用法:
#      bash linux/install_deps.sh
#      bash linux/install_deps.sh --skip-group     # 不动用户组 (CI / Docker)
#      bash linux/install_deps.sh --no-pybind11    # 只装原生 SDK 所需, 跳过 Python 相关
#      bash linux/install_deps.sh --python python3.10   # 指定 Python (多版本环境)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

SKIP_GROUP=0
NO_PYBIND11=0
PY_EXE=python3

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-group)  SKIP_GROUP=1 ;;
        --no-pybind11) NO_PYBIND11=1 ;;
        --python)      shift; PY_EXE="$1" ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "[install_deps] 未知参数: $1" >&2
            exit 1
            ;;
    esac
    shift
done

echo "============================================================"
echo "  fafu_robot_cpp SDK — Ubuntu 依赖安装"
echo "  项目目录: $PROJECT_DIR"
echo "  Python:   $PY_EXE (--no-pybind11=$NO_PYBIND11)"
echo "============================================================"

# ---- 1. 检测系统 ----
if [ ! -f /etc/os-release ]; then
    echo "[install_deps] 无法检测系统 (找不到 /etc/os-release)" >&2
    exit 1
fi
. /etc/os-release
echo "[install_deps] 检测到系统: $PRETTY_NAME"

case "${ID:-}" in
    ubuntu|debian|linuxmint|pop) ;;
    *)
        echo "[install_deps] 警告: 系统是 '$ID', 仅在 Ubuntu/Debian 系测试过."
        read -r -p "                继续 (y/N)? " ans
        [[ "$ans" =~ ^[Yy]$ ]] || exit 1
        ;;
esac

# ---- 2. apt 装包 ----
echo ""
echo "--- 1/4 apt 装编译依赖 ---"

APT_PKGS=(
    build-essential
    cmake
    pkg-config
    git
    ca-certificates
)
if [ "$NO_PYBIND11" -eq 0 ]; then
    # python3-dev: Python.h 头文件, pybind11 编译需要
    # python3-pip: pip install pybind11
    APT_PKGS+=(python3-dev python3-pip)
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends "${APT_PKGS[@]}"

# ---- 3. 工具链版本校验 ----
echo ""
echo "--- 2/4 工具链版本校验 ---"
echo "  gcc:    $(gcc --version | head -n1)"
echo "  g++:    $(g++ --version | head -n1)"
echo "  cmake:  $(cmake --version | head -n1)"

GCC_MAJOR=$(gcc -dumpversion | cut -d. -f1)
if [ "$GCC_MAJOR" -lt 7 ]; then
    echo "[install_deps] 错误: gcc 主版本 $GCC_MAJOR < 7, 不支持 C++17" >&2
    exit 1
fi

CMAKE_VER=$(cmake --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | head -n1)
CMAKE_MAJOR=${CMAKE_VER%.*}
CMAKE_MINOR=${CMAKE_VER#*.}
if [ "$CMAKE_MAJOR" -lt 3 ] || { [ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 18 ]; }; then
    echo "[install_deps] 错误: cmake $CMAKE_VER < 3.18 (CMakeLists.txt 要求)" >&2
    echo "                Ubuntu 20.04+ apt 版本就够; 18.04 需要装 Kitware 官方源:" >&2
    echo "                    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \\ " >&2
    echo "                        | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null" >&2
    echo "                    echo 'deb https://apt.kitware.com/ubuntu/ bionic main' \\ " >&2
    echo "                        | sudo tee /etc/apt/sources.list.d/kitware.list" >&2
    echo "                    sudo apt update && sudo apt install -y cmake" >&2
    exit 1
fi

# ---- 4. pybind11 (Python 侧编译需要) ----
if [ "$NO_PYBIND11" -eq 0 ]; then
    echo ""
    echo "--- 3/4 pybind11 (给 $PY_EXE 装) ---"

    if ! command -v "$PY_EXE" >/dev/null 2>&1; then
        echo "[install_deps] 错误: $PY_EXE 没装" >&2
        echo "               装法: sudo apt install $PY_EXE" >&2
        exit 1
    fi

    PY_VER=$("$PY_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "  Python 版本: $PY_VER  ($PY_EXE)"
    echo "  Python.h:    $("$PY_EXE" -c "import sysconfig; print(sysconfig.get_path('include'))")/Python.h"

    if ! "$PY_EXE" -c "import Python" 2>/dev/null; then
        # 上面这条永远失败, 但把 Python.h 存在性检查一下
        PY_INC=$("$PY_EXE" -c "import sysconfig; print(sysconfig.get_path('include'))")
        if [ ! -f "$PY_INC/Python.h" ]; then
            echo "[install_deps] 警告: Python.h 不在 $PY_INC, python${PY_VER}-dev 可能没装干净"
            echo "                     手动装: sudo apt install python${PY_VER}-dev"
        fi
    fi

    # pybind11: 装到 user site-packages, 不污染系统 Python
    if "$PY_EXE" -c "import pybind11" 2>/dev/null; then
        echo "  pybind11:    $("$PY_EXE" -c "import pybind11; print(pybind11.__version__, '->', pybind11.get_cmake_dir())")"
    else
        echo "  pybind11 没装, 现在装 (--user):"
        "$PY_EXE" -m pip install --user --upgrade pybind11
        echo "  pybind11:    $("$PY_EXE" -c "import pybind11; print(pybind11.__version__, '->', pybind11.get_cmake_dir())")"
    fi

    # 写入环境提示文件, 供 build.sh 复用
    ENV_FILE="$HOME/.fafu_robot_cpp.env"
    {
        echo "# fafu_robot_cpp build env (由 install_deps.sh 生成于 $(date))"
        echo "export FAFU_PY_EXE=$(command -v "$PY_EXE")"
        echo "export FAFU_PYBIND11_DIR=$("$PY_EXE" -c "import pybind11; print(pybind11.get_cmake_dir())")"
    } > "$ENV_FILE"
    echo "  已写入: $ENV_FILE (build.sh 会自动 source 它)"
else
    echo ""
    echo "--- 3/4 pybind11 (--no-pybind11 已跳过) ---"
    echo "  只会编 fafu_robot_sdk 静态库 + 例程, 不编 fafu_motor.so."
fi

# ---- 5. dialout 组 ----
echo ""
echo "--- 4/4 用户组 (访问 /dev/ttyUSB* 必需) ---"

if [ "$SKIP_GROUP" -eq 1 ]; then
    echo "[install_deps] --skip-group, 跳过用户组处理"
else
    if id -nG "$USER" | grep -qw dialout; then
        echo "  用户 $USER 已在 dialout 组"
    else
        echo "  把 $USER 加入 dialout 组..."
        sudo usermod -aG dialout "$USER"
        echo ""
        echo "  *** 重要: 用户组修改需要 *重新登录* 才生效 ***"
        echo "  或用 udev 规则绕过组限制 (推荐):"
        echo "      bash linux/setup_udev.sh"
    fi
fi

echo ""
echo "============================================================"
echo "  依赖安装完成. 下一步:"
echo "    1. (推荐) 装 udev 规则:      bash linux/setup_udev.sh"
echo "    2. 编译 (全部):              bash linux/build.sh"
echo "       只编 fafu_motor.so:  bash linux/build_module_only.sh"
echo "       只编原生 SDK / 例程:     bash linux/build.sh --no-python"
echo "    3. 跑例程:                   ./build_linux/bin/01_smoke"
echo "    4. Python 端验证:            cd ../fafu_robot_python && python3 -c 'import fafu_motor'"
echo "============================================================"
