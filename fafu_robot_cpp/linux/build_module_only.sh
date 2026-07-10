#!/usr/bin/env bash
# =============================================================================
#  build_module_only.sh — 只编 fafu_motor.so (给不同 Python ABI 快速重编)
#
#  与 build.sh --module-only 等价, 但更快: 不重跑 configure, 直接 --target.
#  典型场景: 已经装了新的 Python 环境 (conda activate / pyenv shell),
#           想给它编一个匹配 ABI 的 .so, 又不想动 SDK / examples.
#
#  用法:
#      # 假设已经 conda activate py310
#      bash linux/build_module_only.sh                       # 用当前 python3
#      bash linux/build_module_only.sh --python python3.11   # 指定
#      bash linux/build_module_only.sh --clean               # 强制重 configure
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
BUILD_DIR="$PROJECT_DIR/build_linux"
PY_TARGET_DIR="$PROJECT_DIR/../fafu_robot_python"

DO_CLEAN=0
PY_EXE=python3

while [ $# -gt 0 ]; do
    case "$1" in
        --clean)  DO_CLEAN=1 ;;
        --python) shift; PY_EXE="$1" ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "[build_mod] 未知参数: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if ! command -v "$PY_EXE" >/dev/null 2>&1; then
    echo "[build_mod] $PY_EXE 没装" >&2
    exit 1
fi
if ! "$PY_EXE" -c "import pybind11" 2>/dev/null; then
    echo "[build_mod] $PY_EXE 里没装 pybind11" >&2
    echo "            $PY_EXE -m pip install --user pybind11" >&2
    exit 1
fi

PYBIND11_DIR=$("$PY_EXE" -c "import pybind11; print(pybind11.get_cmake_dir())")
PY_VER=$("$PY_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "============================================================"
echo "  只编 fafu_motor.so"
echo "  Python:        $(command -v "$PY_EXE")  (cp$(echo "$PY_VER" | tr -d .))"
echo "  pybind11_DIR:  $PYBIND11_DIR"
echo "  build_dir:     $BUILD_DIR"
echo "============================================================"

if [ "$DO_CLEAN" -eq 1 ] || [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    rm -rf "$BUILD_DIR"
    cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAFU_BUILD_PYBIND11_MODULE=ON \
        -DFAFU_BUILD_NATIVE_SDK=OFF \
        -DFAFU_BUILD_SDK_EXAMPLES=OFF \
        -DPython3_EXECUTABLE="$(command -v "$PY_EXE")" \
        -DPYTHON_EXECUTABLE="$(command -v "$PY_EXE")" \
        -Dpybind11_DIR="$PYBIND11_DIR"
fi

cmake --build "$BUILD_DIR" --target fafu_motor --parallel "$(nproc 2>/dev/null || echo 4)"

# POST_BUILD 修复 (同 build.sh)
if [ -d "$PY_TARGET_DIR" ] && [ -f "$BUILD_DIR/bin/libserial_cmake.so" ]; then
    cp -f "$BUILD_DIR/bin/libserial_cmake.so" "$PY_TARGET_DIR/"
    MOD_SO=$(find "$PY_TARGET_DIR" -maxdepth 1 -name 'fafu_motor*.so' | head -n1)
    if [ -n "$MOD_SO" ] && command -v patchelf >/dev/null 2>&1; then
        patchelf --set-rpath '$ORIGIN' --force-rpath "$MOD_SO"
    fi
fi

echo ""
echo "构建完成. 验证导入:"
echo "    cd $PY_TARGET_DIR"
echo "    $PY_EXE -c 'import fafu_motor; print(fafu_motor.__file__)'"
