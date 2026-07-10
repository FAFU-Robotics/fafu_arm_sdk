#!/usr/bin/env bash
# =============================================================================
#  build.sh — fafu_robot_cpp Ubuntu 一键构建
#
#  与老版 (motor_example_debug/linux/build.sh) 差异:
#    - 需要 pybind11 (默认编 fafu_motor.so, 可 --no-python 关掉)
#    - 自动从 ~/.fafu_robot_cpp.env 读 FAFU_PYBIND11_DIR (由 install_deps.sh 生成)
#    - POST_BUILD 阶段修 Linux 侧遗漏: 把 libserial_cmake.so 复制到
#      ../fafu_robot_python/, 设 fafu_motor.so 的 RPATH=$ORIGIN
#      (CMakeLists 里只在 WIN32 分支复制 dll, Linux 不复制, 会导致
#       `import fafu_motor` 报 "libserial_cmake.so: cannot open shared object file")
#
#  用法:
#      bash linux/build.sh                    # 全部: pybind11 + SDK + examples (Release)
#      bash linux/build.sh --no-python        # 只编原生 SDK + examples, 不依赖 pybind11
#      bash linux/build.sh --no-examples      # 编模块 + SDK 静态库, 不编例程
#      bash linux/build.sh --module-only      # 只编 pybind11 模块 (快速迭代)
#      bash linux/build.sh --clean            # 删 build_linux/ 重头编
#      bash linux/build.sh --debug            # Debug 构建
#      bash linux/build.sh --jobs 4           # 限制并行数
#      bash linux/build.sh --python python3.10  # 指定 Python (多 ABI 环境)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
BUILD_DIR="$PROJECT_DIR/build_linux"
PY_TARGET_DIR="$PROJECT_DIR/../fafu_robot_python"   # POST_BUILD 目标 (CMakeLists 默认)

BUILD_TYPE=Release
DO_CLEAN=0
NO_PYTHON=0
NO_EXAMPLES=0
MODULE_ONLY=0
NO_NATIVE_SDK=0
JOBS=$(nproc 2>/dev/null || echo 4)
PY_EXE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --debug)         BUILD_TYPE=Debug ;;
        --release)       BUILD_TYPE=Release ;;
        --clean)         DO_CLEAN=1 ;;
        --no-python)     NO_PYTHON=1 ;;
        --no-examples)   NO_EXAMPLES=1 ;;
        --module-only)   MODULE_ONLY=1 ;;
        --no-native-sdk) NO_NATIVE_SDK=1 ;;
        --jobs|-j)       shift; JOBS="$1" ;;
        --python)        shift; PY_EXE="$1" ;;
        -h|--help)
            sed -n '2,25p' "$0"
            exit 0
            ;;
        *)
            echo "[build] 未知参数: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ "$MODULE_ONLY" -eq 1 ]; then
    NO_NATIVE_SDK=1
    NO_EXAMPLES=1
fi

cd "$PROJECT_DIR"

# ---- 工具链检查 ----
for tool in cmake g++; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[build] 错误: $tool 没装. 跑 bash linux/install_deps.sh." >&2
        exit 1
    fi
done

# ---- serial_cmake 路径检查 (self-contained, 应该已 vendor) ----
SERIAL_DIR="$PROJECT_DIR/third_part/serial_cmake"
if [ ! -f "$SERIAL_DIR/CMakeLists.txt" ]; then
    echo "[build] 错误: 找不到 vendored serial_cmake" >&2
    echo "         期望路径: $SERIAL_DIR" >&2
    echo "         这个 SDK 是 self-contained 的, 三方源码不应缺失." >&2
    exit 1
fi

# ---- pybind11 定位 (仅 --no-python 未指定时) ----
CMAKE_ARGS=(
    "-S" "$PROJECT_DIR"
    "-B" "$BUILD_DIR"
    "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    "-DFAFU_BUILD_PYBIND11_MODULE=$([ "$NO_PYTHON" -eq 1 ] && echo OFF || echo ON)"
    "-DFAFU_BUILD_NATIVE_SDK=$([ "$NO_NATIVE_SDK" -eq 1 ] && echo OFF || echo ON)"
    "-DFAFU_BUILD_SDK_EXAMPLES=$([ "$NO_EXAMPLES" -eq 1 ] && echo OFF || echo ON)"
)

PYBIND11_DIR=""
if [ "$NO_PYTHON" -eq 0 ]; then
    # 优先级: --python 参数 > $FAFU_PY_EXE 环境 > ~/.fafu_robot_cpp.env > python3
    if [ -z "$PY_EXE" ]; then
        if [ -f "$HOME/.fafu_robot_cpp.env" ]; then
            # shellcheck disable=SC1091
            source "$HOME/.fafu_robot_cpp.env"
            PY_EXE="${FAFU_PY_EXE:-python3}"
            PYBIND11_DIR="${FAFU_PYBIND11_DIR:-}"
        else
            PY_EXE=python3
        fi
    fi

    if ! command -v "$PY_EXE" >/dev/null 2>&1; then
        echo "[build] 错误: $PY_EXE 没装. 装它, 或者用 --no-python." >&2
        exit 1
    fi

    # PYBIND11_DIR 没定时, 从 Python 里问
    if [ -z "$PYBIND11_DIR" ]; then
        if ! "$PY_EXE" -c "import pybind11" 2>/dev/null; then
            echo "[build] 错误: $PY_EXE 里没装 pybind11." >&2
            echo "        装法: $PY_EXE -m pip install --user pybind11" >&2
            echo "        或用: bash linux/install_deps.sh" >&2
            exit 1
        fi
        PYBIND11_DIR=$("$PY_EXE" -c "import pybind11; print(pybind11.get_cmake_dir())")
    fi

    CMAKE_ARGS+=(
        "-DPython3_EXECUTABLE=$(command -v "$PY_EXE")"
        "-DPYTHON_EXECUTABLE=$(command -v "$PY_EXE")"
        "-Dpybind11_DIR=$PYBIND11_DIR"
    )
fi

echo "============================================================"
echo "  fafu_robot_cpp SDK — Linux 构建"
echo "  项目目录:      $PROJECT_DIR"
echo "  构建目录:      $BUILD_DIR"
echo "  构建类型:      $BUILD_TYPE"
echo "  并行 jobs:     $JOBS"
echo "  cmake:         $(cmake --version | head -n1)"
echo "  g++:           $(g++ --version | head -n1)"
if [ "$NO_PYTHON" -eq 0 ]; then
    PY_VER=$("$PY_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "  Python:        $(command -v "$PY_EXE") (cp$(echo "$PY_VER" | tr -d .))"
    echo "  pybind11_DIR:  $PYBIND11_DIR"
fi
echo "  编 fafu_motor.so:  $([ "$NO_PYTHON" -eq 0 ] && echo yes || echo no)"
echo "  编 fafu_robot_sdk.a:   $([ "$NO_NATIVE_SDK" -eq 0 ] && echo yes || echo no)"
echo "  编 examples:            $([ "$NO_EXAMPLES" -eq 0 ] && echo yes || echo no)"
echo "============================================================"

# ---- clean ----
if [ "$DO_CLEAN" -eq 1 ]; then
    echo "[build] --clean: 删除 $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# ---- configure ----
echo ""
echo "--- cmake configure ---"
cmake "${CMAKE_ARGS[@]}"

# clangd / IDE 自动索引
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    ln -sf "$BUILD_DIR/compile_commands.json" "$PROJECT_DIR/compile_commands.json"
fi

# ---- build ----
echo ""
echo "--- cmake build ---"
cmake --build "$BUILD_DIR" --parallel "$JOBS"

# ---- POST_BUILD 修复: 把 libserial_cmake.so 复制到 fafu_robot_python/ ----
#
# CMakeLists.txt 里只在 if(WIN32) 分支复制 serial_cmake.dll, Linux 下漏掉了
# libserial_cmake.so. 结果 fafu_robot_python/ 目录只有 fafu_motor.so
# 没有 libserial_cmake.so, Python `import fafu_motor` 会报:
#     libserial_cmake.so: cannot open shared object file
#
# 修复策略:
#   1. cp build_linux/bin/libserial_cmake.so 到 ../fafu_robot_python/
#   2. 用 patchelf 把 fafu_motor.so 的 RPATH 设成 $ORIGIN,
#      让动态加载器优先在 .so 自己的目录里找依赖 (跨机器可移植)
#   3. 没装 patchelf 时降级: 提示用户手动 export LD_LIBRARY_PATH
if [ "$NO_PYTHON" -eq 0 ]; then
    echo ""
    echo "--- POST_BUILD: 修复 Linux 侧动态库依赖 ---"

    if [ ! -d "$PY_TARGET_DIR" ]; then
        echo "[build] 警告: $PY_TARGET_DIR 不存在, 跳过 fixup"
    else
        # 1. libserial_cmake.so → ../fafu_robot_python/
        SO_SRC="$BUILD_DIR/bin/libserial_cmake.so"
        if [ -f "$SO_SRC" ]; then
            cp -f "$SO_SRC" "$PY_TARGET_DIR/"
            echo "  ✓ $SO_SRC → $PY_TARGET_DIR/libserial_cmake.so"
        else
            echo "  ✗ 没找到 $SO_SRC (构建失败?)"
        fi

        # 2. patchelf: fafu_motor*.so 的 RPATH = $ORIGIN
        MOD_SO=$(find "$PY_TARGET_DIR" -maxdepth 1 -name 'fafu_motor*.so' | head -n1)
        if [ -n "$MOD_SO" ]; then
            if command -v patchelf >/dev/null 2>&1; then
                # --force-rpath 保证使用 DT_RPATH (老系统兼容); 新系统 DT_RUNPATH 也 work
                patchelf --set-rpath '$ORIGIN' --force-rpath "$MOD_SO"
                CUR_RPATH=$(patchelf --print-rpath "$MOD_SO" 2>/dev/null || echo "?")
                echo "  ✓ $MOD_SO  RPATH = $CUR_RPATH"
            else
                echo "  ! patchelf 没装, 无法设置 RPATH"
                echo "    装法:  sudo apt install patchelf"
                echo "    替代方案 (每次跑前 export):"
                echo "        export LD_LIBRARY_PATH=$PY_TARGET_DIR:\$LD_LIBRARY_PATH"
            fi
        else
            echo "  ✗ 没找到 fafu_motor*.so 在 $PY_TARGET_DIR (POST_BUILD 复制失败?)"
        fi
    fi
fi

# ---- 列出产出 ----
echo ""
echo "============================================================"
echo "  构建完成. 产出在 $BUILD_DIR/bin/"
echo "============================================================"
if [ -d "$BUILD_DIR/bin" ]; then
    ls -lh "$BUILD_DIR/bin/" | grep -v '^d' | grep -v '^total' || true
fi
if [ "$NO_PYTHON" -eq 0 ] && [ -d "$PY_TARGET_DIR" ]; then
    echo ""
    echo "  Python 侧 ($PY_TARGET_DIR):"
    ls -lh "$PY_TARGET_DIR/"*.so 2>/dev/null | sed 's/^/    /' || echo "    (无 .so)"
fi

if [ "$NO_EXAMPLES" -eq 0 ]; then
    echo ""
    echo "  跑原生 C++ 例程 (先插调试板):"
    echo "    cd $BUILD_DIR/bin"
    echo "    ./01_smoke          # 最小连通性测试 (安全, 不动电机)"
    echo "    ./02_move_j         # go_home + 多关节 S-curve (会运动!)"
    echo "    ./03_gripper        # 夹爪 open/close/grasp (会闭合!)"
    echo "    ./04_servo_j        # 100Hz sin 跟踪 + 看门狗 (会运动!)"
    echo ""
    echo "  或用实时优先级 (推荐 04_servo_j):"
    echo "    bash linux/run_rt.sh 04_servo_j"
fi

if [ "$NO_PYTHON" -eq 0 ]; then
    echo ""
    echo "  Python 侧验证:"
    echo "    cd $PY_TARGET_DIR"
    echo "    $PY_EXE -c 'import fafu_motor; print(fafu_motor.__file__)'"
fi
