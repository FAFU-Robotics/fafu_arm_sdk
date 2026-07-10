@echo off
REM ============================================================================
REM  fafu_robot_sdk/fafu_robot_cpp/build.bat
REM  Windows 一键构建 fafu_motor (pybind11) 模块.
REM
REM  用法:
REM    build.bat                 REM Release 构建 (推荐)
REM    build.bat Debug           REM Debug 构建
REM
REM  前置条件:
REM    1. 已装 CMake 3.18+ (cmake --version)
REM    2. 已装 Visual Studio 2019/2022 (含 "C++ 桌面开发" workload)
REM    3. 已装 Python (3.7+) + pybind11:
REM         pip install pybind11
REM
REM  ★ 完全自包含 ★
REM  本目录已经把所有 C++ 源码 (协议层 + serial_cmake 三方库) 都 vendor 进来,
REM  不依赖任何外部仓库. 直接在本文件夹里就能编译.
REM
REM  产物会被 CMakeLists.txt 的 POST_BUILD 自动 copy 到:
REM    ..\fafu_robot_python\fafu_motor.cpXY-win_amd64.pyd
REM    ..\fafu_robot_python\serial_cmake.dll
REM  之后从 fafu_robot_python 上下文里直接 `import fafu_motor` 就 work.
REM ============================================================================
setlocal EnableDelayedExpansion

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

cd /d "%~dp0"

echo.
echo ================================================================
echo  fafu_motor build (%BUILD_TYPE%)  --  fafu_robot_cpp
echo ================================================================
echo.

REM 自动定位 pybind11 cmake 目录
for /f "delims=" %%i in ('python -c "import pybind11; print(pybind11.get_cmake_dir())"') do set PYBIND11_DIR=%%i
if "%PYBIND11_DIR%"=="" (
    echo [ERROR] 找不到 pybind11. 请先 `pip install pybind11`.
    exit /b 1
)
echo pybind11_DIR = %PYBIND11_DIR%

for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set PY_EXE=%%i
echo Python       = %PY_EXE%

if not exist build (
    mkdir build
)

cmake -S . -B build ^
    -G "Visual Studio 17 2022" -A x64 ^
    -Dpybind11_DIR="%PYBIND11_DIR%" ^
    -DPYTHON_EXECUTABLE="%PY_EXE%"
if errorlevel 1 goto :fail

cmake --build build --config %BUILD_TYPE% -j
if errorlevel 1 goto :fail

echo.
echo ================================================================
echo  构建成功
echo ================================================================
echo.
echo 验证导入 (从 ..\fafu_robot_python\ 加载 fafu_motor):
pushd "..\fafu_robot_python"
"%PY_EXE%" -c "import sys, os; sys.path.insert(0, os.getcwd()); import fafu_motor; print('OK', fafu_motor.__file__)"
popd
goto :eof

:fail
echo.
echo [BUILD FAILED]
exit /b 1
