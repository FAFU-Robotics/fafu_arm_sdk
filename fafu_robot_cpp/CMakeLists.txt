# =============================================================================
#  fafu_robot_sdk/fafu_robot_cpp/CMakeLists.txt
#
#  ★ 完全自包含 ★
#  本目录已经把所有需要的源码 (协议层 + serial_cmake 三方串口库) 都 vendor 进来,
#  所以构建过程不引用任何外部仓库. 你可以把整个
#  fafu_robot_sdk/ 单独拷到任何位置, 仍然能在 fafu_robot_cpp/ 内完成构建.
#
#  构建 (Windows / VS 2022, Release, 推荐):
#    cd fafu_robot_sdk\fafu_robot_cpp
#    build.bat
#
#  手工 (跨平台):
#    cd fafu_robot_sdk/fafu_robot_cpp
#    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#          -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
#    cmake --build build --config Release -j
#
#  产物 (POST_BUILD 自动复制到 ../fafu_robot_python/):
#    fafu_motor.cpXY-win_amd64.pyd       ← pybind11 模块
#    serial_cmake.dll                    ← Windows 运行时依赖
#
#  关键 CMake 变量 (按需 -D 覆盖):
#    -DFAFU_ROBOT_PYTHON_DIR=<path>    ← 产物复制目标 (默认 ../fafu_robot_python)
#    -Dpybind11_DIR=<cmake_dir>        ← pybind11 cmake 配置目录
# =============================================================================
cmake_minimum_required(VERSION 3.18)
project(fafu_robot_cpp LANGUAGES CXX)

# 顶层选项
option(FAFU_BUILD_PYBIND11_MODULE  "Build the pybind11 module fafu_motor"       ON)
option(FAFU_BUILD_NATIVE_SDK       "Build the native C++ SDK (sdk/)"            ON)
option(FAFU_BUILD_SDK_EXAMPLES     "Build sdk/examples/*.exe"                   ON)

set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(MSVC)
    add_compile_options(/utf-8 /W3 /bigobj)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# Output everything (libs, .pyd, helpers) under build/bin so .dll 和 .pyd
# 永远在同一目录, 调试时手工跑也方便.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# -----------------------------------------------------------------------------
#  pybind11 (仅当 FAFU_BUILD_PYBIND11_MODULE=ON 才需要)
# -----------------------------------------------------------------------------
if(FAFU_BUILD_PYBIND11_MODULE)
    find_package(pybind11 CONFIG REQUIRED)
endif()

# -----------------------------------------------------------------------------
#  Vendored: serial_cmake (跨平台串口库, 原 wjwwood/serial 的 fork)
# -----------------------------------------------------------------------------
set(_SERIAL_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_part/serial_cmake)
if(NOT EXISTS ${_SERIAL_CMAKE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "未找到 third_part/serial_cmake/. 这是已 vendor 的源码, 不应缺失.\n"
        "如果丢了, 从上游 motor_cpp/third_part/serial_cmake/ 重新拷一份.")
endif()
add_subdirectory(${_SERIAL_CMAKE_DIR} ${CMAKE_BINARY_DIR}/serial_cmake_build)

# -----------------------------------------------------------------------------
#  Vendored: hightorque_serial (调试板 USB 串口协议 C++ 实现)
# -----------------------------------------------------------------------------
add_library(hightorque_serial_debug STATIC
    src/hightorque_serial.cpp
)

target_include_directories(hightorque_serial_debug PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hightorque_serial_debug PUBLIC serial_cmake)

# -----------------------------------------------------------------------------
#  pybind11 模块: fafu_motor
#  (FafuRobotController 里通过 `import fafu_motor` 加载.)
# -----------------------------------------------------------------------------
if(FAFU_BUILD_PYBIND11_MODULE)
    pybind11_add_module(fafu_motor
        bindings.cpp
    )

    target_link_libraries(fafu_motor PRIVATE
        hightorque_serial_debug
    )

    target_include_directories(fafu_motor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# -----------------------------------------------------------------------------
#  原生 C++ SDK: fafu_robot_sdk  (sdk/)
# -----------------------------------------------------------------------------
if(FAFU_BUILD_NATIVE_SDK)
    add_subdirectory(sdk)
endif()

# -----------------------------------------------------------------------------
#  POST_BUILD: 把产物 (.pyd + serial_cmake.dll) 复制到 ../fafu_robot_python/
# -----------------------------------------------------------------------------
if(NOT DEFINED FAFU_ROBOT_PYTHON_DIR)
    set(FAFU_ROBOT_PYTHON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../fafu_robot_python)
endif()

if(FAFU_BUILD_PYBIND11_MODULE)
    add_custom_command(TARGET fafu_motor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${FAFU_ROBOT_PYTHON_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:fafu_motor>
            ${FAFU_ROBOT_PYTHON_DIR}/$<TARGET_FILE_NAME:fafu_motor>
        COMMENT "复制 fafu_motor 模块到 ${FAFU_ROBOT_PYTHON_DIR}/"
    )

    if(WIN32)
        add_custom_command(TARGET fafu_motor POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:serial_cmake>
                ${FAFU_ROBOT_PYTHON_DIR}/$<TARGET_FILE_NAME:serial_cmake>
            COMMENT "复制 serial_cmake.dll 到 ${FAFU_ROBOT_PYTHON_DIR}/"
        )
    endif()
endif()

# -----------------------------------------------------------------------------
#  Configure-time summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "================================================================")
message(STATUS " fafu_robot_cpp 配置完成")
message(STATUS "   FAFU_BUILD_PYBIND11_MODULE = ${FAFU_BUILD_PYBIND11_MODULE}")
message(STATUS "   FAFU_BUILD_NATIVE_SDK      = ${FAFU_BUILD_NATIVE_SDK}")
message(STATUS "   FAFU_BUILD_SDK_EXAMPLES    = ${FAFU_BUILD_SDK_EXAMPLES}")
if(FAFU_BUILD_PYBIND11_MODULE)
    message(STATUS "   pybind11_DIR               = ${pybind11_DIR}")
    message(STATUS "   Python                     = ${PYTHON_EXECUTABLE} (${PYTHON_VERSION_STRING})")
    message(STATUS "   FAFU_ROBOT_PYTHON_DIR      = ${FAFU_ROBOT_PYTHON_DIR}")
endif()
message(STATUS "   Vendored serial_cmake      = ${_SERIAL_CMAKE_DIR}")
message(STATUS "================================================================")
message(STATUS "")
