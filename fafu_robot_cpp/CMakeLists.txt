cmake_minimum_required(VERSION 3.18)
project(fafu_robot_cpp LANGUAGES CXX)

option(FAFU_BUILD_PYBIND11_MODULE "Build the Python extension module" ON)
option(FAFU_BUILD_NATIVE_SDK "Build the native C++ SDK" ON)
option(FAFU_BUILD_SDK_EXAMPLES "Build native SDK examples" ON)
option(FAFU_BUILD_TESTS "Build hardware-free native tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8 /W4 /bigobj)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(FAFU_BUILD_PYBIND11_MODULE)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
endif()

set(_SERIAL_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_part/serial_cmake)
if(NOT EXISTS ${_SERIAL_CMAKE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "Missing vendored third_part/serial_cmake")
endif()
add_subdirectory(${_SERIAL_CMAKE_DIR} ${CMAKE_BINARY_DIR}/serial_cmake_build)

add_library(hightorque_serial_debug STATIC
    src/hightorque_serial.cpp
)
set_target_properties(hightorque_serial_debug PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(hightorque_serial_debug PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(hightorque_serial_debug PUBLIC serial_cmake)

add_library(fafu_robot_core STATIC
    src/core/controller_state.cpp
    src/core/motor_calibration.cpp
    src/core/motor_io.cpp
    src/core/robot_core.cpp
)
set_target_properties(fafu_robot_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(fafu_robot_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fafu_robot_core PUBLIC hightorque_serial_debug)

if(FAFU_BUILD_PYBIND11_MODULE)
    pybind11_add_module(fafu_motor
        bindings.cpp
        bindings_core.cpp
    )
    target_link_libraries(fafu_motor PRIVATE fafu_robot_core)
    target_include_directories(fafu_motor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

if(FAFU_BUILD_NATIVE_SDK)
    add_subdirectory(sdk)
endif()

if(FAFU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NOT DEFINED FAFU_ROBOT_PYTHON_DIR)
    set(FAFU_ROBOT_PYTHON_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/../fafu_robot_python)
endif()

if(FAFU_BUILD_PYBIND11_MODULE)
    add_custom_command(TARGET fafu_motor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
                ${FAFU_ROBOT_PYTHON_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:fafu_motor>
                ${FAFU_ROBOT_PYTHON_DIR}/$<TARGET_FILE_NAME:fafu_motor>
        COMMENT "Copying fafu_motor to ${FAFU_ROBOT_PYTHON_DIR}"
    )

    if(WIN32 AND SERIAL_CMAKE_BUILD_SHARED)
        add_custom_command(TARGET fafu_motor POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:serial_cmake>
                    ${FAFU_ROBOT_PYTHON_DIR}/$<TARGET_FILE_NAME:serial_cmake>
            COMMENT "Copying serial_cmake runtime"
        )
    endif()
endif()

message(STATUS "")
message(STATUS "fafu_robot_cpp configuration")
message(STATUS "  Python module:  ${FAFU_BUILD_PYBIND11_MODULE}")
message(STATUS "  Native SDK:     ${FAFU_BUILD_NATIVE_SDK}")
message(STATUS "  SDK examples:   ${FAFU_BUILD_SDK_EXAMPLES}")
message(STATUS "  Native tests:   ${FAFU_BUILD_TESTS}")
message(STATUS "  Serial shared:  ${SERIAL_CMAKE_BUILD_SHARED}")
message(STATUS "")
