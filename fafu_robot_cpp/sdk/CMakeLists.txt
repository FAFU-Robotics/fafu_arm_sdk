# =============================================================================
#  fafu_robot_sdk/fafu_robot_cpp/sdk/CMakeLists.txt
#
#  原生 C++ 机械臂 SDK. 跟 pybind11 模块 fafu_motor 并行存在,
#  共享父目录的 hightorque_serial_debug + serial_cmake 静态/动态库.
#
#  导出:
#    - 静态库 target: fafu_robot_sdk
#         头文件搜索路径: <include/> (用户写 #include "fafu/fafu_robot_controller.hpp")
#         链接:          hightorque_serial_debug
#    - 例程 (可选, FAFU_BUILD_SDK_EXAMPLES=ON):
#         01_smoke  / 02_move_j  / 03_gripper
#         POST_BUILD 把 ../fafu_robot_python/robot.cfg copy 到 exe 旁边, 双击就能跑.
# =============================================================================

# 静态库
add_library(fafu_robot_sdk STATIC
    src/fafu_robot_controller.cpp
)

target_include_directories(fafu_robot_sdk
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fafu_robot_sdk
    PUBLIC
        hightorque_serial_debug
)

# 例程
if(FAFU_BUILD_SDK_EXAMPLES)
    set(_FAFU_SDK_EXAMPLES
        01_smoke
        02_move_j
        03_gripper
        04_servo_j
    )

    foreach(_ex IN LISTS _FAFU_SDK_EXAMPLES)
        add_executable(${_ex} examples/${_ex}.cpp)
        target_link_libraries(${_ex} PRIVATE fafu_robot_sdk)

        if(WIN32)
            # 把 serial_cmake.dll 复制到 exe 旁边, 双击能跑
            add_custom_command(TARGET ${_ex} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:serial_cmake>
                    $<TARGET_FILE_DIR:${_ex}>
                COMMENT "复制 serial_cmake.dll 到 ${_ex}/")
        endif()

        # 把示例 robot.cfg 复制到 exe 旁边
        set(_CFG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../fafu_robot_python/robot.cfg)
        if(EXISTS ${_CFG_SRC})
            add_custom_command(TARGET ${_ex} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_CFG_SRC}
                    $<TARGET_FILE_DIR:${_ex}>/robot.cfg
                COMMENT "复制 robot.cfg 到 ${_ex}/")
        endif()
    endforeach()
endif()

message(STATUS "   fafu_robot_sdk (static lib): src/fafu_robot_controller.cpp")
if(FAFU_BUILD_SDK_EXAMPLES)
    message(STATUS "   fafu_robot_sdk examples:     ${_FAFU_SDK_EXAMPLES}")
endif()
