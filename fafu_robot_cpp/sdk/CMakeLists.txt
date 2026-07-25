add_library(fafu_robot_sdk STATIC
    src/fafu_robot_controller.cpp
)
target_include_directories(fafu_robot_sdk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(fafu_robot_sdk PUBLIC fafu_robot_core)

if(FAFU_BUILD_SDK_EXAMPLES)
    set(_FAFU_SDK_EXAMPLES
        01_smoke
        02_move_j
        03_gripper
        04_servo_j
    )
    foreach(_example IN LISTS _FAFU_SDK_EXAMPLES)
        add_executable(${_example} examples/${_example}.cpp)
        target_link_libraries(${_example} PRIVATE fafu_robot_sdk)

        set(_CFG_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/../../fafu_robot_python/robot.cfg)
        if(EXISTS ${_CFG_SRC})
            add_custom_command(TARGET ${_example} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${_CFG_SRC}
                        $<TARGET_FILE_DIR:${_example}>/robot.cfg
            )
        endif()
    endforeach()
endif()
