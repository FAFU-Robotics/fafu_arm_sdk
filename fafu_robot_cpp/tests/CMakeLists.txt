add_executable(fafu_robot_core_tests
    test_robot_core.cpp
)
target_link_libraries(fafu_robot_core_tests PRIVATE fafu_robot_core)
target_include_directories(fafu_robot_core_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME fafu_robot_core_tests COMMAND fafu_robot_core_tests)

add_executable(fafu_hightorque_position_encoding_tests
    test_hightorque_position_encoding.cpp
)
target_link_libraries(fafu_hightorque_position_encoding_tests
    PRIVATE hightorque_serial_debug
)
add_test(
    NAME fafu_hightorque_position_encoding_tests
    COMMAND fafu_hightorque_position_encoding_tests
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(fafu_linux_serial_exclusive_tests
        test_linux_serial_exclusive.cpp
    )
    target_link_libraries(fafu_linux_serial_exclusive_tests
        PRIVATE serial_cmake
    )

    add_executable(fafu_linux_serial_polling_tests
        test_linux_serial_polling.cpp
    )
    target_link_libraries(fafu_linux_serial_polling_tests
        PRIVATE hightorque_serial_debug
    )

    add_test(
        NAME fafu_linux_serial_exclusive_tests
        COMMAND fafu_linux_serial_exclusive_tests
    )
    add_test(
        NAME fafu_linux_serial_polling_tests
        COMMAND fafu_linux_serial_polling_tests
    )
endif()
