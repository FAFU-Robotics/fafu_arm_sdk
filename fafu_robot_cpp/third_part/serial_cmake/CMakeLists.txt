cmake_minimum_required(VERSION 3.10)
project(serial_cmake)

option(SERIAL_CMAKE_BUILD_SHARED
       "Build serial_cmake as a shared library" OFF)

include(cmake/version.cmake)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(SERIAL_SRC src/serial.cc)
if(WIN32)
    list(APPEND SERIAL_SRC
        src/impl/win.cc
        src/impl/list_ports/list_ports_win.cc
    )
elseif(APPLE)
    list(APPEND SERIAL_SRC
        src/impl/unix.cc
        src/impl/list_ports/list_ports_osx.cc
    )
else()
    list(APPEND SERIAL_SRC
        src/impl/unix.cc
        src/impl/list_ports/list_ports_linux.cc
    )
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(SERIAL_CMAKE_BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED ${SERIAL_SRC})
else()
    add_library(${PROJECT_NAME} STATIC ${SERIAL_SRC})
    set_target_properties(${PROJECT_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(${PROJECT_NAME} PUBLIC setupapi)
elseif(NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PUBLIC rt)
endif()

install(DIRECTORY include/serial
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(TARGETS ${PROJECT_NAME}
    EXPORT serial_cmakeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(EXPORT serial_cmakeTargets
    FILE serial_cmakeTargets.cmake
    NAMESPACE serial::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/serial_cmake
)
configure_package_config_file(
    cmake/serial_cmakeConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/serial_cmakeConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/serial_cmake
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/serial_cmakeConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/serial_cmake
)
