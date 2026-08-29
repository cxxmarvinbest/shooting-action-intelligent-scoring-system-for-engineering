cmake_minimum_required(VERSION 3.10)
project(mpp_player)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 设置库输出目录为源码根目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec libavformat libavutil libswscale)
pkg_check_modules(MPP REQUIRED rockchip_mpp)

# RGA 手动查找
find_path(RGA_INCLUDE_DIR
    NAMES rga/rga.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES rockchip
)
find_library(RGA_LIBRARY
    NAMES rga librga
    PATHS /usr/lib /usr/local/lib
)
if(NOT RGA_INCLUDE_DIR OR NOT RGA_LIBRARY)
    message(FATAL_ERROR "RGA library or headers not found")
else()
    message(STATUS "Found RGA: ${RGA_LIBRARY} (include ${RGA_INCLUDE_DIR})")
endif()

pybind11_add_module(mpp_player mpp_player.cpp)

target_include_directories(mpp_player PRIVATE
    ${AVCODEC_INCLUDE_DIRS}
    ${MPP_INCLUDE_DIRS}
    ${RGA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mpp_player PRIVATE
    ${AVCODEC_LIBRARIES}
    ${MPP_LIBRARIES}
    ${RGA_LIBRARY}
    ${OpenCV_LIBS}
    pthread
)

