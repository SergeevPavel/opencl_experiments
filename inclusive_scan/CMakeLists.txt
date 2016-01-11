project(inclusive_scan)
cmake_minimum_required(VERSION 2.8)
SET( CMAKE_MODULE_PATH ${inclusive_scan_SOURCE_DIR})
find_package(OPENCL)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Debug)

TARGET_LINK_LIBRARIES (
   ${PROJECT_NAME}
   ${cllib}
   ${OPENCL_LIBRARIES}
)
