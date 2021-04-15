#cmake_minimum_required(VERSION 3.15)
project(acv)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(mainTp1 mainTp1.cpp)
target_link_libraries(mainTp1 ${OpenCV_LIBS})