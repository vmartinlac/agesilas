
cmake_minimum_required(VERSION 3.0)
project(agesilas)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
add_executable(agesilas main.cpp)
target_link_libraries(agesilas PUBLIC opencv_imgcodecs opencv_highgui)

