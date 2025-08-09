cmake_minimum_required(VERSION 3.16)
project(fc_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fc_inference fc_inference.cpp)
# Require SVE2 support during compilation
set_target_properties(fc_inference PROPERTIES
    COMPILE_FLAGS "-O3 -march=armv9-a+sve2"
)
