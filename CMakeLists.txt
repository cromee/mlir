cmake_minimum_required(VERSION 3.10)
project(fc_avx512 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(fc_avx512_inference fc_avx512_inference.cpp)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
if (COMPILER_SUPPORTS_AVX512F)
    target_compile_options(fc_avx512_inference PRIVATE -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl)
else()
    message(FATAL_ERROR "Compiler does not support AVX512F")
endif()
