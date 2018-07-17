cmake_minimum_required(VERSION 3.11)
project(sjb_nn)

set(CMAKE_CXX_STANDARD 11)
set(EIGEN_DIRECTORY third_party/eigen-3.3.4_bin/include/eigen3)

set(INTEL_SIMD_EXTENSIONS "-mavx")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fno-exceptions -march=native -mtune=native -pipe -m64")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -debug all -O0 -traceback")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast ${INTEL_SIMD_EXTENSIONS}")

set(SOURCES
        main.cpp
        mnist_loader.cpp
        mmap.cpp
        numpy_loader.cpp
        numpy_loader.h
        mnist_loader.h
        mmap.h mnist_definitions.h)
# Include EIGEN
include_directories(${EIGEN_DIRECTORY})

add_executable(sjb_nn ${SOURCES})