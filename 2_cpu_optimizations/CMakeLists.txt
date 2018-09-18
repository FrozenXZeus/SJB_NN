cmake_minimum_required(VERSION 3.11)
project(sjb_nn)

set(CMAKE_CXX_STANDARD 14)
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(EIGEN_DIRECTORY ......)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -fno-exceptions -xHost -pipe -m64")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -debug all -O0 -traceback")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")

# Include EIGEN
include_directories(${EIGEN_DIRECTORY})
include_directories(${sjb_nn_SOURCE_DIR})

add_subdirectory(mnist)
add_subdirectory(mnist_eigen)
add_subdirectory(mnist_mkl)
