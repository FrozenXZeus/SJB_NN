set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")

set(SOURCES
        mmap.cpp
        mnist_loader.cpp
        numpy_loader.cpp
        include/mmap.h
        include/mnist_definitions.h
        include/mnist_loader.h
        include/numpy_loader.h)

add_library(mnist STATIC ${SOURCES})