set(SOURCES mkl_wrapper.cpp mkl_wrapper.h loaders.cpp loaders.h main.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl=parallel")

set(MKLROOT ......)

include_directories(${MKLROOT}/include)

add_executable(mnist_mkl ${SOURCES})

target_link_libraries(mnist_mkl mnist
        tbb stdc++ pthread m dl)
