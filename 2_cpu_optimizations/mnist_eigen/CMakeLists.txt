set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")

set(SOURCES main.cpp loaders.cpp loaders.h)

add_executable(mnist_eigen ${SOURCES})

target_link_libraries(mnist_eigen mnist)