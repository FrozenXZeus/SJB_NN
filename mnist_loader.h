#pragma once

#ifndef SJBWL_MNIST_LOADER_H
#define SJBWL_MNIST_LOADER_H

#include <stdint.h>

// Eigen
// Unfortunately needed to add Eigen Return Type
#include <Eigen/Dense>
#include "mnist_definitions.h"

constexpr int MNIST_SUCCESS = 0;
constexpr int MNIST_IMAGE_INVALID = -1;
constexpr int MNIST_LABEL_INVALID = -2;
constexpr int MNIST_LABEL_IMAGE_SIZE_MISMATCH = -3;

typedef Eigen::Matrix<float, MNIST_PIXELS, Eigen::Dynamic> MNIST_INPUT;
typedef Eigen::VectorXi MNIST_LABEL;

int load_mnist_data(void *image_file,
                    int64_t img_filesize,
                    void *label_file,
                    int64_t label_filesize,
                    MNIST_INPUT **input,
                    MNIST_LABEL **label);


#endif //SJBWL_MNIST_LOADER_H