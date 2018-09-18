#pragma once

#ifndef SJB_NN_LOADERS_H
#define SJB_NN_LOADERS_H

#include <Eigen/Dense>

#include "mnist/include/mnist_definitions.h"

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Weight;
typedef Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> Bias;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MNIST_INPUT;
typedef Eigen::VectorXi MNIST_LABEL;

typedef struct Layer {
    Weight *weight = nullptr;
    Bias *bias = nullptr;

    ~Layer() {
        if (weight != nullptr) {
            delete weight;
        }

        if (bias != nullptr) {
            delete bias;
        }
    }
} Layer;

int load_numpy_layer(void *weights_file,
                     int64_t weights_filesize,
                     void *biases_file,
                     int64_t biases_filesize,
                     int input_size,
                     int output_size,
                     Layer *layer);

int load_mnist_data(void *image_file,
                    long img_filesize,
                    void *label_file,
                    long label_filesize,
                    MNIST_INPUT **input,
                    MNIST_LABEL **label);

#endif //SJB_NN_LOADERS_H
