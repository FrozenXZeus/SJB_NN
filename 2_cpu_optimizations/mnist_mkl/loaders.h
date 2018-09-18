#pragma once

#ifndef SJB_NN_LOADERS_H
#define SJB_NN_LOADERS_H

#include <stdint.h>
#include "mkl_wrapper.h"

enum LOADING_STATUS {
    LOAD_SUCCESS = 0,
    LOAD_ALLOCATION_FAILED,
    LOAD_FILE_INVALID,
};

int load_mnist_data(void *image_file,
                    long img_filesize,
                    void *label_file,
                    long label_filesize,
                    MKLMatrix &input,
                    MKLLabel &label);

int load_numpy_layer(void *weights_file,
                     int64_t weights_filesize,
                     void *biases_file,
                     int64_t biases_filesize,
                     int input_size,
                     int output_size,
                     MKLMatrix &weights,
                     MKLMatrix &biases);


#endif //SJB_NN_LOADERS_H
