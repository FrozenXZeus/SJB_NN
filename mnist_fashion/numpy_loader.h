#pragma once

#ifndef SJB_NN_NUMPY_LOADER_H
#define SJB_NN_NUMPY_LOADER_H

#include <stdint.h>
#include "mnist_definitions.h"

int load_numpy_layer(void *weights_file,
                     int64_t weights_filesize,
                     void *biases_file,
                     int64_t biases_filesize,
                     int input_size,
                     int output_size,
                     Layer *layer);

#endif //SJB_NN_NUMPY_LOADER_H
