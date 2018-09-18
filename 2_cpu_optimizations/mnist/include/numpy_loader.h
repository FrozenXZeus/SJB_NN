#pragma once

#ifndef SJB_NN_NUMPY_LOADER_H
#define SJB_NN_NUMPY_LOADER_H

#include <stdint.h>
#include "mnist_definitions.h"

bool verify_numpy_header(void *file_map,
                         int64_t filesize,
                         long min_data_size,
                         long *data_offset);


#endif //SJB_NN_NUMPY_LOADER_H
