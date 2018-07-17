#include "numpy_loader.h"

#include <Eigen/Dense>
#include <stdint.h>
// copied from https://docs.scipy.org/doc/numpy/neps/npy-format.html

const int MAGIC_STRING_SIZE = 6;
const char MAGIC_STRING[] = "\x93NUMPY";
// Minimum header size
int MIN_HEADER_SIZE = 10;

#pragma pack(push, 1)
typedef struct NumpyHeader {
    int8_t magic_string[MAGIC_STRING_SIZE];
    uint8_t major_version;
    uint8_t minor_version;
    uint16_t header_data_len;
} NumpyHeader;
#pragma pack(pop)

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> LayerRMMap;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> BiasRMMap;

bool verify_numpy_header(void *file_map,
        int64_t filesize,
        long min_data_size,
        long *data_offset)
{
    if (filesize < MIN_HEADER_SIZE) {
        return false;
    }

    NumpyHeader *header = (NumpyHeader *) file_map;
    for (int i = 0; i < MAGIC_STRING_SIZE; i++) {
        if (header->magic_string[i] != MAGIC_STRING[i]) {
            return false;
        }
    }
    uint16_t header_len = header->header_data_len;

    if (filesize < (MIN_HEADER_SIZE + header_len + min_data_size)) {
        return false;
    }

    *data_offset = MIN_HEADER_SIZE + (int)header_len;

    return true;
}

float* get_offset(void *addr, size_t offset)
{
    char *weights_start = ((char *)addr) + offset;
    return (float *)weights_start;
}

int load_numpy_layer(void *weights_file,
               int64_t weights_filesize,
               void *biases_file,
               int64_t biases_filesize,
               int input_size,
               int output_size,
               Layer *layer)
{
    long weights_offset;
    long biases_offset;
    bool weights_valid = verify_numpy_header(weights_file,
            weights_filesize,
            input_size * output_size,
            &weights_offset);
    bool biases_valid = verify_numpy_header(biases_file,
            biases_filesize,
            output_size,
            &biases_offset);

    if (!weights_valid || !biases_valid) {
        return -1;
    }

    float *weights_data = get_offset(weights_file, (size_t)weights_offset);

    LayerRMMap weights_map(weights_data, output_size, input_size);
    layer->weight = new Weight(weights_map);

    float *biases_data = get_offset(biases_file, (size_t)biases_offset);
    BiasRMMap biases_map(biases_data, output_size);
    layer->bias = new Bias(biases_map);

    return 0;
}