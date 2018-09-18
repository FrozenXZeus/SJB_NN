#include <Eigen/Dense>
#include <byteswap.h>

#include "loaders.h"
#include "mnist/include/mnist_loader.h"
#include "mnist/include/numpy_loader.h"

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> LayerRMMap;
typedef Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> BiasRMMap;


float* get_offset(void *addr, size_t offset)
{
    char *weights_start = ((char *)addr) + offset;
    return (float *)weights_start;
}

int load_mnist_data(void *image_file,
                    long img_filesize,
                    void *label_file,
                    long label_filesize,
                    MNIST_INPUT **input,
                    MNIST_LABEL **label) {
    if (!image_file_valid(image_file, img_filesize)) {
        return MNIST_IMAGE_INVALID;
    }

    if (!label_file_valid(label_file, label_filesize)) {
        return MNIST_LABEL_INVALID;
    }

    auto img_header = (unsigned int *)image_file;
    auto label_header = (unsigned int *)label_file;

    unsigned int image_count = bswap_32(img_header[1]);
    unsigned int label_count = bswap_32(label_header[1]);

    if (image_count != label_count) {
        return MNIST_LABEL_IMAGE_SIZE_MISMATCH;
    }

    // Load image
    // due to some annoying type conversions, have to use loops
    MNIST_INPUT *image_matrix = new MNIST_INPUT(image_count, MNIST_PIXELS);
    unsigned char *image_start = ((unsigned char *)image_file) + IMAGE_HEADER_SIZE;
    for (int i = 0; i < image_count; ++i) {
        for (int j = 0; j < MNIST_PIXELS; ++j) {
            (*image_matrix)(i, j) = (float)*image_start / 255.0;
            ++image_start;
        }
    }

    // load labels
    unsigned char *label_start = ((unsigned char *)label_file) + LABEL_HEADER_SIZE;
    MNIST_LABEL *label_vector = new MNIST_LABEL(label_count, 1);
    for (int i = 0; i < label_count; i++) {
        (*label_vector)(i, 0) = (int)*label_start;
        ++label_start;
    }

    *input = image_matrix;
    *label = label_vector;

    return MNIST_SUCCESS;
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