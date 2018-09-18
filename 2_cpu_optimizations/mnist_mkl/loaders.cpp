#include "loaders.h"

#include <byteswap.h>
#include <string.h>

#include "mnist/include/numpy_loader.h"
#include "mnist/include/mnist_loader.h"

float* get_offset(void *addr, size_t offset)
{
    char *weights_start = ((char *)addr) + offset;
    return (float *)weights_start;
}

int load_mnist_data(void *image_file,
                    long img_filesize,
                    void *label_file,
                    long label_filesize,
                    MKLMatrix &input,
                    MKLLabel &label)
{
    if (!image_file_valid(image_file, img_filesize)) {
        return LOAD_FILE_INVALID;
    }

    if (!label_file_valid(label_file, label_filesize)) {
        return LOAD_FILE_INVALID;
    }

    auto img_header = (unsigned int *)image_file;
    auto label_header = (unsigned int *)label_file;

    unsigned int image_count = bswap_32(img_header[1]);
    unsigned int label_count = bswap_32(label_header[1]);

    if (image_count != label_count) {
        return LOAD_FILE_INVALID;
    }

    // Load image
    // due to some annoying type conversions, have to use loops
    int res = input.allocate(MNIST_PIXELS, image_count);
    if (res != MKLMATRIX_SUCCESS) {
        return LOAD_ALLOCATION_FAILED;
    }

    unsigned char *image_start = ((unsigned char *)image_file) + IMAGE_HEADER_SIZE;
    size_t total_pixels = MNIST_PIXELS * image_count;
    for (int i = 0; i < total_pixels; i++) {
        input.data[i] = ((float)image_start[i]) / 255.0f;
    }


    // load labels
    unsigned char *label_start = ((unsigned char *)label_file) + LABEL_HEADER_SIZE;
    res = label.allocate(label_count);
    if (res != MKLMATRIX_SUCCESS) {
        return LOAD_ALLOCATION_FAILED;
    }

    memcpy(label.data, label_start, sizeof(uint8_t) * image_count);

    return LOAD_SUCCESS;
}

int load_numpy_layer(void *weights_file,
                     int64_t weights_filesize,
                     void *biases_file,
                     int64_t biases_filesize,
                     int input_size,
                     int output_size,
                     MKLMatrix &weights,
                     MKLMatrix &biases)
{
    long weights_offset;
    long biases_offset;
    bool weights_valid = verify_numpy_header(weights_file,
                                             weights_filesize,
                                             input_size * output_size,
                                             &weights_offset);
    if (!weights_valid ) {
        return LOAD_FILE_INVALID;
    }

    bool biases_valid = verify_numpy_header(biases_file,
                                            biases_filesize,
                                            output_size,
                                            &biases_offset);
    if (!biases_valid) {
        return LOAD_FILE_INVALID;
    }

    int res = weights.allocate(input_size, output_size);
    if (res != MKLMATRIX_SUCCESS) {
        return LOAD_ALLOCATION_FAILED;
    }

    res = biases.allocate(output_size, 1);
    if (res != MKLMATRIX_SUCCESS) {
        return LOAD_ALLOCATION_FAILED;
    }

    // we are going to assume row major storage here
    // numpy stores in row major by default, simple copy should work
    float *weights_data = get_offset(weights_file, (size_t)weights_offset);
    memcpy(weights.data, weights_data, sizeof(float)*input_size*output_size);

    float *biases_data = get_offset(biases_file, (size_t)biases_offset);
    memcpy(biases.data, biases_data, sizeof(float)*output_size);

    return 0;
}