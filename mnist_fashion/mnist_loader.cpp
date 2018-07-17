#include "mnist_loader.h"

// Linux only
#include <byteswap.h>

#include "mnist_definitions.h"
#include "mmap.h"

bool image_file_valid(void *image_file, int64_t img_filesize) {
    if (img_filesize < IMAGE_HEADER_SIZE) {
        return false;
    }

    uint32_t *img_header = (uint32_t *)image_file;

    uint32_t img_magic = bswap_32(*img_header);

    if (img_magic != MNIST_IMAGE_MAGIC) {
        return false;
    }

    uint32_t image_count = bswap_32(img_header[1]);
    uint32_t rows = bswap_32(img_header[2]);
    uint32_t cols = bswap_32(img_header[3]);

    if (rows != MNIST_ROWS || cols != MNIST_COLS) {
        return false;
    }

    // verify MNIST size
    int32_t theoretical_size = (int32_t)rows * (int32_t)cols * (int32_t)image_count + 16;
    if (img_filesize < theoretical_size) {
        return false;
    }

    return true;
}

bool label_file_valid(void *label_file, int64_t label_filesize) {
    if (label_filesize < LABEL_HEADER_SIZE) {
        return false;
    }

    uint32_t *label_header = (uint32_t*)label_file;

    uint32_t label_magic = bswap_32(*label_header);

    if (label_magic != MNIST_LABEL_MAGIC) {
        return false;
    }

    uint32_t label_size = bswap_32(label_header[1]);

    int32_t theoretical_label_size = label_size + 8;
    if (label_filesize < theoretical_label_size) {
        return false;
    }

    return true;
}

int load_mnist_data(void *image_file,
                    int64_t img_filesize,
                    void *label_file,
                    int64_t label_filesize,
                    MNIST_INPUT **input,
                    MNIST_LABEL **label) {
    if (!image_file_valid(image_file, img_filesize)) {
        return MNIST_IMAGE_INVALID;
    }

    if (!label_file_valid(label_file, label_filesize)) {
        return MNIST_LABEL_INVALID;
    }

    auto img_header = (uint32_t *)image_file;
    auto label_header = (uint32_t*)label_file;

    uint32_t image_count = bswap_32(img_header[1]);
    uint32_t label_count = bswap_32(label_header[1]);

    if (image_count != label_count) {
        return MNIST_LABEL_IMAGE_SIZE_MISMATCH;
    }

    // Load image
    // due to some annoying type conversions, have to use loops
    MNIST_INPUT *image_matrix = new MNIST_INPUT(MNIST_PIXELS, image_count);
    unsigned char *image_start = ((unsigned char *)image_file) + IMAGE_HEADER_SIZE;
    for (int i = 0; i < image_count; ++i) {
        for (int j = 0; j < MNIST_PIXELS; ++j) {
            (*image_matrix)(j, i) = (float)*image_start / 255.0;
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