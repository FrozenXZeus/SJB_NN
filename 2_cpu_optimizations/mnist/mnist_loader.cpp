#include "include/mnist_loader.h"

// Linux only
#include <byteswap.h>

#include "include/mnist_definitions.h"
#include "include/mmap.h"

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

