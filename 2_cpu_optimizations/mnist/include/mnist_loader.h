#pragma once

#ifndef SJBWL_MNIST_LOADER_H
#define SJBWL_MNIST_LOADER_H

#include <stdint.h>

#include "mnist_definitions.h"

constexpr int MNIST_SUCCESS = 0;
constexpr int MNIST_IMAGE_INVALID = -1;
constexpr int MNIST_LABEL_INVALID = -2;
constexpr int MNIST_LABEL_IMAGE_SIZE_MISMATCH = -3;

bool image_file_valid(void *image_file, int64_t img_filesize);
bool label_file_valid(void *label_file, int64_t label_filesize);

#endif //SJBWL_MNIST_LOADER_H