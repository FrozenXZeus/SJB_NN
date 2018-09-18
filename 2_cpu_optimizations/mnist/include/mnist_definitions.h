#pragma once

#ifndef SJB_NN_MNIST_DEFINITIONS_H
#define SJB_NN_MNIST_DEFINITIONS_H

#include <stdint.h>

constexpr uint32_t MNIST_LABEL_MAGIC = 0x00000801;
constexpr uint32_t MNIST_IMAGE_MAGIC = 0x00000803;

constexpr uint32_t MNIST_ROWS = 28;
constexpr uint32_t MNIST_COLS = 28;
constexpr uint32_t MNIST_PIXELS = MNIST_ROWS * MNIST_COLS;

constexpr int IMAGE_HEADER_SIZE = 16;
constexpr int LABEL_HEADER_SIZE = 8;

constexpr int HIDDEN_SIZE = 100;
constexpr int OUTPUT_SIZE = 10;

#endif //SJB_NN_MNIST_DEFINITIONS_H
