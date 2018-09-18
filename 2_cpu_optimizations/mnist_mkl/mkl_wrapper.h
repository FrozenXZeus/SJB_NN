#pragma once

#ifndef SJB_NN_LAYER_H
#define SJB_NN_LAYER_H

#include <cstddef>

enum MKLMATRIX_RETURN {
    MKLMATRIX_SUCCESS = 0,
    MKLMATRIX_ALLOCATION_FAILED
};

enum FORWARD_PROP {
    FORWARD_PROP_SUCCESS = 0,
    FORWARD_PROP_MISMATCHED_DIMENSIONS,
    FORWARD_PROP_ALLOCATION_FAILURE
};

class MKLMatrix {
public:
    size_t width;
    size_t height;
    float *data;

    MKLMatrix() : width(), height(), data(nullptr) {}

    int allocate(size_t width, size_t height);
    MKLMatrix copy();

    ~MKLMatrix();
};

class MKLLabel {
public:
    size_t length;
    unsigned char *data;

    MKLLabel() : length(), data(nullptr) {}

    int allocate(size_t length);

    ~MKLLabel();
};

int forward_prop(MKLMatrix &input, MKLMatrix &weights, MKLMatrix &biases, MKLMatrix &out);
void relu(float *array, size_t size);
void softmax(float *array, size_t width, size_t height);
void argmax(float *data_in, size_t width, size_t height, int *results);

void warmup();

#endif //SJB_NN_LAYER_H
