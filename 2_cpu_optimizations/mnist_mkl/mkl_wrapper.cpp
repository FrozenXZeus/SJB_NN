#include "mkl_wrapper.h"

#include <math.h>
#include <string.h>
#include "mkl.h"

constexpr int INTEL_CACHE_SIZE = 64;

int MKLMatrix::allocate(size_t width, size_t height) {
    float *data_allocated = (float *)mkl_malloc(width * height * sizeof(float), INTEL_CACHE_SIZE);
    if (data_allocated == NULL) {
        return MKLMATRIX_ALLOCATION_FAILED;
    }
    data = data_allocated;
    this->width = width;
    this->height = height;
    return MKLMATRIX_SUCCESS;
}

MKLMatrix::~MKLMatrix() {
    if (data != nullptr) {
        mkl_free(data);
    }
}

MKLMatrix MKLMatrix::copy() {
    MKLMatrix mat;
    mat.allocate(width, height);
    memcpy(mat.data, data, sizeof(float)*width*height);
    return mat;
}


int MKLLabel::allocate(size_t length) {
    unsigned char *data_allocated = (unsigned char*) mkl_malloc(length * sizeof(unsigned char), INTEL_CACHE_SIZE);
    if (data_allocated == NULL) {
        return MKLMATRIX_ALLOCATION_FAILED;
    }
    data = data_allocated;
    this->length = length;
    return MKLMATRIX_SUCCESS;
}

MKLLabel::~MKLLabel() {
    if (data != nullptr) {
        mkl_free(data);
    }
}

// dumb version for now, could probably use multithreading for this
// cbf at this current stage, considering matrix multiplication is the main bottleneck here
void relu(float *array, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        array[i] = fmaxf(array[i], 0.0f);
    }
}

// another dumb version, also cbf improving now
// note: chainer has some interesting code that first of all makes all result (-inf, 0]
// Presumably since exp(0) = 1, this is a nice safeguard against overflow
void softmax(float *array, size_t width, size_t height) {
    float *row_ptr = array;
    for (size_t row = 0; row < height; ++row) {
        // find the max
        float row_max = 0.0f;
        for (size_t col = 0; col < width; ++col) {
            if (row_ptr[col] > row_max) {
                row_max = row_ptr[col];
            }
        }

        for (size_t col = 0; col < width; ++col) {
            row_ptr[col] -= row_max;
        }

        row_ptr += width;
    }

    vsExp(width*height, array, array);

    row_ptr = array;
    for (size_t row = 0; row < height; ++row) {
        float row_total = 0.0f;
        for (size_t col = 0; col < width; ++col) {
            row_total += row_ptr[col];
        }

        for (size_t col = 0; col < width; ++col) {
            row_ptr[col] /= row_total;
        }
        row_ptr += width;
    }
}

void argmax(float *data_in, size_t width, size_t height, int *results) {
    float *row_ptr = data_in;
    int *res_ptr = results;
    for (size_t row = 0; row < height; ++row) {
        int max_index = 0;
        float max_seen = *row_ptr;
        for (size_t col = 0; col < width; ++col) {
            if (row_ptr[col] > max_seen) {
                max_seen = row_ptr[col];
                max_index = col;
            }
        }

        *res_ptr = max_index;

        row_ptr += width;
        ++res_ptr;
    }
}

int forward_prop(MKLMatrix &input, MKLMatrix &weights, MKLMatrix &biases, MKLMatrix &out) {
    if (input.width != weights.width) {
        return FORWARD_PROP_MISMATCHED_DIMENSIONS;
    }

    if (weights.height != biases.width) {
        return FORWARD_PROP_MISMATCHED_DIMENSIONS;
    }

    // allocate output
    int res = out.allocate(biases.width, input.height);
    if (res != MKLMATRIX_SUCCESS) {
        return FORWARD_PROP_ALLOCATION_FAILURE;
    }

    // pad the allocated array with result, to add later on
    float *output_array = out.data;
    for (int i = 0; i < input.height; i++) {
        memcpy(output_array, biases.data, sizeof(float) * biases.width);
        output_array += biases.width;
    }

    cblas_sgemm(CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            input.height,
            weights.height,
            input.width,
            1,
            input.data,
            input.width,
            weights.data,
            input.width,
            1,
            out.data,
            weights.height);

    return FORWARD_PROP_SUCCESS;
}

// TODO
// turns out, doing a matrix multiplication does some kind of initialization
// Running Intel VTune didn't really give any insights, eh, whatever
void warmup()
{
    float *a = (float *)mkl_malloc(100*sizeof(float), INTEL_CACHE_SIZE);
    float *b = (float *)mkl_malloc(100*sizeof(float), INTEL_CACHE_SIZE);
    float *c = (float *)mkl_malloc(100*sizeof(float), INTEL_CACHE_SIZE);

    // lol what is initialization
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                10,
                10,
                10,
                1,
                a,
                10,
                b,
                10,
                0,
                c,
                10);

    mkl_free(a);
    mkl_free(b);
    mkl_free(c);
}