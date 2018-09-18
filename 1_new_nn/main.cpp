#include <stdio.h>
#include <string.h>

#include <string>
#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "mnist_definitions.h"
#include "mnist_loader.h"
#include "numpy_loader.h"
#include "mmap.h"

const char MNIST_TRAIN_PATH[] = "";
const char TRAIN_IMAGES_NAME[] = "train-images-idx3-ubyte";
const char TRAIN_LABELS_NAME[] = "train-labels-idx1-ubyte";

const char MNIST_TEST_PATH[] = "";
const char TEST_IMAGES_NAME[] = "t10k-images-idx3-ubyte";
const char TEST_LABELS_NAME[] = "t10k-labels-idx1-ubyte";

const char LAYER1_PATH[] = "";
const char LAYER2_PATH[] = "";
const char WEIGHTS_FILENAME[] = "W.npy";
const char BIASES_FILENAME[] = "b.npy";

constexpr int LOAD_SUCCESS = 0;
constexpr int LOAD_FAIL = -1;

int load_mnist(const char *mnist_folder,
               const char *images_name,
               const char *labels_name,
               MNIST_INPUT **input,
               MNIST_LABEL **label) {
    std::string mnist_folder_path(mnist_folder);
    std::string img_path = mnist_folder_path + images_name;
    std::string label_path = mnist_folder_path + labels_name;

    LinuxMMap img_map(img_path.c_str());
    LinuxMMap label_map(label_path.c_str());

    void *img_mapping = img_map.open_map();
    void *label_mapping = label_map.open_map();
    if (img_mapping == nullptr || label_mapping == nullptr) {
        return LOAD_FAIL;
    }

    int res = load_mnist_data(img_mapping,
                              img_map.get_file_size(),
                              label_mapping,
                              label_map.get_file_size(),
                              input,
                              label);

    if (res != 0) {
        return LOAD_FAIL;
    }

    return LOAD_SUCCESS;
}

int load_layer(const char *layer_folder,
        const char *weights_name,
        const char *biases_name,
        int input_size,
        int output_size,
        Layer *layer) {
    std::string weights_folder(layer_folder);
    std::string weight_filename = weights_folder + weights_name;
    std::string biases_filename = weights_folder + biases_name;

    LinuxMMap weight_layer1_map(weight_filename.c_str());
    LinuxMMap bias_layer1_map(biases_filename.c_str());

    void *weights_mapping = weight_layer1_map.open_map();
    void *bias_mapping = bias_layer1_map.open_map();

    if (weights_mapping == nullptr || bias_mapping == nullptr) {
        return LOAD_FAIL;
    }

    int res = load_numpy_layer(weights_mapping,
                           weight_layer1_map.get_file_size(),
                           bias_mapping,
                           bias_layer1_map.get_file_size(),
                           input_size,
                           output_size,
                           layer);
    if (res != 0) {
        return LOAD_FAIL;
    }

    return LOAD_SUCCESS;
}

void do_mnist() {
    MNIST_INPUT *image_matrix = nullptr;
    MNIST_LABEL *label_vector = nullptr;
    int res = load_mnist(MNIST_TEST_PATH,
            TEST_IMAGES_NAME,
            TEST_LABELS_NAME,
            &image_matrix,
            &label_vector);

    if (res != LOAD_SUCCESS) {
        return;
    }

    Layer layer1;
    Layer layer2;
    res = load_layer(LAYER1_PATH,
            WEIGHTS_FILENAME,
            BIASES_FILENAME,
            MNIST_PIXELS,
            HIDDEN_SIZE,
            &layer1);

    if (res != 0) {
        printf("Problem occured when loading layer 1");
        return;
    }

    // Memory management;
    std::unique_ptr<MNIST_INPUT> image_matrix_managed(image_matrix);
    std::unique_ptr<MNIST_LABEL> label_vector_managed(label_vector);

    res = load_layer(LAYER2_PATH,
                     WEIGHTS_FILENAME,
                     BIASES_FILENAME,
                     HIDDEN_SIZE,
                     OUTPUT_SIZE,
                     &layer2);

    if (res != 0) {
        printf("Problem occured when loading layer 2");
        return;
    }

    // Let's do some neural networks, yay
    auto layer_1_res = (*layer1.weight) * (*image_matrix);
    auto layer_1_result = layer_1_res.colwise() + (*layer1.bias);

    // ReLU
    auto layer_1_relu = layer_1_result.cwiseMax(0);

    auto layer_2_res = (*layer2.weight) * layer_1_relu;
    auto layer_2_result = layer_2_res.colwise() + (*layer2.bias);

    // layer 2 results, before softmax
    auto final_exp = layer_2_result.array().exp();
    auto exp_sum = final_exp.colwise().sum();

    auto softmaxed_result = final_exp.rowwise() / exp_sum;

    auto final_result = softmaxed_result.eval();

    Eigen::VectorXi predictions(final_result.cols());

    // argmax
    for (int col = 0; col < final_result.cols(); ++col) {
        float max_seen = -1;
        int max_index = 0;
        for (int row = 0; row < final_result.rows(); ++row) {
            if (final_result(row, col) > max_seen) {
                max_seen = final_result(row, col);
                max_index = row;
            }
        }
        predictions(col, 0) = max_index;
    }

    printf("First 10 predictions\n");
    // print the first 10 predictions
    for (int i = 0; i < 10; i++) {
        printf("%d\t", predictions(i, 0));
    }
    printf("\n");

    auto correct_predictions = predictions.cwiseEqual(*label_vector).count();
    auto accuracy = (double)correct_predictions * 100/ final_result.cols();
    printf("Total Accuracy: %4.2f%%", accuracy);
}

int main() {
    do_mnist();
}
