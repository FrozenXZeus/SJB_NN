#include <stdio.h>
#include <chrono>
#include <string>
#include <memory>
#include <bits/unique_ptr.h>

#include "loaders.h"

#include "mkl_wrapper.h"
#include "mnist/include/mnist_definitions.h"
#include "mnist/include/mmap.h"

#define TIMESTAMP(var) auto var = std::chrono::high_resolution_clock::now()

const char MNIST_TRAIN_PATH[] = "/home/herp/deep_learning_datasets/fashion_mnist/train/";
const char TRAIN_IMAGES_NAME[] = "train-images-idx3-ubyte";
const char TRAIN_LABELS_NAME[] = "train-labels-idx1-ubyte";

const char MNIST_TEST_PATH[] = "/home/herp/deep_learning_datasets/fashion_mnist/test/";
const char TEST_IMAGES_NAME[] = "t10k-images-idx3-ubyte";
const char TEST_LABELS_NAME[] = "t10k-labels-idx1-ubyte";

const char LAYER1_PATH[] = "/home/herp/CLionProjects/sjb_nn/l1/";
const char LAYER2_PATH[] = "/home/herp/CLionProjects/sjb_nn/l2/";
const char WEIGHTS_FILENAME[] = "W.npy";
const char BIASES_FILENAME[] = "b.npy";

enum LAYER_LOAD_STATUS {
    LAYER_LOAD_SUCEESS = 0,
    LAYER_LOAD_FAIL
};


int load_mnist(const char *mnist_folder,
               const char *images_name,
               const char *labels_name,
               MKLMatrix &input,
               MKLLabel &label)
{
    std::string mnist_folder_path(mnist_folder);
    std::string img_path = mnist_folder_path + images_name;
    std::string label_path = mnist_folder_path + labels_name;

    DumbMMap img_map(img_path.c_str());
    DumbMMap label_map(label_path.c_str());

    void *img_mapping = img_map.open_map();
    void *label_mapping = label_map.open_map();
    if (img_mapping == nullptr || label_mapping == nullptr) {
        return LAYER_LOAD_FAIL;
    }

    int res = load_mnist_data(img_mapping,
                              img_map.get_file_size(),
                              label_mapping,
                              label_map.get_file_size(),
                              input,
                              label);

    if (res != 0) {
        return LAYER_LOAD_FAIL;
    }

    return LAYER_LOAD_SUCEESS;
}

int load_layer(const char *layer_folder,
               const char *weights_name,
               const char *biases_name,
               int input_size,
               int output_size,
               MKLMatrix &weights,
               MKLMatrix &biases)
{
    std::string weights_folder(layer_folder);
    std::string weight_filename = weights_folder + weights_name;
    std::string biases_filename = weights_folder + biases_name;

    DumbMMap weight_layer1_map(weight_filename.c_str());
    DumbMMap bias_layer1_map(biases_filename.c_str());

    void *weights_mapping = weight_layer1_map.open_map();
    void *bias_mapping = bias_layer1_map.open_map();

    if (weights_mapping == nullptr || bias_mapping == nullptr) {
        return LAYER_LOAD_FAIL;
    }

    int res = load_numpy_layer(weights_mapping,
                               weight_layer1_map.get_file_size(),
                               bias_mapping,
                               bias_layer1_map.get_file_size(),
                               input_size,
                               output_size,
                               weights,
                               biases);

    if (res != 0) {
        return LAYER_LOAD_FAIL;
    }

    return LAYER_LOAD_SUCEESS;
}



void do_mnist() {
    MKLMatrix mnist_input;
    MKLLabel mnist_label;

    TIMESTAMP(timer_start);

    int res = load_mnist(MNIST_TEST_PATH,
                         TEST_IMAGES_NAME,
                         TEST_LABELS_NAME,
                         mnist_input,
                         mnist_label);

    if (res != LAYER_LOAD_SUCEESS) {
        return;
    }

    TIMESTAMP(timer_dataset_loaded);

    MKLMatrix layer1_weights;
    MKLMatrix layer1_biases;
    MKLMatrix layer2_weights;
    MKLMatrix layer2_biases;

    res = load_layer(LAYER1_PATH,
                     WEIGHTS_FILENAME,
                     BIASES_FILENAME,
                     MNIST_PIXELS,
                     HIDDEN_SIZE,
                     layer1_weights,
                     layer1_biases);

    if (res != LAYER_LOAD_SUCEESS) {
        return;
    }

    res = load_layer(LAYER2_PATH,
                    WEIGHTS_FILENAME,
                    BIASES_FILENAME,
                    HIDDEN_SIZE,
                    OUTPUT_SIZE,
                    layer2_weights,
                    layer2_biases);

    if (res != LAYER_LOAD_SUCEESS) {
        return;
    }


    TIMESTAMP(timer_weights_loaded);
    // neural nets, yay
    MKLMatrix layer1_output;
    MKLMatrix layer2_output;

    res = forward_prop(mnist_input, layer1_weights, layer1_biases, layer1_output);
    if (res != FORWARD_PROP_SUCCESS) {
        return;
    }
    relu(layer1_output.data, layer1_output.width * layer1_output.height);
    res = forward_prop(layer1_output, layer2_weights, layer2_biases, layer2_output);
    if (res != FORWARD_PROP_SUCCESS) {
        return;
    }

    TIMESTAMP(timer_forward_prop);

    MKLMatrix final_result = layer2_output.copy();
    softmax(final_result.data, final_result.width, final_result.height);

    std::unique_ptr<int[]> result_vector = std::make_unique<int[]>(final_result.height);

    argmax(final_result.data, final_result.width, final_result.height, result_vector.get());

    TIMESTAMP(timer_predictions_made);

    printf("First 10 predictions\n");
    // print the first 10 predictions
    for (int i = 0; i < 10; i++) {
        printf("%d\t", result_vector[i]);
    }
    printf("\n");

    int correct_predictions = 0;
    for (unsigned int i = 0; i < mnist_label.length; ++i) {
        if (mnist_label.data[i] == result_vector[i]) {
            ++correct_predictions;
        }
    }
    auto accuracy = (double)correct_predictions * 100 / mnist_label.length;
    printf("Total Accuracy: %4.2f%% \n", accuracy);

    // Print timer info
    auto total_time = timer_predictions_made - timer_start;
    auto dataset_load = timer_dataset_loaded - timer_start;
    auto weight_load = timer_weights_loaded - timer_dataset_loaded;
    auto forward_pass = timer_forward_prop - timer_weights_loaded;
    auto prediction_time = timer_predictions_made - timer_forward_prop;
    constexpr intmax_t MILLISECOND_RESOLUTION = 1000000L;

    printf("Total time taken: %ld milliseconds\n", total_time.count()/MILLISECOND_RESOLUTION);
    printf("Dataset load:\t %ld milliseconds\n", dataset_load.count()/MILLISECOND_RESOLUTION);
    printf("Weight load:\t %ld milliseconds\n", weight_load.count()/MILLISECOND_RESOLUTION);
    printf("Forward pass:\t %ld milliseconds\n", forward_pass.count()/MILLISECOND_RESOLUTION);
    printf("Prediction:\t\t %ld milliseconds\n", prediction_time.count()/MILLISECOND_RESOLUTION);
}

int main() {
    warmup();
    do_mnist();
}