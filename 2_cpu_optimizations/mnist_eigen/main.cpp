#include <stdio.h>
#include <string.h>
#include <chrono>
#include <string>
#include <iostream>
#include <memory>

#include <Eigen/Dense>
#include <float.h>

#include "mnist/include/mmap.h"
#include "loaders.h"

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

    DumbMMap img_map(img_path.c_str());
    DumbMMap label_map(label_path.c_str());

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

    DumbMMap weight_layer1_map(weight_filename.c_str());
    DumbMMap bias_layer1_map(biases_filename.c_str());

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

    auto timer_start = std::chrono::high_resolution_clock::now();

    int res = load_mnist(MNIST_TEST_PATH,
            TEST_IMAGES_NAME,
            TEST_LABELS_NAME,
            &image_matrix,
            &label_vector);

    if (res != LOAD_SUCCESS) {
        return;
    }

    // Memory management hack;
    std::unique_ptr<MNIST_INPUT> image_matrix_managed(image_matrix);
    std::unique_ptr<MNIST_LABEL> label_vector_managed(label_vector);

    auto timer_dataset_loaded = std::chrono::high_resolution_clock::now();

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

    auto timer_weights_loaded = std::chrono::high_resolution_clock::now();

    // Let's do some neural networks, yay
    auto layer_1_relu = (((*image_matrix) * (layer1.weight->transpose())).rowwise() + (*layer1.bias)).cwiseMax(0);

    auto layer_2_res = ((layer_1_relu * layer2.weight->transpose())).rowwise() + (*layer2.bias);

    auto nn_output = layer_2_res.eval();

    auto timer_forward_prop = std::chrono::high_resolution_clock::now();

    // layer 2 results, before softmax
    auto final_exp = nn_output.array().exp();
    auto exp_sum = final_exp.rowwise().sum();

    auto softmaxed_result = final_exp.colwise() / exp_sum;

    auto final_result = softmaxed_result.eval();

    Eigen::VectorXi predictions(final_result.rows());

    // argmax
    for (int row = 0; row < final_result.rows(); row++) {
        float max_seen = FLT_MIN;
        int max_index = 0;
        for (int col = 0; col < final_result.cols(); col++) {
            if (final_result(row, col) > max_seen) {
                max_seen = final_result(row, col);
                max_index = col;
            }
        }
        predictions(row, 0) = max_index;
    }

    auto timer_predictions_made = std::chrono::high_resolution_clock::now();

    printf("First 10 predictions\n");
    // print the first 10 predictions
    for (int i = 0; i < 10; i++) {
        printf("%d\t", predictions(i, 0));
    }
    printf("\n");

    auto correct_predictions = predictions.cwiseEqual(*label_vector).count();
    auto accuracy = (double)correct_predictions * 100/ final_result.rows();
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

void warmup() {
    Eigen::MatrixXf a = Eigen::MatrixXf::Random(1000,1000);
    Eigen::MatrixXf b = Eigen::MatrixXf::Random(1000,1000);
    auto c = a*b;
    // print it out just in case compilers try to dead code eliminate this
    printf("Warmup: %f\n", c.sum());
}

int main() {
    warmup();
    do_mnist();
}