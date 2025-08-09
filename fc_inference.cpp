#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <algorithm>

std::vector<float> read_data(const std::string &path, size_t expected) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open " << path << std::endl;
        std::exit(1);
    }
    std::vector<float> data;
    float v;
    while (in >> v) {
        data.push_back(v);
    }
    if (data.size() != expected) {
        std::cerr << "Unexpected size for " << path << " (got " << data.size()
                  << ", expected " << expected << ")" << std::endl;
        std::exit(1);
    }
    return data;
}

int main(int argc, char **argv) {
    const int input_dim = 10;
    const int hidden_dim = 20;
    const int output_dim = 1;
    int iters = 10000;
    if (argc > 1) iters = std::stoi(argv[1]);

    auto fc1_w = read_data("fc1_weight.txt", hidden_dim * input_dim);
    auto fc1_b = read_data("fc1_bias.txt", hidden_dim);
    auto fc2_w = read_data("fc2_weight.txt", output_dim * hidden_dim);
    auto fc2_b = read_data("fc2_bias.txt", output_dim);
    auto input = read_data("input.txt", input_dim);

    std::vector<float> hidden(hidden_dim);
    std::vector<float> output(output_dim);

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iters; ++iter) {
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = fc1_b[i];
            for (int j = 0; j < input_dim; ++j) {
                sum += fc1_w[i * input_dim + j] * input[j];
            }
            hidden[i] = std::max(0.0f, sum);
        }
        for (int i = 0; i < output_dim; ++i) {
            float sum = fc2_b[i];
            for (int j = 0; j < hidden_dim; ++j) {
                sum += fc2_w[i * hidden_dim + j] * hidden[j];
            }
            output[i] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    std::ofstream out_file("cpp_output.txt");
    for (float v : output) out_file << v << " ";
    std::cout << (total_us / iters) << std::endl;
    return 0;
}

