#include <arm_sve.h>
#include <array>
#include <chrono>
#include <iostream>
#include <random>
#include "fc_weights.h"

// Linear layer: y = x * W^T + b
void linear_layer(const float* input, const float* weight, const float* bias,
                  float* output, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        const float* w_row = weight + o * in_dim;
        svfloat32_t vacc = svdup_f32(bias[o]);
        int i = 0;
        while (i < in_dim) {
            svbool_t pg = svwhilelt_b32(i, in_dim);
            svfloat32_t w = svld1(pg, &w_row[i]);
            svfloat32_t x = svld1(pg, &input[i]);
            vacc = svmla_f32_m(pg, vacc, w, x);
            i += svcntw();
        }
        output[o] = svaddv_f32(svptrue_b32(), vacc);
    }
}

// ReLU activation in-place
void relu_layer(float* data, int size) {
    svfloat32_t vzero = svdup_f32(0.0f);
    for (int i = 0; i < size; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, size);
        svfloat32_t x = svld1(pg, &data[i]);
        svfloat32_t y = svmax_f32_m(pg, x, vzero);
        svst1(pg, &data[i], y);
    }
}

int main() {
    constexpr int input_dim = 10;
    constexpr int hidden_dim = 20;
    constexpr int output_dim = 1;
    std::array<float, input_dim> input{};
    std::array<float, hidden_dim> hidden{};
    std::array<float, output_dim> output{};

    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : input) v = dist(gen);

    constexpr int runs = 100000;  // number of inferences for timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        linear_layer(input.data(), FC1_WEIGHT.data(), FC1_BIAS.data(),
                     hidden.data(), input_dim, hidden_dim);
        relu_layer(hidden.data(), hidden_dim);
        linear_layer(hidden.data(), FC2_WEIGHT.data(), FC2_BIAS.data(),
                     output.data(), hidden_dim, output_dim);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Total time for " << runs << " runs: " << elapsed.count()
              << " ms\n";
    std::cout << "Average inference time: " << (elapsed.count() / runs)
              << " ms\n";
    std::cout << "Output: " << output[0] << '\n';
    return 0;
}
