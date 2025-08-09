#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// Linear layer implemented with AVX-512 intrinsics
void linear_avx512(const float* input, const float* weight, const float* bias,
                   float* output, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; ++o) {
        __m512 sum = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= in_dim; i += 16) {
            __m512 w = _mm512_loadu_ps(weight + o * in_dim + i);
            __m512 x = _mm512_loadu_ps(input + i);
            sum = _mm512_fmadd_ps(w, x, sum);
        }
        if (i < in_dim) {
            __mmask16 mask = (1u << (in_dim - i)) - 1;
            __m512 w = _mm512_maskz_loadu_ps(mask, weight + o * in_dim + i);
            __m512 x = _mm512_maskz_loadu_ps(mask, input + i);
            sum = _mm512_fmadd_ps(w, x, sum);
        }
        float temp[16];
        _mm512_storeu_ps(temp, sum);
        float total = bias[o];
        for (int t = 0; t < 16; ++t) total += temp[t];
        output[o] = total;
    }
}

// ReLU activation using AVX-512
void relu_avx512(float* data, int len) {
    __m512 zero = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 x = _mm512_loadu_ps(data + i);
        __m512 y = _mm512_max_ps(x, zero);
        _mm512_storeu_ps(data + i, y);
    }
    if (i < len) {
        __mmask16 mask = (1u << (len - i)) - 1;
        __m512 x = _mm512_maskz_loadu_ps(mask, data + i);
        __m512 y = _mm512_max_ps(x, zero);
        _mm512_mask_storeu_ps(data + i, mask, y);
    }
}

int main() {
    constexpr int input_dim = 10;
    constexpr int hidden_dim = 20;
    constexpr int output_dim = 1;

    std::vector<float> input(input_dim);
    std::vector<float> fc1_w(hidden_dim * input_dim);
    std::vector<float> fc1_b(hidden_dim);
    std::vector<float> fc2_w(output_dim * hidden_dim);
    std::vector<float> fc2_b(output_dim);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : input) v = dist(rng);
    for (auto &v : fc1_w) v = dist(rng);
    for (auto &v : fc1_b) v = dist(rng);
    for (auto &v : fc2_w) v = dist(rng);
    for (auto &v : fc2_b) v = dist(rng);

    std::vector<float> hidden(hidden_dim);
    std::vector<float> output(output_dim);

    const int iterations = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        linear_avx512(input.data(), fc1_w.data(), fc1_b.data(), hidden.data(), input_dim, hidden_dim);
        relu_avx512(hidden.data(), hidden_dim);
        linear_avx512(hidden.data(), fc2_w.data(), fc2_b.data(), output.data(), hidden_dim, output_dim);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Output: " << output[0] << '\n';
    std::cout << "Average inference time: " << (elapsed.count() / iterations) << " ms" << std::endl;
    return 0;
}

