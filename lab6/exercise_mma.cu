#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <tuple>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

constexpr int32_t mma_size_i = 16;
constexpr int32_t mma_size_j = 8;
constexpr int32_t mma_size_k = 8;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 2: Tensor Core Warm-Up Exercise

__global__ void mma_16x8x8_kernel(float const *a, float const *b, float *c) {
    // TODO: your GPU code here (using inline PTX)
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

void mma_16x8x8_launch(float const *a, float const *b, float *c) {
    mma_16x8x8_kernel<<<1, 32>>>(a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

void mma_16x8x8_ref_cpu(float const *a, float const *b, float *c) {
    for (int32_t i = 0; i < mma_size_i; i++) {
        for (int32_t j = 0; j < mma_size_j; j++) {
            for (int32_t k = 0; k < mma_size_k; k++) {
                c[i * mma_size_j + j] += a[i * mma_size_k + k] * b[k * mma_size_j + j];
            }
        }
    }
}

void print_matrix(int32_t n_row, int32_t n_col, std::vector<float> const &matrix) {
    for (int32_t i = 0; i < n_row; i++) {
        printf("    ");
        for (int32_t j = 0; j < n_col; j++) {
            printf("%10.5f ", matrix.at(i * n_col + j));
        }
        printf("\n");
    }
}

bool test_mma_16x8x8() {
    auto base_test_cases =
        std::vector<std::tuple<std::vector<float>, std::vector<float>>>();

    auto sequential_a = std::vector<float>(mma_size_i * mma_size_k, 0.0f);
    for (int32_t i = 0; i < mma_size_i * mma_size_k; i++) {
        sequential_a.at(i) = static_cast<float>(i);
    }

    auto sequential_b = std::vector<float>(mma_size_k * mma_size_j, 0.0f);
    for (int32_t i = 0; i < mma_size_k * mma_size_j; i++) {
        sequential_b.at(i) = static_cast<float>(i);
    }

    for (int32_t shift_j = 0; shift_j < mma_size_j; shift_j++) {
        auto b = std::vector<float>(mma_size_k * mma_size_j, 0.0f);
        for (int32_t k = 0; k < mma_size_k; k++) {
            int32_t j = (k + shift_j) % mma_size_j;
            b.at(k * mma_size_j + j) = 1.0f;
        }
        base_test_cases.push_back({sequential_a, b});
    }

    for (int32_t shift_k = 0; shift_k < mma_size_k; shift_k++) {
        auto a = std::vector<float>(mma_size_i * mma_size_k, 0.0f);
        for (int32_t i = 0; i < mma_size_i; i++) {
            int32_t k = (i + shift_k) % mma_size_k;
            a.at(i * mma_size_k + k) = 1.0f;
        }
        base_test_cases.push_back({a, sequential_b});
    }

    for (int32_t i = 0; i < mma_size_i; i++) {
        for (int32_t j = 0; j < mma_size_j; j++) {
            for (int32_t k = 0; k < mma_size_k; k++) {
                auto a = std::vector<float>(mma_size_i * mma_size_k, 0.0f);
                auto b = std::vector<float>(mma_size_k * mma_size_j, 0.0f);
                a.at(i * mma_size_k + k) = 1.0f;
                b.at(k * mma_size_j + j) = 1.0f;
                base_test_cases.push_back({std::move(a), std::move(b)});
            }
        }
    }

    // clang-format off
    auto rand_a = std::vector<float>{
        -0.029,  2.105, -1.886,  0.295, -0.114,  0.181,  0.889,  0.996,
        -0.045, -0.544,  0.583, -0.853, -1.933, -1.307, -0.455, -0.192,
        -0.971, -1.172, -0.597, -2.666, -0.322,  1.346, -0.027,  0.411,
         0.436, -1.798,  0.362, -0.899,  0.176, -0.616, -0.230,  0.486,
        -0.304, -0.406, -0.001, -1.545,  1.060,  1.139,  1.473,  0.522,
        -1.135, -0.768,  1.767,  1.431, -0.728,  0.155, -1.767, -1.696,
         1.407, -1.319,  0.601,  1.388, -1.371, -0.532,  0.004,  1.250,
         0.850, -0.154, -1.392,  0.170, -1.029,  1.483,  0.023,  0.225,
        -0.185,  1.259,  1.232,  0.600,  0.099,  0.080,  1.711, -1.145,
         0.010, -1.740, -1.447, -0.055,  0.306,  0.794,  0.611, -1.414,
         1.035, -0.164,  0.523, -1.527, -0.028,  1.389,  0.404,  1.185,
        -0.434, -0.201, -0.580, -1.135,  1.136, -1.856, -0.687, -0.285,
         0.145, -1.240,  0.017,  0.566, -0.148, -0.640, -1.786,  0.334,
         0.071, -0.002, -1.080, -0.769,  1.097, -0.233, -0.210,  1.345,
        -1.869,  0.477,  1.172,  0.433,  2.041, -0.187,  0.025, -0.158,
        -0.587, -0.525, -1.792, -0.261,  1.541, -0.011,  0.527, -0.814
    };
    // clang-format on

    // clang-format off
    auto rand_b = std::vector<float>{
         0.151, -0.551,  0.251,  0.594, -0.380,  0.119,  0.035, -0.814,
        -0.008, -0.017,  0.356,  0.609, -0.129, -0.311, -0.193, -0.213,
        -0.678,  0.153,  0.628,  0.265,  0.272,  0.657, -0.238, -0.130,
         0.582, -0.177, -0.392,  0.219,  0.246, -0.036,  0.132,  0.200,
        -0.084,  0.004,  0.251,  0.742, -0.296, -0.047,  0.000,  0.235,
         0.087,  0.094,  0.147, -0.448, -0.046,  0.322, -0.108, -0.011,
        -0.057, -0.006, -0.546, -0.200,  0.568,  0.115,  0.095, -0.390,
         0.053, -0.003,  0.045, -0.101,  0.494, -0.305,  0.722,  0.349
    };
    // clang-format on

    base_test_cases.push_back({std::move(rand_a), std::move(rand_b)});

    auto zero_c = std::vector<float>(mma_size_i * mma_size_j, 0.0f);

    // clang-format off
    auto rand_init_c = std::vector<float>{
        -0.077, -0.377,  0.367,  0.196,  0.196, -0.492,  0.076,  0.061,
        -0.080, -0.575,  0.582, -0.312,  1.398, -0.974,  0.702,  0.149,
         0.286,  1.697,  1.222,  1.819,  0.535,  0.485,  0.400,  1.442,
        -0.965,  1.804, -2.018, -0.115,  0.445, -0.073, -0.926,  2.370,
        -0.492,  1.280, -0.754,  0.718, -0.559, -0.818, -2.285, -0.152,
        -1.667,  0.659, -0.357,  0.883,  0.643, -0.061,  1.028, -0.280,
         1.952, -1.909, -1.774,  0.315, -0.300,  0.558, -0.425,  0.237,
        -0.624, -2.727, -0.988, -1.164,  1.633,  1.442,  1.539,  0.159,
        -0.755, -1.314, -0.414,  1.452, -0.664, -0.282, -0.051,  1.162,
        -0.232, -1.735, -1.378, -1.172, -1.932,  0.403, -0.002,  0.282,
         0.904,  1.210,  0.532, -0.018, -0.609,  0.441,  1.047, -0.649,
        -0.896,  0.251, -0.363, -1.750,  1.364,  0.332,  0.552,  0.080,
        -0.788, -0.291,  1.004, -0.037, -0.064,  1.263, -0.552,  0.701,
         0.542, -1.023,  0.520, -1.334,  2.252,  0.453,  0.675, -0.316,
         0.404, -1.913,  0.261,  1.448,  0.687,  2.093, -0.149,  0.346,
        -0.537, -0.520, -0.283,  1.143,  0.350, -0.357, -1.604,  0.260
    };
    // clang-format on

    auto test_cases = std::vector<
        std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>>();
    for (auto const &test_case : base_test_cases) {
        auto a = std::get<0>(test_case);
        auto b = std::get<1>(test_case);
        test_cases.push_back({a, b, zero_c});
    }
    for (auto const &test_case : base_test_cases) {
        auto a = std::get<0>(test_case);
        auto b = std::get<1>(test_case);
        test_cases.push_back({a, b, rand_init_c});
    }

    float *a_device;
    float *b_device;
    float *c_device;

    // Pad each buffer and fill with nans to make it easier to detect index out of bounds
    // errors
    constexpr int32_t pad_factor = 8;
    CUDA_CHECK(
        cudaMalloc(&a_device, sizeof(float) * mma_size_i * mma_size_k * pad_factor));
    CUDA_CHECK(
        cudaMalloc(&b_device, sizeof(float) * mma_size_k * mma_size_j * pad_factor));
    CUDA_CHECK(
        cudaMalloc(&c_device, sizeof(float) * mma_size_i * mma_size_j * pad_factor));

    CUDA_CHECK(
        cudaMemset(a_device, 0xff, sizeof(float) * mma_size_i * mma_size_k * pad_factor));
    CUDA_CHECK(
        cudaMemset(b_device, 0xff, sizeof(float) * mma_size_k * mma_size_j * pad_factor));
    CUDA_CHECK(
        cudaMemset(c_device, 0xff, sizeof(float) * mma_size_i * mma_size_j * pad_factor));

    int32_t test_idx = 0;
    for (auto const &test_case : test_cases) {
        auto const &a = std::get<0>(test_case);
        auto const &b = std::get<1>(test_case);
        auto const &c = std::get<2>(test_case);

        auto expected = c;

        mma_16x8x8_ref_cpu(a.data(), b.data(), expected.data());

        if (a.size() != mma_size_i * mma_size_k || b.size() != mma_size_k * mma_size_j ||
            c.size() != mma_size_i * mma_size_j) {
            printf("internal error in test harness: buffer size mismatch\n");
            return false;
        }

        CUDA_CHECK(cudaMemcpy(
            a_device,
            a.data(),
            sizeof(float) * mma_size_i * mma_size_k,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_device,
            b.data(),
            sizeof(float) * mma_size_k * mma_size_j,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            c_device,
            c.data(),
            sizeof(float) * mma_size_i * mma_size_j,
            cudaMemcpyHostToDevice));

        mma_16x8x8_launch(a_device, b_device, c_device);

        std::vector<float> actual(mma_size_i * mma_size_j);
        CUDA_CHECK(cudaMemcpy(
            actual.data(),
            c_device,
            sizeof(float) * mma_size_i * mma_size_j,
            cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int32_t i = 0; i < mma_size_i; i++) {
            for (int32_t j = 0; j < mma_size_j; j++) {
                float diff =
                    std::abs(expected[i * mma_size_j + j] - actual[i * mma_size_j + j]);
                if (!(diff < 0.5e-2)) {
                    ok = false;
                    printf("\nmma_16x8x8 test %d failed:\n\n", test_idx);
                    printf("  mismatch in output position i = %d, j = %d\n", i, j);
                    break;
                }
            }
            if (!ok) {
                break;
            }
        }

        if (!ok) {
            printf("\n");
            printf("  a:\n");
            print_matrix(mma_size_i, mma_size_k, a);
            printf("\n");
            printf("  b:\n");
            print_matrix(mma_size_k, mma_size_j, b);
            printf("\n");
            printf("  initial c:\n");
            print_matrix(mma_size_i, mma_size_j, c);
            printf("\n");
            printf("  expected final c:\n");
            print_matrix(mma_size_i, mma_size_j, expected);
            printf("\n");
            printf("  actual final c:\n");
            print_matrix(mma_size_i, mma_size_j, actual);
            printf("\n");
            return false;
        }

        test_idx++;
    }

    CUDA_CHECK(cudaFree(a_device));
    CUDA_CHECK(cudaFree(b_device));
    CUDA_CHECK(cudaFree(c_device));

    return true;
}

int main(int argc, char const *const *argv) {
    printf("Testing mma_16x8x8 kernel...\n");
    bool ok = test_mma_16x8x8();
    if (ok) {
        printf("mma_16x8x8 tests passed\n");
    }
    return !ok;
}