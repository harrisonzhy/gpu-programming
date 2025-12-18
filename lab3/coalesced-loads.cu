// Memory Coalescing Benchmark Kernels
//
// This file contains CUDA kernels to demonstrate and benchmark the performance
// difference between coalesced and non-coalesced memory access patterns:
// - Non-coalesced memory access pattern
// - Coalesced memory access pattern

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdlib.h>
#include <vector>

using data_type = float;

#define THREADS_PER_WARP 32
#define WARPS 128
#define X 1024 // elements per thread

static constexpr size_t kNumOfOuterIterations = 5;
static constexpr size_t kNumOfInnerIterations = 3;

////////////////////////////////////////////////////////////////////////////////
// Non-Coalesced Memory Access Pattern

__global__ void non_coalesced_load(data_type *dst, data_type *src, int x) {

    // <!-- TODO: your code here -->
}

////////////////////////////////////////////////////////////////////////////////
// Coalesced Memory Access Pattern

__global__ void coalesced_load(data_type *dst, data_type *src, int x) {

    // <!-- TODO: your code here -->
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

// CUDA error checking macro
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error " << static_cast<int>(err) << " (" \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// BENCHPRESS macro for sophisticated benchmarking
#define BENCHPRESS(kernel_name, kNumOfOuterIterations, kNumOfInnerIterations, ...) \
    do { \
        std::cout << "Running " << #kernel_name << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                kernel_name<<<WARPS, THREADS_PER_WARP>>>(__VA_ARGS__); \
            } \
            CUDA_CHECK(cudaDeviceSynchronize()); \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::cout << "  Runtime: " << times[0] / 1'000'000 << " ms" << std::endl; \
    } while (0)

int main() {
    // Initialize data and allocate memory
    int total_threads = THREADS_PER_WARP * WARPS;
    int total_elements = total_threads * X;

    data_type *h_src = (data_type *)malloc(sizeof(data_type) * total_elements);
    data_type *h_dst = (data_type *)malloc(sizeof(data_type) * total_elements);

    for (int i = 0; i < total_elements; i++) {
        h_src[i] = static_cast<data_type>(i);
    }

    data_type *d_src = nullptr;
    data_type *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, total_elements * sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(&d_dst, total_elements * sizeof(data_type)));
    CUDA_CHECK(cudaMemcpy(
        d_src,
        h_src,
        total_elements * sizeof(data_type),
        cudaMemcpyHostToDevice));

    // Run benchmarks and test correctness
    CUDA_CHECK(cudaMemset(d_dst, 0, total_elements * sizeof(data_type)));
    BENCHPRESS(
        non_coalesced_load,
        kNumOfOuterIterations,
        kNumOfInnerIterations,
        d_dst,
        d_src,
        X);

    CUDA_CHECK(cudaMemcpy(
        h_dst,
        d_dst,
        total_elements * sizeof(data_type),
        cudaMemcpyDeviceToHost));

    bool non_coalesced_correct = true;
    for (int i = 0; i < total_elements; i++) {
        if (h_dst[i] != h_src[i]) {
            non_coalesced_correct = false;
            break;
        }
    }
    std::cout << "non_coalesced_load: " << (non_coalesced_correct ? "PASSED" : "FAILED")
              << std::endl;

    CUDA_CHECK(cudaMemset(d_dst, 0, total_elements * sizeof(data_type)));
    BENCHPRESS(
        coalesced_load,
        kNumOfOuterIterations,
        kNumOfInnerIterations,
        d_dst,
        d_src,
        X);

    CUDA_CHECK(cudaMemcpy(
        h_dst,
        d_dst,
        total_elements * sizeof(data_type),
        cudaMemcpyDeviceToHost));

    bool coalesced_correct = true;
    for (int i = 0; i < total_elements; i++) {
        if (h_dst[i] != h_src[i]) {
            coalesced_correct = false;
            break;
        }
    }
    std::cout << "coalesced_load: " << (coalesced_correct ? "PASSED" : "FAILED")
              << std::endl;

    // Clean up memory
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    free(h_src);
    free(h_dst);
}
