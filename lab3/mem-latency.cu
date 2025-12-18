// Memory Latency Measurement Kernels
//
// This file contains various CUDA kernels to measure the latency of
// memory accesses at different levels of the memory hierarchy:
// - L1 cache latency measurement
// - L2 cache latency measurement
// - Global memory latency measurement

#include <cuda_runtime.h>
#include <iostream>

using data_type = float;

////////////////////////////////////////////////////////////////////////////////
// L1 Cache Memory Latency

__attribute__((optimize("O0"))) __global__ void l1_mem_latency(
    unsigned long *time_start,
    unsigned long *time_end,
    data_type *array_1,
    data_type *array_2) {
    unsigned long start_time, end_time;
    data_type value1 = 0.0f, result;
    unsigned long temp_addr;

    asm volatile(
        // L1 cache setup - load a zero and create offset address
        "ld.global.ca.u64 %0, [%5];\n\t"
        "add.u64 %0, %0, %5;\n\t"

        // Measure memory load latency
        "mov.u64 %1, %%clock64;\n\t"

        // <!-- TODO: your code here -->

        "mov.u64 %4, %%clock64;\n\t"

        "st.global.f32 [%5], %2;\n\t"
        : "=l"(temp_addr), "=l"(start_time), "=f"(result), "+f"(value1), "=l"(end_time)
        : "l"(array_2)
        : "memory");

    *time_start = start_time;
    *time_end = end_time;
    array_1[0] = result;
}

////////////////////////////////////////////////////////////////////////////////
// L2 Cache Memory Latency

__attribute__((optimize("O0"))) __global__ void l2_mem_latency(
    unsigned long *time_start,
    unsigned long *time_end,
    data_type *array_1,
    data_type *array_2) {
    unsigned long start_time, end_time;
    data_type value1 = 0.0f, result;
    unsigned long temp_addr;

    asm volatile(
        // L2 cache setup - load a zero and create offset address
        "ld.global.cg.u64 %0, [%5];\n\t"
        "add.u64 %0, %0, %5;\n\t"
        "membar.gl;\n\t"

        // Measure memory load latency
        "mov.u64 %1, %%clock64;\n\t"

        // <!-- TODO: your code here -->

        "mov.u64 %4, %%clock64;\n\t"

        "st.global.f32 [%5], %2;\n\t"
        : "=l"(temp_addr), "=l"(start_time), "=f"(result), "+f"(value1), "=l"(end_time)
        : "l"(array_2)
        : "memory");

    *time_start = start_time;
    *time_end = end_time;
    array_1[0] = result;
}

////////////////////////////////////////////////////////////////////////////////
// Global Memory Latency

__attribute__((optimize("O0"))) __global__ void global_mem_latency(
    unsigned long *time_start,
    unsigned long *time_end,
    volatile data_type *array_1,
    volatile data_type *array_2) {
    unsigned long start_time, end_time;
    data_type value1 = 0.0f, result;

    asm volatile(
        // Measure memory load latency directly - no warm-up access
        "membar.gl;\n\t"
        "mov.u64 %0, %%clock64;\n\t"

        // <!-- TODO: your code here -->

        "mov.u64 %4, %%clock64;\n\t"
        "st.global.f32 [%3], %1;\n\t"
        : "=l"(start_time), "=f"(result), "+f"(value1), "+l"(array_2), "=l"(end_time)
        :
        : "memory");

    *time_start = start_time;
    *time_end = end_time;
    array_1[0] = result;
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

// Macro to run kernel and print timing results
#define run_kernel_and_print(kernel_name) \
    do { \
        unsigned long h_time_start, h_time_end; \
        kernel_name<<<1, 1>>>(d_time_start, d_time_end, array_1, array_2); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_start, \
            d_time_start, \
            sizeof(unsigned long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_end, \
            d_time_end, \
            sizeof(unsigned long), \
            cudaMemcpyDeviceToHost)); \
        unsigned long long latency = h_time_end - h_time_start - 2; \
        std::cout << #kernel_name " latency = \t" << latency << " cycles" << std::endl; \
    } while (0)

int main() {
    // Initialize CUDA and allocate device memory
    unsigned long *d_time_start = nullptr;
    unsigned long *d_time_end = nullptr;
    data_type *array_1 = nullptr;
    data_type *array_2 = nullptr;

    unsigned long host_init_time = 0ull;
    data_type host_zero = 0.0f;

    CUDA_CHECK(cudaMalloc(&d_time_start, sizeof(unsigned long)));
    CUDA_CHECK(cudaMalloc(&d_time_end, sizeof(unsigned long)));
    CUDA_CHECK(cudaMalloc(&array_1, sizeof(data_type) * 60));
    CUDA_CHECK(cudaMalloc(&array_2, sizeof(data_type) * 60));

    CUDA_CHECK(
        cudaMemcpy(array_2, &host_zero, sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_time_start,
        &host_init_time,
        sizeof(unsigned long),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_time_end,
        &host_init_time,
        sizeof(unsigned long),
        cudaMemcpyHostToDevice));

    run_kernel_and_print(global_mem_latency);
    run_kernel_and_print(l2_mem_latency);
    run_kernel_and_print(l1_mem_latency);

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_time_start));
    CUDA_CHECK(cudaFree(d_time_end));
    CUDA_CHECK(cudaFree(array_1));
    CUDA_CHECK(cudaFree(array_2));
    return 0;
}