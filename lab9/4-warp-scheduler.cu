// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 4: Bring Your Own Warp Scheduler
////////////////////////////////////////////////////////////////////////////////

__global__ void
tma_multiwarp_pipeline(__grid_constant__ const CUtensorMap tensor_map,
                       __grid_constant__ const CUtensorMap dest_tensor_map,
                       const int N) {
    /* TODO: your TMA memcpy kernel here... */
}

void launch_multiwarp_pipeline(bf16 *dest, bf16 *src, const int N) {
    /*
     * IMPORTANT REQUIREMENT FOR PART 4:
     *
     * To receive credit for this part, you MUST launch the kernel with maximum
     * shared memory allocated.
     *
     * Use cudaFuncSetAttribute() with
     * cudaFuncAttributeMaxDynamicSharedMemorySize to configure the maximum
     * available shared memory before launching the kernel, and then **launch**
     * it with the maximum amount.
     */

    /* TODO: your launch code here... */
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

const int elem_per_block = 16384;
__global__ void simple_vector_copy(bf16 *__restrict__ dest,
                                   const bf16 *__restrict__ src, int N) {
    constexpr int VEC_ELEMS = 8;
    using VecT = uint4;

    int total_vecs = elem_per_block / VEC_ELEMS;
    int start_vec = (blockIdx.x * blockDim.x) * total_vecs;

    const VecT *src_vec = reinterpret_cast<const VecT *>(src);
    VecT *dest_vec = reinterpret_cast<VecT *>(dest);

    for (int i = threadIdx.x; i < blockDim.x * total_vecs; i += blockDim.x) {
        dest_vec[start_vec + i] = src_vec[start_vec + i];
    }
}

#define BENCHMARK_KERNEL(kernel_call, num_iters, size_bytes, label)            \
    do {                                                                       \
        cudaEvent_t start, stop;                                               \
        CUDA_CHECK(cudaEventCreate(&start));                                   \
        CUDA_CHECK(cudaEventCreate(&stop));                                    \
        CUDA_CHECK(cudaEventRecord(start));                                    \
        for (int i = 0; i < num_iters; i++) {                                  \
            kernel_call;                                                       \
        }                                                                      \
        CUDA_CHECK(cudaEventRecord(stop));                                     \
        CUDA_CHECK(cudaEventSynchronize(stop));                                \
        float elapsed_time;                                                    \
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));          \
        float time_per_iter = elapsed_time / num_iters;                        \
        float bandwidth_gb_s = (2.0 * size_bytes * 1e-6 / time_per_iter);      \
        printf("%s - Time: %.4f ms, Bandwidth: %.2f GB/s\n", label,            \
               time_per_iter, bandwidth_gb_s);                                 \
        CUDA_CHECK(cudaEventDestroy(start));                                   \
        CUDA_CHECK(cudaEventDestroy(stop));                                    \
    } while (0)

int main() {
    const size_t size = 132 * 10 * 32 * 128 * 128;

    // Allocate and initialize host memory
    bf16 *matrix = (bf16 *)malloc(size * sizeof(bf16));
    const int N = 128;
    for (int idx = 0; idx < size; idx++) {
        int i = idx / N;
        int j = idx % N;
        float val = fmodf((i * 123 + j * 37) * 0.001f, 2.0f) - 1.0f;
        matrix[idx] = __float2bfloat16(val);
    }

    // Allocate device memory
    bf16 *d_src, *d_dest;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&d_dest, size * sizeof(bf16)));
    CUDA_CHECK(
        cudaMemcpy(d_src, matrix, size * sizeof(bf16), cudaMemcpyHostToDevice));

    // Test TMA copy correctness
    printf("Testing TMA copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    launch_multiwarp_pipeline(d_dest, d_src, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *tma_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(tma_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool tma_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (tma_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));
            tma_correct = false;
            break;
        }
    }
    printf("TMA Copy: %s\n\n", tma_correct ? "PASSED" : "FAILED");
    free(tma_result);

    // Test simple copy correctness
    printf("Testing simple copy correctness...\n");
    CUDA_CHECK(cudaMemset(d_dest, 0, size * sizeof(bf16)));
    simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(d_dest, d_src,
                                                             size);
    CUDA_CHECK(cudaDeviceSynchronize());

    bf16 *simple_result = (bf16 *)malloc(size * sizeof(bf16));
    CUDA_CHECK(cudaMemcpy(simple_result, d_dest, size * sizeof(bf16),
                          cudaMemcpyDeviceToHost));

    bool simple_correct = true;
    for (int idx = 0; idx < size; idx++) {
        if (simple_result[idx] != matrix[idx]) {
            printf("First mismatch at [%d]: %.4f != %.4f\n", idx,
                   __bfloat162float(tma_result[idx]),
                   __bfloat162float(matrix[idx]));

            simple_correct = false;
            break;
        }
    }
    printf("Simple Copy: %s\n\n", simple_correct ? "PASSED" : "FAILED");
    free(simple_result);

    // Benchmark both kernels
    const int num_iters = 10;
    const size_t size_bytes = size * sizeof(bf16);

    if (tma_correct) {
        BENCHMARK_KERNEL((launch_multiwarp_pipeline(d_dest, d_src, size)),
                         num_iters, size_bytes, "TMA Copy");
    }

    if (simple_correct) {
        BENCHMARK_KERNEL(
            (simple_vector_copy<<<size / (elem_per_block * 32), 32>>>(
                 d_dest, d_src, size),
             cudaDeviceSynchronize()),
            num_iters, size_bytes, "Simple Copy");
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dest));
    free(matrix);
    return 0;
}