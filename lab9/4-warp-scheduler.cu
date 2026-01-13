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

static constexpr int32_t tile_size = 256; // per-warp, and should be 128-aligned
static constexpr int32_t warps_per_block = 16; // parallelism factor
static constexpr int32_t tiles_per_warp = 32;
static constexpr int32_t n_prefetch = 8;

__device__ __host__ __forceinline__ int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

__global__ void
tma_multiwarp_pipeline(__grid_constant__ const CUtensorMap tensor_map,
                       __grid_constant__ const CUtensorMap dest_tensor_map,
                       const int N) {
    /* TODO: your TMA memcpy kernel here... */

    const int32_t lane = threadIdx.x;
    const int32_t block_tid = threadIdx.y * blockDim.x + lane;
    const int32_t warp_id = threadIdx.y;

    extern __shared__ __align__(128) bf16 shared_mem[];
    __shared__ __align__(8) uint64_t bar[warps_per_block];
    
    if (block_tid == 0) {
        for (int32_t p = 0; p < warps_per_block; ++p) {
            init_barrier(&bar[p], warps_per_block);
        }
    }
    async_proxy_fence();
    __syncwarp();

    int32_t parity[warps_per_block] = {0};

    // prefetch
    for (int32_t t = 0; t < n_prefetch; ++t) {
        if (lane == 0) {
            const int32_t g_src_idx = blockIdx.x * (tiles_per_warp * warps_per_block * tile_size) + t * (warps_per_block * tile_size) + warp_id * tile_size;
            if (g_src_idx + tile_size <= N) {
                cp_async_bulk_tensor_1d_global_to_shared(
                    shared_mem + t * warps_per_block * tile_size + warp_id * tile_size,
                    &tensor_map,
                    g_src_idx,
                    &bar[t]);
                expect_bytes_and_arrive(&bar[t], tile_size * sizeof(bf16));
            } else {
                arrive(&bar[t], 1);
            }
        }
    }

    for (int32_t t = 0; t < tiles_per_warp; ++t) {
        const int32_t c = (t % warps_per_block);
        const int32_t p = (t + n_prefetch) % warps_per_block;

        if (lane == 0) {
            wait(&bar[c], parity[c]);
            parity[c] ^= 1;
        }
        __syncwarp();

        if (lane == 0) {
            const int32_t g_dst_idx = blockIdx.x * (tiles_per_warp * warps_per_block * tile_size) + t * (warps_per_block * tile_size) + warp_id * tile_size;
            if (g_dst_idx + tile_size <= N) {
                cp_async_bulk_tensor_1d_shared_to_global(
                    &dest_tensor_map,
                    g_dst_idx,
                    shared_mem + c * warps_per_block * tile_size + warp_id * tile_size
                );
            }
        }

        if (lane == 0) {
            tma_commit_group();
            tma_wait_until_pending<n_prefetch - 1>();
        }
        __syncwarp();

        if (lane == 0) {
            const int32_t g_src_idx = blockIdx.x * (tiles_per_warp * warps_per_block * tile_size) + (t + n_prefetch) * (warps_per_block * tile_size) + warp_id * tile_size;
            if (g_src_idx + tile_size <= N) {
                cp_async_bulk_tensor_1d_global_to_shared(
                    shared_mem + p * warps_per_block * tile_size + warp_id * tile_size,
                    &tensor_map,
                    g_src_idx,
                    &bar[p]);
                expect_bytes_and_arrive(&bar[p], tile_size * sizeof(bf16));
            } else {
                arrive(&bar[p], 1);
            }
        } 
    }
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

    const cuuint64_t global_dim[1] = {static_cast<cuuint64_t>(N)};
    const cuuint64_t global_strides[1] = {sizeof(bf16)};
    const cuuint32_t box_dim[1] = {tile_size};
    const cuuint32_t element_strides[1] = {1};

    CUtensorMap src_map;
    CUtensorMap dst_map;

    cuTensorMapEncodeTiled(
        &src_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        1, // 1D
        src,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    cuTensorMapEncodeTiled(
        &dst_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        1, // 1D
        dest,
        global_dim,
        global_strides,
        box_dim,
        element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    dim3 block(32, warps_per_block, 1);
    const int32_t grid = ceil_div(N, tiles_per_warp * warps_per_block * tile_size);
    const int32_t shmem_size = 232448 - 1024;

    cudaFuncSetAttribute(
        tma_multiwarp_pipeline,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size
    );
    tma_multiwarp_pipeline<<<grid, block, shmem_size>>>(src_map, dst_map, N);
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