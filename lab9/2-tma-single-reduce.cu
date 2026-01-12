// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}
// TL {"workspace_files": []}

#include <cuda.h>
#include <cuda_bf16.h>
#include <random>
#include <stdio.h>

#include "tma-interface.cuh"

// Type alias for bfloat16
typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 2: Single Block, Single Tile TMA Reduce
////////////////////////////////////////////////////////////////////////////////

// Feel free to change the interface to this function if you
// are using a different tile dimension that 2d.
__device__ static __forceinline__ void
cp_async_reduce_add_bulk_tensor_2d_shared_to_global(
    const CUtensorMap *tensor_map, int c0, int c1, const void *src) {
    /* TODO: your TMA reduce intrinsic here... */
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group "
        "[%0, {%1, %2}], [%3];\n"
        :
        : "l"(tensor_map),
          "r"(c0),
          "r"(c1),
          "r"(smem_addr)
        : "memory");
}

template <int TILE_M, int TILE_N>
__global__ void
single_tma_reduce(__grid_constant__ const CUtensorMap src_map,
                  __grid_constant__ const CUtensorMap dest_map) {
    /* TODO: your TMA store code here... */
    
    const int32_t lane = threadIdx.x;
    const int32_t block_tid = threadIdx.y * blockDim.x + lane;

    extern __shared__ __align__(128) bf16 shared_mem[];
    __shared__ __align__(8) uint64_t bar;
    
    if (block_tid == 0) {
        init_barrier(&bar, blockDim.x * blockDim.y);
    }
    async_proxy_fence();
    __syncthreads();

    if (block_tid == 0) {
        cp_async_bulk_tensor_2d_global_to_shared(
            shared_mem,
            &src_map,
            0,
            0,
            &bar);
        expect_bytes(&bar, TILE_M * TILE_N * sizeof(bf16));
    }
    arrive(&bar, 1);

    wait(&bar, 0);
    __syncthreads();

    if (block_tid == 0) {
        cp_async_reduce_add_bulk_tensor_2d_shared_to_global(
            &dest_map,
            0,
            0,
            shared_mem
        );
        tma_commit_group();
    }
    tma_wait_until_pending<0>();
}

template <int TILE_M, int TILE_N>
void launch_single_tma_reduce(bf16 *src, bf16 *dest) {
    /* TODO: your launch code here... */

    const cuuint64_t global_dim[2] = {TILE_N, TILE_M};
    const cuuint64_t global_strides[1] = {TILE_N * sizeof(bf16)}; // has size tensor_rank - 1
    const cuuint32_t box_dim[2] = {TILE_N, TILE_M};
    const cuuint32_t element_strides[2] = {1, 1};

    CUtensorMap src_map;
    CUtensorMap dst_map;

    cuTensorMapEncodeTiled(
        &src_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, // 2D
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
        2, // 2D
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

    dim3 block(32, 1, 1);
    dim3 grid(1, 1, 1);
    const int32_t shmem_size = (TILE_M * TILE_N) * sizeof(bf16);
    single_tma_reduce<TILE_M, TILE_N><<<grid, block, shmem_size>>>(src_map, dst_map);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 128;
    const uint64_t total_size = M * N;

    // Allocate host and device memory
    bf16 *matrix = (bf16 *)malloc(total_size * sizeof(bf16));
    bf16 *d_matrix;
    bf16 *d_dest;
    cudaMalloc(&d_matrix, total_size * sizeof(bf16));
    cudaMalloc(&d_dest, total_size * sizeof(bf16));

    // Copy in 1s for the reduction.
    for (int i = 0; i < total_size; i++) {
        matrix[i] = 1;
    }
    cudaMemcpy(d_dest, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    // Initialize source matrix on host
    std::default_random_engine generator(0);
    std::normal_distribution<float> dist(0, 1);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = dist(generator);
            matrix[i * N + j] = __float2bfloat16(val);
        }
    }
    cudaMemcpy(d_matrix, matrix, total_size * sizeof(bf16),
               cudaMemcpyHostToDevice);

    printf("\n\nRunning TMA reduce kernel...\n\n");

    // Launch the TMA kernel
    launch_single_tma_reduce<M, N>(d_matrix, d_dest);

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    bf16 *final_output = (bf16 *)malloc(total_size * sizeof(bf16));
    cudaMemcpy(final_output, d_dest, total_size * sizeof(bf16),
               cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int x = 0; x < M * N; x++) {
        int i = x / N;
        int j = x % N;
        float ref = (float)matrix[i * N + j] + 1.0f;
        float computed = (float)final_output[i * N + j];
        float diff = std::fabs(ref - computed);
        if (diff > 0.1) {
            correct = false;
            printf("Mismatch at (%d, %d): expected %f, got %f \n", i, j, ref,
                   computed);
            break;
        }
    }
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    // Cleanup resources
    cudaFree(d_matrix);
    cudaFree(d_dest);
    free(matrix);
    free(final_output);

    return 0;
}