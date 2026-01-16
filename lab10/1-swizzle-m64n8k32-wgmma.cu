// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdio.h>

#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 0: 64B Swizzle WGGMA load for M = 64, N = 8, K = 32
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void swizzle_wgmma_m64n8k32(
    __grid_constant__ const CUtensorMap src_map_a, 
    __grid_constant__ const CUtensorMap src_map_b,
    float* c) {

    // <--- your code here --->

    const int32_t lane = threadIdx.x;
    const int32_t warp_id = threadIdx.y;
    const int32_t block_tid = warp_id * 32 + lane;

    __shared__ alignas(128) bf16 shared_mem_a[TILE_M * TILE_K];
    __shared__ alignas(128) bf16 shared_mem_b[TILE_N * TILE_K];

    __shared__ alignas(8) uint64_t bar;

    if (block_tid == 0) {
        init_barrier(&bar, 1);
    }
    async_proxy_fence();
    __syncthreads();

    if (block_tid == 0) {
        void* smem_dst = (void*)__cvta_generic_to_shared(shared_mem_a);
        cp_async_bulk_tensor_2d_global_to_shared(
            smem_dst,
            &src_map_a,
            0,
            0,
            &bar);
        expect_bytes_and_arrive(&bar, TILE_M * TILE_K * sizeof(bf16));
        wait(&bar, 0);
    }
    __syncthreads();

    if (block_tid == 0) {
        void* smem_dst = (void*)__cvta_generic_to_shared(shared_mem_b);
        cp_async_bulk_tensor_2d_global_to_shared(
            smem_dst,
            &src_map_b,
            0,
            0,
            &bar);
        expect_bytes_and_arrive(&bar, TILE_N * TILE_K * sizeof(bf16));
        wait(&bar, 1);
    }
    __syncthreads();

    constexpr int core_matrix_rows = 8;
    constexpr int core_matrix_cols = 16 / sizeof(bf16);
    constexpr int core_matrix_elements = core_matrix_rows * core_matrix_cols;
    constexpr int CORE_K = TILE_K / core_matrix_cols;

    async_proxy_fence();
    warpgroup_arrive();

    float d[4];
    {
        constexpr uint64_t a_sbo = core_matrix_elements * CORE_K * sizeof(bf16);
        constexpr uint64_t b_sbo = core_matrix_elements * CORE_K * sizeof(bf16);
        const uint64_t desc_a = make_smem_desc<SWIZZLE_64B>(shared_mem_a, 1 /* ignored in swizzled */, a_sbo);
        const uint64_t desc_b = make_smem_desc<SWIZZLE_64B>(shared_mem_b, 1 /* ignored in swizzled */, b_sbo);

        wgmma_n8<0, 1, 1, 0, 0>(desc_a, desc_b, d);
    }
    {
        constexpr uint64_t a_sbo = core_matrix_elements * CORE_K * sizeof(bf16);
        constexpr uint64_t b_sbo = core_matrix_elements * CORE_K * sizeof(bf16);
        const uint64_t desc_a = make_smem_desc<SWIZZLE_64B>(shared_mem_a + 16, 1 /* ignored in swizzled */, a_sbo);
        const uint64_t desc_b = make_smem_desc<SWIZZLE_64B>(shared_mem_b + 16, 1 /* ignored in swizzled */, b_sbo);

        wgmma_n8<1, 1, 1, 0, 0>(desc_a, desc_b, d);
    }
    wgmma_commit();
    wgmma_wait<0>();

    const int32_t m_base = 16 * warp_id + (lane / 4);
    const int32_t m0 = m_base + 0;
    const int32_t m1 = m_base + 8;
    const int32_t n0 = 2 * (lane % 4);
    const int32_t n1 = 2 * (lane % 4) + 1;

    c[n0 * TILE_M + m0] = d[0];
    c[n1 * TILE_M + m0] = d[1];
    c[n0 * TILE_M + m1] = d[2];
    c[n1 * TILE_M + m1] = d[3];
}

template <int TILE_M, int TILE_N, int TILE_K>
void launch_swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {

    // <--- your code here --->

    CUtensorMap src_map_a;
    {
        const cuuint64_t global_dim[2] = {TILE_K, TILE_M};
        const cuuint64_t global_strides[1] = {TILE_K * sizeof(bf16)};
        const cuuint32_t box_dim[2] = {TILE_K, TILE_M};
        const cuuint32_t element_strides[2] = {1, 1};

        cuTensorMapEncodeTiled(
            &src_map_a,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, // 2D
            a,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
    }
    CUtensorMap src_map_b;
    {
        const cuuint64_t global_dim[2] = {TILE_K, TILE_N};
        const cuuint64_t global_strides[1] = {TILE_K * sizeof(bf16)};
        const cuuint32_t box_dim[2] = {TILE_K, TILE_N};
        const cuuint32_t element_strides[2] = {1, 1};

        cuTensorMapEncodeTiled(
            &src_map_b,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, // 2D
            b,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
    }

    const dim3 block(32, 4, 1);
    const dim3 grid(1, 1, 1);
    swizzle_wgmma_m64n8k32<TILE_M, TILE_N, TILE_K><<<grid, block>>>(src_map_a, src_map_b, c);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 32;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = (i + j) / 10.0f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = (i + j) / 10.0f;
        }
    }

    float *d_c;
    bf16 *d_a, *d_b;
    cudaMalloc(&d_a, M * K * sizeof(bf16));
    cudaMalloc(&d_b, N * K * sizeof(bf16));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a, M * K * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(bf16), cudaMemcpyHostToDevice);

    // Compute CPU reference
    float *cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_row = (float)a[i * K + k];
                float a_col = (float)b[k + j * K];
                temp += a_row * a_col;
            }
            cpu_output[j * M + i] = temp;
        }
    }

    float *gpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        gpu_output[i] = 0;
    }
    cudaMemcpy(d_c, gpu_output, M * N * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\nRunning Swizzle WGMMA M=64, N=8, K-32...\n\n");
    launch_swizzle_wgmma_m64n8k32<M, N, K>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(gpu_output, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // check results
    bool correct = true;
    for (int idx = 0; idx < M * N; idx++) {
        if (fabs(cpu_output[idx] - gpu_output[idx]) > 0.01f) {
            correct = false;
            int j = idx / M;
            int i = idx % M;
            printf(
                "\nFirst mismatch at (%d, %d): CPU=%.0f, GPU=%.0f\n",
                i,
                j,
                cpu_output[idx],
                gpu_output[idx]);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);

    return 0;
}
