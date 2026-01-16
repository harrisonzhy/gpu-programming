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
// Part 0: No Swizzle WGGMA load for M = 64, N = 8, K = 16
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {

    const int32_t lane = threadIdx.x;
    const int32_t warp_id = threadIdx.y;
    const int32_t tid = warp_id * 32 + lane;

    __shared__ __align__(128) bf16 shared_mem_a[TILE_M * TILE_K];
    __shared__ __align__(128) bf16 shared_mem_b[TILE_N * TILE_K];

    constexpr int core_matrix_rows = 8;
    constexpr int core_matrix_cols = 16 / sizeof(bf16);
    constexpr int core_matrix_elements = core_matrix_rows * core_matrix_cols;

    constexpr int CORE_M = TILE_M / core_matrix_rows;
    constexpr int CORE_K = TILE_K / core_matrix_cols;
    constexpr int CORE_N = TILE_N / core_matrix_rows;

    // copy A
    for (int32_t idx = tid; idx < CORE_M * CORE_K; idx += 128) {
        const int32_t m = idx / CORE_K;
        const int32_t k = idx % CORE_K;

        for (int32_t m_in = 0; m_in < core_matrix_rows; ++m_in) {
            for (int32_t k_in = 0; k_in < core_matrix_cols; ++k_in) {
                const int32_t tile_idx = m * CORE_K + k;
                const int32_t dst_idx = tile_idx * core_matrix_elements + (m_in * core_matrix_cols + k_in);
                const int32_t src_idx = (m * core_matrix_rows + m_in) * TILE_K + (k * core_matrix_cols + k_in);
                shared_mem_a[dst_idx] = a[src_idx];
            }
        }
    }

    // copy B
    for (int32_t idx = tid; idx < CORE_N * CORE_K; idx += 128) {
        const int32_t n = idx / CORE_K;
        const int32_t k = idx % CORE_K;

        for (int32_t n_in = 0; n_in < core_matrix_rows; ++n_in) {
            for (int32_t k_in = 0; k_in < core_matrix_cols; ++k_in) {
                const int32_t tile_idx = n * CORE_K + k;
                const int32_t dst_idx = tile_idx * core_matrix_elements + (n_in * core_matrix_cols + k_in);
                const int32_t src_idx = (n * core_matrix_rows + n_in) * TILE_K + (k * core_matrix_cols + k_in);
                shared_mem_b[dst_idx] = b[src_idx];
            }
        }
    }

    constexpr uint64_t a_lbo = core_matrix_elements * sizeof(bf16);
    constexpr uint64_t a_sbo = core_matrix_elements * CORE_K * sizeof(bf16);
    constexpr uint64_t b_lbo = core_matrix_elements * sizeof(bf16);
    constexpr uint64_t b_sbo = core_matrix_elements * CORE_K * sizeof(bf16);

    const uint64_t desc_a = make_smem_desc<NO_SWIZZLE>(shared_mem_a, a_lbo, a_sbo);
    const uint64_t desc_b = make_smem_desc<NO_SWIZZLE>(shared_mem_b, b_lbo, b_sbo);

    float d[4] = {0};

    warpgroup_arrive();
    wgmma_n8<0, 1, 1, 0, 0>(desc_a, desc_b, d);
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
void launch_wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {
    
    // <--- your code here --->

    const dim3 block(32, 4, 1);
    const dim3 grid(1, 1, 1);
    wgmma_m64n8k16<TILE_M, TILE_N, TILE_K><<<grid, block>>>(a, b, c);
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 16;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = i + j;
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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("\n\nWarmup M=%d, N=%d, K=%d...\n", M, N, K);
    {
        launch_wgmma_m64n8k16<M, N, K>(d_a, d_b, d_c);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("\n\nRunning No Swizzle WGMMA M=%d, N=%d, K=%d...\n\n", M, N, K);
    {
        CUDA_CHECK(cudaEventRecord(start, 0));

        constexpr int32_t iters = 1000;
        for (int i = 0; i < iters; ++i)
            launch_wgmma_m64n8k16<M, N, K>(d_a, d_b, d_c);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("kernel time: %.6f ms\n", ms / iters);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

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
