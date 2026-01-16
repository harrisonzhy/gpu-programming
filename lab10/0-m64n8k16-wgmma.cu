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
    __shared__ __align__(128) bf16 shared_mem_b[TILE_K * TILE_N];

    // copy A
    for (int32_t idx = tid; idx < TILE_M * TILE_K; idx += 128) {
        const int32_t m = idx / TILE_K;
        const int32_t k = idx % TILE_K;

        const int32_t m_core = m / 8; // which 8-row block
        const int32_t k_tile = k / 8; // which 8-wide K tile
        const int32_t k_in = k % 8; // column within 8-wide tile
        const int32_t m_in = m % 8; // row within 8-row block

        const int32_t tile_idx = m_core * (TILE_K / 8) + k_tile;
        const int32_t dst_idx = tile_idx * 64 + (m_in * 8 + k_in);
        shared_mem_a[dst_idx] = a[idx];
    }

    // copy B
    for (int32_t idx = tid; idx < TILE_N * TILE_K; idx += 128) {
        const int32_t n = idx / TILE_K;
        const int32_t k = idx % TILE_K;

        const int32_t n_core = n / 8; // which 8-row block
        const int32_t k_tile = k / 8; // which 8-wide K tile
        const int32_t k_in = k % 8; // column within 8-wide tile
        const int32_t n_in = n % 8; // row within 8-row block

        const int32_t tile_idx = n_core * (TILE_K / 8) + k_tile;
        const int32_t dst_idx = tile_idx * 64 /* block elements before this one */ + (n_in * 8 + k_in);
        shared_mem_b[dst_idx] = b[idx];
    }

    constexpr uint64_t a_lbo = 128;
    constexpr uint64_t a_sbo = 128 * (TILE_K / 8);
    constexpr uint64_t b_lbo = 128;
    constexpr uint64_t b_sbo = 128 * (TILE_K / 8);

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

    printf("\n\nRunning No Swizzle WGMMA M=%d, N=%d, K=%d...\n\n", M, N, K);
    launch_wgmma_m64n8k16<M, N, K>(d_a, d_b, d_c);
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
