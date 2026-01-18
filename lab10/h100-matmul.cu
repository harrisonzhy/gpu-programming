// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}

#include <algorithm>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>
#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

////////////////////////////////////////////////////////////////////////////////
// Part 1: Matrix Multiplication for M = 8192, N = 8192, K = 8192
////////////////////////////////////////////////////////////////////////////////

__device__ __host__ __forceinline__ int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

static constexpr int32_t num_warps_per_block = 12;

static constexpr int32_t TILE_M = 64;
static constexpr int32_t TILE_K = 64;
static constexpr int32_t TILE_N = 256;

static constexpr int32_t N_BUFFERS = 1;

__global__ void h100_matmul(
    int M, int N, int K,
    __grid_constant__ const CUtensorMap src_map_a, 
    __grid_constant__ const CUtensorMap src_map_b,
    bf16* c
) {

    // <--- your code here --->

    const int32_t lane = threadIdx.x;
    const int32_t warp_id = threadIdx.y;
    const int32_t block_tid = warp_id * 32 + lane;
    const int32_t warp_group_id = warp_id / 4;
    const int32_t local_warp_id = warp_id % 4;

    constexpr int core_matrix_rows = 8;
    constexpr int core_matrix_cols = 128 / sizeof(bf16); // using 128-swizzle
    constexpr int core_matrix_elements = core_matrix_rows * core_matrix_cols;
    constexpr int CORE_K = TILE_K / core_matrix_cols;

    extern __shared__ __align__(128) bf16 shared_mem_[];
    
    // N-buffer
    bf16* shared_mem_a = shared_mem_;
    bf16* shared_mem_b = shared_mem_a + N_BUFFERS * TILE_M * TILE_K;

    __shared__ alignas(8) uint64_t bar_a;
    __shared__ alignas(8) uint64_t bar_b;

    int32_t parity_a = 0;
    int32_t parity_b = 0;

    if (block_tid == 0) {
        init_barrier(&bar_a, 1);
        init_barrier(&bar_b, 1);
    }
    async_proxy_fence();
    __syncthreads();

    const int32_t M0 = blockIdx.x * TILE_M;
    const int32_t N0 = blockIdx.y * TILE_N;

    float d[16][8] = {}; // N=256, i.e. each thread holds (64x256)/128 output values

    constexpr int32_t loader_warp = 0;
    constexpr int32_t compute_warp_group = 1;

    for (int32_t k = 0; k < K / TILE_K; ++k) {
        const int32_t tile_k_idx = k * TILE_K;
        const int32_t buf = (k % N_BUFFERS);

        // load a tile of A (NxK)
        if (warp_id == loader_warp && lane == 0) {
            bf16* shared_mem_a_dst = shared_mem_a + buf * (TILE_M * TILE_K);
            void* smem_dst = (void*)__cvta_generic_to_shared(shared_mem_a_dst);
            cp_async_bulk_tensor_2d_global_to_shared(
                smem_dst,
                &src_map_a,
                tile_k_idx, // faster moving index (first dim)
                M0,
                &bar_a);
            expect_bytes_and_arrive(&bar_a, TILE_M * TILE_K * sizeof(bf16));
        }

        // load a tile of B (NxK)
        if (warp_id == loader_warp && lane == 0) {
            bf16* shared_mem_b_dst = shared_mem_b + buf * (TILE_N * TILE_K);
            void* smem_dst = (void*)__cvta_generic_to_shared(shared_mem_b_dst);
            cp_async_bulk_tensor_2d_global_to_shared(
                smem_dst,
                &src_map_b,
                tile_k_idx, // faster moving index (first dim)
                N0,
                &bar_b);
            expect_bytes_and_arrive(&bar_b, TILE_N * TILE_K * sizeof(bf16));
        }

        if (warp_id == loader_warp && lane == 0) {
            wait(&bar_a, parity_a);
            parity_a ^= 1;
            wait(&bar_b, parity_b);
            parity_b ^= 1;
        }
        __syncthreads();

        if (warp_group_id == compute_warp_group) {
            bf16* shared_mem_a_base = shared_mem_a + buf * (TILE_M * TILE_K);
            bf16* shared_mem_b_base = shared_mem_b + buf * (TILE_N * TILE_K);

            for (int32_t kk = 0; kk < TILE_K / 16; ++kk) {
                constexpr uint64_t a_sbo = core_matrix_elements * CORE_K * sizeof(bf16);
                constexpr uint64_t b_sbo = core_matrix_elements * CORE_K * sizeof(bf16);
                const uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(shared_mem_a_base + 16 * kk, 1 /* ignored in swizzled */, a_sbo);
                const uint64_t desc_b = make_smem_desc<SWIZZLE_128B>(shared_mem_b_base + 16 * kk, 1 /* ignored in swizzled */, b_sbo);

                wgmma_n256<1, 1, 1, 0, 0>(desc_a, desc_b, d);
            }
            wgmma_commit();
            wgmma_wait<0>();
        }
        __syncthreads();
    }

    if (warp_group_id != compute_warp_group) {
        return;
    }


    // write to output
    const int32_t m_base = 16 * local_warp_id + (lane / 4);
    const int32_t m0 = m_base + 0;
    const int32_t m1 = m_base + 8;

    const int32_t n0 = 2 * (lane % 4);
    const int32_t n1 = 2 * (lane % 4) + 1;

    float* df = &d[0][0];
    for (int32_t frag = 0; frag < 32; ++frag) {
        const int32_t n_base = frag * 8;
        const int32_t d_base = frag * 4;

        const int32_t n00 = N0 + n_base + n0;
        const int32_t n01 = N0 + n_base + n1;

        const int32_t m00 = M0 + m0;
        const int32_t m01 = M0 + m1;

        c[m00 * N + n00] = df[d_base + 0];
        c[m00 * N + n01] = df[d_base + 1];
        c[m01 * N + n00] = df[d_base + 2];
        c[m01 * N + n01] = df[d_base + 3];
    }
}

static inline void CHECK_CU(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* desc = nullptr;
        cuGetErrorName(r, &name);
        cuGetErrorString(r, &desc);
        printf("%s failed: %s (%s)\n",
               what,
               name ? name : "UNKNOWN",
               desc ? desc : "no description");
    }
}

void launch_h100_matmul(unsigned M, unsigned N, unsigned K, bf16 *A, bf16 *B, bf16 *C) {

    // <--- your code here --->

    CUtensorMap src_map_a;
    {
        const cuuint64_t global_dim[2] = {K, M};
        const cuuint64_t global_strides[1] = {K * sizeof(bf16)};
        const cuuint32_t box_dim[2] = {TILE_K, TILE_M};
        const cuuint32_t element_strides[2] = {1, 1};

        auto r = cuTensorMapEncodeTiled(
            &src_map_a,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, // 2D
            A,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        CHECK_CU(r, "src_map_a");
    }
    
    CUtensorMap src_map_b;
    {
        const cuuint64_t global_dim[2] = {K, N};
        const cuuint64_t global_strides[1] = {K * sizeof(bf16)};
        const cuuint32_t box_dim[2] = {TILE_K, TILE_N};
        const cuuint32_t element_strides[2] = {1, 1};

        auto r = cuTensorMapEncodeTiled(
            &src_map_b,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, // 2D
            B,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        CHECK_CU(r, "src_map_b");
    }

    {
        dim3 block(32, num_warps_per_block, 1);
        dim3 grid(ceil_div(M, TILE_M), ceil_div(N, TILE_N), 1);
        const int32_t shmem_size = (N_BUFFERS * TILE_M * TILE_K + N_BUFFERS * TILE_K * TILE_N) * sizeof(bf16);

        CUDA_CHECK(cudaFuncSetAttribute(
            h100_matmul,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size));

        h100_matmul<<<grid, block, shmem_size>>>(M, N, K, src_map_a, src_map_b, C);

        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(e));
        }
        e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            printf("Kernel execution failed: %s\n", cudaGetErrorString(e));
        }
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

static constexpr size_t kNumOfWarmupIterations = 2;
static constexpr size_t kNumOfOuterIterations = 1;
static constexpr size_t kNumOfInnerIterations = 10;


#define BENCHPRESS(func, flops, ...)                                           \
    do {                                                                       \
        std::cout << "Running " << #func << " ...\n";                          \
        for (size_t i = 0; i < kNumOfWarmupIterations; ++i) {                  \
            func(__VA_ARGS__);                                                 \
        }                                                                      \
        cudaDeviceSynchronize();                                               \
        std::vector<float> times(kNumOfOuterIterations);                       \
        cudaEvent_t start, stop;                                               \
        cudaEventCreate(&start);                                               \
        cudaEventCreate(&stop);                                                \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) {                   \
            cudaEventRecord(start);                                            \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) {               \
                func(__VA_ARGS__);                                             \
            }                                                                  \
            cudaEventRecord(stop);                                             \
            cudaEventSynchronize(stop);                                        \
            float elapsed_time;                                                \
            cudaEventElapsedTime(&elapsed_time, start, stop);                  \
            times[i] = elapsed_time / kNumOfInnerIterations;                   \
        }                                                                      \
        cudaEventDestroy(start);                                               \
        cudaEventDestroy(stop);                                                \
        std::sort(times.begin(), times.end());                                 \
        float best_time_ms = times[0];                                         \
        float tflops = (flops * 1e-9) / best_time_ms;                          \
        std::cout << "  Runtime: " << best_time_ms << " ms" << std::endl;      \
        std::cout << "  TFLOP/s: " << tflops << std::endl;                     \
    } while (0)

void runCublasRef(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1, beta = 0;
    cublasStatus_t status =
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                     A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, &beta, C,
                     CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error: " << status << std::endl;
        exit(1);
    }
}

void init_matrix(bf16 *mat, int N) {
    std::default_random_engine generator(0);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < N; i++) {
        mat[i] = distribution(generator);
    }
}

bool check_correctness(bf16 *ref, bf16 *test, int N, float tolerance = 0.1f) {
    int mismatches = 0;
    int total = N;
    for (int i = 0; i < N; i++) {
        float ref_val = __bfloat162float(ref[i]);
        float test_val = __bfloat162float(test[i]);
        float diff = std::abs(ref_val - test_val);
        if (diff > tolerance) {
            if (mismatches < 10) { // Print first 10 mismatches
                std::cout << "  Mismatch at index " << i << ": ref=" << ref_val
                          << ", test=" << test_val << ", diff=" << diff
                          << std::endl;
            }
            mismatches++;
        }
    }
    std::cout << "Total mismatches: " << mismatches << " / " << total << " ("
              << (100.0 * mismatches / total) << "%)" << std::endl;
    return mismatches == 0;
}

int main() {

    const int M = 8192, N = 8192, K = 8192;

    bf16 *A = (bf16 *)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16 *)malloc(sizeof(bf16) * K * N);
    bf16 *C = (bf16 *)malloc(sizeof(bf16) * M * N);

    init_matrix(A, M * K);
    init_matrix(B, K * N);
    memset(C, 0, sizeof(bf16) * M * N);

    bf16 *dA;
    bf16 *dB;
    bf16 *dC;
    bf16 *dCublas;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(bf16) * M * N));
    CUDA_CHECK(cudaMalloc(&dCublas, sizeof(bf16) * M * N));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(bf16) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dCublas, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    bf16 *hCublas = (bf16 *)malloc(sizeof(bf16) * M * N);
    bf16 *hOurs = (bf16 *)malloc(sizeof(bf16) * M * N);

    runCublasRef(M, N, K, dA, dB, dCublas);
    launch_h100_matmul(M, N, K, dA, dB, dC);

    CUDA_CHECK(cudaMemcpy(hCublas, dCublas, sizeof(bf16) * M * N,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hOurs, dC, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));

    bool correct = check_correctness(hCublas, hOurs, M * N, 0.01f);
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    long flops = 2LL * M * N * K;
    BENCHPRESS(runCublasRef, flops, M, N, K, dA, dB, dCublas);

    BENCHPRESS(launch_h100_matmul, flops, M, N, K, dA, dB, dC);

    free(hCublas);
    free(hOurs);

    return 0;
}
