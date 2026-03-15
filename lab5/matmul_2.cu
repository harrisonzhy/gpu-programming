#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
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

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

    // OPTIONAL: Uncomment this block to include your kernel implementation
    // from Lab 4 for easy comparison.

    ////////////////////////////////////////////////////////////////////////////////
    // GPU Implementation with Reuse in L1/Shmem and Registers (Baseline from Lab 4)

    #define HAS_LAB_4_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 4 kernel!

namespace matmul_l1_reg {

__device__ void init_hilbert_block(int32_t* __restrict__ bx, int32_t* __restrict__ by) {
    const int32_t n = 1 << (32 - __clz(max(gridDim.x, gridDim.y) - 1));
    const int32_t block_id = blockIdx.x * gridDim.y + blockIdx.y;

    *bx = 0;
    *by = 0;
    for (int32_t s = 1, d = block_id; s < n; s <<= 1, d >>= 2) {
        const int32_t rx = 1 & (d >> 1);
        const int32_t ry = 1 & (d ^ rx);
        if (ry == 0) {
            if (rx == 1) {
                *bx = s - 1 - *bx;
                *by = s - 1 - *by;
            }
            int32_t tmp = *bx; 
            *bx = *by; 
            *by = tmp; // reflect
        }
        *bx += s * rx;
        *by += s * ry;
    }
}

static constexpr int32_t T = 4;

static constexpr int32_t BIG_TILE_M = 64;
static constexpr int32_t BIG_TILE_N = 64;

static constexpr int32_t TILE_M = BIG_TILE_M / 1;
static constexpr int32_t TILE_N = BIG_TILE_N / 1;
static constexpr int32_t TILE_K = 64;

static constexpr int32_t N_OVERLAY = 1;
static constexpr int32_t SMALL_TILE_K = 16;

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your GPU code here */

    extern __shared__ float shared_mem[];
    float* shared_a = shared_mem;
    float* shared_b = shared_mem + N_OVERLAY * TILE_M * TILE_K;

    const int32_t block_lin = threadIdx.y * blockDim.x + threadIdx.x;
    
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    for (int32_t M0 = bx * BIG_TILE_M; M0 < (bx + 1) * BIG_TILE_M; M0 += TILE_M) {
        for (int32_t N0 = by * BIG_TILE_N; N0 < (by + 1) * BIG_TILE_N; N0 += TILE_N) {

            float result[T][T] = {};

            float reg_a[T * SMALL_TILE_K];
            float reg_b[T * SMALL_TILE_K];

            // prefetch the first A (TILE_M x TILE_K) and B tiles (TILE_K x TILE_N)
            {
                int32_t buf = 0 % N_OVERLAY;
                float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);
                const float* global_a_src = a + (M0 * size_k + 0 * TILE_K);
                const float* global_b_src = b + (0 * TILE_K) * size_j + N0;

                for (int32_t i = block_lin; i < TILE_M * TILE_K; i += (blockDim.x * blockDim.y)) {
                    int32_t row = i / TILE_K;
                    int32_t col = i % TILE_K;
                    shared_a_dst[i] = global_a_src[row * size_k + col];
                }
                for (int32_t i = block_lin; i < TILE_K * TILE_N; i += (blockDim.x * blockDim.y)) {
                    int32_t row = i / TILE_N;
                    int32_t col = i % TILE_N;
                    shared_b_dst[i] = global_b_src[row * size_j + col];
                }
            }

            for (int32_t k = 1; k < size_k / TILE_K + 1; ++k) {
                // do computation. every thread is responsible for (T x T) output tiles
                __syncthreads();
                {
                    int32_t buf = (k - 1) % N_OVERLAY; 
                    float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                    float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);

                    for (int32_t kk_ = 0; kk_ < TILE_K; kk_ += SMALL_TILE_K) {
                        for (int32_t kkk = 0; kkk < SMALL_TILE_K; ++kkk) {
                            int32_t kk = kk_ + kkk;

                            for (int32_t ty = 0; ty < T; ++ty) {
                                int32_t m = threadIdx.y * T + ty;
                                reg_a[kkk * T + ty] = shared_a_dst[m * TILE_K + kk];
                            }
                            for (int32_t tx = 0; tx < T; ++tx) {
                                int32_t n = threadIdx.x * T + tx;
                                reg_b[kkk * T + tx] = shared_b_dst[kk * TILE_N + n];
                            }
                            for (int32_t ty = 0; ty < T; ++ty) {
                                for (int32_t tx = 0; tx < T; ++tx) {
                                    result[ty][tx] += reg_a[kkk * T + ty] * reg_b[kkk * T + tx];
                                }
                            }
                        }
                    }
                }
                __syncthreads();

                // load next tile of A and B
                {
                    if (k < size_k / TILE_K) {
                        int32_t buf = k % N_OVERLAY;
                        float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                        float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);
                        const float* global_a_src = a + (M0 * size_k + k * TILE_K);
                        const float* global_b_src = b + (k * TILE_K) * size_j + N0;

                        for (int32_t i = block_lin; i < TILE_M * TILE_K; i += (blockDim.x * blockDim.y)) {
                            int32_t row = i / TILE_K, col = i % TILE_K;
                            shared_a_dst[i] = global_a_src[row * size_k + col];
                        }
                        for (int32_t i = block_lin; i < TILE_K * TILE_N; i += (blockDim.x * blockDim.y)) {
                            int32_t row = i / TILE_N, col = i % TILE_N;
                            shared_b_dst[i] = global_b_src[row * size_j + col];
                        }
                    }
                }
            }

            // writeback results to DRAM
            for (int32_t ty = 0; ty < T; ++ty) {
                for (int32_t tx = 0; tx < T; ++tx) { 
                    int32_t m = M0 + threadIdx.y * T + ty;
                    int32_t n = N0 + threadIdx.x * T + tx;
                    if (m >= size_i || n >= size_j) {
                        break;
                    }
                    c[m * size_j + n] = result[ty][tx];
                }
            }
        }
    }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */

    dim3 block(ceil_div(TILE_M, T), ceil_div(TILE_N, T));
    dim3 grid(ceil_div(size_i, BIG_TILE_M), ceil_div(size_j, BIG_TILE_N), 1);
    
    static constexpr int32_t shmem_size = N_OVERLAY * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    // printf("shmem_size: %d", shmem_size);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_l1,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size));

    matmul_l1<<<grid, block, shmem_size>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}; // namespace matmul_l1_reg


////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace matmul_improved {

static constexpr int32_t T = 4;

static constexpr int32_t BIG_TILE_M = 64;
static constexpr int32_t BIG_TILE_N = 64;

static constexpr int32_t TILE_M = BIG_TILE_M / 1;
static constexpr int32_t TILE_N = BIG_TILE_N / 1;
static constexpr int32_t TILE_K = 64;

static constexpr int32_t N_OVERLAY = 1;
static constexpr int32_t SMALL_TILE_K = 16;

__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {
    /* TODO: your GPU code here */

    extern __shared__ __align__(16) float shared_mem[];
    float* shared_a = shared_mem;
    float* shared_b = shared_mem + N_OVERLAY * TILE_M * TILE_K;

    const int32_t block_lin = threadIdx.y * blockDim.x + threadIdx.x;
    
    const int32_t bx = blockIdx.x;
    const int32_t by = blockIdx.y;

    for (int32_t M0 = bx * BIG_TILE_M; M0 < (bx + 1) * BIG_TILE_M; M0 += TILE_M) {
        for (int32_t N0 = by * BIG_TILE_N; N0 < (by + 1) * BIG_TILE_N; N0 += TILE_N) {

            __align__(16) float result[T][T] = {};

            __align__(16) float reg_a[T * SMALL_TILE_K];
            __align__(16) float reg_b[T * SMALL_TILE_K];

            // prefetch the first A (TILE_M x TILE_K) and B tiles (TILE_K x TILE_N)
            {
                int32_t buf = 0 % N_OVERLAY;
                float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);
                const float* global_a_src = a + (M0 * size_k + 0 * TILE_K);
                const float* global_b_src = b + (0 * TILE_K) * size_j + N0;

                for (int32_t i = block_lin * 4; i < TILE_M * TILE_K; i += (blockDim.x * blockDim.y) * 4) {
                    int32_t row = i / TILE_K;
                    int32_t col = i % TILE_K;
                    // shared_a_dst[i] = global_a_src[row * size_k + col];

                    cp_async4(&shared_a_dst[i], &global_a_src[row * size_k + col]);
                }
                for (int32_t i = block_lin * 4; i < TILE_K * TILE_N; i += (blockDim.x * blockDim.y) * 4) {
                    int32_t row = i / TILE_N;
                    int32_t col = i % TILE_N;
                    // shared_b_dst[i] = global_b_src[row * size_j + col];

                    cp_async4(&shared_b_dst[i], &global_b_src[row * size_j + col]);
                }
                async_commit_group();
            }

            for (int32_t k = 1; k < size_k / TILE_K + 1; ++k) {
                // do computation. every thread is responsible for (T x T) output tiles
                async_wait_pending<0>();
                __syncthreads();
                {
                    int32_t buf = (k - 1) % N_OVERLAY;
                    float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                    float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);

                    for (int32_t kk_ = 0; kk_ < TILE_K; kk_ += SMALL_TILE_K) {
                        for (int32_t kkk = 0; kkk < SMALL_TILE_K; ++kkk) {
                            int32_t kk = kk_ + kkk;

                            for (int32_t ty = 0; ty < T; ++ty) {
                                int32_t m = threadIdx.y * T + ty;
                                reg_a[kkk * T + ty] = shared_a_dst[m * TILE_K + kk];
                            }
                            for (int32_t tx = 0; tx < T; ++tx) {
                                int32_t n = threadIdx.x * T + tx;
                                reg_b[kkk * T + tx] = shared_b_dst[kk * TILE_N + n];
                            }
                            for (int32_t ty = 0; ty < T; ++ty) {
                                for (int32_t tx = 0; tx < T; ++tx) {
                                    result[ty][tx] += reg_a[kkk * T + ty] * reg_b[kkk * T + tx];
                                }
                            }
                        }
                    }
                }
                __syncthreads();

                // load next tile of A and B
                {
                    if (k < size_k / TILE_K) {
                        int32_t buf = k % N_OVERLAY;
                        float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                        float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);
                        const float* global_a_src = a + (M0 * size_k + k * TILE_K);
                        const float* global_b_src = b + (k * TILE_K) * size_j + N0;

                        for (int32_t i = block_lin * 4; i < TILE_M * TILE_K; i += (blockDim.x * blockDim.y) * 4) {
                            int32_t row = i / TILE_K, col = i % TILE_K;
                            // shared_a_dst[i] = global_a_src[row * size_k + col];
                            cp_async4(&shared_a_dst[i], &global_a_src[row * size_k + col]);
                        }
                        for (int32_t i = block_lin * 4; i < TILE_K * TILE_N; i += (blockDim.x * blockDim.y) * 4) {
                            int32_t row = i / TILE_N, col = i % TILE_N;
                            // shared_b_dst[i] = global_b_src[row * size_j + col];
                            cp_async4(&shared_b_dst[i], &global_b_src[row * size_j + col]);
                        }
                        async_commit_group();
                    }
                }
            }

            // writeback results to DRAM
            for (int32_t ty = 0; ty < T; ++ty) {
                int32_t m = M0 + threadIdx.y * T + ty;
                if (m >= size_i) {
                    continue;
                }
                int32_t n0 = N0 + threadIdx.x * T;
                if (n0 < size_j) {
                    int32_t tx = 0;
                    for (; tx + 4 < T; tx += 4) {
                        *reinterpret_cast<float4*>(&c[m * size_j + n0 + tx]) = *reinterpret_cast<float4*>(&result[ty][tx]);
                    }
                    for (; tx < T; ++tx) {
                        c[m * size_j + n0 + tx] = result[ty][tx];
                    }
                }
            }
        }
    }
}

void launch_matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */

    dim3 block(ceil_div(TILE_M, T), ceil_div(TILE_N, T));
    dim3 grid(ceil_div(size_i, BIG_TILE_M), ceil_div(size_j, BIG_TILE_N), 1);
    
    static constexpr int32_t shmem_size = N_OVERLAY * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    // printf("shmem_size: %d", shmem_size);

    CUDA_CHECK(cudaFuncSetAttribute(
        matmul_improved,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size));

    matmul_improved<<<grid, block, shmem_size>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}; // namespace matmul_improved

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k

namespace matmul_improved_reduce {

static constexpr int32_t T = 4;

static constexpr int32_t BIG_TILE_M = 64;
static constexpr int32_t BIG_TILE_N = 64;

static constexpr int32_t TILE_M = BIG_TILE_M / 1;
static constexpr int32_t TILE_N = BIG_TILE_N / 1;
static constexpr int32_t TILE_K = 64;

static constexpr int32_t N_OVERLAY = 2;
static constexpr int32_t SMALL_TILE_K = 8;
static constexpr int32_t CHUNK_K = TILE_K * 4;

__global__ void matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    int32_t num_slices,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *partial_sums /* pointer to GPU memory */ 
) {
    const int32_t block_lin = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ __align__(16) float shared_mem[];
    float* shared_a = shared_mem;
    float* shared_b = shared_mem + N_OVERLAY * TILE_M * TILE_K;
    
    for (int32_t M0 = blockIdx.x * BIG_TILE_M; M0 < (blockIdx.x + 1) * BIG_TILE_M; M0 += TILE_M) {
        for (int32_t N0 = blockIdx.y * BIG_TILE_N; N0 < (blockIdx.y + 1) * BIG_TILE_N; N0 += TILE_N) {

            __align__(16) float result[T][T] = {};
            __align__(16) float reg_a[T * SMALL_TILE_K];
            __align__(16) float reg_b[T * SMALL_TILE_K];

            // prefetch the first A (TILE_M x TILE_K) and B tiles (TILE_K x TILE_N)
            {
                int32_t k_init = CHUNK_K * blockIdx.z;
                int32_t buf = k_init % N_OVERLAY;
                float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);
                const float* global_a_src = a + (M0 * size_k + k_init);
                const float* global_b_src = b + (k_init) * size_j + N0;

                for (int32_t i = block_lin * 4; i < TILE_M * TILE_K; i += (blockDim.x * blockDim.y) * 4) {
                    int32_t row = i / TILE_K;
                    int32_t col = i % TILE_K;
                    // shared_a_dst[i] = global_a_src[row * size_k + col];
                    cp_async4(&shared_a_dst[i], &global_a_src[row * size_k + col]);
                }
                for (int32_t i = block_lin * 4; i < TILE_K * TILE_N; i += (blockDim.x * blockDim.y) * 4) {
                    int32_t row = i / TILE_N;
                    int32_t col = i % TILE_N;
                    // shared_b_dst[i] = global_b_src[row * size_j + col];
                    cp_async4(&shared_b_dst[i], &global_b_src[row * size_j + col]);
                }
                async_commit_group();
            }

            for (int32_t k = CHUNK_K * blockIdx.z + TILE_K; k < CHUNK_K * (blockIdx.z + 1) + TILE_K; k += TILE_K) {
                async_wait_pending<0>();
                __syncthreads();
                {
                    // compute on the previous tile
                    int32_t buf = (k - TILE_K) % N_OVERLAY;
                    float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                    float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);

                    for (int32_t kk_ = 0; kk_ < TILE_K; kk_ += SMALL_TILE_K) {
                        for (int32_t kkk = 0; kkk < SMALL_TILE_K; ++kkk) {
                            int32_t kk = kk_ + kkk;
                            // copy to registers
                            for (int32_t ty = 0; ty < T; ++ty) {
                                int32_t m = threadIdx.y * T + ty;
                                reg_a[kkk * T + ty] = shared_a_dst[m * TILE_K + kk];
                            }
                            for (int32_t tx = 0; tx < T; ++tx) {
                                int32_t n = threadIdx.x * T + tx;
                                reg_b[kkk * T + tx] = shared_b_dst[kk * TILE_N + n];
                            }
                            // compute
                            for (int32_t ty = 0; ty < T; ++ty) {
                                for (int32_t tx = 0; tx < T; ++tx) {
                                    result[ty][tx] += reg_a[kkk * T + ty] * reg_b[kkk * T + tx];
                                }
                            }
                        }
                    }
                }

                {
                    if (k < CHUNK_K * (blockIdx.z + 1)) {
                        // fetch new tile
                        int32_t buf = k % N_OVERLAY;
                        float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                        float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);
                        const float* global_a_src = a + (M0 * size_k + k);
                        const float* global_b_src = b + k * size_j + N0;

                        for (int32_t i = block_lin * 4; i < TILE_M * TILE_K; i += (blockDim.x * blockDim.y) * 4) {
                            int32_t row = i / TILE_K;
                            int32_t col = i % TILE_K;
                            // shared_a_dst[i] = global_a_src[row * size_k + col];
                            cp_async4(&shared_a_dst[i], &global_a_src[row * size_k + col]);
                        }
                        for (int32_t i = block_lin * 4; i < TILE_K * TILE_N; i += (blockDim.x * blockDim.y) * 4) {
                            int32_t row = i / TILE_N;
                            int32_t col = i % TILE_N;
                            // shared_b_dst[i] = global_b_src[row * size_j + col];
                            cp_async4(&shared_b_dst[i], &global_b_src[row * size_j + col]);
                        }
                        async_commit_group();
                    }
                }

                for (int32_t ty = 0; ty < T; ++ty) {
                    int32_t m = M0 + threadIdx.y * T + ty;
                    if (m >= size_i) {
                        continue;
                    }
                    int32_t psum_offset = blockIdx.z /* k-chunk offset */ * (size_i * size_j);
                    int32_t n0 = N0 + threadIdx.x * T;
                    if (n0 < size_j) {
                        int32_t tx = 0;
                        for (; tx + 4 < T; tx += 4) {
                            *reinterpret_cast<float4*>(&partial_sums[psum_offset + m * size_j + n0 + tx]) = *reinterpret_cast<float4*>(&result[ty][tx]);
                        }
                        for (; tx < T; ++tx) {
                            partial_sums[psum_offset + m * size_j + n0 + tx] = result[ty][tx];
                        }
                    }
                }
            }
        }
    }
}

__global__ void reduce_basic(
    int32_t size_i,
    int32_t size_j,
    int32_t num_k_chunks,
    const float *partial_sums, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */ 
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size_i * size_j) {
        return;
    }

    float sum = 0;
    const float* base = partial_sums + idx;
    for (int32_t s = 0; s < num_k_chunks; ++s) {
        sum += base[s * (size_i * size_j)];
    }

    c[idx] = sum;
}

__forceinline__ int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

/* TODO: your GPU kernels here... */

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    /* TODO: your CPU code here */
    const int32_t num_psum_bufs = ceil_div(size_k, CHUNK_K);
    const int32_t num_elements = num_psum_bufs * size_i * size_j;
    return num_elements * sizeof(float);
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    /* TODO: your CPU code here */
    float* partial_sums = reinterpret_cast<float*>(workspace);
    int32_t num_k_chunks = ceil_div(size_k, CHUNK_K);

    {
        dim3 block(ceil_div(TILE_M, T), ceil_div(TILE_N, T), 1);
        dim3 grid(ceil_div(size_i, BIG_TILE_M), ceil_div(size_j, BIG_TILE_N), num_k_chunks);

        static constexpr int32_t shmem_size = N_OVERLAY * (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);

        cudaFuncSetAttribute(
            matmul_improved_reduce,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );
        matmul_improved_reduce<<<grid, block, shmem_size>>>(size_i, size_j, size_k, num_k_chunks, a, b, partial_sums);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch matmul error: %s\n", cudaGetErrorString(e));
        }
    }

    {
        static constexpr int32_t block = 256;
        const int32_t grid = ceil_div(size_i * size_j, block);
        reduce_basic<<<grid, block>>>(size_i, size_j, num_k_chunks, partial_sums, c);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch reduce error: %s\n", cudaGetErrorString(e));
        }
    }
}

}; // namespace matmul_improved_reduce

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
            mse += diff * diff;
            ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
        }
    }
    mse /= size_i * size_j;
    ref_mean_square /= size_i * size_j;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-5) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        if (phase == Phase::BENCHMARK) {
            elapsed_ms = benchmark_ms(
                target_time_ms,
                1,
                [&]() {
                    if (workspace_size > 0) {
                        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                    }
                    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
                },
                [&]() {
                    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
                });
        } else {
            elapsed_ms = benchmark_ms(
                target_time_ms,
                1,
                [&]() {
                    if (workspace_size > 0) {
                        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                    }
                    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
                },
                [&]() {
                    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
                }); 
        }

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_4_BASELINE_IMPL

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

#endif

struct MatmulImproved {
    constexpr static char const *name = "matmul_improved";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_4_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulL1Reg>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulImproved>(phase, data, configs));
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";


    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
        {1, 3072, 3072},
        {256, 256, 256},
        {256, 256, 1024},
        {256, 256, 8192},
        {128, 128, 32768},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            auto const &first = results.at(i);
            auto const &second = results.at(j);
            printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
            printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
            printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
            for (auto const &config : configs) {
                auto size_i = config.size_i;
                auto size_j = config.size_j;
                auto size_k = config.size_k;
                printf("  %6d  %6d  %6d", size_i, size_j, size_k);
                auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
                auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
                if (it_first != first.elapsed_ms.end() &&
                    it_second != second.elapsed_ms.end()) {
                    printf("  %6.02fx", it_first->second / it_second->second);
                } else {
                    printf("  %7s", "-");
                }
                printf("\n");
            }
        }
    }

    write_json_results("out/results.json", results);

    return 0;
}
