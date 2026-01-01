#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
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
// from Lab 5 for easy comparison.

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k (Baseline from Lab 5)

#define HAS_LAB_5_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 5 kernel!

namespace matmul_improved_reduce {

__forceinline__ int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

static constexpr int32_t T = 4; // thread processes TxT output tile
static constexpr int32_t K = 64;

__device__ __forceinline__
void cp_async_float(float* smem_dst, const float* gmem_src, bool ignore_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    int pred = ignore_src ? 1 : 0;

    asm volatile(
        "{ .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  cp.async.ca.shared.global [%0], [%1], 4, p;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_addr), "r"(pred)
        : "memory"
    );
}

__device__ __forceinline__
void cp_async_float4(float* smem_dst, const float* gmem_src, bool ignore_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    int pred = ignore_src ? 1 : 0;

    asm volatile(
        "{ .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  cp.async.cg.shared.global [%0], [%1], 16, p;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_addr), "r"(pred)
        : "memory"
    );
}

__device__ __forceinline__
void cp_async_float4_(float* smem_dst, const float* gmem_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_addr)
        : "memory"
    );
}

// TODO: your GPU kernels here...

__global__ void reduce_basic(
    int32_t size_i,
    int32_t size_j,
    int32_t num_slices,
    const float *partial_sums, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */ 
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size_i * size_j) {
        return;
    }

    float sum = 0;

    const float* base = partial_sums + idx;
    for (int32_t s = 0; s < num_slices; ++s) {
        sum += base[s * (size_i * size_j)];
    }

    c[idx] = sum;
}

__global__ void matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    int32_t num_slices,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *partial_sums /* pointer to GPU memory */ 
) {
    const auto tile_height = blockDim.y * T;
    const auto tile_width = blockDim.x * T;
    const auto block_i0 = tile_height * blockIdx.y;
    const auto block_j0 = tile_width * blockIdx.x;
    
    const auto thread_i0 = threadIdx.y * T;
    const auto thread_j0 = threadIdx.x * T;
    
    const auto group = blockIdx.z;
    const auto slice0 = 2 * group;
    const auto slice1 = 2 * group + 1;

    const int32_t block_k0_0 = slice0 * K;
    const int32_t block_k0_1 = slice1 * K;

    if (block_k0_0 >= size_k) {
        return;
    }

    extern __shared__ __align__(16) float shared_mem[];
    float* shared_a0 = shared_mem + 0 * K * K;
    float* shared_b0 = shared_mem + 1 * K * K;
    float* shared_a1 = shared_mem + 2 * K * K;
    float* shared_b1 = shared_mem + 3 * K * K;

    auto do_cp_async = [&](int32_t block_k0, float* shared_a, float* shared_b) {
        for (int32_t shared_base = threadIdx.y * blockDim.x + threadIdx.x; shared_base < K * K / 4; shared_base += blockDim.x * blockDim.y) {
            const auto base = shared_base * 4;
            const int32_t row = base / K;
            const int32_t col = base % K;

            const int32_t global_i = block_i0 + row;
            const int32_t global_j = block_j0 + col;
            const int32_t global_k = block_k0 + col;
            const int32_t global_k_ = block_k0 + row;

            const bool a_oob = !(global_i < size_i && global_k + 3 < size_k);
            const bool b_oob = !(global_k_ < size_k && global_j + 3 < size_j);

            const float* a_src = (a_oob) ? a : &a[global_i * size_k + global_k];
            const float* b_src = (b_oob) ? b : &b[global_k_ * size_j + global_j];

            if (!a_oob) {
                cp_async_float4_(&shared_a[base], a_src);
            }
            if (!b_oob) {
                cp_async_float4_(&shared_b[base], b_src);
            }
        }
    };

    do_cp_async(block_k0_0, shared_a0, shared_b0);
    async_commit_group();

    const bool has_k0_1 = (block_k0_1 < size_k);
    if (has_k0_1) {
        do_cp_async(block_k0_1, shared_a1, shared_b1);
        async_commit_group(); 
    }

    static_assert(T == 4);
    float partial_sums_reg[T * T] = {0};
    auto compute_slice = [&](int32_t slice, const float* shared_a, const float* shared_b) {
        for (int32_t kk = 0; kk < K; ++kk) {
            float reg_a[T];
            float reg_b[T];
            for (int32_t ii = 0; ii < T; ++ii) {
                const int32_t row = thread_i0 + ii;
                reg_a[ii] = shared_a[row * K + kk];
            }
            for (int32_t jj = 0; jj < T; ++jj) {
                const int32_t col = thread_j0 + jj;
                reg_b[jj] = shared_b[kk * K + col];
            }
            for (int32_t ii = 0; ii < T; ++ii) {
                for (int32_t jj = 0; jj < T; ++jj) {
                    partial_sums_reg[ii * T + jj] += reg_a[ii] * reg_b[jj];
                }
            }
        }
    };

    async_wait_pending<1>();
    __syncthreads();
    compute_slice(slice0, shared_a0, shared_b0);

    if (has_k0_1) {
        async_wait_pending<0>();
        __syncthreads();
        compute_slice(slice1, shared_a1, shared_b1);
    }

    // writeback to DRAM
    const int32_t i0 = block_i0 + thread_i0;
    const int32_t j0 = block_j0 + thread_j0;

    static_assert(T == 4);

    const bool full_tile = (i0 + 3) < size_i && (j0 + 3) < size_j;
    float* out = partial_sums + group * (size_i * size_j);
    if (full_tile) {
        for (int32_t idx = 0; idx < T; ++idx) {
            float4 v = *reinterpret_cast<float4*>(&partial_sums_reg[idx * 4]);
            *reinterpret_cast<float4*>(&out[(i0 + idx) * size_j + j0]) = v;
        }
    } else {
        for (int32_t ii = 0; ii < T; ++ii) {
            const int32_t i = i0 + ii;
            if (i >= size_i) {
                continue;
            }
            for (int32_t jj = 0; jj < T; ++jj) {
                const int32_t j = j0 + jj;
                if (j >= size_j) {
                    continue;
                }
                out[i * size_j + j] = partial_sums_reg[ii * 4 + jj];
            }
        }
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    // TODO: your CPU code here
    const int32_t num_psum_bufs = ceil_div(ceil_div(size_k, K), 2);
    const int32_t num_elements = num_psum_bufs * size_i * size_j;
    return num_elements * sizeof(float);
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, // pointer to GPU memory
    float const *b, // pointer to GPU memory
    float *c,       // pointer to GPU memory
    void *workspace // pointer to GPU memory
) {
    // TODO: your CPU code here
    const int32_t num_slices = ceil_div(ceil_div(size_k, K), 2);
    float* partial_sums = reinterpret_cast<float*>(workspace);

    {
        dim3 block(ceil_div(K, T), ceil_div(K, T), 1);
        dim3 grid(ceil_div(size_j, block.x * T), ceil_div(size_i, block.y * T), num_slices);
        
        static constexpr int32_t shmem_size = 2 * 2 * K * K * sizeof(float);

        cudaFuncSetAttribute(
            matmul_improved_reduce,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );
        matmul_improved_reduce<<<grid, block, shmem_size>>>(size_i, size_j, size_k, num_slices, a, b, partial_sums);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch matmul error: %s\n", cudaGetErrorString(e));
        }
    }

    {
        static constexpr int32_t num_threads = 256;
        const int32_t num_blocks = ceil_div(size_i * size_j, num_threads);
        reduce_basic<<<num_blocks, num_threads>>>(size_i, size_j, num_slices, partial_sums, c);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch reduce error: %s\n", cudaGetErrorString(e));
        }
    }
}

} // namespace matmul_improved_reduce

////////////////////////////////////////////////////////////////////////////////
// Tensor Core GPU Implementation

namespace matmul_tensor {

// these should not be changed
static constexpr int32_t PROB_A_ELEMS = 16 * 8;
static constexpr int32_t PROB_B_ELEMS = (8 * 8) * 2;

static constexpr int32_t K = 8; // how many (16x8)x(2x8x8) tiles to process at once
static constexpr int32_t K_BLOCK = 192;
static constexpr int32_t warps_m = 4;
static constexpr int32_t warps_n = 4;

__host__ __device__ __forceinline__ int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__
void cp_async_float(float* smem_dst, const float* gmem_src, bool ignore_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    int pred = ignore_src ? 1 : 0;

    asm volatile(
        "{ .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  cp.async.ca.shared.global [%0], [%1], 4, p;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_addr), "r"(pred)
        : "memory"
    );
}

__device__ __forceinline__
void cp_async_float_(float* smem_dst, const float* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_addr)
        : "memory"
    );
}

template<int32_t MaxN>
__device__ __forceinline__ void async_wait_pending_dynamic(int32_t n) {
    if (n <= 0) {
        async_wait_pending<0>();
        return;
    }
    if (n >= MaxN) {
        async_wait_pending<MaxN>();
        return;
    }
    if constexpr (MaxN > 0) {
        if (n == MaxN) {
            async_wait_pending<MaxN>();
        } else {
            async_wait_pending_dynamic<MaxN - 1>(n);
        }
    }
}


/* TODO: your GPU kernels here... */

__global__ void reduce_basic(
    int32_t size_i,
    int32_t size_j,
    int32_t num_groups,
    const float *partial_sums, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */ 
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size_i * size_j) {
        return;
    }
    
    float sum = 0;
    const float* base = partial_sums + idx;
    for (int32_t s = 0; s < num_groups; ++s) {
        sum += base[s * (size_i * size_j)];
    }
    c[idx] = sum;
}

__global__ void matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    int32_t num_k_tiles,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c, /* pointer to GPU memory */
    float *partial_sums /* pointer to GPU memory */
) {
    const int32_t lane = threadIdx.x;
    const int32_t warp_linear = threadIdx.y;
    const int32_t warp_i = warp_linear / warps_n;
    const int32_t warp_j = warp_linear % warps_n;

    // output C coords
    const int32_t block_i0 = blockIdx.x * (16 * warps_m);
    const int32_t block_j0 = blockIdx.y * (16 * warps_n);
    const int32_t warp_i0 = block_i0 + warp_i * 16;
    const int32_t warp_j0 = block_j0 + warp_j * 16;

    extern __shared__ __align__(16) float shared_mem[];

    // base indices for this thread block
    float* shared_a_warp = shared_mem;
    float* shared_b_warp = shared_a_warp + K * warps_m * PROB_A_ELEMS;

    const int32_t cd_idx[4] = {2 * lane + 0, 
                               2 * lane + 1, 
                               2 * lane + 64, 
                               2 * lane + 65};
    const int32_t a_idx[4] = {(lane / 4) * 8 + (lane % 4) + 0, 
                              (lane / 4) * 8 + (lane % 4) + 64, 
                              (lane / 4) * 8 + (lane % 4) + 4, 
                              (lane / 4) * 8 + (lane % 4) + 68};
    const int32_t b_idx[4] = {(lane % 4) * 16 + (lane / 4), 
                              ((lane % 4) + 4) * 16 + (lane / 4),
                              (lane % 4) * 16 + (8 + lane / 4),
                              ((lane % 4) + 4) * 16 + (8 + lane / 4)};

    auto do_cp_async = [&](int32_t block_k0, float* shared_a, float* shared_b) {
        if (warp_j == 0) {
            for (int32_t i = 0; i < 4; ++i) {
                // A (16x8)
                const int32_t linear_idx = a_idx[i];
                const int32_t row = linear_idx / 8;
                const int32_t col = linear_idx % 8;
                const int32_t global_i = warp_i0 + row;
                const int32_t global_k = block_k0 + col;

                const bool oob = !(global_i < size_i && global_k < size_k);
                if (!oob) {
                    cp_async_float_(&shared_a[linear_idx], &a[global_i * size_k + global_k]);
                }
            } 
        }
        if (warp_i == 0) {
            for (int32_t i = 0; i < 4; ++i) {
                // B (2 x (8x8))
                const int32_t linear_idx = b_idx[i];
                const int32_t row = linear_idx / 16;
                const int32_t col = linear_idx % 16;

                const int32_t global_k = block_k0 + row;
                const int32_t global_j = warp_j0 + col;

                const bool oob = !(global_k < size_k && global_j < size_j);
                if (!oob) {
                    cp_async_float_(&shared_b[linear_idx], &b[global_k * size_j + global_j]);
                }
            }
        }
    };

    float c_reg[8] = {0};

    auto compute_slice = [&](float* shared_a, float* shared_b) {
        uint32_t a_reg[4] = {
            __float_as_uint(shared_a[a_idx[0]]),
            __float_as_uint(shared_a[a_idx[1]]),
            __float_as_uint(shared_a[a_idx[2]]),
            __float_as_uint(shared_a[a_idx[3]])
        };
        uint32_t b_reg[4] = {
            __float_as_uint(shared_b[b_idx[0]]),
            __float_as_uint(shared_b[b_idx[1]]),
            __float_as_uint(shared_b[b_idx[2]]),
            __float_as_uint(shared_b[b_idx[3]])
        };
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : 
            "+f"(c_reg[0]),
            "+f"(c_reg[1]),
            "+f"(c_reg[2]),
            "+f"(c_reg[3])
            :
            "r"(a_reg[0]),
            "r"(a_reg[1]),
            "r"(a_reg[2]),
            "r"(a_reg[3]),
            "r"(b_reg[0]),
            "r"(b_reg[1])
        );
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : 
            "+f"(c_reg[4]),
            "+f"(c_reg[5]),
            "+f"(c_reg[6]),
            "+f"(c_reg[7])
            :
            "r"(a_reg[0]),
            "r"(a_reg[1]),
            "r"(a_reg[2]),
            "r"(a_reg[3]),
            "r"(b_reg[2]),
            "r"(b_reg[3])
        );
    };

    const int32_t group = blockIdx.z;
    const int32_t tiles_begin = group * K_BLOCK;
    const int32_t tiles_end = min(tiles_begin + K_BLOCK, num_k_tiles);
    const int32_t tiles_count = max(tiles_end - tiles_begin, 0);

    // load first K tiles
    if (warp_i == 0 || warp_j == 0) {
        for (int32_t t = 0; t < K; ++t) {
            const int32_t k0 = (tiles_begin + t) * 8;
            float* shared_ai = shared_a_warp + t * (warps_m * PROB_A_ELEMS) + warp_i * PROB_A_ELEMS;
            float* shared_bi = shared_b_warp + t * (warps_n * PROB_B_ELEMS) + warp_j * PROB_B_ELEMS;
            do_cp_async(k0, shared_ai, shared_bi);
            async_commit_group();
        }
    }

    // do main computation for this K_BLOCK
    int32_t blocks_left = K;
    for (int32_t t = 0; t < tiles_count; ++t) {
        const int32_t s = t % K;
        float* shared_ai = shared_a_warp + s * (warps_m * PROB_A_ELEMS) + warp_i * PROB_A_ELEMS;
        float* shared_bi = shared_b_warp + s * (warps_n * PROB_B_ELEMS) + warp_j * PROB_B_ELEMS;
      
        async_wait_pending_dynamic<K - 1>(blocks_left - 1);
        __syncthreads();
        compute_slice(shared_ai, shared_bi);
        __syncthreads();

        blocks_left -= 1;

        const int next_tile = t + K;
        if (next_tile < tiles_count) {
            if (warp_i == 0 || warp_j == 0) {
                const int32_t next_k0 = (tiles_begin + next_tile) * 8;
                do_cp_async(next_k0, shared_ai, shared_bi);
                async_commit_group();
                blocks_left += 1;
            }
        }
    }

    // write to partial sum tile
    float* out = partial_sums + group * (size_i * size_j);
    for (int t = 0; t < 4; ++t) {
        const int32_t idx = cd_idx[t];
        const int32_t row = idx / 8;
        const int32_t col = idx % 8;

        const int32_t global_i = warp_i0 + row;
        const int32_t global_j0 = warp_j0 + col;
        const int32_t global_j1 = warp_j0 + 8 + col;

        const bool oob0 = !(global_i < size_i && global_j0 < size_j);
        const bool oob1 = !(global_i < size_i && global_j1 < size_j);
        if (!oob0) {
            out[global_i * size_j + global_j0] = c_reg[t];
        }
        if (!oob1) {
            out[global_i * size_j + global_j1] = c_reg[4 + t];
        }
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    /* TODO: your CPU code here */
    const int32_t num_elements = size_i * size_j;
    const int32_t num_k_tiles = ceil_div(size_k, 8);
    const int32_t num_groups = ceil_div(num_k_tiles, K_BLOCK);
    return num_elements * num_groups * sizeof(float);
}

void launch_matmul_tensor(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    /* TODO: your CPU code here */

    float *partial_sums = reinterpret_cast<float*>(workspace);
    const int32_t num_k_tiles = ceil_div(size_k, 8);
    const int32_t num_groups = ceil_div(num_k_tiles, K_BLOCK);
    const int32_t num_warps = warps_m * warps_n;

    {
        dim3 block(32, num_warps, 1);
        dim3 grid(ceil_div(size_i, 16 * warps_m), ceil_div(size_j, 16 * warps_n), num_groups);
        const int32_t shmem_size = K * (warps_m * PROB_A_ELEMS + warps_n * PROB_B_ELEMS) * sizeof(float);
        cudaFuncSetAttribute(
            matmul_improved_reduce,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );
        matmul_improved_reduce<<<grid, block, shmem_size>>>(size_i, size_j, size_k, num_k_tiles, a, b, c, partial_sums);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch matmul error: %s\n", cudaGetErrorString(e));
        }
    }
    {
        static constexpr int32_t num_threads = 128;
        const int32_t num_blocks = ceil_div(size_i * size_j, num_threads);
        reduce_basic<<<num_blocks, num_threads>>>(size_i, size_j, num_groups, partial_sums, c);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch reduce error: %s\n", cudaGetErrorString(e));
        }
    }
}

}; // namespace matmul_tensor

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

    if (rel_rmse > 1e-3) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
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
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
    CUDA_CHECK(cudaFree(flush_gpu));
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

#ifdef HAS_LAB_5_BASELINE_IMPL

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

#endif

struct MatmulTensor {
    constexpr static char const *name = "matmul_tensor";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_tensor::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_tensor::launch_matmul_tensor(size_i, size_j, size_k, a, b, c, workspace);
    }
};

BenchmarkResults get_cublas_fma_results() {
    // Hard-coded data collected on A4000 GPU
    return BenchmarkResults{
        "cublas_fma",
        {
            {{3072, 3072, 3072}, 3.152},
            {{2048, 3072, 3072}, 2.174},
            {{1024, 3072, 3072}, 1.090},
            {{512, 3072, 3072}, 0.559},
            {{256, 3072, 3072}, 0.356},
            {{128, 3072, 3072}, 0.256},
            {{64, 3072, 3072}, 0.194},
            {{32, 3072, 3072}, 0.181},
            {{16, 3072, 3072}, 0.181},
        }};
}

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_5_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulTensor>(phase, data, configs));
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

void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &second) {
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
        if (it_first != first.elapsed_ms.end() && it_second != second.elapsed_ms.end()) {
            printf("  %6.02fx", it_first->second / it_second->second);
        } else {
            printf("  %7s", "-");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";


    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {2048, 3072, 3072},
        {1024, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            print_speedup(configs, results.at(i), results.at(j));
        }
    }

    printf("\n-----------------------------------------------------------\n");
    printf("---- Comparison to non-tensor-core cuBLAS performance: ----\n");
    printf("-----------------------------------------------------------\n");

    print_speedup(configs, get_cublas_fma_results(), results.at(results.size() - 1));

    write_json_results("out/results.json", results);

    return 0;
}
