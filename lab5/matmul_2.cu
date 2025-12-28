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

static constexpr int32_t T = 4; // thread processes TxT output tile
static constexpr int32_t K = 64;

__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    // TODO: your GPU code here

    extern __shared__ float shared_mem[];
    float* shared_a = shared_mem;
    float* shared_b = shared_mem + K * K;

    // global indices
    const int32_t tile_height = blockDim.y * T;
    const int32_t tile_width = blockDim.x * T;
    const int32_t block_i0 = tile_height * blockIdx.y;
    const int32_t block_j0 = tile_width * blockIdx.x;

    // offset/scratchpad indices
    const int32_t thread_i0 = threadIdx.y * T;
    const int32_t thread_j0 = threadIdx.x * T;

    float results[T * T] = {0};

    for (int32_t block_k0 = 0; block_k0 < size_k; block_k0 += K) {
        // load scratchpad with elements of A and B
        for (int32_t shared_base = threadIdx.y * blockDim.x + threadIdx.x; 
                    shared_base < K * K; 
                    shared_base += blockDim.x * blockDim.y) {
            const int32_t row = shared_base / K;
            const int32_t col = shared_base % K;

            // indices to load A
            const int32_t global_i = block_i0 + row;
            const int32_t global_k = block_k0 + col;

            // indices to load B
            const int32_t global_k_ = block_k0 + row;
            const int32_t global_j = block_j0 + col;

            if (global_i < size_i && global_k < size_k) {
                shared_a[shared_base] = a[global_i * size_k + global_k];
            } else {
                shared_a[shared_base] = 0;
            }
            if (global_k_ < size_k && global_j < size_j) {
                shared_b[shared_base] = b[global_k_ * size_j + global_j];
            } else {
                shared_b[shared_base] = 0;
            }
        }

        __syncthreads();

        // to exploit register reuse, do outer product
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
                    results[ii * T + jj] += reg_a[ii] * reg_b[jj];
                }
            }
        }

        __syncthreads();
    }

    // writeback results to DRAM
    for (int32_t ii = 0; ii < T; ++ii) {
        for (int32_t jj = 0; jj < T; ++jj) {
            const int32_t i = block_i0 + thread_i0 + ii;
            const int32_t j = block_j0 + thread_j0 + jj;
            if (i >= size_i || j >= size_j) {
                continue;
            }
            c[i * size_j + j] = results[ii * T + jj];
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
    // TODO: your CPU code here

    auto ceil_div = [](int32_t a, int32_t b) -> int32_t { return (a + b - 1) / b; };

    dim3 block(ceil_div(K, T), ceil_div(K, T), 1);
    dim3 grid(ceil_div(size_j, block.x * T), ceil_div(size_i, block.y * T), 1);
    
    static constexpr int32_t shmem_size = 2 * K * K * sizeof(float);
    matmul_l1_reg<<<grid, block, shmem_size>>>(size_i, size_j, size_k, a, b, c);
}

} // namespace matmul_l1_reg

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace matmul_improved {

static constexpr int32_t T = 4; // thread processes TxT output tile
static constexpr int32_t K = 64;

__device__ __forceinline__
void cp_async_float(float* smem_dst, const float* gmem_src, bool ignore_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    int32_t pred = ignore_src ? 1 : 0;

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

__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {
    /* TODO: your GPU code here */

    extern __shared__ float shared_mem[];
    float* shared_a = shared_mem;
    float* shared_b = shared_mem + K * K;

    // global indices
    const int32_t tile_height = blockDim.y * T;
    const int32_t tile_width = blockDim.x * T;
    const int32_t block_i0 = tile_height * blockIdx.y;
    const int32_t block_j0 = tile_width * blockIdx.x;

    // offset/scratchpad indices
    const int32_t thread_i0 = threadIdx.y * T;
    const int32_t thread_j0 = threadIdx.x * T;

    float results[T * T] = {0};

    for (int32_t block_k0 = 0; block_k0 < size_k; block_k0 += K) {
        // load scratchpad with elements of A and B
        for (int32_t shared_base = threadIdx.y * blockDim.x + threadIdx.x; 
                    shared_base < K * K; 
                    shared_base += blockDim.x * blockDim.y) {
            const int32_t row = shared_base / K;
            const int32_t col = shared_base % K;

            // indices to load A
            const int32_t global_i = block_i0 + row;
            const int32_t global_k = block_k0 + col;

            // indices to load B
            const int32_t global_k_ = block_k0 + row;
            const int32_t global_j = block_j0 + col;

            // async copy
            bool a_oob = !(global_i < size_i && global_k < size_k);
            bool b_oob = !(global_k_ < size_k && global_j < size_j);

            const float* a_src = (a_oob) ? a : &a[global_i * size_k + global_k];
            const float* b_src = (b_oob) ? b : &b[global_k_ * size_j + global_j];

            cp_async_float(&shared_a[shared_base], a_src, a_oob);
            cp_async_float(&shared_b[shared_base], b_src, b_oob);
        }

        async_commit_group();
        async_wait_pending<0>();

        __syncthreads();

        // to exploit register reuse, do outer product
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
                    results[ii * T + jj] += reg_a[ii] * reg_b[jj];
                }
            }
        }

        __syncthreads();
    }

    // writeback results to DRAM
    for (int32_t ii = 0; ii < T; ++ii) {
        for (int32_t jj = 0; jj < T; ++jj) {
            const int32_t i = block_i0 + thread_i0 + ii;
            const int32_t j = block_j0 + thread_j0 + jj;
            if (i >= size_i || j >= size_j) {
                continue;
            }
            c[i * size_j + j] = results[ii * T + jj];
        }
    }
}

void launch_matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {
    /* TODO: your CPU code here */

    auto ceil_div = [](int32_t a, int32_t b) -> int32_t { return (a + b - 1) / b; };

    dim3 block(ceil_div(K, T), ceil_div(K, T), 1);
    dim3 grid(ceil_div(size_j, block.x * T), ceil_div(size_i, block.y * T), 1);
    
    static constexpr int32_t shmem_size = 2 * K * K * sizeof(float);
    matmul_improved<<<grid, block, shmem_size>>>(size_i, size_j, size_k, a, b, c);
}

}; // namespace matmul_improved

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k

namespace matmul_improved_reduce {

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
        "  cp.async.ca.shared.global [%0], [%1], 16, p;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(gmem_addr), "r"(pred)
        : "memory"
    );
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
    
    const auto slice = blockIdx.z;
    const int32_t block_k0 = slice * K;

    if (block_k0 >= size_k) {
        return;
    }

    extern __shared__ __align__(16) float shared_mem[];
    float* shared_a = shared_mem;
    float* shared_b = shared_mem + K * K;

    int32_t shared_base = threadIdx.y * blockDim.x + threadIdx.x;
    for (; shared_base < K * K / 4; shared_base += blockDim.x * blockDim.y) {
        const int32_t base = shared_base * 4;
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

        cp_async_float4(&shared_a[base], a_src, a_oob);
        cp_async_float4(&shared_b[base], b_src, b_oob);
    }

    async_commit_group();
    async_wait_pending<0>();
    __syncthreads();

    float partial_sums_reg[T * T] = {0};
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

    static_assert(T == 4);

    // writeback to DRAM
    const int32_t i0 = block_i0 + thread_i0;
    const int32_t j0 = block_j0 + thread_j0;

    const bool full_tile = (i0 + 3) < size_i && (j0 + 3) < size_j;
    float* out = partial_sums + slice * (size_i * size_j);

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

    __syncthreads();
}

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

__global__ void reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t num_slices,
    const float *partial_sums, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */ 
) {
    const int32_t N = size_i * size_j;
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    extern __shared__ float shared_mem[];

    float sum = 0;
    for (int32_t slice_base = 0; slice_base < num_slices; slice_base += blockDim.y /* sheet stride */) {
        const int32_t s = slice_base + threadIdx.y;

        float v = 0;
        if (s < num_slices) {
            v = partial_sums[s * N + idx];
        }

        shared_mem[threadIdx.y * blockDim.x + threadIdx.x] = v;
        __syncthreads();

        if (threadIdx.y == 0) {
            float chunk = 0;
            // reduce along y-axis
            for (int32_t yy = 0; yy < blockDim.y; ++yy) {
                chunk += shared_mem[yy * blockDim.x + threadIdx.x];
            }
            sum += chunk;
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) {
        c[idx] = sum;
    }
}

__forceinline__ int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

/* TODO: your GPU kernels here... */

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    /* TODO: your CPU code here */
    const int32_t num_psum_bufs = ceil_div(size_k, K);
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

    const int32_t num_slices = ceil_div(size_k, K);
    float* partial_sums = reinterpret_cast<float*>(workspace);

    {
        dim3 block(ceil_div(K, T), ceil_div(K, T), 1);
        dim3 grid(ceil_div(size_j, block.x * T), ceil_div(size_i, block.y * T), num_slices);
        
        static constexpr int32_t shmem_size = 2 * K * K * sizeof(float);

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

    // Using fancier reduction
    {
        dim3 block_reduce(128, 8, 1);
        const int32_t num_blocks = ceil_div(size_i * size_j, block_reduce.x);
        const int32_t shmem_size = block_reduce.x * block_reduce.y * sizeof(float); 
        reduce<<<num_blocks, block_reduce, shmem_size>>>(size_i, size_j, num_slices, partial_sums, c);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch reduce error: %s\n", cudaGetErrorString(e));
        }
    }

    /* // Using basic reduction
    {
        static constexpr int32_t num_threads = 1024;
        const int32_t num_blocks = ceil_div(size_i * size_j, num_threads);
        reduce_basic<<<num_blocks, num_threads>>>(size_i, size_j, num_slices, partial_sums, c);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch reduce error: %s\n", cudaGetErrorString(e));
        }
    }
    */
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
