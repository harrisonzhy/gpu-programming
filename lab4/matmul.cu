#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
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

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
void matmul_cpu_naive(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float sum = 0.0;
            for (int32_t k = 0; k < size_k; ++k) {
                sum += a[i * size_k + k] * b[k * size_j + j];
            }
            c[i * size_j + j] = sum;
        }
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)

namespace matmul_l1 {

static constexpr int32_t T = 2;

static constexpr int32_t BIG_TILE_M = 128;
static constexpr int32_t BIG_TILE_N = 128;

static constexpr int32_t TILE_M = BIG_TILE_M / 2;
static constexpr int32_t TILE_N = BIG_TILE_N / 2;
static constexpr int32_t TILE_K = 64;

static constexpr int32_t N_OVERLAY = 2;

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
    
    for (int32_t M0 = blockIdx.x * BIG_TILE_M; M0 < (blockIdx.x + 1) * BIG_TILE_M; M0 += TILE_M) {
        for (int32_t N0 = blockIdx.y * BIG_TILE_N; N0 < (blockIdx.y + 1) * BIG_TILE_N; N0 += TILE_N) {

            float result[T][T] = {};

            // prefetch the first A (TILE_M x TILE_K) and B tiles (TILE_K x TILE_N)
            {
                int32_t buf = 0 % N_OVERLAY;
                float* shared_a_dst = shared_a + buf * (TILE_M * TILE_K);
                float* shared_b_dst = shared_b + buf * (TILE_K * TILE_N);
                const float* global_a_src = a + (M0 * size_k + 0 * TILE_K);
                const float* global_b_src = b + (0 * TILE_K) * size_j + N0;

                for (int32_t i = block_lin; i < TILE_M * TILE_K; i += (blockDim.x * blockDim.y)) {
                    int32_t row = i / TILE_K, col = i % TILE_K;
                    shared_a_dst[i] = global_a_src[row * size_k + col];
                }
                for (int32_t i = block_lin; i < TILE_K * TILE_N; i += (blockDim.x * blockDim.y)) {
                    int32_t row = i / TILE_N, col = i % TILE_N;
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

                    for (int32_t ty = 0; ty < T; ++ty) {
                        for (int32_t tx = 0; tx < T; ++tx) {

                            int32_t m = threadIdx.y * T + ty;
                            int32_t n = threadIdx.x * T + tx;

                            for (int32_t kk = 0; kk < TILE_K; ++kk) {
                                result[ty][tx] += shared_a_dst[m * TILE_K + kk] * shared_b_dst[kk * TILE_N + n];
                            }
                        }
                    }
                }

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
                    c[m * size_j + n] = result[ty][tx];
                }
            }
        }
    }
}

void launch_matmul_l1(
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

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

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

static constexpr int32_t BIG_TILE_M = 96;
static constexpr int32_t BIG_TILE_N = 96;

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
    const int32_t grid_size = 1 << (32 - __clz(max(gridDim.x, gridDim.y) - 1));
    
    int32_t bx, by;
    init_hilbert_block(&bx, &by);

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
                        }

                        for (int32_t kkk = 0; kkk < SMALL_TILE_K; ++kkk) {
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

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
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

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

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

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

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

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results) {
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

int main(int argc, char **argv) {
    std::string test_data_dir = ".";

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
