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

static constexpr int32_t T = 2; // thread processes TxT output tile
static constexpr int32_t W = 32;
static constexpr int32_t H = 32;
static constexpr int32_t K = 32;

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
    float* shared_b = shared_mem + K * K;

    // global indices
    const int32_t tile_height = blockDim.y * T;
    const int32_t tile_width = blockDim.x * T;
    const int32_t block_i0 = tile_height * blockIdx.y;
    const int32_t block_j0 = tile_width * blockIdx.x;

    // thread offset/scratchpad indices
    const int32_t thread_i0 = threadIdx.y * T;
    const int32_t thread_j0 = threadIdx.x * T;

    float results[T * T] = {0};

    // accumulate outer product in chunks of K
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

        // accumulate using scratchpad
        for (int32_t ii = 0; ii < T; ++ii) {
            for (int32_t jj = 0; jj < T; ++jj) {
                const int32_t i = thread_i0 + ii;
                const int32_t j = thread_j0 + jj;
                for (int32_t k = 0; k < K; ++k) {
                    results[ii * T + jj] += shared_a[i * K + k] * shared_b[k * K + j];
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

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    /* TODO: your CPU code here */

    auto ceil_div = [](int32_t a, int32_t b) -> int32_t { return (a + b - 1) / b; };

    dim3 grid(ceil_div(size_i, matmul_l1::W), ceil_div(size_k, matmul_l1::H), 1);
    dim3 block(ceil_div(matmul_l1::K, T), ceil_div(matmul_l1::K, T), 1);

    static constexpr int32_t shmem_size = 2 * K * K * sizeof(float);
    matmul_l1<<<grid, block, shmem_size>>>(size_i, size_j, size_k, a, b, c);
}

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

namespace matmul_l1_reg {

static constexpr int32_t T = 4; // thread processes TxT output tile
static constexpr int32_t W = 32;
static constexpr int32_t H = 32;
static constexpr int32_t K = 32;

__global__ void matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
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
    /* TODO: your CPU code here */

    auto ceil_div = [](int32_t a, int32_t b) -> int32_t { return (a + b - 1) / b; };

    dim3 grid(ceil_div(size_i, matmul_l1_reg::W), ceil_div(size_k, matmul_l1_reg::H), 1);
    dim3 block(ceil_div(matmul_l1_reg::K, T), ceil_div(matmul_l1_reg::K, T), 1);

    static constexpr int32_t shmem_size = 2 * K * K * sizeof(float);
    matmul_l1_reg<<<grid, block, shmem_size>>>(size_i, size_j, size_k, a, b, c);
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
