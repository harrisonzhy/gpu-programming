#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

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

template <typename Op>
void print_array(
    size_t n,
    typename Op::Data const *x // allowed to be either a CPU or GPU pointer
);

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

template <typename Op>
void scan_cpu(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();
    for (size_t i = 0; i < n; i++) {
        accumulator = Op::combine(accumulator, x[i]);
        out[i] = accumulator;
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace scan_gpu {

static constexpr int32_t T = 4;
static constexpr int32_t W = 32 * T; // each warp does 32 * T elements

static constexpr int32_t num_warps_per_block = 32;
static constexpr int32_t stage_stride = num_warps_per_block * 32;

static constexpr int32_t num_warmup = 16;

__forceinline__ int32_t ceil_div(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template<typename Op>
__device__ __forceinline__ 
void cp_async_(typename Op::Data* smem_dst, const typename Op::Data* gmem_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_addr)
        : "memory"
    );
}

template<typename Op>
__device__ __forceinline__ 
void cp_async4_(typename Op::Data* smem_dst, const typename Op::Data* gmem_src) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_dst);
    unsigned long long gmem_addr = reinterpret_cast<unsigned long long>(gmem_src);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
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

template <typename Op>
__global__ void partial_reduce(
    size_t n,
    typename Op::Data *x,
    typename Op::Data *inter
) {
    using Data = typename Op::Data;

    const int32_t lane = threadIdx.x;
    const int32_t warp_idx = threadIdx.y;
    const int32_t warp_idx0 = blockIdx.x * (num_warps_per_block * W) + warp_idx * W;

    extern __shared__ __align__(16) char shared_mem_[];
    Data* shared_mem = reinterpret_cast<Data*>(shared_mem_);

    auto do_cp_async = [&](int32_t warp_offset /* 0..T-1 */, Data* shared_dst) { 
        if (lane < 8) {
            // per-warp copy
            const int32_t dst_idx = warp_idx * 32;
            const int32_t src_idx = warp_idx0 + 32 * warp_offset;
            Data* dst_tile = shared_dst + dst_idx;
            const Data* src_tile = x + src_idx;
            const bool bound_ok = (src_idx + lane * 4 + 3 < n);

            if (bound_ok) {
                cp_async4_<Op>(dst_tile + lane * 4, src_tile + lane * 4);
            } else {
                for (int32_t idx = 0; idx < 4; ++idx) {
                    if (src_idx + lane * 4 + idx < n) {
                        cp_async_<Op>(dst_tile + lane * 4 + idx, src_tile + lane * 4 + idx);
                    }
                }
            }
        }
    };

    Data reg[T];
    auto do_shfl = [&](int32_t warp_offset, Data init_val, const Data* shared_src) {
        reg[warp_offset] = shared_src[warp_idx * 32 + lane];
        Data sum;
        for (int32_t idx = 1; idx < 32; idx = idx * 2) {
            sum = __shfl_up_sync(0xFFFFFFFF, reg[warp_offset], idx);
            if (lane >= idx) {
                reg[warp_offset] = Op::combine(sum, reg[warp_offset]);
            }
        }
        reg[warp_offset] = Op::combine(init_val, reg[warp_offset]);
        inter[warp_idx0 + 32 * warp_offset + lane] = reg[warp_offset];
    };

    for (int32_t t = 0; t < num_warmup; ++t) {
        do_cp_async(t, shared_mem + stage_stride * t);
        async_commit_group();
    }
    async_wait_pending<num_warmup - T + 1>();

    for (int32_t t = 0; t < T; ++t) {
        if (t == 0) {
            do_shfl(t, Op::identity(), shared_mem + stage_stride * t);
        } else {
            Data carry = __shfl_sync(0xFFFFFFFF, reg[t - 1], 31);
            do_shfl(t, carry, shared_mem + stage_stride * t);
        }
    }
}

template <typename Op>
__global__ void warp_fixup(
    size_t n,
    const typename Op::Data *inter,
    typename Op::Data *block_carries
) {
    using Data = typename Op::Data;

    const int32_t lane = threadIdx.x; // 0..32
    const int32_t warp_idx = threadIdx.y;
    const int32_t warp_idx0 = blockIdx.x * (num_warps_per_block * W) + warp_idx * W;

    extern __shared__ __align__(16) char shared_mem_[];
    Data* shared_mem = reinterpret_cast<Data*>(shared_mem_);
 
    auto do_cp_async = [&](int32_t warp_offset /* 0..T-1 */, Data* shared_dst) {
        // per-warp copy
        int32_t dst_idx = warp_idx * 32;
        int32_t src_idx = warp_idx0 + 32 * warp_offset;
        Data* dst_tile = shared_dst + dst_idx;
        const Data* src_tile = inter + src_idx;
        bool bound_ok = (src_idx + lane * 4 + 3 < n);
        if (lane < 8) {
            if (bound_ok) {
                cp_async4_<Op>(dst_tile + lane * 4, src_tile + lane * 4);
            } else {
                for (int32_t idx = 0; idx < 4; ++idx) {
                    if (src_idx + lane * 4 + idx < n) {
                        cp_async_<Op>(dst_tile + lane * 4 + idx, src_tile + lane * 4 + idx);
                    }
                }
            }
        }
    };

    auto fetch_last_partials = [&]() {
        if (warp_idx == 0 && lane == 0) {
            const int32_t block_base = blockIdx.x * (num_warps_per_block * W);
            for (int32_t w = 0; w < num_warps_per_block; ++w) {
                const int32_t warp_base = block_base + w * W;
                const int32_t end = min(warp_base + W, (int32_t)n) - 1;
                if (end >= warp_base) {
                    cp_async_<Op>(shared_mem + w, inter + end);
                }
            }
        }
    };

    auto scan_shared_partials = [&]() {
        if (warp_idx == 0) {
            Data val = (lane < num_warps_per_block) ? shared_mem[lane] : Op::identity();
            Data sum;
            for (int32_t idx = 1; idx < 32; idx = idx * 2) {
                sum = __shfl_up_sync(0xFFFFFFFF, val, idx);
                if (lane >= idx) {
                    val = Op::combine(sum, val);
                }
            }
            if (lane < num_warps_per_block) {
                shared_mem[lane] = val;
            }
        }
    };

    fetch_last_partials();
    async_commit_group();
    async_wait_pending<0>();

    scan_shared_partials();

    if (warp_idx == 0 && lane == 0) {
        block_carries[blockIdx.x] = shared_mem[num_warps_per_block - 1];
    }
}

template <typename Op>
__global__ void finalize(
    size_t n,
    typename Op::Data *inter,
    const typename Op::Data *block_carries
) {
    using Data = typename Op::Data;

    const int32_t lane = threadIdx.x; // 0..32
    const int32_t warp_idx = threadIdx.y;
    const int32_t warp_idx0 = blockIdx.x * (num_warps_per_block * W) + warp_idx * W;

    extern __shared__ __align__(16) char shared_mem_[];
    Data* shared_mem = reinterpret_cast<Data*>(shared_mem_);
    Data* shared_mem_partials = shared_mem + T * stage_stride;
    Data* shared_mem_block_carries = shared_mem_partials + num_warps_per_block;
 
    auto do_cp_async = [&](int32_t warp_offset /* 0..T-1 */, Data* shared_dst) {
        // per-warp copy
        int32_t dst_idx = warp_idx * 32;
        int32_t src_idx = warp_idx0 + 32 * warp_offset;
        Data* dst_tile = shared_dst + dst_idx;
        const Data* src_tile = inter + src_idx;
        bool bound_ok = (src_idx + lane * 4 + 3 < n);
        if (lane < 8) {
            if (bound_ok) {
                cp_async4_<Op>(dst_tile + lane * 4, src_tile + lane * 4);
            } else {
                for (int32_t idx = 0; idx < 4; ++idx) {
                    if (src_idx + lane * 4 + idx < n) {
                        cp_async_<Op>(dst_tile + lane * 4 + idx, src_tile + lane * 4 + idx);
                    }
                }
            }
        }
    };

    auto fetch_last_partials = [&]() {
        if (warp_idx == 0) {
            const int32_t block_base = blockIdx.x * (num_warps_per_block * W);
            if (lane < num_warps_per_block) {
                const int32_t warp_base = block_base + lane * W;
                const int32_t end = min(warp_base + W, (int32_t)n) - 1;
                if (end >= warp_base) {
                    cp_async_<Op>(shared_mem_partials + lane, inter + end);
                }
            }
        }
    };

    auto scan_shared_partials = [&]() {
        if (warp_idx == 0) {
            Data val = (lane < num_warps_per_block) ? shared_mem_partials[lane] : Op::identity();
            Data sum;
            for (int32_t idx = 1; idx < 32; idx = idx * 2) {
                sum = __shfl_up_sync(0xFFFFFFFF, val, idx);
                if (lane >= idx) {
                    val = Op::combine(sum, val);
                }
            }
            if (lane < num_warps_per_block) {
                shared_mem_partials[lane] = val;
            }
        }
    };

    if (blockIdx.x > 0 && warp_idx == 0) {
        cp_async_<Op>(shared_mem_block_carries, block_carries + blockIdx.x - 1);
        async_commit_group();
    }

    fetch_last_partials();
    async_commit_group();
    async_wait_pending<0>();

    scan_shared_partials();
    __syncthreads();

    const Data block_carry = (blockIdx.x == 0) ? Op::identity() : shared_mem_block_carries[0];
    const Data partial_reg = shared_mem_partials[warp_idx - 1];

    for (int32_t t = 0; t < T; ++t) {
        if (t == 0) {
            // initiate prefetch for t = 0
            do_cp_async(t, shared_mem + stage_stride * t);
            async_commit_group();
        }

        async_wait_pending<0>();
        __syncthreads();

        const int32_t next = t + 1;
        if (next < T) {
            do_cp_async(next, shared_mem + stage_stride * next);
            async_commit_group();
        }

        const Data* stage = shared_mem + stage_stride * t;
        Data val = stage[warp_idx * 32 + lane];

        __syncthreads();

        const int32_t idx = warp_idx0 + 32 * t + lane;
        if (idx < n) {
            Data carry = (warp_idx == 0) ? Op::identity() : partial_reg;
            val = Op::combine(carry, val);
            inter[idx] = Op::combine(val, block_carry);
        }
    }
}

// Returns desired size of scratch buffer in bytes.
template <typename Op> size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;
    /* TODO: your CPU code here... */
    // note: factor of 2 accomodates block_carries matrix sufficiently
    return 2 * sizeof(Data) * n;
}

// 'launch_scan'
//
// Input:
//
//   'n': Number of elements in the input array 'x'.
//
//   'x': Input array in GPU memory. The 'launch_scan' function is allowed to
//   overwrite the contents of this buffer.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size<Op>(n)'.
//
// Output:
//
//   Returns a pointer to GPU memory which will contain the results of the scan
//   after all launched kernels have completed. Must be either a pointer to the
//   'x' buffer or to an offset within the 'workspace' buffer.
//
//   The contents of the output array should be "partial reductions" of the
//   input; each element 'i' of the output array should be given by:
//
//     output[i] = Op::combine(x[0], x[1], ..., x[i])
//
//   where 'Op::combine(...)' of more than two arguments is defined in terms of
//   repeatedly combining pairs of arguments. Note that 'Op::combine' is
//   guaranteed to be associative, but not necessarily commutative, so
//
//        Op::combine(a, b, c)              // conceptual notation; not real C++
//     == Op::combine(a, Op::combine(b, c)) // real C++
//     == Op::combine(Op::combine(a, b), c) // real C++
//
//  but we don't necessarily have
//
//    Op::combine(a, b) == Op::combine(b, a) // not true in general!
//
template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    void *workspace_       // pointer to GPU memory
) {
    using Data = typename Op::Data;
    /* TODO: your CPU code here... */

    Data* out = reinterpret_cast<Data*>(workspace_);
    const int32_t shmem_size_tiles = max(T, num_warmup) * stage_stride /* (this is num_warps_per_block * 32) */ * sizeof(Data);

    const dim3 block(32, num_warps_per_block, 1);
    const int32_t grid = ceil_div(n, num_warps_per_block * W);
    {
        cudaFuncSetAttribute(
            partial_reduce<Op>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size_tiles
        );
        partial_reduce<Op><<<grid, block, shmem_size_tiles>>>(n, x, out);
    }
    
    Data* block_carries = reinterpret_cast<Data*>(out) + n;
    const int32_t shmem_size_partial = num_warps_per_block * sizeof(Data);
    {
        const int32_t shmem_size = shmem_size_partial;
        cudaFuncSetAttribute(
            warp_fixup<Op>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );
        warp_fixup<Op><<<grid, block, shmem_size>>>(n, out, block_carries);
    }
    {
        std::vector<Data> block_carries_(grid);
        cudaMemcpy(block_carries_.data(), block_carries, grid * sizeof(Data), cudaMemcpyDeviceToHost);
        for (int32_t idx = 1; idx < grid; ++idx) {
            block_carries_[idx] = Op::combine(block_carries_[idx - 1], block_carries_[idx]);
        }
        cudaMemcpy(block_carries, block_carries_.data(), grid * sizeof(Data), cudaMemcpyHostToDevice);
    }
    const int32_t shmem_size_carry = 1 * sizeof(Data);
    {
        const int32_t shmem_size = shmem_size_tiles + shmem_size_partial + shmem_size_carry;
        cudaFuncSetAttribute(
            finalize<Op>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );
        finalize<Op><<<grid, block, shmem_size>>>(n, out, block_carries);
    }

    return out; // replace with an appropriate pointer
}

} // namespace scan_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

struct DebugRange {
    uint32_t lo;
    uint32_t hi;

    static constexpr uint32_t INVALID = 0xffffffff;

    static __host__ __device__ __forceinline__ DebugRange invalid() {
        return {INVALID, INVALID};
    }

    __host__ __device__ __forceinline__ bool operator==(const DebugRange &other) const {
        return lo == other.lo && hi == other.hi;
    }

    __host__ __device__ __forceinline__ bool operator!=(const DebugRange &other) const {
        return !(*this == other);
    }

    __host__ __device__ bool is_empty() const { return lo == hi; }

    __host__ __device__ bool is_valid() const { return lo != INVALID; }

    std::string to_string() const {
        if (lo == INVALID) {
            return "INVALID";
        } else {
            return std::to_string(lo) + ":" + std::to_string(hi);
        }
    }
};

struct DebugRangeConcatOp {
    using Data = DebugRange;

    static __host__ __device__ __forceinline__ Data identity() { return {0, 0}; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        if (a.is_empty()) {
            return b;
        } else if (b.is_empty()) {
            return a;
        } else if (a.is_valid() && b.is_valid() && a.hi == b.lo) {
            return {a.lo, b.hi};
        } else {
            return Data::invalid();
        }
    }

    static std::string to_string(Data d) { return d.to_string(); }
};

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

constexpr size_t max_print_array_output = 1025;
static thread_local size_t total_print_array_output = 0;

template <typename Op> void print_array(size_t n, typename Op::Data const *x) {
    using Data = typename Op::Data;

    // copy 'x' from device to host if necessary
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, x));
    auto x_host_buf = std::vector<Data>();
    Data const *x_host_ptr = nullptr;
    if (attr.type == cudaMemoryTypeDevice) {
        x_host_buf.resize(n);
        x_host_ptr = x_host_buf.data();
        CUDA_CHECK(
            cudaMemcpy(x_host_buf.data(), x, n * sizeof(Data), cudaMemcpyDeviceToHost));
    } else {
        x_host_ptr = x;
    }

    if (total_print_array_output >= max_print_array_output) {
        return;
    }

    printf("[\n");
    for (size_t i = 0; i < n; i++) {
        auto s = Op::to_string(x_host_ptr[i]);
        printf("  [%zu] = %s,\n", i, s.c_str());
        total_print_array_output++;
        if (total_print_array_output > max_print_array_output) {
            printf("  ... (output truncated)\n");
            break;
        }
    }
    printf("]\n");

    if (total_print_array_output >= max_print_array_output) {
        printf("(Reached maximum limit on 'print_array' output; skipping further calls "
               "to 'print_array')\n");
    }

    total_print_array_output++;
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
    double bandwidth_gb_per_sec;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename Op>
Results run_config(Mode mode, std::vector<typename Op::Data> const &x) {
    // Allocate buffers
    using Data = typename Op::Data;
    size_t n = x.size();
    size_t workspace_size = scan_gpu::get_workspace_size<Op>(n);
    Data *x_gpu;
    Data *workspace_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
    CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));

    // Test correctness
    auto expected = std::vector<Data>(n);
    scan_cpu<Op>(n, x.data(), expected.data());
    auto out_gpu = scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu);
    if (out_gpu == nullptr) {
        printf("'launch_scan' function not yet implemented (returned nullptr)\n");
        exit(1);
    }
    auto actual = std::vector<Data>(n);
    CUDA_CHECK(
        cudaMemcpy(actual.data(), out_gpu, n * sizeof(Data), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (actual.at(i) != expected.at(i)) {
            auto actual_str = Op::to_string(actual.at(i));
            auto expected_str = Op::to_string(expected.at(i));
            printf(
                "Mismatch at position %zu: %s != %s\n",
                i,
                actual_str.c_str(),
                expected_str.c_str());
            if (n <= 128) {
                printf("Input:\n");
                print_array<Op>(n, x.data());
                printf("\nExpected:\n");
                print_array<Op>(n, expected.data());
                printf("\nActual:\n");
                print_array<Op>(n, actual.data());
            }
            exit(1);
        }
    }
    if (mode == Mode::TEST) {
        return {0.0, 0.0};
    }

    // Benchmark
    double target_time_ms = 200.0;
    double time_ms = benchmark_ms(
        target_time_ms,
        [&]() {
            CUDA_CHECK(
                cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
        },
        [&]() { scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu); });
    double bytes_processed = n * sizeof(Data) * 2;
    double bandwidth_gb_per_sec = bytes_processed / time_ms / 1e6;

    // Cleanup
    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(workspace_gpu));

    return {time_ms, bandwidth_gb_per_sec};
}

std::vector<DebugRange> gen_debug_ranges(uint32_t n) {
    auto ranges = std::vector<DebugRange>();
    for (uint32_t i = 0; i < n; ++i) {
        ranges.push_back({i, i + 1});
    }
    return ranges;
}

template <typename Rng> std::vector<uint32_t> gen_random_data(Rng &rng, uint32_t n) {
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);
    auto data = std::vector<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        data.push_back(uniform(rng));
    }
    return data;
}

template <typename Op, typename GenData>
void run_tests(std::vector<uint32_t> const &sizes, GenData &&gen_data) {
    for (auto size : sizes) {
        auto data = gen_data(size);
        printf("  Testing size %8u\n", size);
        run_config<Op>(Mode::TEST, data);
        printf("  OK\n\n");
    }
}

int main(int argc, char const *const *argv) {
    auto correctness_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1024,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
        64 << 20,
    };

    auto rng = std::mt19937(0xCA7CAFE);

    printf("Correctness:\n\n");
    // printf("Testing scan operation: debug range concatenation\n\n");
    // run_tests<DebugRangeConcatOp>(correctness_sizes, gen_debug_ranges);
    printf("Testing scan operation: integer sum\n\n");
    run_tests<SumOp>(correctness_sizes, [&](uint32_t n) {
        return gen_random_data(rng, n);
    });

    printf("Performance:\n\n");

    size_t n = 64 << 20;
    auto data = gen_random_data(rng, n);

    printf("Benchmarking scan operation: integer sum, size %zu\n\n", n);

    // Warmup
    run_config<SumOp>(Mode::BENCHMARK, data);
    // Benchmark
    auto results = run_config<SumOp>(Mode::BENCHMARK, data);
    printf("  Time: %.2f ms\n", results.time_ms);
    printf("  Throughput: %.2f GB/s\n", results.bandwidth_gb_per_sec);

    return 0;
}