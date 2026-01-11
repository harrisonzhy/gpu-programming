#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
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

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void rle_compress_cpu(
    uint32_t raw_count,
    char const *raw,
    std::vector<char> &compressed_data,
    std::vector<uint32_t> &compressed_lengths) {
    compressed_data.clear();
    compressed_lengths.clear();

    uint32_t i = 0;
    while (i < raw_count) {
        char c = raw[i];
        uint32_t run_length = 1;
        i++;
        while (i < raw_count && raw[i] == c) {
            run_length++;
            i++;
        }
        compressed_data.push_back(c);
        compressed_lengths.push_back(run_length);
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

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
void cp_async_(void* smem_dst, const void* gmem_src) {
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
void cp_async4_(void* smem_dst, const void* gmem_src) {
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

namespace rle_gpu {

namespace scan_gpu_indicator {

static constexpr int32_t T = 4;
static constexpr int32_t W = 32 * T; // each warp does 32 * T elements
static constexpr int32_t num_warps_per_block = 32;
static constexpr int32_t block_stride = num_warps_per_block * 32;

static constexpr int32_t num_warmup = 16;

/* TODO: your GPU kernels here... */

template <typename Op>
__global__ void partial_reduce(
    size_t n,
    const char *x, /* note: always aligned to >= 256 bytes per CUDA standard */
    typename Op::Data *inter
) {
    using Data = typename Op::Data;

    const int32_t lane = threadIdx.x;
    const int32_t warp_idx = threadIdx.y;
    const int32_t warp_idx0 = blockIdx.x * (num_warps_per_block * W) + warp_idx * W;

    extern __shared__ __align__(16) char shared_mem_[];
    char* shared_mem = reinterpret_cast<char*>(shared_mem_);

    auto do_cp_async = [&](int32_t warp_offset /* 0..T-1 */, char* shared_dst) { 
        // TODO: use cp_async4_
        if (lane < 8) {
            // per-warp copy
            const int32_t dst_idx = warp_idx * 32;
            const int32_t src_idx = warp_idx0 + 32 * warp_offset;
            char* dst_tile = shared_dst + dst_idx;
            const char* src_tile = x + src_idx;
            const bool bound_ok = (src_idx + lane * 4 + 3 < n);

            if (bound_ok) {
                cp_async_<Op>(dst_tile + lane * 4, src_tile + lane * 4);
            } else {
                for (int32_t idx = 0; idx < 4; ++idx) {
                    const int32_t lane_idx = lane * 4 + idx;
                    if (src_idx + lane_idx < n) {
                        dst_tile[lane_idx] = src_tile[lane_idx];
                    }
                }
            }
        }
    };

    __shared__ char block_prev;
    if (warp_idx == 0 && lane == 0) {
        block_prev = (warp_idx0 > 0) ? x[warp_idx0 - 1] : 0;
    }
    __syncthreads();

    Data reg[T];
    auto do_shfl = [&](int32_t warp_offset, Data init_val, const char* shared_src) {
        const int32_t lane_idx = warp_idx * 32 + lane;
        const int32_t global_idx = warp_idx0 + 32 * warp_offset + lane;
        
        Data ind = Op::identity();
        if (global_idx < n) {
            if (global_idx == 0) {
                ind = 1;
            } else {
                char cur = shared_src[lane_idx];
                char prev;

                if (lane > 0) {
                    prev = shared_src[lane_idx - 1];
                } else {
                    if (warp_offset > 0) {
                        const char* prev_tile = shared_mem + block_stride * (warp_offset - 1);
                        prev = prev_tile[warp_idx * 32 + 31];
                    } else if (warp_idx > 0) {
                        const char* prev_warp_last_tile = shared_mem + block_stride * (T - 1);
                        prev = prev_warp_last_tile[(warp_idx - 1) * 32 + 31];
                        // prev = shared_src[lane_idx - 1];
                    } else {
                        prev = block_prev;
                    }
                }
                ind = (Data)(cur != prev);
            }
        }
        reg[warp_offset] = ind;
        
        Data sum;
        for (int32_t idx = 1; idx < 32; idx = idx * 2) {
            sum = __shfl_up_sync(0xFFFFFFFF, reg[warp_offset], idx);
            if (lane >= idx) {
                reg[warp_offset] = Op::combine(sum, reg[warp_offset]);
            }
        }
        reg[warp_offset] = Op::combine(init_val, reg[warp_offset]);
        if (global_idx < n) {
            inter[global_idx] = reg[warp_offset];
        }
    };

    for (int32_t t = 0; t < num_warmup; ++t) {
        do_cp_async(t, shared_mem + block_stride * t);
        async_commit_group();
    }
    async_wait_pending<num_warmup - T + 1>();

    for (int32_t t = 0; t < T; ++t) {
        if (t == 0) {
            do_shfl(t, Op::identity(), shared_mem + block_stride * t);
        } else {
            Data carry = __shfl_sync(0xFFFFFFFF, reg[t - 1], 31);
            do_shfl(t, carry, shared_mem + block_stride * t);
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
    Data* shared_mem_partials = shared_mem + T * block_stride;
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
            do_cp_async(t, shared_mem + block_stride * t);
            async_commit_group();
        }

        async_wait_pending<0>();
        __syncthreads();

        const int32_t next = t + 1;
        if (next < T) {
            do_cp_async(next, shared_mem + block_stride * next);
            async_commit_group();
        }

        const Data* stage = shared_mem + block_stride * t;
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

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    char *x, // pointer to GPU memory
    void *workspace_       // pointer to GPU memory
) {
    using Data = typename Op::Data;
    /* TODO: your CPU code here... */

    Data* out = reinterpret_cast<Data*>(workspace_);
    const int32_t shmem_size_tiles = max(T, num_warmup) * block_stride /* (this is num_warps_per_block * 32) */ * sizeof(Data);

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

} // namespace scan_gpu_indicator

namespace scan_gpu_normal {

static constexpr int32_t T = 4;
static constexpr int32_t W = 32 * T; // each warp does 32 * T elements
static constexpr int32_t num_warps_per_block = 16;
static constexpr int32_t block_stride = num_warps_per_block * 32;

template <typename Op>
__global__ void partial_reduce(
    size_t n,
    const typename Op::Data *x,
    typename Op::Data *inter
) {
    using Data = typename Op::Data;

    const int32_t lane = threadIdx.x;
    const int32_t warp_idx = threadIdx.y;
    const int32_t warp_idx0 = blockIdx.x * (num_warps_per_block * W) + warp_idx * W;

    extern __shared__ __align__(16) char shared_mem_[];
    Data* shared_mem = reinterpret_cast<Data*>(shared_mem_);

    auto do_cp_async = [&](int32_t warp_offset /* 0..T-1 */, Data* shared_dst) { 
        // per-warp copy
        const int32_t dst_idx = warp_idx * W + 128 * warp_offset;
        const int32_t src_idx = warp_idx0 + 128 * warp_offset;
        
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
    };

    Data reg[T];
    auto do_shfl = [&](int32_t warp_offset, Data init_val, const Data* shared_src) {
        // reg[warp_offset] = shared_src[warp_idx * 32 + lane];
        reg[warp_offset] = shared_src[lane];
        Data sum;
        for (int32_t idx = 1; idx < 32; idx = idx * 2) {
            sum = __shfl_up_sync(0xFFFFFFFF, reg[warp_offset], idx);
            if (lane >= idx) {
                reg[warp_offset] = Op::combine(sum, reg[warp_offset]);
            }
        }
        reg[warp_offset] = Op::combine(init_val, reg[warp_offset]);
        const int32_t global_idx = warp_idx0 + 32 * warp_offset + lane;
        if (global_idx < n) {
            inter[global_idx] = reg[warp_offset];
        }
    };

    for (int32_t t = 0; t < T / 4; ++t) {
        do_cp_async(t, shared_mem);
        async_commit_group();
    }
    async_wait_pending<0>();

    for (int32_t t = 0; t < T; ++t) {
        Data* shared_mem_tile = shared_mem + warp_idx * W + t * 32;
        if (t == 0) {
            do_shfl(t, Op::identity(), shared_mem_tile);
        } else {
            Data carry = __shfl_sync(0xFFFFFFFF, reg[t - 1], 31);
            do_shfl(t, carry, shared_mem_tile);
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

    extern __shared__ __align__(16) char shared_mem_[];
    Data* shared_mem = reinterpret_cast<Data*>(shared_mem_);

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
    Data* shared_mem_partials = shared_mem + T * block_stride;
    Data* shared_mem_block_carries = shared_mem_partials + num_warps_per_block;
 
    auto do_cp_async = [&](int32_t warp_offset /* 0..T-1 */, Data* shared_dst) {
        // per-warp copy
        const int32_t dst_idx = warp_idx * 128 * (T / 4) + 128 * warp_offset;
        const int32_t src_idx = warp_idx0 + 128 * warp_offset;
        
        Data* dst_tile = shared_dst + dst_idx;
        const Data* src_tile = inter + src_idx;
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
    const Data partial_reg = (warp_idx == 0) ? Op::identity() : shared_mem_partials[warp_idx - 1];

    for (int32_t t = 0; t < T / 4; ++t) {
        do_cp_async(t, shared_mem);
        async_commit_group();
    }
    async_wait_pending<0>();
    __syncthreads();

    for (int32_t t = 0; t < T; ++t) {
        Data* shared_mem_tile = shared_mem + warp_idx * (128 * (T / 4)) + (t / 4) * 128 + (t % 4) * 32;

        Data val = shared_mem_tile[lane];
        const int32_t idx = warp_idx0 + 32 * t + lane;
        if (idx < n) {
            val = Op::combine(partial_reg, val);
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

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    const typename Op::Data *x, // pointer to GPU memory
    void *workspace_       // pointer to GPU memory
) {
    using Data = typename Op::Data;
    /* TODO: your CPU code here... */

    Data* out = reinterpret_cast<Data*>(workspace_);
    const int32_t shmem_size_tiles = T * block_stride * sizeof(Data);

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

} // namespace scan_gpu_normal

static constexpr int32_t T = 4;
static constexpr int32_t W = 32 * T; // each warp does 32 * T elements
static constexpr int32_t num_warps_per_block = 16;

/* TODO: your GPU kernels here... */

template<typename Op>
__global__ void accum_counts(
    uint32_t n,
    const uint32_t *ind_scan_results,             // pointer to GPU buffer
    uint32_t *compressed_lengths // pointer to GPU buffer
) {
    using Data = Op::Data;

    const int32_t lane = threadIdx.x;
    const int32_t warp_idx = threadIdx.y;
    const int32_t warp_idx0 = blockIdx.x * (num_warps_per_block * W) + warp_idx * W;

    extern __shared__ __align__(16) char shared_mem_[];

    Data* shared_mem = reinterpret_cast<Data*>(shared_mem_);
    Data* shared_mem_lengths = shared_mem + (num_warps_per_block * W) + warp_idx * W;

    for (int32_t i = lane; i < W; i += blockDim.x) {
        shared_mem_lengths[i] = 0;
    }

    auto do_cp_async = [&](int32_t warp_offset, Data* shared_dst) {
        const int32_t dst_idx = warp_idx * W + 128 * warp_offset;
        const int32_t src_idx = warp_idx0 + 128 * warp_offset;

        Data* dst_tile = shared_dst + dst_idx;
        const Data* src_tile = ind_scan_results + src_idx;
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
    };

    const uint32_t num_runs = ind_scan_results[n - 1];

    for (int32_t t = 0; t < T / 4; ++t) {
        do_cp_async(t, shared_mem);
        async_commit_group();
    }
    async_wait_pending<0>();
    __syncthreads();

    const Data warp_count0 = shared_mem[warp_idx * W];
    auto count_shared = [&](int32_t t) {
        Data* shared_mem_tile = shared_mem + warp_idx * W + t * 32;
        const int32_t idx = t * 32 + lane;
        if (warp_idx0 + idx < n) {
            Data count = shared_mem_tile[lane] - warp_count0;
            if (count < W) {
                atomicAdd(&shared_mem_lengths[count], 1);
            }
        }
    };

    for (int32_t t = 0; t < T; ++t) {
        count_shared(t);
    }

    for (int32_t idx = lane; idx < W; idx += blockDim.x) {
        const int32_t out = idx + warp_count0 - 1;
        if (out < num_runs) {
            atomicAdd(&compressed_lengths[out], shared_mem_lengths[idx]);
        }
    }
}

static constexpr int32_t T_sparse = 16;

__global__ void sparse_memcpy(
    const uint32_t* sparse_indices,
    int32_t sparse_count,
    const char* src,
    char* dst
) {
    const int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    const int tid_global = blockIdx.x * (blockDim.x * blockDim.y) + tid_in_block;
    const int idx_T = tid_global * T_sparse;

    if (idx_T == 0) {
        dst[0] = src[0];
    }

    for (int32_t idx_ = 0; idx_ < T_sparse; ++idx_) {
        const int32_t idx = idx_T + idx_;
        if (idx < sparse_count) {
            const uint32_t sparse_idx = sparse_indices[idx] - 1;
            dst[idx] = src[sparse_idx];
        }
    }
}

// Returns desired size of scratch buffer in bytes.
size_t get_workspace_size(uint32_t raw_count) {
    /* TODO: your CPU code here... */
    const size_t size = scan_gpu_indicator::get_workspace_size<SumOp>(raw_count) 
                      + scan_gpu_normal::get_workspace_size<SumOp>(raw_count) + 16;
    return size;
}

// 'launch_rle_compress'
//
// Input:
//
//   'raw_count': Number of bytes in the input buffer 'raw'.
//
//   'raw': Uncompressed bytes in GPU memory.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size'.
//
// Output:
//
//   Returns: 'compressed_count', the number of runs in the compressed data.
//
//   'compressed_data': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' bytes of this buffer
//   with the compressed data.
//
//   'compressed_lengths': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' integers in this buffer
//   with the lengths of the runs in the compressed data.
//
uint32_t launch_rle_compress(
    uint32_t raw_count,
    char *raw,             // pointer to GPU buffer
    void *workspace,             // pointer to GPU buffer
    char *compressed_data,       // pointer to GPU buffer
    uint32_t *compressed_lengths // pointer to GPU buffer
) {
    /* TODO: your CPU code here... */
    using Data = SumOp::Data;

    uint32_t compressed_count;

    uint32_t* ind_scan_results;
    {
        ind_scan_results = scan_gpu_indicator::launch_scan<SumOp>(raw_count, raw, workspace);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch ind scan error: %s\n", cudaGetErrorString(e));
            exit(1);
        }
        cudaMemcpy(&compressed_count, ind_scan_results + (raw_count - 1), sizeof(Data), cudaMemcpyDeviceToHost);
    }
    {
        const dim3 block(32, num_warps_per_block, 1);
        const int32_t grid = ceil_div(raw_count, num_warps_per_block * W);
        const int32_t shmem_size = 2 * (num_warps_per_block * W) * sizeof(Data);

        cudaFuncSetAttribute(
            accum_counts<SumOp>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size 
        );
        accum_counts<SumOp><<<grid, block, shmem_size>>>(raw_count, ind_scan_results, compressed_lengths);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch accum error: %s\n", cudaGetErrorString(e));
            exit(1);
        }
    }
    uint32_t* cc_scan_results;
    {
        void* cc_workspace_ = reinterpret_cast<uint8_t*>(workspace) + scan_gpu_indicator::get_workspace_size<SumOp>(raw_count);
        void* cc_workspace = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(cc_workspace_) + 15) & ~15);
        cc_scan_results = scan_gpu_normal::launch_scan<SumOp>(compressed_count, compressed_lengths, cc_workspace);
    
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch accum error: %s\n", cudaGetErrorString(e));
            exit(1);
        }
    }
    {
        static constexpr int32_t num_warps_per_block_ = 8;
        const dim3 block(32, num_warps_per_block_, 1);
        const int32_t grid = ceil_div(compressed_count, block.x * block.y * T_sparse);
        sparse_memcpy<<<grid, block>>>(cc_scan_results, compressed_count, raw, compressed_data);

        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("Launch accum error: %s\n", cudaGetErrorString(e));
            exit(1);
        }
    }
    return compressed_count;
}

} // namespace rle_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

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
};

enum class Mode {
    TEST,
    BENCHMARK,
};

Results run_config(Mode mode, std::vector<char> const &raw) {
    // Allocate buffers
    size_t workspace_size = rle_gpu::get_workspace_size(raw.size());
    char *raw_gpu;
    void *workspace;
    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;
    CUDA_CHECK(cudaMalloc(&raw_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&compressed_lengths_gpu, raw.size() * sizeof(uint32_t)));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(raw_gpu, raw.data(), raw.size(), cudaMemcpyHostToDevice));

    auto reset = [&]() {
        CUDA_CHECK(cudaMemset(compressed_data_gpu, 0, raw.size()));
        CUDA_CHECK(cudaMemset(compressed_lengths_gpu, 0, raw.size() * sizeof(uint32_t)));
    };

    auto f = [&]() {
        rle_gpu::launch_rle_compress(
            raw.size(),
            raw_gpu,
            workspace,
            compressed_data_gpu,
            compressed_lengths_gpu);
    };

    // Test correctness
    reset();
    uint32_t compressed_count = rle_gpu::launch_rle_compress(
        raw.size(),
        raw_gpu,
        workspace,
        compressed_data_gpu,
        compressed_lengths_gpu);
    std::vector<char> compressed_data(compressed_count);
    std::vector<uint32_t> compressed_lengths(compressed_count);
    CUDA_CHECK(cudaMemcpy(
        compressed_data.data(),
        compressed_data_gpu,
        compressed_count,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        compressed_lengths.data(),
        compressed_lengths_gpu,
        compressed_count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    std::vector<char> compressed_data_expected;
    std::vector<uint32_t> compressed_lengths_expected;
    rle_compress_cpu(
        raw.size(),
        raw.data(),
        compressed_data_expected,
        compressed_lengths_expected);

    bool correct = true;
    if (compressed_count != compressed_data_expected.size()) {
        printf("Mismatch in compressed count:\n");
        printf("  Expected: %zu\n", compressed_data_expected.size());
        printf("  Actual:   %u\n", compressed_count);
        correct = false;
    }
    if (correct) {
        for (size_t i = 0; i < compressed_data_expected.size(); i++) {
            if (compressed_data[i] != compressed_data_expected[i]) {
                printf("Mismatch in compressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(compressed_data_expected[i]));
                printf(
                    "  Actual:   0x%02x\n",
                    static_cast<unsigned char>(compressed_data[i]));
                correct = false;
                break;
            }
            if (compressed_lengths[i] != compressed_lengths_expected[i]) {
                printf("Mismatch in compressed lengths at index %zu:\n", i);
                printf("  Expected: %u\n", compressed_lengths_expected[i]);
                printf("  Actual:   %u\n", compressed_lengths[i]);
                correct = false;
                break;
            }
        }
    }
    if (!correct) {
        if (raw.size() <= 1024) {
            printf("\nInput:\n");
            for (size_t i = 0; i < raw.size(); i++) {
                printf("  [%4zu] = 0x%02x\n", i, static_cast<unsigned char>(raw[i]));
            }
            printf("\nExpected:\n");
            for (size_t i = 0; i < compressed_data_expected.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data_expected[i]),
                    compressed_lengths_expected[i]);
            }
            printf("\nActual:\n");
            if (compressed_data.size() == 0) {
                printf("  (empty)\n");
            }
            for (size_t i = 0; i < compressed_data.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data[i]),
                    compressed_lengths[i]);
            }
        }
        exit(1);
    }

    if (mode == Mode::TEST) {
        return {};
    }

    // Benchmark
    double target_time_ms = 1000.0;
    double time_ms = benchmark_ms(target_time_ms, reset, f);

    // Cleanup
    CUDA_CHECK(cudaFree(raw_gpu));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(compressed_data_gpu));
    CUDA_CHECK(cudaFree(compressed_lengths_gpu));

    return {time_ms};
}

template <typename Rng> std::vector<char> generate_test_data(uint32_t size, Rng &rng) {
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());
    constexpr uint32_t alphabet_size = 4;
    auto alphabet = std::vector<char>();
    for (uint32_t i = 0; i < alphabet_size; i++) {
        alphabet.push_back(random_byte(rng));
    }
    auto random_symbol = std::uniform_int_distribution<uint32_t>(0, alphabet_size - 1);
    auto data = std::vector<char>();
    for (uint32_t i = 0; i < size; i++) {
        data.push_back(alphabet.at(random_symbol(rng)));
    }
    return data;
}

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto test_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1 << 10,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
    };

    printf("Correctness:\n\n");
    for (auto test_size : test_sizes) {
        auto raw = generate_test_data(test_size, rng);
        printf("  Testing compression for size %u\n", test_size);
        run_config(Mode::TEST, raw);
        printf("  OK\n\n");
    }

    auto test_data_search_paths = std::vector<std::string>{".", "/"};
    std::string test_data_path;
    for (auto test_data_search_path : test_data_search_paths) {
        auto candidate_path = test_data_search_path + "/rle_raw.bmp";
        if (std::filesystem::exists(candidate_path)) {
            test_data_path = candidate_path;
            break;
        }
    }
    if (test_data_path.empty()) {
        printf("Could not find test data file.\n");
        exit(1);
    }

    auto raw = std::vector<char>();
    {
        auto file = std::ifstream(test_data_path, std::ios::binary);
        if (!file) {
            printf("Could not open test data file '%s'.\n", test_data_path.c_str());
            exit(1);
        }
        file.seekg(0, std::ios::end);
        raw.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(raw.data(), raw.size());
    }

    printf("Performance:\n\n");
    printf("  Testing compression on file 'rle_raw.bmp' (size %zu)\n", raw.size());
    auto results = run_config(Mode::BENCHMARK, raw);
    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}