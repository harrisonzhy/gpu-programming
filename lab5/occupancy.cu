#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Simple Memcpy Kernel
////////////////////////////////////////////////////////////////////////////////
template <int TILE> __global__ void memcpy(float *dst, const float *src) {
    int iblock = blockIdx.x + blockIdx.y * gridDim.x;
    int index = threadIdx.x + TILE * iblock * blockDim.x;

    float a[TILE];

#pragma unroll
    for (int i = 0; i < TILE; i++) {
        a[i] = src[index + i * blockDim.x];
    }

#pragma unroll
    for (int i = 0; i < TILE; i++) {
        dst[index + i * blockDim.x] = a[i];
    }
}


////////////////////////////////////////////////////////////////////////////////
// Prelab Question 3: Fill in the shared memory sizes you want to run the
// kernel with. Changing these values will limit the occupancy of the kernel.
////////////////////////////////////////////////////////////////////////////////
inline std::vector<int> shared_memory_configuration() { return {0}; }

////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int size = 100 * 1024 * 1024;
    float *src, *dst;
    cudaMalloc(&src, size * sizeof(float));
    cudaMalloc(&dst, size * sizeof(float));

    // Host buffer for L2 invalidation
    float *h_src = (float *)malloc(size * sizeof(float));

    const int TILE = 4; // TILE size for memcpy kernel
    const int threads = 64;
    const int blocks = (size + TILE * threads - 1) / (TILE * threads);

    // Shared memory configurations
    std::vector<int> smem_sizes = shared_memory_configuration();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Print table header
    printf(
        "\n%-12s %-10s %-12s %-12s %-10s %-10s %-14s\n",
        "SharedMem",
        "Time(ms)",
        "BW(GB/s)",
        "Eff(%)",
        "Occ(%)",
        "Blocks/SM",
        "Bytes in flight");
    printf("-----------------------------------------------------------------------------"
           "----------------\n");

    cudaFuncSetAttribute(
        memcpy<TILE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        prop.sharedMemPerBlockOptin);

    for (int shared_mem : smem_sizes) {

        // Benchmark
        cudaEventRecord(start);
        memcpy<TILE><<<blocks, threads, shared_mem>>>(dst, src);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time;
        cudaEventElapsedTime(&time, start, stop);

        double bytes = 2.0 * size * sizeof(float); // read + write
        double bw = bytes * 1000.0 / (time * 1e9);

        int numBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks,
            memcpy<TILE>,
            threads,
            shared_mem);

        float occupancy = ((numBlocks * threads / 32) / 48.0f) * 100.0f;
        int bytes_per_thread = TILE * (int)sizeof(float) * 48 * 64 * numBlocks;

        printf(
            "%-12d %-10.3f %-12.3f %-12.1f %-10.1f %-10d %-14d\n",
            shared_mem,
            time,
            bw,
            100.0 * bw / (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6),
            occupancy,
            numBlocks,
            bytes_per_thread);
        // Invalidate L2 cache by copying "fresh" data to device
        for (int i = 0; i < size; i++) {
            h_src[i] = (float)(i % 997) * 0.123f;
        }
        cudaMemcpy(src, h_src, size * sizeof(float), cudaMemcpyHostToDevice);
    }

    free(h_src);
    cudaFree(src);
    cudaFree(dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
