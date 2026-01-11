// Tell telerun that we are trying to run on an h100
// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}
// TL {"workspace_files": []}

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

__global__ void hello() { printf("Hello from a H100!\n"); }

int main() {
  hello<<<1, 1>>>();
  cudaDeviceSynchronize();

  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g1bf9d625a931d657e08db2b4391170f0
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int maxSharedMemPerSM, maxRegsPerSM;
  cudaDeviceGetAttribute(&maxSharedMemPerSM,
                         cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  cudaDeviceGetAttribute(&maxRegsPerSM,
                         cudaDevAttrMaxRegistersPerMultiprocessor, 0);

  printf("\n============ BASIC DEVICE INFORMATION ============\n");
  printf("Device: %s\n", prop.name);
  printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
  printf("Boost Clock Rate: %d MHz\n", prop.clockRate / 1000);

  printf("\n============ COMPUTE INFORMATION ============\n");
  printf("Number of SMs: %d\n", prop.multiProcessorCount);
  printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
  printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

  printf("\n============ MEMORY INFORMATION ============\n");
  printf("\n*** Device Specific ***\n");
  printf("Total Global Memory: %.2f GB (%zu bytes)\n",
         (double)prop.totalGlobalMem / (1024 * 1024 * 1024),
         prop.totalGlobalMem);
  printf("L2 Cache Size: %.2f MB (%d bytes)\n",
         (double)prop.l2CacheSize / (1024 * 1024), prop.l2CacheSize);
  // Compute bandwitdh of memory (GB/s)
  // Divide by 8 for bits to byte
  // Multiply by 2 because of DDR, transfers twice per clock cycle
  double bandwidth =
      (double)prop.memoryClockRate * (prop.memoryBusWidth / 8.0) * 2 / 1.0e6;
  printf("Theoretical Memory Bandwidth: %.2f GB/s\n", bandwidth);

  printf("\n*** SM Specific ***\n");
  printf("Default Shared Memory per SM: %.2f KB (%zu bytes)\n",
         (double)prop.sharedMemPerMultiprocessor / 1024,
         prop.sharedMemPerMultiprocessor);
  printf("Maximum Shared Memory per SM: %.2f KB (%d bytes)\n",
         (double)maxSharedMemPerSM / 1024, maxSharedMemPerSM);
  printf("Maximum Registers per SM: %d\n", maxRegsPerSM);

  printf("\n*** Block Specific ***\n");
  printf("Default Shared Memory per Block: %.2f KB (%zu bytes)\n",
         (double)prop.sharedMemPerBlock / 1024, prop.sharedMemPerBlock);
  printf("Maximum Shared Memory per Block: %.2f KB (%zu bytes)\n",
         (double)prop.sharedMemPerBlockOptin / 1024,
         prop.sharedMemPerBlockOptin);
  printf("Registers per Block: %d\n", prop.regsPerBlock);

  return 0;
}