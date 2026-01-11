// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "tma-interface.cuh"

using datatype = uint8_t;

////////////////////////////////////////////////////////////////////////////////
// Part 5: Reverse-Engineering TMA 64B Swizzle
////////////////////////////////////////////////////////////////////////////////

/// <--- your code here --->

template <int TILE_M, int TILE_N, int OFFSET>
__global__ void tma_swizzle(__grid_constant__ const CUtensorMap src_map,
                            datatype *dest) {
    /*
     * IMPORTANT REQUIREMENT FOR PART 5:
     *
     * To get credit, you need to use smem_buffer to store your TMA data.
     * Do not edit the setup for smem_buffer.
     */
    __shared__ alignas(128)
        datatype smem_buffer_abs[TILE_M * TILE_N + 128 * OFFSET];
    datatype *smem_buffer = &smem_buffer_abs[128 * OFFSET];

    // Cast to a "shared pointer" so that it works with
    // cp_async_bulk_tensor_2d_global_to_shared.

    /* TODO: your launch code here... */
}

template <int TILE_M, int TILE_N, int OFFSET>
void launch_tma_swizzle(datatype *src, datatype *dest) {

    /*
     * IMPORTANT REQUIREMENT FOR PART 5:
     *
     * To get credit for this part, launch the tma_swizzle
     * kernel with the CU_TENSOR_MAP_SWIZZLE_64B setting.
     */

    /* TODO: your launch code here... */
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

template <int M, int N, int OFFSET>
void set_up_test(datatype *matrix, datatype *d_matrix, datatype *d_dest) {

    printf("Testing offset %d...\n", OFFSET);

    const uint64_t total_size = M * N;
    datatype *final_output = (datatype *)malloc(total_size * sizeof(datatype));
    // Zero out destination buffer
    for (int i = 0; i < total_size; i++) {
        final_output[i] = 0;
    }
    cudaMemcpy(d_dest, final_output, total_size * sizeof(datatype),
               cudaMemcpyHostToDevice);
    // Launch kernel
    launch_tma_swizzle<M, N, OFFSET>(d_matrix, d_dest);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    cudaMemcpy(final_output, d_dest, total_size * sizeof(datatype),
               cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int x = 0; x < M * N; x++) {
        int i = x / N;
        int j = x % N;
        float ref = (float)matrix[i * N + j];
        float computed = (float)final_output[i * N + j];
        if (ref != computed) {
            correct = false;
            printf("Mismatch at (%d, %d): expected %f, got %f \n", i, j, ref,
                   computed);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    free(final_output);
}

template <int M, int N>
void run_test(datatype *matrix, datatype *d_matrix, datatype *d_dest) {
    // Test with different offsets
    set_up_test<M, N, 0>(matrix, d_matrix, d_dest);
    set_up_test<M, N, 1>(matrix, d_matrix, d_dest);
    set_up_test<M, N, 2>(matrix, d_matrix, d_dest);
    set_up_test<M, N, 3>(matrix, d_matrix, d_dest);
}

int main() {
    const int M = 1;
    const int N = 64;
    const uint64_t total_size = M * N;

    // Allocate host and device memory
    datatype *matrix = (datatype *)malloc(total_size * sizeof(datatype));
    datatype *d_matrix;
    datatype *d_dest;
    cudaMalloc(&d_matrix, total_size * sizeof(datatype));
    cudaMalloc(&d_dest, total_size * sizeof(datatype));

    // Initialize source matrix on host
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = i + j;
        }
    }
    cudaMemcpy(d_matrix, matrix, total_size * sizeof(datatype),
               cudaMemcpyHostToDevice);

    printf("\n\nRunning TMA swizzle tests...\n\n");

    run_test<M, N>(matrix, d_matrix, d_dest);

    // Cleanup resources
    cudaFree(d_matrix);
    cudaFree(d_dest);
    free(matrix);

    return 0;
}
