#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
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
// CPU Reference Implementation (Already Written)

// 'wave_cpu_step':
//
// Input:
//
//     t -- time coordinate
//     u(t - dt) in array 'u0' of size 'n_cells_y * n_cells_x'
//     u(t) in array 'u1' of size 'n_cells_y * n_cells_x'
//
// Output:
//
//     u(t + dt) in array 'u0' (overwrites the input)
//
template <typename Scene> void wave_cpu_step(float t, float *u0, float const *u1) {
    constexpr int32_t n_cells_x = Scene::n_cells_x;
    constexpr int32_t n_cells_y = Scene::n_cells_y;
    constexpr float c = Scene::c;
    constexpr float dx = Scene::dx;
    constexpr float dt = Scene::dt;

    for (int32_t idx_y = 0; idx_y < n_cells_y; ++idx_y) {
        for (int32_t idx_x = 0; idx_x < n_cells_x; ++idx_x) {
            int32_t idx = idx_y * n_cells_x + idx_x;
            bool is_border =
                (idx_x == 0 || idx_x == n_cells_x - 1 || idx_y == 0 ||
                 idx_y == n_cells_y - 1);
            float u_next_val;
            if (is_border || Scene::is_wall(idx_x, idx_y)) {
                u_next_val = 0.0f;
            } else if (Scene::is_source(idx_x, idx_y)) {
                u_next_val = Scene::source_value(idx_x, idx_y, t);
            } else {
                constexpr float coeff = c * c * dt * dt / (dx * dx);
                float damping = Scene::damping(idx_x, idx_y);
                u_next_val =
                    ((2.0f - damping - 4.0f * coeff) * u1[idx] -
                     (1.0f - damping) * u0[idx] +
                     coeff *
                         (u1[idx - 1] + u1[idx + 1] + u1[idx - n_cells_x] +
                          u1[idx + n_cells_x]));
            }
            u0[idx] = u_next_val;
        }
    }
}

// 'wave_cpu':
//
// Input:
//
//     t0 -- initial time coordinate
//     n_steps -- number of time steps to simulate
//     u(t0 - dt) in array 'u0' of size 'n_cells_y * n_cells_x'
//     u(t0) in array 'u1' of size 'n_cells_y * n_cells_x'
//
// Output:
//
//     Overwrites contents of memory pointed to by 'u0' and 'u1'
//
//     Returns pointers to buffers containing the final states of the wave
//     u(t0 + (n_steps - 1) * dt) and u(t0 + n_steps * dt).
//
template <typename Scene>
std::pair<float *, float *> wave_cpu(float t0, int32_t n_steps, float *u0, float *u1) {
    for (int32_t idx_step = 0; idx_step < n_steps; idx_step++) {
        float t = t0 + idx_step * Scene::dt;
        wave_cpu_step<Scene>(t, u0, u1);
        std::swap(u0, u1);
    }
    return {u0, u1};
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (Naive)

// 'wave_gpu_step':
//
// Input:
//
//     t -- time coordinate
//     u(t - dt) in GPU array 'u0' of size 'n_cells_y * n_cells_x'
//     u(t) in GPU array 'u1' of size 'n_cells_y * n_cells_x'
//
// Output:
//
//     u(t + dt) in GPU array 'u0' (overwrites the input)
//
template <typename Scene>
__global__ void wave_gpu_naive_step(
    float t,
    float *u0,      /* pointer to GPU memory */
    float const *u1 /* pointer to GPU memory */
) {
    /* TODO: your GPU code here... */
}

// 'wave_gpu_naive':
//
// Input:
//
//     t0 -- initial time coordinate
//     n_steps -- number of time steps to simulate
//     u(t0 - dt) in GPU array 'u0' of size 'n_cells_y * n_cells_x'
//     u(t0) in GPU array 'u1' of size 'n_cells_y * n_cells_x'
//
// Output:
//
//     Launches kernels to overwrite the GPU memory pointed to by 'u0' and 'u1'
//
//     Returns pointers to GPU buffers which will contain the final states of
//     the wave u(t0 + (n_steps - 1) * dt) and u(t0 + n_steps * dt) after all
//     launched kernels have completed.
//
template <typename Scene>
std::pair<float *, float *> wave_gpu_naive(
    float t0,
    int32_t n_steps,
    float *u0, /* pointer to GPU memory */
    float *u1  /* pointer to GPU memory */
) {
    /* TODO: your CPU code here... */
    return {u0, u1};
}

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Shared Memory)

template <typename Scene>
__global__ void wave_gpu_shmem_multistep(
    /* TODO: your arguments here... */
) {
    /* TODO: your GPU code here... */
}

// 'wave_gpu_shmem':
//
// Input:
//
//     t0 -- initial time coordinate
//
//     n_steps -- number of time steps to simulate
//
//     u(t0 - dt) in GPU array 'u0' of size 'n_cells_y * n_cells_x'
///
//     u(t0) in GPU array 'u1' of size 'n_cells_y * n_cells_x'
//
//     Scratch buffers 'extra0' and 'extra1' of size 'n_cells_y * n_cells_x'
//
// Output:
//
//     Launches kernels to (potentially) overwrite the GPU memory pointed to
//     by 'u0' and 'u1', 'extra0', and 'extra1'.
//
//     Returns pointers to GPU buffers which will contain the final states of
//     the wave u(t0 + (n_steps - 1) * dt) and u(t0 + n_steps * dt) after all
//     launched kernels have completed. These buffers can be any of 'u0', 'u1',
//     'extra0', or 'extra1'.
//
template <typename Scene>
std::pair<float *, float *> wave_gpu_shmem(
    float t0,
    int32_t n_steps,
    float *u0,     /* pointer to GPU memory */
    float *u1,     /* pointer to GPU memory */
    float *extra0, /* pointer to GPU memory */
    float *extra1  /* pointer to GPU memory */
) {
    /* TODO: your CPU code here... */
    return {u0, u1};
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

struct BaseScene {
    constexpr static int32_t n_cells_x = 3201;
    constexpr static int32_t n_cells_y = 3201;
    constexpr static float c = 1.0f;
    constexpr static float dx = 1.0f / float(n_cells_x - 1);
    constexpr static float dy = 1.0f / float(n_cells_y - 1);
    constexpr static float dt = 0.25f * dx / c;
    constexpr static float t_end = 1.0f;
};

struct BaseSceneSmallScale {
    constexpr static int32_t n_cells_x = 201;
    constexpr static int32_t n_cells_y = 201;
    constexpr static float c = 1.0f;
    constexpr static float dx = 1.0f / float(n_cells_x - 1);
    constexpr static float dy = 1.0f / float(n_cells_y - 1);
    constexpr static float dt = 0.25f * dx / c;
    constexpr static float t_end = 1.0f;
};

float __host__ __device__ __forceinline__ boundary_damping(
    int32_t n_cells_x,
    int32_t n_cells_y,
    float ramp_size,
    float max_damping,
    int32_t idx_x,
    int32_t idx_y) {
    float x = float(idx_x) / (n_cells_x - 1);
    float y = float(idx_y) / (n_cells_y - 1);
    float fx = 1.0f - min(min(x, 1.0f - x), ramp_size) / ramp_size;
    float fy = 1.0f - min(min(y, 1.0f - y), ramp_size) / ramp_size;
    float f = max(fx, fy);
    return max_damping * f * f;
}

struct PointSource : public BaseScene {
    static __host__ __device__ __forceinline__ bool
    is_wall(int32_t idx_x, int32_t idx_y) {
        return false;
    }

    static __host__ __device__ __forceinline__ bool
    is_source(int32_t idx_x, int32_t idx_y) {
        return idx_x == (n_cells_x - 1) / 2 && idx_y == (n_cells_y - 1) / 2;
    }

    static __host__ __device__ __forceinline__ float
    source_value(int32_t idx_x, int32_t idx_y, float t) {
        return 10.0f * sinf(2.0f * 3.14159265359f * 20.0f * t);
    }

    static __host__ __device__ __forceinline__ float
    damping(int32_t idx_x, int32_t idx_y) {
        return boundary_damping(n_cells_x, n_cells_y, 0.1f, 0.5f, idx_x, idx_y);
    }
};

struct Slit : public BaseScene {
    constexpr static float slit_width = 0.05f;

    static __host__ __device__ __forceinline__ bool
    is_wall(int32_t idx_x, int32_t idx_y) {
        float y = float(idx_y) / (n_cells_y - 1);
        return idx_x == (n_cells_x - 1) / 2 &&
            (y < 0.5f - slit_width / 2 || y > 0.5f + slit_width / 2);
    }

    static __host__ __device__ __forceinline__ bool
    is_source(int32_t idx_x, int32_t idx_y) {
        return idx_x == (n_cells_x - 1) / 4 && idx_y == (n_cells_y - 1) / 2;
    }

    static __host__ __device__ __forceinline__ float
    source_value(int32_t idx_x, int32_t idx_y, float t) {
        return 10.0f * sinf(2.0f * 3.14159265359f * 40.0f * t);
    }

    static __host__ __device__ __forceinline__ float
    damping(int32_t idx_x, int32_t idx_y) {
        return boundary_damping(n_cells_x, n_cells_y, 0.1f, 0.5f, idx_x, idx_y);
    }
};

struct DoubleSlit : public BaseScene {
    constexpr static float slit_width = 0.03f;

    static __host__ __device__ __forceinline__ bool
    is_wall(int32_t idx_x, int32_t idx_y) {
        float y = float(idx_y) / (n_cells_y - 1);
        return (idx_x == (n_cells_x - 1) * 2 / 3) &&
            !((y >= 0.45f - slit_width / 2 && y <= 0.45f + slit_width / 2) ||
              (y >= 0.55f - slit_width / 2 && y <= 0.55f + slit_width / 2));
    }

    static __host__ __device__ __forceinline__ bool
    is_source(int32_t idx_x, int32_t idx_y) {
        return idx_x == (n_cells_x - 1) / 6 && idx_y == (n_cells_y - 1) / 2;
    }

    static __host__ __device__ __forceinline__ float
    source_value(int32_t idx_x, int32_t idx_y, float t) {
        return 10.0f * sinf(2.0f * 3.14159265359f * 20.0f * t);
    }

    static __host__ __device__ __forceinline__ float
    damping(int32_t idx_x, int32_t idx_y) {
        return boundary_damping(n_cells_x, n_cells_y, 0.1f, 0.5f, idx_x, idx_y);
    }
};

struct DoubleSlitSmallScale : public BaseSceneSmallScale {
    constexpr static float slit_width = 0.03f;

    static __host__ __device__ __forceinline__ bool
    is_wall(int32_t idx_x, int32_t idx_y) {
        constexpr float EPS = 1e-6;
        float y = float(idx_y) / (n_cells_y - 1);
        return (idx_x == (n_cells_x - 1) * 2 / 3) &&
            !((y >= 0.45f - slit_width / 2 - EPS && y <= 0.45f + slit_width / 2 + EPS) ||
              (y >= 0.55f - slit_width / 2 - EPS && y <= 0.55f + slit_width / 2 + EPS));
    }

    static __host__ __device__ __forceinline__ bool
    is_source(int32_t idx_x, int32_t idx_y) {
        return idx_x == (n_cells_x - 1) / 6 && idx_y == (n_cells_y - 1) / 2;
    }

    static __host__ __device__ __forceinline__ float
    source_value(int32_t idx_x, int32_t idx_y, float t) {
        return 10.0f * sinf(2.0f * 3.14159265359f * 20.0f * t);
    }

    static __host__ __device__ __forceinline__ float
    damping(int32_t idx_x, int32_t idx_y) {
        return boundary_damping(n_cells_x, n_cells_y, 0.1f, 0.5f, idx_x, idx_y);
    }
};

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(
    const char *fname,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t> &pixels) {
    BMPHeader header;
    header.width = width;
    header.height = height;

    uint32_t rowSize = (width * 3 + 3) & (~3); // Align to 4 bytes
    header.imageSize = rowSize * height;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // Write pixel data with padding
    std::vector<uint8_t> padding(rowSize - width * 3, 0);
    for (int32_t idx_y = height - 1; idx_y >= 0;
         --idx_y) { // BMP stores pixels from bottom to top
        const uint8_t *row = &pixels[idx_y * width * 3];
        file.write(reinterpret_cast<const char *>(row), width * 3);
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char *>(padding.data()), padding.size());
        }
    }
}

// If trunc - cut the border of the image.
template <typename Scene>
std::vector<uint8_t> render_wave(const float *u, int trunc = 0) {
    constexpr int32_t n_cells_x = Scene::n_cells_x;
    constexpr int32_t n_cells_y = Scene::n_cells_y;

    std::vector<uint8_t> pixels((n_cells_x - trunc) * (n_cells_y - trunc) * 3);
    for (int32_t idx_y = 0; idx_y < n_cells_y - trunc; ++idx_y) {
        for (int32_t idx_x = 0; idx_x < n_cells_x - trunc; ++idx_x) {
            int32_t idx = idx_y * (n_cells_x - trunc) + idx_x;
            int32_t u_idx = idx_y * n_cells_x + idx_x;
            float val = u[u_idx];
            bool is_wall = Scene::is_wall(idx_x, idx_y);
            // BMP stores pixels in BGR order
            if (is_wall) {
                pixels[idx * 3 + 2] = 0;
                pixels[idx * 3 + 1] = 0;
                pixels[idx * 3 + 0] = 0;
            } else if (val > 0.0f) {
                pixels[idx * 3 + 2] = 255;
                pixels[idx * 3 + 1] = 255 - uint8_t(min(val * 255.0f, 255.0f));
                pixels[idx * 3 + 0] = 255 - uint8_t(min(val * 255.0f, 255.0f));
            } else {
                pixels[idx * 3 + 2] = 255 - uint8_t(min(-val * 255.0f, 255.0f));
                pixels[idx * 3 + 1] = 255 - uint8_t(min(-val * 255.0f, 255.0f));
                pixels[idx * 3 + 0] = 255;
            }
        }
    }
    return pixels;
}

struct Results {
    std::vector<float> u0_final;
    std::vector<float> u1_final;
    double time_ms;
};

template <typename Scene, typename F>
Results run_cpu(
    float t0,
    int32_t n_steps,
    int32_t num_iters_outer,
    int32_t num_iters_inner,
    F func) {
    auto u0 = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);
    auto u1 = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);

    std::pair<float *, float *> u_final;

    double best_time = std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < num_iters_outer; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t j = 0; j < num_iters_inner; ++j) {
            std::fill(u0.begin(), u0.end(), 0.0f);
            std::fill(u1.begin(), u1.end(), 0.0f);
            u_final = func(t0, n_steps, u0.data(), u1.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() /
            num_iters_inner;
        best_time = std::min(best_time, time_ms);
    }

    if (u_final.first == u1.data() && u_final.second == u0.data()) {
        std::swap(u0, u1);
    } else if (!(u_final.first == u0.data() && u_final.second == u1.data())) {
        std::cerr << "Unexpected return values from 'func'" << std::endl;
        std::abort();
    }

    return {std::move(u0), std::move(u1), best_time};
}

template <typename Scene, typename F>
Results run_gpu(
    float t0,
    int32_t n_steps,
    int32_t num_iters_outer,
    int32_t num_iters_inner,
    bool use_extra,
    F func) {
    float *u0;
    float *u1;
    float *extra0 = nullptr;
    float *extra1 = nullptr;

    CUDA_CHECK(cudaMalloc(&u0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&u1, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));

    if (use_extra) {
        CUDA_CHECK(
            cudaMalloc(&extra0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
        CUDA_CHECK(
            cudaMalloc(&extra1, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    }

    std::pair<float *, float *> u_final;

    double best_time = std::numeric_limits<double>::infinity();
    for (int32_t i = 0; i < num_iters_outer; ++i) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t j = 0; j < num_iters_inner; ++j) {
            CUDA_CHECK(
                cudaMemset(u0, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
            CUDA_CHECK(
                cudaMemset(u1, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
            if (use_extra) {
                CUDA_CHECK(cudaMemset(
                    extra0,
                    0,
                    Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
                CUDA_CHECK(cudaMemset(
                    extra1,
                    0,
                    Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
            }
            u_final = func(t0, n_steps, u0, u1, extra0, extra1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count() /
            num_iters_inner;
        best_time = std::min(best_time, time_ms);
    }

    if (u_final.first != u0 && u_final.first != u1 &&
        (extra0 == nullptr || u_final.first != extra0) &&
        (extra1 == nullptr || u_final.first != extra1)) {
        std::cerr << "Unexpected final 'u0' pointer returned from GPU implementation"
                  << std::endl;
        std::abort();
    }

    if (u_final.second != u0 && u_final.second != u1 &&
        (extra0 == nullptr || u_final.second != extra0) &&
        (extra1 == nullptr || u_final.second != extra1)) {
        std::cerr << "Unexpected final 'u1' pointer returned from GPU implementation"
                  << std::endl;
        std::abort();
    }

    auto u0_cpu = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);
    auto u1_cpu = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);
    CUDA_CHECK(cudaMemcpy(
        u0_cpu.data(),
        u_final.first,
        Scene::n_cells_x * Scene::n_cells_y * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        u1_cpu.data(),
        u_final.second,
        Scene::n_cells_x * Scene::n_cells_y * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(u0));
    CUDA_CHECK(cudaFree(u1));
    if (use_extra) {
        CUDA_CHECK(cudaFree(extra0));
        CUDA_CHECK(cudaFree(extra1));
    }

    return {std::move(u0_cpu), std::move(u1_cpu), best_time};
}

template <typename Scene, typename F>
Results run_gpu_extra(
    float t0,
    int32_t n_steps,
    int32_t num_iters_outer,
    int32_t num_iters_inner,
    F func) {
    return run_gpu<Scene>(t0, n_steps, num_iters_outer, num_iters_inner, true, func);
}

template <typename Scene, typename F>
Results run_gpu_no_extra(
    float t0,
    int32_t n_steps,
    int32_t num_iters_outer,
    int32_t num_iters_inner,
    F func) {
    return run_gpu<Scene>(
        t0,
        n_steps,
        num_iters_outer,
        num_iters_inner,
        false,
        [&](float t0,
            int32_t n_steps,
            float *u0,
            float *u1,
            float *extra0,
            float *extra1) { return func(t0, n_steps, u0, u1); });
}

double rel_rmse(std::vector<float> const &a, std::vector<float> const &b) {
    if (a.size() != b.size()) {
        std::cerr << "Mismatched sizes in 'rel_rmse'" << std::endl;
        std::abort();
    }
    double ref_sum = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        ref_sum += double(a.at(i)) * double(a.at(i));
        double diff = double(a.at(i)) - double(b.at(i));
        sum += diff * diff;
    }
    return sqrt(sum / a.size()) / sqrt(ref_sum / a.size());
}

// FFmpeg implementations.
typedef std::vector<std::vector<uint8_t>> FFmpegFrames;

// CPU implementation with FFmpeg framing.
template <typename Scene>
void wave_ffmpeg(float t0, int32_t n_steps, FFmpegFrames &frames) {
    static constexpr int32_t frame_step = 2;
    auto u0_v = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);
    auto u1_v = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);
    auto u0 = u0_v.data();
    auto u1 = u1_v.data();
    for (int32_t idx_step = 0; idx_step < n_steps; idx_step += frame_step) {
        auto r = wave_cpu<Scene>(t0 + idx_step * Scene::dt, frame_step, u0, u1);
        u0 = r.first;
        u1 = r.second;
        frames.push_back(render_wave<Scene>(u1, 1));
    }
}

template <typename Scene>
void wave_ffmpeg_gpu(float t0, int32_t n_steps, FFmpegFrames &frames) {
    static constexpr int32_t frame_step = 2;
    auto u1_cpu = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);
    float *u0;
    float *u1;
    CUDA_CHECK(cudaMalloc(&u0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&u1, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMemset(u0, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMemset(u1, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    for (int32_t idx_step = 0; idx_step < n_steps; idx_step += frame_step) {
        auto r = wave_gpu_naive<Scene>(t0 + idx_step * Scene::dt, frame_step, u0, u1);
        u0 = r.first;
        u1 = r.second;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            u1_cpu.data(),
            u1,
            Scene::n_cells_x * Scene::n_cells_y * sizeof(float),
            cudaMemcpyDeviceToHost));
        frames.push_back(render_wave<Scene>(u1_cpu.data(), 1));
    }
}

template <typename Scene>
void wave_ffmpeg_gpu_shmem(float t0, int32_t n_steps, FFmpegFrames &frames) {
    static constexpr int32_t frame_step = 2;
    auto u1_cpu = std::vector<float>(Scene::n_cells_x * Scene::n_cells_y);
    float *u0;
    float *u1;
    float *extra0;
    float *extra1;
    CUDA_CHECK(cudaMalloc(&u0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&u1, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMemset(u0, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMemset(u1, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&extra0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&extra1, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
    float *buffers[] = {u0, u1, extra0, extra1};
    for (int32_t idx_step = 0; idx_step < n_steps; idx_step += frame_step) {
        CUDA_CHECK(
            cudaMemset(extra0, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(extra1, 0, Scene::n_cells_x * Scene::n_cells_y * sizeof(float)));
        auto r = wave_gpu_shmem<Scene>(
            t0 + idx_step * Scene::dt,
            frame_step,
            u0,
            u1,
            extra0,
            extra1);
        u0 = r.first;
        u1 = r.second;
        for (int i = 0; i < 4; ++i) {
            if (buffers[i] != u0 && buffers[i] != u1) {
                extra0 = buffers[i];
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (buffers[i] != u0 && buffers[i] != u1 && buffers[i] != extra0) {
                extra1 = buffers[i];
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
            u1_cpu.data(),
            u1,
            Scene::n_cells_x * Scene::n_cells_y * sizeof(float),
            cudaMemcpyDeviceToHost));
        frames.push_back(render_wave<Scene>(u1_cpu.data(), 1));
    }
}

template <typename Scene>
int generate_animation(const FFmpegFrames &frames, std::string fname) {
    std::string ffmpeg_command = "ffmpeg -y "
                                 "-f rawvideo "
                                 "-pixel_format rgb24 "
                                 "-video_size " +
        std::to_string(Scene::n_cells_x - 1) + "x" +
        std::to_string(Scene::n_cells_y - 1) +
        " "
        "-framerate " +
        std::to_string(30) +
        " "
        "-i - "
        "-c:v libx264 "
        "-pix_fmt yuv420p " +
        fname + ".mp4" + " 2> /dev/null";

    FILE *pipe = popen(ffmpeg_command.c_str(), "w");
    if (!pipe) {
        std::cerr << "Failed to open pipe to FFmpeg." << std::endl;
        return 1;
    }

    for (auto &frame : frames) {
        if (fwrite(frame.data(), 1, frame.size(), pipe) != frame.size()) {
            std::cerr << "Failed to write frame to FFmpeg." << std::endl;
            return 1;
        }
    }

    pclose(pipe);
    return 0;
}

int main(int argc, char **argv) {
    // Small scale tests: mainly for correctness.
    double tolerance = 3e-2;
    bool gpu_naive_correct = false;
    bool gpu_shmem_correct = false;
    {
        printf("Small scale tests (on scene 'DoubleSlitSmallScale'):\n");
        using Scene = DoubleSlitSmallScale;

        // CPU.
        int32_t n_steps = Scene::t_end / Scene::dt;
        auto cpu_results = run_cpu<Scene>(0.0f, n_steps, 1, 1, wave_cpu<Scene>);
        writeBMP(
            "out/wave_cpu_small_scale.bmp",
            Scene::n_cells_x,
            Scene::n_cells_y,
            render_wave<Scene>(cpu_results.u0_final.data()));
        printf("  CPU sequential implementation:\n");
        printf("    run time: %.2f ms\n", cpu_results.time_ms);
        printf("\n");

        // GPU: wave_gpu_naive.
        auto gpu_naive_results =
            run_gpu_no_extra<Scene>(0.0f, n_steps, 1, 1, wave_gpu_naive<Scene>);
        writeBMP(
            "out/wave_gpu_naive_small_scale.bmp",
            Scene::n_cells_x,
            Scene::n_cells_y,
            render_wave<Scene>(gpu_naive_results.u0_final.data()));
        double naive_rel_rmse =
            rel_rmse(cpu_results.u0_final, gpu_naive_results.u0_final);
        if (naive_rel_rmse < tolerance) {
            gpu_naive_correct = true;
        }
        printf("  GPU naive implementation:\n");
        printf("    run time: %.2f ms\n", gpu_naive_results.time_ms);
        printf("    correctness: %.2e relative RMSE\n", naive_rel_rmse);
        printf("\n");

        // GPU: wave_gpu_shmem.
        auto gpu_shmem_results =
            run_gpu_extra<Scene>(0.0f, n_steps, 1, 1, wave_gpu_shmem<Scene>);
        writeBMP(
            "out/wave_gpu_shmem_small_scale.bmp",
            Scene::n_cells_x,
            Scene::n_cells_y,
            render_wave<Scene>(gpu_shmem_results.u0_final.data()));
        double shmem_rel_rmse =
            rel_rmse(cpu_results.u0_final, gpu_shmem_results.u0_final);
        if (shmem_rel_rmse < tolerance) {
            gpu_shmem_correct = true;
        }
        printf("  GPU shared memory implementation:\n");
        printf("    run time: %.2f ms\n", gpu_shmem_results.time_ms);
        printf("    correctness: %.2e relative RMSE\n", shmem_rel_rmse);
        printf("\n");

        if (gpu_naive_correct) {
            printf(
                "  CPU -> GPU naive speedup: %.2fx\n",
                cpu_results.time_ms / gpu_naive_results.time_ms);
        }
        if (gpu_shmem_correct) {
            printf(
                "  CPU -> GPU shared memory speedup: %.2fx\n",
                cpu_results.time_ms / gpu_shmem_results.time_ms);
        }
        if (gpu_naive_correct && gpu_shmem_correct) {
            printf(
                "  GPU naive -> GPU shared memory speedup: %.2fx\n",
                gpu_naive_results.time_ms / gpu_shmem_results.time_ms);
        }
        printf("\n");
    }

    // Run performance tests if requested.
    bool run_perf_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-p") == 0) {
            run_perf_tests = true;
            break;
        }
    }

    // Large scale tests: mainly for performance.
    if (run_perf_tests) {
        printf("Large scale tests (on scene 'DoubleSlit'):\n");
        using Scene = DoubleSlit;

        int32_t n_steps = Scene::t_end / Scene::dt;
        int32_t num_iters_outer_gpu = 1;
        int32_t num_iters_inner_gpu = 1;

        // GPU: wave_gpu_naive.
        Results gpu_naive_results;
        if (gpu_naive_correct) {
            gpu_naive_results = run_gpu_no_extra<Scene>(
                0.0f,
                n_steps,
                num_iters_outer_gpu,
                num_iters_inner_gpu,
                wave_gpu_naive<Scene>);
            printf("  GPU naive implementation:\n");
            printf("    run time: %.2f ms\n", gpu_naive_results.time_ms);
            printf("\n");
            auto pixels_gpu_naive = render_wave<Scene>(gpu_naive_results.u0_final.data());
            writeBMP(
                "out/wave_gpu_naive_large_scale.bmp",
                Scene::n_cells_x,
                Scene::n_cells_y,
                pixels_gpu_naive);
        } else {
            printf("  Skipping GPU naive implementation (incorrect)\n");
        }

        // GPU: wave_gpu_shmem.
        Results gpu_shmem_results;
        double naive_shmem_rel_rmse = 0.0;
        if (gpu_shmem_correct) {
            gpu_shmem_results = run_gpu_extra<Scene>(
                0.0f,
                n_steps,
                num_iters_outer_gpu,
                num_iters_inner_gpu,
                wave_gpu_shmem<Scene>);
            naive_shmem_rel_rmse =
                rel_rmse(gpu_naive_results.u0_final, gpu_shmem_results.u0_final);
            printf("  GPU shared memory implementation:\n");
            printf("    run time: %.2f ms\n", gpu_shmem_results.time_ms);
            printf(
                "    correctness (w.r.t. GPU naive): %.2e relative RMSE\n",
                naive_shmem_rel_rmse);
            printf("\n");
            auto pixels_gpu_shmem = render_wave<Scene>(gpu_shmem_results.u0_final.data());
            writeBMP(
                "out/wave_gpu_shmem_large_scale.bmp",
                Scene::n_cells_x,
                Scene::n_cells_y,
                pixels_gpu_shmem);
        } else {
            printf("  Skipping GPU shared memory implementation (incorrect)\n");
        }

        if (gpu_naive_correct && gpu_shmem_correct && naive_shmem_rel_rmse < tolerance) {
            printf(
                "  GPU naive -> GPU shared memory speedup: %.2fx\n",
                gpu_naive_results.time_ms / gpu_shmem_results.time_ms);

        } else {
            printf("  GPU naive -> GPU shared memory speedup: N/A (incorrect)\n");
        }
        printf("\n");
    }

    // Generate animation if requested.
    bool a_flag = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-a") == 0) {
            a_flag = true;
            break;
        }
    }

    if (a_flag) {
        using Scene = DoubleSlitSmallScale;
        int32_t n_steps = Scene::t_end / Scene::dt;

        // CPU.
        FFmpegFrames cpu_frames;
        wave_ffmpeg<Scene>(0.0f, n_steps, cpu_frames);
        if (generate_animation<Scene>(cpu_frames, "out/wave_cpu") != 0) {
            std::cout << "CPU animation error: Failed to generate animation."
                      << std::endl;
        } else {
            std::cout << "CPU video has been generated." << std::endl;
        }

        // GPU naive.
        FFmpegFrames gpu_naive_frames;
        wave_ffmpeg_gpu<Scene>(0.0f, n_steps, gpu_naive_frames);
        if (generate_animation<Scene>(gpu_naive_frames, "out/wave_gpu_naive") != 0) {
            std::cout << "GPU_naive animation error: Failed to generate animation."
                      << std::endl;
        } else {
            std::cout << "GPU_naive video has been generated." << std::endl;
        }

        // GPU shared memory.
        FFmpegFrames gpu_shmem_frames;
        wave_ffmpeg_gpu_shmem<Scene>(0.0f, n_steps, gpu_shmem_frames);
        if (generate_animation<Scene>(gpu_shmem_frames, "out/wave_gpu_shmem") != 0) {
            std::cout << "GPU_shem animation error: Failed to generate animation."
                      << std::endl;
        } else {
            std::cout << "GPU_shmem video has been generated." << std::endl;
        }
    }

    return 0;
}
