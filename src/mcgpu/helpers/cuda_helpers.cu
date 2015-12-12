/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 01:27:33 PM CET
 *   modified: Fri 11 Dec 2015 05:35:41 PM CET
 */

#include <vector>
#include <chrono>
#include <thread>
#include "mcgpu/helpers/cuda_call.hpp"
#include "device.hpp"


__global__ void var_kernel(float *gpu_array, float mean) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    gpu_array[i] = (gpu_array[i] - mean) * (gpu_array[i] - mean);
}

__host__ __device__ unsigned int msb32(unsigned int x) {
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return (x & ~(x >> 1));
}

__global__ void reduction_kernel(float *gpu_array, float *gpu_output,
                                 int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    int width = msb32(blockDim.x - 1);

    __shared__ float cache[CUDA_maxThreadsPerBlock];

    float thread_sum = 0.0;
    while (i < size) {
        thread_sum += gpu_array[i];
        i += step;
    }

    cache[threadIdx.x] = thread_sum;

    __syncthreads();

    while (width > 0) {
        if (threadIdx.x < width) {
            int end = threadIdx.x + width;
            if (end < blockDim.x) cache[threadIdx.x] += cache[end];
        }
        width /= 2;
        __syncthreads();
    }

    if (threadIdx.x == 0) gpu_output[blockIdx.x] = cache[0];
}

namespace mcgpu {
namespace helpers {

float reduction_gpu(float *gpu_array, int size, int blocks, int threads) {
    float *gpu_output;
    std::vector<float> cpu_array(blocks);

    CUDA_CALL(cudaMalloc((void **)&gpu_output, blocks * sizeof(float)));

    reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>
        (gpu_array, gpu_output, size);

    CUDA_CALL(cudaMemcpy(cpu_array.data(), gpu_output, blocks * sizeof(float),
                         cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(gpu_output));

    float sumgpu = 0.0F;
    for (auto e : cpu_array) {
        sumgpu += e;
    }
    sumgpu /= size;

    return sumgpu;
}

void invoke_kernel_variance(float *gpu_array, float mean, int blocks,
                            int threads) {
    var_kernel<<<blocks, threads>>> (gpu_array, mean);
}
}
}
