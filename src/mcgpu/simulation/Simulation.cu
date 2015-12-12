/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 06:56:51 AM CET
 */

#include <vector>
#include <sys/time.h>
#include <iostream>

#include "mcgpu/simulation/Simulation.hpp"
#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/helpers/cuda_helpers.hpp"

#include "device.hpp"

namespace mcgpu {
namespace simulation {

Simulation::Simulation(int paths_, int points_) : paths(paths_), points(points_) {
    threads = CUDA_maxThreadsPerBlock;
    blocks = (paths + CUDA_maxThreadsPerBlock - 1) / CUDA_maxThreadsPerBlock;
    paths = blocks * threads;

    CUDA_CALL(cudaMalloc((void **)&gpu_array, paths * sizeof(float)));
    CUDA_CALL(cudaMalloc((void **)&gpu_seeds, paths * sizeof(int)));

    std::vector<unsigned int> cpu_seeds(paths);
    init_seeds(cpu_seeds.data(), paths);
    CUDA_CALL(cudaMemcpy(gpu_seeds, cpu_seeds.data(), paths * sizeof(int),
                         cudaMemcpyHostToDevice));
}

Simulation::~Simulation() {
    CUDA_CALL(cudaFree(gpu_seeds));
    CUDA_CALL(cudaFree(gpu_array));
}

void Simulation::init_seeds(unsigned int *mem, int n) {
    struct timeval tv;
    gettimeofday(&tv, 0);
    srand((unsigned int)tv.tv_usec);

    for (int i = 0; i < n; ++i) {
        mem[i] = rand();
    }
}

Result Simulation::finish() {
    float mean =
        mcgpu::helpers::reduction_gpu(gpu_array, paths, blocks, threads);

    mcgpu::helpers::invoke_kernel_variance(gpu_array, mean, blocks, threads);

    float var =
        mcgpu::helpers::reduction_gpu(gpu_array, paths, blocks, threads);

    return {mean, var};
}
}
}
