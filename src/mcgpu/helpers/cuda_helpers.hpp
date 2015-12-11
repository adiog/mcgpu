/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 05:21:22 AM CET
 */

#ifndef MCGPU_HELPERS_CUDA_HELPERS_HPP
#define MCGPU_HELPERS_CUDA_HELPERS_HPP

namespace mcgpu {
namespace helpers {

void invoke_kernel_variance(float *gpu_array, float mean, int blocks,
                            int threads);
float reduction_gpu(float *gpu_array, int size, int blocks, int threads);
}
}

#endif
