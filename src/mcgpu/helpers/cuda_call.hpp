/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 05:17:34 AM CET
 */

#ifndef MCGPU_HELPERS_CUDA_CALL_HPP
#define MCGPU_HELPERS_CUDA_CALL_HPP

#include <curand.h>
#include <curand_kernel.h>

#include <cuda_runtime_api.h>
#include <cstdio>

#define CUDA_CALL(command)                        \
    do {                                          \
        gpuAssert((command), __FILE__, __LINE__); \
    } while (0);
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s [at %s:%d]\n", cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

#endif
