/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Fri 11 Dec 2015 05:42:06 PM CET
 */

#include <chrono>
#include <thread>
#include "mcgpu/payoff/european/EuropeanCall.hpp"
#include "mcgpu/helpers/cuda_call.hpp"

__device__ float european_call_apply(float stock, float T, void *data) {
    float K = *((float *)data);
    return ((stock > K) ? (stock - K) : 0.0);
};

__global__ void get_european_call_apply(gpu_euro_apply *apply_ptr) {
    *apply_ptr = european_call_apply;
}

namespace mcgpu {
namespace payoff {
namespace european {

EuropeanCall::EuropeanCall(float K_) : K(K_) {
    gpu_euro_apply *function_to_pointer_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply)));
    get_european_call_apply<<<1, 1>>> (function_to_pointer_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_apply, function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_on_device_ptr));

    CUDA_CALL(cudaMalloc((void **)&gpu_apply_args, sizeof(float)));

    float cpu_apply_args[1];
    cpu_apply_args[0] = K;
    CUDA_CALL(cudaMemcpy(gpu_apply_args, cpu_apply_args, sizeof(float),
                         cudaMemcpyHostToDevice));
}

EuropeanCall::~EuropeanCall() { CUDA_CALL(cudaFree(gpu_apply_args)); }
}
}
}
