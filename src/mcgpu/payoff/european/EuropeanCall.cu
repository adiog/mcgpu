/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Sat 12 Dec 2015 09:16:23 PM CET
 */

#include "mcgpu/payoff/european/EuropeanCall.hpp"
#include "mcgpu/helpers/cuda_call.hpp"

__device__ float european_call_apply(float stock, float T, void *data) {
    float K = ((EuropeanCallApplyArgs *)data)->K;
    return ((stock > K) ? (stock - K) : 0.0F);
};

__global__ void get_european_call_apply(gpu_euro_apply *apply_ptr) {
    *apply_ptr = european_call_apply;
}

namespace mcgpu {
namespace payoff {
namespace european {

EuropeanCall::EuropeanCall(float K) : applyArgs{K} {
    gpu_euro_apply *function_to_pointer_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply)));
    get_european_call_apply<<<1, 1>>>(function_to_pointer_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_apply, function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_on_device_ptr));

    CUDA_CALL(
        cudaMalloc((void **)&gpu_apply_args, sizeof(EuropeanCallApplyArgs)));

    CUDA_CALL(cudaMemcpy(gpu_apply_args, &applyArgs,
                         sizeof(EuropeanCallApplyArgs),
                         cudaMemcpyHostToDevice));
}

EuropeanCall::~EuropeanCall() { CUDA_CALL(cudaFree(gpu_apply_args)); }
}
}
}
