/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Sat 12 Dec 2015 09:15:39 PM CET
 */

#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/payoff/european/EuropeanBinary.hpp"

__device__ float european_binary_apply(float stock, float T, void *data) {
    float K = ((EuropeanBinaryApplyArgs *)data)->K;
    float pay = ((EuropeanBinaryApplyArgs *)data)->pay;
    bool activateAbove = ((EuropeanBinaryApplyArgs *)data)->activateAbove;
    if (activateAbove) {
        return (stock > K) ? pay : 0.0F;
    } else {
        return (stock < K) ? pay : 0.0F;
    }
};

__global__ void get_european_binary_apply(gpu_euro_apply *apply_ptr) {
    *apply_ptr = european_binary_apply;
}

namespace mcgpu {
namespace payoff {
namespace european {

EuropeanBinary::EuropeanBinary(float K, float pay, bool activateAbove)
    : applyArgs{K, pay, activateAbove} {
    gpu_euro_apply *function_to_pointer_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply)));
    get_european_binary_apply<<<1, 1>>>(function_to_pointer_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_apply, function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_on_device_ptr));

    CUDA_CALL(
        cudaMalloc((void **)&gpu_apply_args, sizeof(EuropeanBinaryApplyArgs)));

    CUDA_CALL(cudaMemcpy(gpu_apply_args, &applyArgs,
                         sizeof(EuropeanBinaryApplyArgs),
                         cudaMemcpyHostToDevice));
}

EuropeanBinary::~EuropeanBinary() { CUDA_CALL(cudaFree(gpu_apply_args)); }
}
}
}
