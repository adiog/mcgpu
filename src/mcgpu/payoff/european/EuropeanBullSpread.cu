/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Sat 12 Dec 2015 09:15:48 PM CET
 */

#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/payoff/european/EuropeanBullSpread.hpp"

__device__ float european_bull_apply(float stock, float T, void *data) {
    float K1 = ((EuropeanBullSpreadApplyArgs *)data)->K1;
    float K2 = ((EuropeanBullSpreadApplyArgs *)data)->K2;
    return ((stock > K1) ? (stock - K1) : 0.0F) -
           ((stock > K2) ? (stock - K2) : 0.0F);
};

__global__ void get_european_bull_apply(gpu_euro_apply *apply_ptr) {
    *apply_ptr = european_bull_apply;
}

namespace mcgpu {
namespace payoff {
namespace european {

EuropeanBullSpread::EuropeanBullSpread(float K1, float K2) : applyArgs{K1, K2} {
    gpu_euro_apply *function_to_pointer_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply)));
    get_european_bull_apply<<<1, 1>>>(function_to_pointer_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_apply, function_to_pointer_on_device_ptr,
                         sizeof(gpu_euro_apply), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_on_device_ptr));

    CUDA_CALL(
        cudaMalloc((void **)&gpu_apply_args, sizeof(EuropeanBinaryApplyArgs)));

    CUDA_CALL(cudaMemcpy(gpu_apply_args, &applyArgs,
                         sizeof(EuropeanBinaryApplyArgs),
                         cudaMemcpyHostToDevice));
}

EuropeanBullSpread::~EuropeanBullSpread() {
    CUDA_CALL(cudaFree(gpu_apply_args));
}
}
}
}
