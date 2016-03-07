/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:44:54 PM CET
 *   modified: Fri 11 Dec 2015 07:21:29 PM CET
 */

#include <cmath>
#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/payoff/asian/AsianCall.hpp"

__device__ float asian_call_apply(float stock, float acc, float T, void *data) {
    float K = ((AsianCallApplyArgs *)data)->K;
    float contgeoavg = exp((1.0F / T) * acc);
    return ((contgeoavg > K) ? (contgeoavg - K) : 0.0F);
};

__global__ void get_asian_call_apply(gpu_asian_apply *apply_ptr) {
    *apply_ptr = asian_call_apply;
}

__device__ float asian_call_fold(float stock, float acc, float t, float dT,
                                 void *data) {
    return acc + log(stock) * dT;
};

__global__ void get_asian_call_fold(gpu_asian_fold *fold_ptr) {
    *fold_ptr = asian_call_fold;
}

namespace mcgpu {
namespace payoff {
namespace asian {

AsianCall::AsianCall(float K) : applyArgs{K} {
    gpu_asian_apply *function_to_pointer_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_on_device_ptr,
                         sizeof(gpu_asian_apply)));
    get_asian_call_apply<<<1, 1>>>(function_to_pointer_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_apply, function_to_pointer_on_device_ptr,
                         sizeof(gpu_asian_apply), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_on_device_ptr));

    gpu_asian_fold *function_to_pointer_fold_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_fold_on_device_ptr,
                         sizeof(gpu_asian_fold)));
    get_asian_call_fold<<<1, 1>>>(function_to_pointer_fold_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_fold, function_to_pointer_fold_on_device_ptr,
                         sizeof(gpu_asian_fold), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_fold_on_device_ptr));

    CUDA_CALL(cudaMalloc((void **)&gpu_apply_args, sizeof(AsianCallApplyArgs)));
    init_acc = 0.0F;

    gpu_fold_args = (void *)0;
    CUDA_CALL(cudaMemcpy(gpu_apply_args, &applyArgs, sizeof(AsianCallApplyArgs),
                         cudaMemcpyHostToDevice));
}

AsianCall::~AsianCall() { CUDA_CALL(cudaFree(gpu_apply_args)); }
}
}
}
