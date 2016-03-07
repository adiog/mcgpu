/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:01 PM CET
 *   modified: Mon Apr  2 20:40:10 2012
 */

#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/payoff/asian/BarrierUpAndOutCall.hpp"

__device__ float asian_barrier_apply(float stock, float acc, float T,
                                     void *data) {
    float K = ((float *)data)[0];
    return (((acc == 1.0F) && (stock > K)) ? (stock - K) : 0.0F);
};

__global__ void get_asian_barrier_apply(gpu_asian_apply *apply_ptr) {
    *apply_ptr = asian_barrier_apply;
}

__device__ float asian_barrier_fold(float stock, float acc, float t, float dT,
                                    void *data) {
    float B = ((float *)data)[0];
    return ((acc == 1.0F) && (stock < B)) ? 1.0F : 0.0F;
};

__global__ void get_asian_barrier_fold(gpu_asian_fold *fold_ptr) {
    *fold_ptr = asian_barrier_fold;
}

namespace mcgpu {
namespace payoff {
namespace asian {

BarrierUpAndOutCall::BarrierUpAndOutCall(float K, float B)
    : applyArgs{K}, foldArgs{B} {
    gpu_asian_apply *function_to_pointer_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_on_device_ptr,
                         sizeof(gpu_asian_apply)));
    get_asian_barrier_apply<<<1, 1>>>(function_to_pointer_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_apply, function_to_pointer_on_device_ptr,
                         sizeof(gpu_asian_apply), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_on_device_ptr));

    gpu_asian_fold *function_to_pointer_fold_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_fold_on_device_ptr,
                         sizeof(gpu_asian_fold)));
    get_asian_barrier_fold<<<1, 1>>>(function_to_pointer_fold_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_fold, function_to_pointer_fold_on_device_ptr,
                         sizeof(gpu_asian_fold), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_fold_on_device_ptr));

    CUDA_CALL(cudaMalloc((void **)&gpu_apply_args,
                         sizeof(BarrierUpAndOutCallApplyArgs)));
    CUDA_CALL(cudaMalloc((void **)&gpu_fold_args,
                         sizeof(BarrierUpAndOutCallFoldArgs)));
    init_acc = 1.0F;

    CUDA_CALL(cudaMemcpy(gpu_apply_args, &applyArgs,
                         sizeof(BarrierUpAndOutCallApplyArgs),
                         cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(gpu_fold_args, &foldArgs,
                         sizeof(BarrierUpAndOutCallFoldArgs),
                         cudaMemcpyHostToDevice));
}

BarrierUpAndOutCall::~BarrierUpAndOutCall() {
    CUDA_CALL(cudaFree(gpu_apply_args));
    CUDA_CALL(cudaFree(gpu_fold_args));
}
}
}
}
