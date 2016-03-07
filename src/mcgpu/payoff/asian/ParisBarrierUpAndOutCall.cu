/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Mon Apr  2 20:39:41 2012
 */

#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/payoff/asian/ParisBarrierUpAndOutCall.hpp"

__device__ float asian_paris_apply(float stock, float acc, float T,
                                   void *data) {
    float K = ((float *)data)[0];
    return (((acc == 1.0F) && (stock > K)) ? (stock - K) : 0.0F);
};

__global__ void get_asian_paris_apply(gpu_asian_apply *apply_ptr) {
    *apply_ptr = asian_paris_apply;
}

__device__ float asian_paris_fold(float stock, float acc, float t, float dT,
                                  void *data) {
    float B = ((float *)data)[0];
    float timein = ((float *)data)[1];
    return ((acc == 1.0F) && ((t < timein) || (stock < B)));
};

__global__ void get_asian_paris_fold(gpu_asian_fold *fold_ptr) {
    *fold_ptr = asian_paris_fold;
}

namespace mcgpu {
namespace payoff {
namespace asian {
ParisBarrierUpAndOutCall::ParisBarrierUpAndOutCall(float K, float B,
                                                   float timein)
    : applyArgs{K}, foldArgs{B, timein} {
    gpu_asian_apply *function_to_pointer_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_on_device_ptr,
                         sizeof(gpu_asian_apply)));
    get_asian_paris_apply<<<1, 1>>>(function_to_pointer_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_apply, function_to_pointer_on_device_ptr,
                         sizeof(gpu_asian_apply), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_on_device_ptr));

    gpu_asian_fold *function_to_pointer_fold_on_device_ptr;
    CUDA_CALL(cudaMalloc((void **)&function_to_pointer_fold_on_device_ptr,
                         sizeof(gpu_asian_fold)));
    get_asian_paris_fold<<<1, 1>>>(function_to_pointer_fold_on_device_ptr);
    CUDA_CALL(cudaMemcpy(&gpu_fold, function_to_pointer_fold_on_device_ptr,
                         sizeof(gpu_asian_fold), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(function_to_pointer_fold_on_device_ptr));

    CUDA_CALL(cudaMalloc((void **)&gpu_apply_args,
                         sizeof(ParisBarrierUpAndOutCallApplyArgs)));
    CUDA_CALL(cudaMalloc((void **)&gpu_fold_args,
                         sizeof(ParisBarrierUpAndOutCallFoldArgs)));

    init_acc = 1.0F;

    CUDA_CALL(cudaMemcpy(gpu_apply_args, &applyArgs,
                         sizeof(ParisBarrierUpAndOutCallApplyArgs),
                         cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemcpy(gpu_fold_args, &foldArgs,
                         sizeof(ParisBarrierUpAndOutCallFoldArgs),
                         cudaMemcpyHostToDevice));
}

ParisBarrierUpAndOutCall::~ParisBarrierUpAndOutCall() {
    CUDA_CALL(cudaFree(gpu_apply_args));
    CUDA_CALL(cudaFree(gpu_fold_args));
}
}
}
}
