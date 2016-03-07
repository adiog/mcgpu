/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Mon Apr  2 05:24:43 2012
 */

#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/model/BlackScholes.hpp"
#include "mcgpu/payoff/asian/Asian.hpp"
#include "mcgpu/payoff/european/European.hpp"
#include "mcgpu/simulation/Simulation.hpp"

__global__ void kernel_blackscholes_eulermaruyama_european(
    float S0, float r, float sigma, float T, int points, float *prices,
    unsigned int *rands, gpu_euro_apply apply, void *apply_args) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float S = S0;
    float dt = T / points;

    // initialize random number generator
    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    for (int t = 0; t < points; ++t) {
        // perform step of Euler-Maruyama numerical scheme
        S += r * S * dt + sigma * S * sqrt(dt) * curand_normal(&state);
    }

    // store prices in global memory
    prices[i] = apply(S, T, apply_args);
};

__global__ void kernel_blackscholes_eulermaruyama_asian(
    float S0, float r, float sigma, float T, int points, float *prices,
    unsigned int *rands, gpu_asian_apply apply, void *apply_args,
    gpu_asian_fold fold, void *fold_args, float init_acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S = S0;
    float acc = init_acc;
    float dt = T / points;

    // initialize random number generator
    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    for (int t = 0; t < points; ++t) {
        // perform step of Euler-Maruyama numerical scheme
        S += r * S * dt + sigma * S * sqrt(dt) * curand_normal(&state);
        // apply fold function
        acc = (*fold)(S, acc, dt * t, dt, fold_args);
    }

    // store prices in global memory
    prices[i] = (*apply)(S, acc, T, apply_args);
};

__global__ void kernel_blackscholes_milstein_european(
    float S0, float r, float sigma, float T, int points, float *prices,
    unsigned int *rands, gpu_euro_apply apply, void *apply_args) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S = S0;
    float dt = T / points;
    float dW;

    // initialize random number generator
    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    for (int t = 0; t < points; ++t) {
        // perform step of Milstein numerical scheme
        dW = sqrt(dt) * curand_normal(&state);
        S += r * S * dt + sigma * S * dW +
             (0.5F * (sigma * sigma) * S * ((dW * dW) - dt));
    }

    // store prices in global memory
    prices[i] = (*apply)(S, T, apply_args);
};

__global__ void kernel_blackscholes_milstein_asian(
    float S0, float r, float sigma, float T, int points, float *prices,
    unsigned int *rands, gpu_asian_apply apply, void *apply_args,
    gpu_asian_fold fold, void *fold_args, float init_acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S = S0;
    float acc = init_acc;
    float dt = T / points;
    float dW;

    // initialize random number generator
    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    for (int t = 0; t < points; ++t) {
        // perform step of Milstein numerical scheme
        dW = sqrt(dt) * curand_normal(&state);
        S += r * S * dt + sigma * S * dW +
             (0.5F * (sigma * sigma) * S * ((dW * dW) - dt));
        // apply fold function
        acc = (*fold)(S, acc, dt * t, dt, fold_args);
    }

    // store prices in global memory
    prices[i] = (*apply)(S, acc, T, apply_args);
};

namespace mcgpu {
namespace model {

void BlackScholes::runEulerSimulation(
    const mcgpu::payoff::european::European *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_blackscholes_eulermaruyama_european<<<simulation->get_blocks(),
                                                 simulation->get_threads()>>>(
        S0, r, sigma, T, simulation->get_points(), simulation->get_gpu_array(),
        simulation->get_gpu_seeds(), payoff->get_apply(),
        payoff->get_apply_args());
}

void BlackScholes::runEulerSimulation(
    const mcgpu::payoff::asian::Asian *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_blackscholes_eulermaruyama_asian<<<simulation->get_blocks(),
                                              simulation->get_threads()>>>(
        S0, r, sigma, T, simulation->get_points(), simulation->get_gpu_array(),
        simulation->get_gpu_seeds(), payoff->get_apply(),
        payoff->get_apply_args(), payoff->get_fold(), payoff->get_fold_args(),
        payoff->get_init_acc());
}

void BlackScholes::runMilsteinSimulation(
    const mcgpu::payoff::european::European *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_blackscholes_milstein_european<<<simulation->get_blocks(),
                                            simulation->get_threads()>>>(
        S0, r, sigma, T, simulation->get_points(), simulation->get_gpu_array(),
        simulation->get_gpu_seeds(), payoff->get_apply(),
        payoff->get_apply_args());
}

void BlackScholes::runMilsteinSimulation(
    const mcgpu::payoff::asian::Asian *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_blackscholes_milstein_asian<<<simulation->get_blocks(),
                                         simulation->get_threads()>>>(
        S0, r, sigma, T, simulation->get_points(), simulation->get_gpu_array(),
        simulation->get_gpu_seeds(), payoff->get_apply(),
        payoff->get_apply_args(), payoff->get_fold(), payoff->get_fold_args(),
        payoff->get_init_acc());
}
}
}
