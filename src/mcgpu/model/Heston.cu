/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Mon Apr  2 04:24:42 2012
 */

#include <cmath>
#include "mcgpu/helpers/cuda_call.hpp"
#include "mcgpu/model/Heston.hpp"
#include "mcgpu/payoff/asian/Asian.hpp"
#include "mcgpu/payoff/european/European.hpp"
#include "mcgpu/simulation/Simulation.hpp"

__global__ void kernel_heston_eulermaruyama_european(
    float S0, float r, float V0, float kappa, float theta, float xi, float rho,
    float T, int points, float *prices, unsigned int *rands,
    gpu_euro_apply apply, void *apply_args) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S = S0;
    float V = V0;
    float dt = T / points;

    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    float dW1, dW2;
    float dWs, dWv;

    for (int t = 0; t < points; ++t) {
        dW1 = curand_normal(&state) * sqrt(dt);
        dW2 = curand_normal(&state) * sqrt(dt);
        // introducing correlation
        dWs = dW1 + rho * dW2;
        dWv = sqrt(1 - rho * rho) * dW2;
        // apply scheme
        V += kappa * (theta - V) * dt + xi * sqrt(V) * dWv;
        S += r * S * dt + sqrt(V) * S * dWs;
    }

    // store prices in global memory
    prices[i] = (*apply)(S, T, apply_args);
};

__global__ void kernel_heston_eulermaruyama_asian(
    float S0, float r, float V0, float kappa, float theta, float xi, float rho,
    float T, int points, float *prices, unsigned int *rands,
    gpu_asian_apply apply, void *apply_args, gpu_asian_fold fold,
    void *fold_args, float init_acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S = S0;
    float V = V0;
    float acc = init_acc;
    float dt = T / points;

    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    float dW1, dW2;
    float dWs, dWv;

    for (int t = 0; t < points; ++t) {
        dW1 = curand_normal(&state) * sqrt(dt);
        dW2 = curand_normal(&state) * sqrt(dt);
        // introducing correlation
        dWs = dW1 + rho * dW2;
        dWv = sqrt(1 - rho * rho) * dW2;
        // apply scheme
        V += kappa * (theta - V) * dt + xi * sqrt(V) * dWv;
        S += r * S * dt + sqrt(V) * S * dWs;
        acc = (*fold)(S, acc, dt * t, dt, fold_args);
    }

    // store prices in global memory
    prices[i] = (*apply)(S, acc, T, apply_args);
};

__global__ void kernel_heston_milstein_european(
    float S0, float r, float V0, float kappa, float theta, float xi, float rho,
    float T, int points, float *prices, unsigned int *rands,
    gpu_euro_apply apply, void *apply_args) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S = S0;
    float V = V0;
    float dt = T / points;

    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    float dW1, dW2;
    float dWs, dWv;

    for (int t = 0; t < points; ++t) {
        dW1 = curand_normal(&state) * sqrt(dt);
        dW2 = curand_normal(&state) * sqrt(dt);
        // introducing correlation
        dWs = dW1 + rho * dW2;
        dWv = sqrt(1 - rho * rho) * dW2;
        // apply scheme
        V += kappa * (theta - V) * dt + xi * sqrt(V) * dWv -
             0.25 * xi * xi * (dWv * dWv - dt);
        S += r * S * dt + sqrt(V) * S * dWs + 0.5 * V * S * (dWs * dWs - dt);
    }

    // store prices in global memory
    prices[i] = (*apply)(S, T, apply_args);
};

__global__ void kernel_heston_milstein_asian(
    float S0, float r, float V0, float kappa, float theta, float xi, float rho,
    float T, int points, float *prices, unsigned int *rands,
    gpu_asian_apply apply, void *apply_args, gpu_asian_fold fold,
    void *fold_args, float init_acc) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    float S = S0;
    float V = V0;
    float acc = init_acc;
    float dt = T / points;

    curandStateXORWOW_t state;
    curand_init(rands[i], i, 0, &state);

    float dW1, dW2;
    float dWs, dWv;

    for (int t = 0; t < points; ++t) {
        dW1 = curand_normal(&state) * sqrt(dt);
        dW2 = curand_normal(&state) * sqrt(dt);
        // introducing correlation
        dWs = dW1 + rho * dW2;
        dWv = sqrt(1 - rho * rho) * dW2;
        // apply scheme
        V += kappa * (theta - V) * dt + xi * sqrt(V) * dWv -
             0.25 * xi * xi * (dWv * dWv - dt);
        S += r * S * dt + sqrt(V) * S * dWs + 0.5 * V * S * (dWs * dWs - dt);
        acc = (*fold)(S, acc, dt * t, dt, fold_args);
    }

    // store prices in global memory
    prices[i] = (*apply)(S, acc, T, apply_args);
};

namespace mcgpu {
namespace model {

void Heston::runEulerSimulation(
    const mcgpu::payoff::european::European *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_heston_eulermaruyama_european<<<simulation->get_blocks(),
                                           simulation->get_threads()>>>(
        S0, r, V0, kappa, theta, xi, rho, T, simulation->get_points(),
        simulation->get_gpu_array(), simulation->get_gpu_seeds(),
        payoff->get_apply(), payoff->get_apply_args());
}

void Heston::runEulerSimulation(
    const mcgpu::payoff::asian::Asian *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_heston_eulermaruyama_asian<<<simulation->get_blocks(),
                                        simulation->get_threads()>>>(
        S0, r, V0, kappa, theta, xi, rho, T, simulation->get_points(),
        simulation->get_gpu_array(), simulation->get_gpu_seeds(),
        payoff->get_apply(), payoff->get_apply_args(), payoff->get_fold(),
        payoff->get_fold_args(), payoff->get_init_acc());
}

void Heston::runMilsteinSimulation(
    const mcgpu::payoff::european::European *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_heston_milstein_european<<<simulation->get_blocks(),
                                      simulation->get_threads()>>>(
        S0, r, V0, kappa, theta, xi, rho, T, simulation->get_points(),
        simulation->get_gpu_array(), simulation->get_gpu_seeds(),
        payoff->get_apply(), payoff->get_apply_args());
}

void Heston::runMilsteinSimulation(
    const mcgpu::payoff::asian::Asian *payoff,
    const mcgpu::simulation::Simulation *simulation) const {
    kernel_heston_milstein_asian<<<simulation->get_blocks(),
                                   simulation->get_threads()>>>(
        S0, r, V0, kappa, theta, xi, rho, T, simulation->get_points(),
        simulation->get_gpu_array(), simulation->get_gpu_seeds(),
        payoff->get_apply(), payoff->get_apply_args(), payoff->get_fold(),
        payoff->get_fold_args(), payoff->get_init_acc());
}
}
}
