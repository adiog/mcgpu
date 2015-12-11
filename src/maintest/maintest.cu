/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 05:35:02 PM CET
 */

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <sys/time.h>

#include "mcgpu/host/Timer.hpp"
#include "mcgpu/host/Random.hpp"
#include "mcgpu/host/Theory.hpp"

#include "mcgpu/model/MarketModel.hpp"
#include "mcgpu/model/BlackScholes.hpp"
#include "mcgpu/payoff/european/EuropeanCall.hpp"
#include "mcgpu/simulation/Simulation.hpp"

#include <cuda_runtime_api.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <mcgpu/helpers/cuda_call.hpp>
#include <thread>

int main(int argc, char **argv) {
    int deviceCount;
    int device = 0;

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA device found." << std::endl;
        return 1;
    }

    CUDA_CALL(cudaSetDevice(device));

    std::unique_ptr<mcgpu::model::MarketModel> mm(
        new mcgpu::model::BlackScholes());
    std::unique_ptr<mcgpu::payoff::european::European> payoff(
        new mcgpu::payoff::european::EuropeanCall());

    std::unique_ptr<mcgpu::simulation::Simulation> simulation(
        new mcgpu::simulation::Simulation());

    mm->runEulerSimulation(payoff.get(), simulation.get());

    std::pair<float, float> result = simulation->finish();

    float expectedResult =
        mcgpu::host::european_call_price(50.0F, 0.05F, 0.3F, 1.0F, 50.0F);

    std::cout << expectedResult << " " << result.first << " " << result.second << std::endl;

    return 0;
}
