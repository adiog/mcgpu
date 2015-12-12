/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 05:35:02 PM CET
 */

#include <iostream>
#include <memory>

#include "mcgpu/model/MarketModel.hpp"
#include "mcgpu/model/BlackScholes.hpp"
#include "mcgpu/payoff/european/EuropeanBullSpread.hpp"

int main(int argc, char **argv) {
    std::unique_ptr<mcgpu::model::MarketModel> mm(
        new mcgpu::model::BlackScholes());
    std::unique_ptr<mcgpu::payoff::european::European> payoff(
        new mcgpu::payoff::european::EuropeanBullSpread());

    std::unique_ptr<mcgpu::simulation::Simulation> simulation(
        new mcgpu::simulation::Simulation());

    mm->runEulerSimulation(payoff.get(), simulation.get());

    mcgpu::simulation::Result result = simulation->finish();

    std::cout << result.getMean() << " " << result.getVariance() << std::endl;

    return 0;
}
