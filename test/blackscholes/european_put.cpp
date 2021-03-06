/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 05:35:02 PM CET
 */

#include <iostream>
#include <memory>

#include "mcgpu/host/Timer.hpp"
#include "mcgpu/host/Random.hpp"
#include "mcgpu/host/Theory.hpp"

#include "mcgpu/model/MarketModel.hpp"
#include "mcgpu/model/BlackScholes.hpp"
#include "mcgpu/payoff/european/EuropeanPut.hpp"

#include <gtest/gtest.h>

TEST(BlackScholesTest, EuropeanPutEulerTest) {
    std::unique_ptr<mcgpu::model::MarketModel> mm(
            new mcgpu::model::BlackScholes());
    std::unique_ptr<mcgpu::payoff::european::European> payoff(
            new mcgpu::payoff::european::EuropeanPut());

    std::unique_ptr<mcgpu::simulation::Simulation> simulation(
            new mcgpu::simulation::Simulation());

    mm->runEulerSimulation(payoff.get(), simulation.get());

    mcgpu::simulation::Result result = simulation->finish();

    float expectedResult =
            mcgpu::host::european_put_price(50.0F, 0.05F, 0.3F, 1.0F, 50.0F);

    std::cout << expectedResult << " " << result.getMean() << " " << result.getVariance() << std::endl;
}

TEST(BlackScholesTest, EuropeanPutMilsteinTest) {
    std::unique_ptr<mcgpu::model::MarketModel> mm(
            new mcgpu::model::BlackScholes());
    std::unique_ptr<mcgpu::payoff::european::European> payoff(
            new mcgpu::payoff::european::EuropeanPut());

    std::unique_ptr<mcgpu::simulation::Simulation> simulation(
            new mcgpu::simulation::Simulation());

    mm->runMilsteinSimulation(payoff.get(), simulation.get());

    mcgpu::simulation::Result result = simulation->finish();

    float expectedResult =
            mcgpu::host::european_put_price(50.0F, 0.05F, 0.3F, 1.0F, 50.0F);

    std::cout << expectedResult << " " << result.getMean() << " " << result.getVariance() << std::endl;
}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

