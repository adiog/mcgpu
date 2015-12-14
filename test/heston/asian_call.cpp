/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 05:35:02 PM CET
 */

#include <iostream>
#include <memory>

#include "mcgpu/model/MarketModel.hpp"
#include "mcgpu/model/Heston.hpp"
#include "mcgpu/payoff/asian/AsianCall.hpp"


#include <gtest/gtest.h>

TEST(HestonTest, AsianCallEulerTest) {
    std::unique_ptr<mcgpu::model::MarketModel> mm(
            new mcgpu::model::Heston());
    std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
            new mcgpu::payoff::asian::AsianCall());

    std::unique_ptr<mcgpu::simulation::Simulation> simulation(
            new mcgpu::simulation::Simulation());

    mm->runEulerSimulation(payoff.get(), simulation.get());

    mcgpu::simulation::Result result = simulation->finish();

    std::cout << result.getMean() << " " << result.getVariance() << std::endl;
}

TEST(HestonTest, AsianCallMilsteinTest) {
    std::unique_ptr<mcgpu::model::MarketModel> mm(
            new mcgpu::model::Heston());
    std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
            new mcgpu::payoff::asian::AsianCall());

    std::unique_ptr<mcgpu::simulation::Simulation> simulation(
            new mcgpu::simulation::Simulation());

    mm->runMilsteinSimulation(payoff.get(), simulation.get());

    mcgpu::simulation::Result result = simulation->finish();

    std::cout << result.getMean() << " " << result.getVariance() << std::endl;
}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
