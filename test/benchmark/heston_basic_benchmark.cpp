/*
 * Copyright 2015 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 12 Dec 2015 05:26:13 PM CET
 *   modified: Sat 12 Dec 2015 05:26:36 PM CET
 */

#include "mcgpu/model/MarketModel.hpp"
#include "mcgpu/model/Heston.hpp"
#include "mcgpu/payoff/european/EuropeanBinary.hpp"
#include "mcgpu/payoff/european/EuropeanBullSpread.hpp"
#include "mcgpu/payoff/european/EuropeanCall.hpp"
#include "mcgpu/payoff/european/EuropeanPut.hpp"
#include "mcgpu/payoff/asian/AsianCall.hpp"
#include "mcgpu/payoff/asian/BarrierUpAndOutCall.hpp"
#include "mcgpu/payoff/asian/ParisBarrierUpAndOutCall.hpp"

#include <memory>
#include <benchmark/benchmark_api.h>

#define SIMULATION_BENCHMARK(bm_function) \
    BENCHMARK(bm_function)                \
        ->ArgPair(1000, 100)              \
        ->ArgPair(10000, 100)             \
        ->ArgPair(100000, 100)            \
        ->ArgPair(1000, 1000)             \
        ->ArgPair(10000, 1000);

void BM_Heston_EuropeanCall_Euler(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
            new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
            new mcgpu::payoff::european::EuropeanCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
            new mcgpu::simulation::Simulation(state.range_x(),
                                              state.range_y()));

        mm->runEulerSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_EuropeanCall_Milstein(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
            new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
            new mcgpu::payoff::european::EuropeanCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
            new mcgpu::simulation::Simulation(state.range_x(),
                                              state.range_y()));

        mm->runMilsteinSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_EuropeanPut_Euler(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
                new mcgpu::payoff::european::EuropeanPut());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runEulerSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_EuropeanPut_Milstein(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
                new mcgpu::payoff::european::EuropeanPut());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runMilsteinSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_EuropeanBinary_Euler(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
                new mcgpu::payoff::european::EuropeanBinary());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runEulerSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_EuropeanBinary_Milstein(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
                new mcgpu::payoff::european::EuropeanBinary());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runMilsteinSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_EuropeanBullSpread_Euler(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
                new mcgpu::payoff::european::EuropeanBullSpread());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runEulerSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_EuropeanBullSpread_Milstein(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::european::European> payoff(
                new mcgpu::payoff::european::EuropeanBullSpread());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runMilsteinSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_AsianCall_Euler(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
                new mcgpu::payoff::asian::AsianCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runEulerSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_AsianCall_Milstein(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
                new mcgpu::payoff::asian::AsianCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runMilsteinSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_AsianBarrier_Euler(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
                new mcgpu::payoff::asian::BarrierUpAndOutCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runEulerSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_AsianBarrier_Milstein(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
                new mcgpu::payoff::asian::BarrierUpAndOutCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runMilsteinSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_AsianParis_Euler(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
                new mcgpu::payoff::asian::ParisBarrierUpAndOutCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runEulerSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

void BM_Heston_AsianParis_Milstein(benchmark::State& state) {
    while (state.KeepRunning()) {
        std::unique_ptr<mcgpu::model::MarketModel> mm(
                new mcgpu::model::Heston());
        std::unique_ptr<mcgpu::payoff::asian::Asian> payoff(
                new mcgpu::payoff::asian::ParisBarrierUpAndOutCall());

        std::unique_ptr<mcgpu::simulation::Simulation> simulation(
                new mcgpu::simulation::Simulation(state.range_x(),
                                                  state.range_y()));

        mm->runMilsteinSimulation(payoff.get(), simulation.get());

        mcgpu::simulation::Result result = simulation->finish();
    }
}

SIMULATION_BENCHMARK(BM_Heston_EuropeanCall_Euler);
SIMULATION_BENCHMARK(BM_Heston_EuropeanCall_Milstein);
SIMULATION_BENCHMARK(BM_Heston_EuropeanPut_Euler);
SIMULATION_BENCHMARK(BM_Heston_EuropeanPut_Milstein);
SIMULATION_BENCHMARK(BM_Heston_EuropeanBinary_Euler);
SIMULATION_BENCHMARK(BM_Heston_EuropeanBinary_Milstein);
SIMULATION_BENCHMARK(BM_Heston_EuropeanBullSpread_Euler);
SIMULATION_BENCHMARK(BM_Heston_EuropeanBullSpread_Milstein);

SIMULATION_BENCHMARK(BM_Heston_AsianCall_Euler);
SIMULATION_BENCHMARK(BM_Heston_AsianCall_Milstein);
SIMULATION_BENCHMARK(BM_Heston_AsianBarrier_Euler);
SIMULATION_BENCHMARK(BM_Heston_AsianBarrier_Milstein);
SIMULATION_BENCHMARK(BM_Heston_AsianParis_Euler);
SIMULATION_BENCHMARK(BM_Heston_AsianParis_Milstein);

BENCHMARK_MAIN()
