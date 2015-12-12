/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 05:39:28 PM CET
 */

#ifndef MCGPU_MODEL_MARKETMODEL_HPP
#define MCGPU_MODEL_MARKETMODEL_HPP

#include "mcgpu/payoff/european/European.hpp"
#include "mcgpu/payoff/asian/Asian.hpp"
#include "mcgpu/simulation/Simulation.hpp"

namespace mcgpu {
namespace model {

class MarketModel {
  public:
    MarketModel(float S0 = 50.0F, float T = 1.0F) : S0(S0), T(T) {}
    MarketModel(const MarketModel &mm) = delete;
    MarketModel(MarketModel &&mm) = delete;
    MarketModel &operator=(const MarketModel &mm) = delete;
    MarketModel &operator=(MarketModel &&mm) = delete;
    virtual ~MarketModel() = default;

    virtual void runEulerSimulation(
        const mcgpu::payoff::european::European *payoff,
        const mcgpu::simulation::Simulation *simulation) const = 0;
    virtual void runEulerSimulation(
        const mcgpu::payoff::asian::Asian *payoff,
        const mcgpu::simulation::Simulation *simulation) const = 0;

    virtual void runMilsteinSimulation(
        const mcgpu::payoff::european::European *payoff,
        const mcgpu::simulation::Simulation *simulation) const = 0;
    virtual void runMilsteinSimulation(
        const mcgpu::payoff::asian::Asian *payoff,
        const mcgpu::simulation::Simulation *simulation) const = 0;

  protected:
    /// initial stock price
    float S0;
    /// time to maturity
    float T;
};
}
}

#endif
