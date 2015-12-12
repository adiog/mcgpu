/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 05:39:40 PM CET
 */

#ifndef MCGPU_MODEL_BLACKSCHOLES_HPP
#define MCGPU_MODEL_BLACKSCHOLES_HPP

#include "mcgpu/model/MarketModel.hpp"

namespace mcgpu {
namespace model {

class BlackScholes final : public MarketModel {
  public:
    BlackScholes(float r = 0.05F, float sigma = 0.3F) : r(r), sigma(sigma){};
    BlackScholes(const BlackScholes &bs) = delete;
    BlackScholes(BlackScholes &&bs) = delete;
    BlackScholes &operator=(const BlackScholes &bs) = delete;
    BlackScholes &operator=(BlackScholes &&bs) = delete;
    ~BlackScholes() = default;

    virtual void runEulerSimulation(
        const mcgpu::payoff::european::European *payoff,
        const mcgpu::simulation::Simulation *simulation) const override final;
    virtual void runEulerSimulation(
        const mcgpu::payoff::asian::Asian *payoff,
        const mcgpu::simulation::Simulation *simulation) const override final;

    virtual void runMilsteinSimulation(
        const mcgpu::payoff::european::European *payoff,
        const mcgpu::simulation::Simulation *simulation) const override final;
    virtual void runMilsteinSimulation(
        const mcgpu::payoff::asian::Asian *payoff,
        const mcgpu::simulation::Simulation *simulation) const override final;

  private:
    /// interest free rate
    float r;
    /// volatility
    float sigma;
};
}
}

#endif
