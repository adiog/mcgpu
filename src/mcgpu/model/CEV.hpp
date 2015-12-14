/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Wed Jun  6 09:33:04 2012
 */

#ifndef MCGPU_MODEL_CEV_HPP
#define MCGPU_MODEL_CEV_HPP

#include "mcgpu/model/MarketModel.hpp"

namespace mcgpu {
namespace model {

class CEV : public MarketModel {
  public:
    CEV(float r = 0.05F, float sigma = 0.3F, float alpha = 0.7F) : r(r), sigma(sigma), alpha(alpha) {};
    CEV(const CEV &cev) = delete;
    CEV(CEV &&cev) = delete;
    CEV &operator=(const CEV &cev) = delete;
    CEV &operator=(CEV &&cev) = delete;
    virtual ~CEV() = default;

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
    /// elasticity factor
    float alpha;
};
}
}

#endif
