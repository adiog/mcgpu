/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Wed Jun  6 09:33:04 2012
 */

#ifndef MCGPU_MODEL_HESTON_HPP
#define MCGPU_MODEL_HESTON_HPP

#include "mcgpu/model/MarketModel.hpp"

namespace mcgpu {
namespace model {

class Heston : public MarketModel {
  public:
    Heston(float r = 0.05F, float V0 = 0.2F, float kappa = 0.8F,
           float theta = 0.3F, float xi = 0.1F, float rho = 0.5F)
        : r(r), V0(V0), kappa(kappa), theta(theta), xi(xi), rho(rho){};
    Heston(const Heston &heston) = delete;
    Heston(Heston &&heston) = delete;
    Heston &operator=(const Heston &heston) = delete;
    Heston &operator=(Heston &&heston) = delete;
    virtual ~Heston() = default;

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
    /// initial volatility
    float V0;
    /// theta-vol reverts rate
    float kappa;
    // long variance
    float theta;
    // volatility of volatility
    float xi;
    // correlation coefficient
    float rho;
};
}
}

#endif
