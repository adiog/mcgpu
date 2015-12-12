/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Sat 12 Dec 2015 09:17:35 PM CET
 */

#ifndef MCGPU_PAYOFF_EUROPEAN_EUROPEANPUT_HPP
#define MCGPU_PAYOFF_EUROPEAN_EUROPEANPUT_HPP

#include "mcgpu/payoff/european/EuropeanPut.hpp"

struct EuropeanPutApplyArgs {
    /// strike price
    float K;
};

namespace mcgpu {
namespace payoff {
namespace european {

/**
 * The value of the put option at expire, is max(K-S, 0).
 */
class EuropeanPut : public mcgpu::payoff::european::European {
  public:
    EuropeanPut(float K = 50.0F);
    EuropeanPut(const EuropeanPut &europeanPut) = delete;
    EuropeanPut(EuropeanPut &&europeanPut) = delete;
    EuropeanPut &operator=(const EuropeanPut &europeanPut) = delete;
    EuropeanPut &operator=(EuropeanPut &&europeanPut) = delete;

    virtual ~EuropeanPut();

  private:
    EuropeanPutApplyArgs applyArgs;
};
}
}
}

#endif
