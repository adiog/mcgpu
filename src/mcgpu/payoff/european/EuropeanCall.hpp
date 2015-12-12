/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Sat 12 Dec 2015 09:16:27 PM CET
 */

#ifndef MCGPU_PAYOFF_EUROPEAN_EUROPEANCALL_HPP
#define MCGPU_PAYOFF_EUROPEAN_EUROPEANCALL_HPP

#include "mcgpu/payoff/european/European.hpp"

struct EuropeanCallApplyArgs {
    /// strike price
    float K;
};

namespace mcgpu {
namespace payoff {
namespace european {

/**
 * The value of the call option at expire, is max(S-K, 0).
 */
class EuropeanCall : public mcgpu::payoff::european::European {
  public:
    EuropeanCall(float K = 50.0F);
    EuropeanCall(const EuropeanCall &europeanCall) = delete;
    EuropeanCall(EuropeanCall &&europeanCall) = delete;
    EuropeanCall &operator=(const EuropeanCall &europeanCall) = delete;
    EuropeanCall &operator=(EuropeanCall &&europeanCall) = delete;

    virtual ~EuropeanCall();

  private:
    EuropeanCallApplyArgs applyArgs;
};
}
}
}

#endif
