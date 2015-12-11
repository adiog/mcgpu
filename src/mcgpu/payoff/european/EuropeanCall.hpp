/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Fri 11 Dec 2015 05:42:10 PM CET
 */

#ifndef MCGPU_PAYOFF_EUROPEAN_EUROPEANCALL_HPP
#define MCGPU_PAYOFF_EUROPEAN_EUROPEANCALL_HPP

#include "mcgpu/payoff/european/European.hpp"

namespace mcgpu {
namespace payoff {
namespace european {

/**
 * The value of the call option at expire, is max(S-K, 0).
 */
class EuropeanCall : public mcgpu::payoff::european::European {
  public:
    EuropeanCall(float K = 50.0F);
    virtual ~EuropeanCall();

  private:
    /// strike price
    float K;
};
}
}
}

#endif
