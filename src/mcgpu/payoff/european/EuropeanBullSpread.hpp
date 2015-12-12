/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Sat 12 Dec 2015 09:16:12 PM CET
 */

#ifndef MCGPU_PAYOFF_EUROPEAN_EUROPEANBULLSPREAD_HPP
#define MCGPU_PAYOFF_EUROPEAN_EUROPEANBULLSPREAD_HPP

#include "mcgpu/payoff/european/European.hpp"

struct EuropeanBullSpreadApplyArgs {
    /// strike price (lower)
    float K1;
    /// strike price (upper)
    float K2;
};

namespace mcgpu {
namespace payoff {
namespace european {

/**
 * The value of the bull option at expiry, is [max(S-K1, 0) - max(S-K2, 0)].
 */
class EuropeanBullSpread : public European {
  public:
    EuropeanBullSpread(float K1 = 50.0F, float K2 = 60.0F);
    EuropeanBullSpread(const EuropeanBullSpread &europeanBullSpread) = delete;
    EuropeanBullSpread(EuropeanBullSpread &&europeanBullSpread) = delete;
    EuropeanBullSpread &operator=(
        const EuropeanBullSpread &europeanBullSpread) = delete;
    EuropeanBullSpread &operator=(EuropeanBullSpread &&europeanBullSpread) =
        delete;
    virtual ~EuropeanBullSpread();

  private:
    EuropeanBullSpreadApplyArgs applyArgs;
};
}
}
}

#endif
