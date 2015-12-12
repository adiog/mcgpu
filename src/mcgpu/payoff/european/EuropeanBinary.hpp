/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Sat 12 Dec 2015 09:15:46 PM CET
 */

#ifndef MCGPU_PAYOFF_EUROPEAN_EUROPEANBINARY_HPP
#define MCGPU_PAYOFF_EUROPEAN_EUROPEANBINARY_HPP

#include "mcgpu/payoff/european/European.hpp"

struct EuropeanBinaryApplyArgs {
    /// strike price
    float K;
    /// value of cash unit
    float pay;
    /// indicates type of activation
    bool activateAbove;
};

namespace mcgpu {
namespace payoff {
namespace european {
/**
 * The binary cash or nothing option at expiry pays out one unit of cash if the
 * spot is above or below the strike at maturity.
 */
class EuropeanBinary : public European {
  public:
    EuropeanBinary(float K = 50.0F, float pay = 10.0F, bool activateAbove = true);
    EuropeanBinary(const EuropeanBinary &europeanBinary) = delete;
    EuropeanBinary(EuropeanBinary &&europeanBinary) = delete;
    EuropeanBinary &operator=(const EuropeanBinary &europeanBinary) = delete;
    EuropeanBinary &operator=(EuropeanBinary &&europeanBinary) = delete;
    virtual ~EuropeanBinary();

  private:
    EuropeanBinaryApplyArgs applyArgs;
};
}
}
}

#endif
