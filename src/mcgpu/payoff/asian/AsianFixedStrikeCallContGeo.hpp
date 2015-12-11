/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 07:21:31 PM CET
 */

#ifndef MCGPU_PAYOFF_ASIAN_ASIANFIXEDSTRIKECALLCONTGEO_HPP
#define MCGPU_PAYOFF_ASIAN_ASIANFIXEDSTRIKECALLCONTGEO_HPP

#include "mcgpu/payoff/asian/Asian.hpp"

namespace mcgpu {
namespace payoff {
namespace asian {

class AsianFixedStrikeCallContGeo : public Asian {
  public:
    AsianFixedStrikeCallContGeo(float K);
    virtual ~AsianFixedStrikeCallContGeo();

  private:
    /// strike price
    float K;
};
}
}
}

#endif
