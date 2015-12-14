/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 07:21:31 PM CET
 */

#ifndef MCGPU_PAYOFF_ASIAN_AsianCall_HPP
#define MCGPU_PAYOFF_ASIAN_AsianCall_HPP

#include "mcgpu/payoff/asian/Asian.hpp"

struct AsianCallApplyArgs {
    /// strike price
    float K;
};

namespace mcgpu {
namespace payoff {
namespace asian {

/**
 * Asian Fixed Strike Call with Continuous Geometric Average
 */
class AsianCall : public Asian {
  public:
    AsianCall(float K = 50.0F);
    AsianCall(const AsianCall &asianCall) = delete;
    AsianCall(AsianCall &&asianCall) = delete;
    AsianCall &operator=(const AsianCall &asianCall) = delete;
    AsianCall &operator=(AsianCall &&asianCall) = delete;
    virtual ~AsianCall();

  private:
    AsianCallApplyArgs applyArgs;
};
}
}
}

#endif
