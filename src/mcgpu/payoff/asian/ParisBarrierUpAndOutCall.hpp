/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Mon Apr  2 20:39:41 2012
 */

#ifndef MCGPU_PAYOFF_ASIAN_PARISBARRIERUPANDOUTCALL_HPP
#define MCGPU_PAYOFF_ASIAN_PARISBARRIERUPANDOUTCALL_HPP

#include "mcgpu/payoff/asian/Asian.hpp"

struct ParisBarrierUpAndOutCallApplyArgs {
    /// strike price
    float K;
};

struct ParisBarrierUpAndOutCallFoldArgs {
    /// barrier price
    float B;
    /// time range needed to activate Paris Barrier
    float timein;
};

namespace mcgpu {
namespace payoff {
namespace asian {
/**
 * Continuous Parisian Barrier Up-and-Out Call
 */
class ParisBarrierUpAndOutCall : public mcgpu::payoff::asian::Asian {
  public:
    ParisBarrierUpAndOutCall(float K = 55.0F, float B = 57.0F,
                             float timein = 0.5F);
    ParisBarrierUpAndOutCall(
        const ParisBarrierUpAndOutCall &parisBarrierUpAndOutCall) = delete;
    ParisBarrierUpAndOutCall(
        ParisBarrierUpAndOutCall &&parisBarrierUpAndOutCall) = delete;
    ParisBarrierUpAndOutCall &operator=(
        const ParisBarrierUpAndOutCall &parisBarrierUpAndOutCall) = delete;
    ParisBarrierUpAndOutCall &operator=(
        ParisBarrierUpAndOutCall &&parisBarrierUpAndOutCall) = delete;
    virtual ~ParisBarrierUpAndOutCall();

  private:
    ParisBarrierUpAndOutCallApplyArgs applyArgs;
    ParisBarrierUpAndOutCallFoldArgs foldArgs;
};
}
}
}
#endif
