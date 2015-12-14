/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:01 PM CET
 *   modified: Mon Apr  2 20:40:10 2012
 */

#ifndef MCGPU_PAYOFF_ASIAN_BARRIERUPANDOUTCALL_HPP
#define MCGPU_PAYOFF_ASIAN_BARRIERUPANDOUTCALL_HPP

#include "mcgpu/payoff/asian/Asian.hpp"

struct BarrierUpAndOutCallApplyArgs {
    /// strike price
    float K;
};

struct BarrierUpAndOutCallFoldArgs {
    /// barrier price
    float B;
};

namespace mcgpu {
namespace payoff {
namespace asian {

class BarrierUpAndOutCall : public mcgpu::payoff::asian::Asian {
  public:
    BarrierUpAndOutCall(float K = 55.0F, float B = 57.0F);
    BarrierUpAndOutCall(const BarrierUpAndOutCall &barrierUpAndOutCall) =
        delete;
    BarrierUpAndOutCall(BarrierUpAndOutCall &&barrierUpAndOutCall) = delete;
    BarrierUpAndOutCall &operator=(
        const BarrierUpAndOutCall &barrierUpAndOutCall) = delete;
    BarrierUpAndOutCall &operator=(BarrierUpAndOutCall &&barrierUpAndOutCall) =
        delete;
    virtual ~BarrierUpAndOutCall();

  private:
    BarrierUpAndOutCallApplyArgs applyArgs;
    BarrierUpAndOutCallFoldArgs foldArgs;
};
}
}
}

#endif
