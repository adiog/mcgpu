/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:45:31 PM CET
 *   modified: Fri 11 Dec 2015 05:41:47 PM CET
 */

#ifndef MCGPU_PAYOFF_EUROPEAN_EUROPEAN_HPP
#define MCGPU_PAYOFF_EUROPEAN_EUROPEAN_HPP

#include "mcgpu/payoff/Payoff.hpp"

typedef float (*gpu_euro_apply)(float, float, void*);

namespace mcgpu {
namespace payoff {
namespace european {

class European : public mcgpu::payoff::Payoff {
  public:
    European(){};
    virtual ~European(){};

    virtual gpu_euro_apply get_apply() const { return gpu_apply; }
    virtual void* get_apply_args() const { return gpu_apply_args; }

  protected:
    gpu_euro_apply gpu_apply;
    void* gpu_apply_args;
};
}
}
}

#endif
