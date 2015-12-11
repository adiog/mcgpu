/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Thu 22 Mar 2012 01:17:43 PM CET
 *   modified: Fri 11 Dec 2015 07:21:46 PM CET
 */

#ifndef MCGPU_PAYOFF_ASIAN_ASIAN_HPP
#define MCGPU_PAYOFF_ASIAN_ASIAN_HPP

#include "mcgpu/payoff/Payoff.hpp"

typedef float (*gpu_asian_apply)(float, float, float, void*);
typedef float (*gpu_asian_fold)(float, float, float, float, void*);

namespace mcgpu {
namespace payoff {
namespace asian {

class Asian : public mcgpu::payoff::Payoff {
  public:
    Asian(){};
    virtual ~Asian(){};
    bool isAsian();

    gpu_asian_apply get_apply() const { return gpu_apply; }
    gpu_asian_fold get_fold() const { return gpu_fold; }

    void* get_apply_args() const { return gpu_apply_args; }
    void* get_fold_args() const { return gpu_fold_args; }

    float get_init_acc() const { return init_acc; };

  protected:
    gpu_asian_apply gpu_apply;
    gpu_asian_fold gpu_fold;

    void* gpu_apply_args;
    void* gpu_fold_args;

    float init_acc;
};
}
}
}

#endif
