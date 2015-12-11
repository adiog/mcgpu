/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Sat 24 Mar 2012 08:54:47 PM CET
 *   modified: Fri 11 Dec 2015 05:40:49 PM CET
 */

#ifndef MCGPU_PAYOFF_PAYOFF_HPP
#define MCGPU_PAYOFF_PAYOFF_HPP

namespace mcgpu {
namespace payoff {

class Payoff {
  public:
    Payoff() = default;
    Payoff(const Payoff &p) = delete;
    Payoff(Payoff &&p) = delete;
    Payoff &operator=(const Payoff &p) = delete;
    Payoff &operator=(Payoff &&p) = delete;
    virtual ~Payoff() = default;
};
}
}

#endif
