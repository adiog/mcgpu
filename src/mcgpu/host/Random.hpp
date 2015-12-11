/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 05:24:08 AM CET
 */

#ifndef MCGPU_HOST_RANDOM_HPP
#define MCGPU_HOST_RANDOM_HPP

namespace mcgpu {
namespace host {

extern const float PI;

void init_rand(void);
float cpu_uniform(void);
float rand_normal(void);
}
}

#endif
