/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 05:24:39 AM CET
 */

#ifndef MCGPU_HOST_THEORY_HPP
#define MCGPU_HOST_THEORY_HPP

namespace mcgpu {
namespace host {

float distrib(float f);

float european_call_price(float S0, float r, float sigma, float T, float K);

float european_put_price(float S0, float r, float sigma, float T, float K);

void display_confidence_intervals(float mean, float var, int paths,
                                  float level);
}
}

#endif
