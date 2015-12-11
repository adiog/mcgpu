/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 05:23:24 AM CET
 */

#include <cstdlib>
#include <sys/time.h>
#include <cmath>

namespace mcgpu {
namespace host {

const float PI = 3.14159265358979323846264338327950f;

void init_rand() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    srand((unsigned int)tv.tv_usec);
}

float cpu_uniform() { return ((float)rand()) / ((float)RAND_MAX); }

float rand_normal() {
    return (sqrt(-2 * log(cpu_uniform())) * sin(2 * PI * cpu_uniform()));
}
}
}
