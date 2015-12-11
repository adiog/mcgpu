/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Mon 26 Mar 2012 05:06:14 PM CET
 *   modified: Fri 11 Dec 2015 05:37:00 PM CET
 */

#include <iostream>
#include <cmath>

namespace mcgpu {
namespace host {

float distrib(float f) {
    // numerical approximation of normal gaussian distribution function
    const double A1 = 0.31938153;
    const double A2 = -0.356563782;
    const double A3 = 1.781477937;
    const double A4 = -1.821255978;
    const double A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(f));

    double cnd = RSQRT2PI * exp(-0.5 * f * f) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (f > 0) cnd = 1.0 - cnd;

    return cnd;
};

float european_call_price(float S0, float r, float sigma, float T, float K) {
    float d1 =
        (log(S0 / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt(T));
    float d2 =
        (log(S0 / K) + (r - sigma * sigma / 2.0) * T) / (sigma * sqrt(T));

    return S0 * distrib(d1) - K * exp(-r * T) * distrib(d2);
}

float european_put_price(float S0, float r, float sigma, float T, float K) {
    float d1 =
        (log(S0 / K) + (r + sigma * sigma / 2.0) * T) / (sigma * sqrt(T));
    float d2 =
        (log(S0 / K) + (r - sigma * sigma / 2.0) * T) / (sigma * sqrt(T));

    return K * exp(-r * T) * (1.0 - distrib(d2)) - S0 * (1.0 - distrib(d1));
}

void display_confidence_intervals(float mean, float var, int paths,
                                  float level) {
    float ts = 1.960;
    switch ((int)(level * 10)) {
        case 900:
            ts = 1.645;
            break;
        case 950:
            ts = 1.960;
            break;
        case 980:
            ts = 2.326;
            break;
        case 990:
            ts = 2.576;
            break;
        case 995:
            ts = 2.807;
            break;
        case 999:
            ts = 3.291;
            break;
        default:
            return;
    }
    std::cout << "Confidence interval " << level << "%: ["
              << (mean - ts * var / sqrt(paths)) << ", "
              << (mean + ts * var / sqrt(paths)) << "]" << std::endl;
}
}
}
