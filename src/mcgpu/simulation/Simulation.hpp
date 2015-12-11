/*
 * Copyright 2012 Aleksander Gajewski <adiog@brainfuck.pl>
 *   created:  Tue 27 Mar 2012 02:36:28 PM CET
 *   modified: Fri 11 Dec 2015 05:40:16 PM CET
 */

#ifndef MCGPU_SIMULATION_SIMULATION_HPP
#define MCGPU_SIMULATION_SIMULATION_HPP

#include <cstdint>
#include "mcgpu/payoff/Payoff.hpp"

namespace mcgpu {
namespace simulation {

enum class NumericalScheme : uint8_t { EulerMaruyama, Milstein };

class Simulation {
  public:
    Simulation(int paths = 100000, int points = 100);
    Simulation(const Simulation &simulation) = delete;
    Simulation(Simulation &&simulation) = delete;
    Simulation &operator=(const Simulation &simulation) = delete;
    Simulation &operator=(Simulation &&simulation) = delete;
    ~Simulation();

    float *get_gpu_array() const { return gpu_array; }
    unsigned int *get_gpu_seeds() const { return gpu_seeds; }
    int get_blocks() const { return blocks; }
    int get_threads() const { return threads; }
    int get_points() const { return points; }

    std::pair<float, float> finish();

  private:
    void init_seeds(unsigned int *mem, int n);

  private:
    /// main simulation parameter - number of blocks
    int blocks;
    /// main simulation parameter - number of threads per block
    int threads;

    /// number of paths
    int paths;
    /// number of point per trajectory
    int points;

    /// pointer to results array (on device)
    float *gpu_array;

    /// pointer to initial seeds (on device)
    unsigned int *gpu_seeds;

    /// pointer to initial seeds (on host)
    unsigned int *cpu_seeds;
};
}
}

#endif
