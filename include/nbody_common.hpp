#pragma once

#include <vector>
#include <cstddef>
#include <string>
#include <cmath>

// Precision selection: double by default; float if NB_FLOAT_PRECISION defined.
#ifdef NB_FLOAT_PRECISION
using nb_float = float;
#else
using nb_float = double;
#endif

// Gravitational constant and softening parameter.
// G: Newtonian gravitational constant (scaled units if desired).
// epsilon2: softening term squared to avoid singularities when r -> 0.
struct SimConstants {
    nb_float G;
    nb_float epsilon2;
};

// Structure-of-Arrays (SoA) layout for better cache and GPU coalescing.
// Arrays hold per-body positions, velocities, accelerations, and masses.
struct Bodies {
    std::vector<nb_float> x, y, z;
    std::vector<nb_float> vx, vy, vz;
    std::vector<nb_float> ax, ay, az;
    std::vector<nb_float> m;
};

// Simulation configuration provided via CLI.
// N: number of bodies; dt: time step; steps: number of iterations to run.
// backend: selected compute backend ("seq", "omp", "cuda").
// threads: requested CPU threads for OpenMP (if enabled).
// blockSize: CUDA block dimension (if enabled).
// inputPath: file path to initial conditions.
struct SimConfig {
    std::size_t N{0};
    nb_float dt{0.0};
    std::size_t steps{0};
    std::string backend{"seq"};
    int threads{1};
    int blockSize{256};
    std::string inputPath;
};

// Simple invariants and safety checks utilities.
inline bool is_finite(nb_float v) {
    return std::isfinite(static_cast<double>(v));
}
