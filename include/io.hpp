#pragma once

#include "nbody_common.hpp"
#include <string>

// load_bodies_from_csv
// Arguments:
// - path: input CSV file path (columns: x,y,z,vx,vy,vz,m)
// - out: Bodies to populate; out vectors are resized to match file rows
// Behavior:
// - Validates column count and numeric finiteness
// - Returns true on success, false on failure
bool load_bodies_from_csv(const std::string& path, Bodies& out);

// generate_random_bodies
// Arguments:
// - N: number of bodies to generate
// - mass: mass for each body (or use simple distribution)
// - scale: positional range scale (positions uniform in [-scale, scale])
// - out: Bodies to populate with random initial conditions, zero accel
// Behavior:
// - Deterministic with fixed seed for reproducibility
void generate_random_bodies(std::size_t N, nb_float mass, nb_float scale, Bodies& out);
