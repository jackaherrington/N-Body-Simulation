#pragma once

#include "nbody_common.hpp"

// compute_accelerations_seq
// Arguments:
// - bodies: input/output Bodies; writes accelerations (ax, ay, az)
// - constants: simulation constants (G, epsilon2)
// Behavior:
// - Brute-force O(N^2) pairwise gravitational acceleration using softening
// - Safe for finite values; skips self-interaction
void compute_accelerations_seq(Bodies& bodies, const SimConstants& constants);

// step_velocity_verlet_seq
// Arguments:
// - bodies: input/output Bodies (positions, velocities updated)
// - dt: time step
// - constants: simulation constants (used to recompute accelerations)
// Behavior:
// - Velocity Verlet integrator: x_{t+dt} = x_t + v_t*dt + 0.5*a_t*dt^2
//   v_{t+dt} = v_t + 0.5*(a_t + a_{t+dt})*dt
// - Calls compute_accelerations_seq twice: before and after position update
void step_velocity_verlet_seq(Bodies& bodies, nb_float dt, const SimConstants& constants);
