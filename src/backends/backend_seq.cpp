#include "nbody_common.hpp"
#include "integrator.hpp"

// run_sim_seq
// Arguments:
// - bodies: initial bodies; modified in-place during simulation
// - cfg: simulation config (N, dt, steps)
// - constants: physical constants (G, epsilon2)
// Behavior:
// - Runs sequential Velocity Verlet for cfg.steps
// - Returns true if completed all steps without detecting non-finite values
bool run_sim_seq(Bodies& bodies, const SimConfig& cfg, const SimConstants& constants) {
    for (std::size_t s = 0; s < cfg.steps; ++s) {
        step_velocity_verlet_seq(bodies, cfg.dt, constants);
        // Safety: check finiteness
        for (std::size_t i = 0; i < bodies.x.size(); ++i) {
            if (!is_finite(bodies.x[i]) || !is_finite(bodies.y[i]) || !is_finite(bodies.z[i]) ||
                !is_finite(bodies.vx[i]) || !is_finite(bodies.vy[i]) || !is_finite(bodies.vz[i])) {
                return false;
            }
        }
    }
    return true;
}
