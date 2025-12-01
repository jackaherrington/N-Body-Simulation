#include "nbody_common.hpp"
#include "integrator.hpp"
#ifdef ENABLE_OMP
#include <omp.h>
#endif
#include <algorithm>

// compute_accelerations_omp
// Arguments:
// - bodies: input/output Bodies; writes accelerations (ax, ay, az)
// - constants: simulation constants (G, epsilon2)
// - threads: number of OpenMP threads to use (if OpenMP enabled)
// Behavior:
// - Parallelizes outer loop over i with dynamic scheduling for better load balancing
// - Uses per-thread private accumulators; writes ax[i], ay[i], az[i]
// - Optimized for cache locality and reduced overhead
static void compute_accelerations_omp(Bodies& b, const SimConstants& c, int threads) {
    const std::size_t N = b.x.size();
    std::fill(b.ax.begin(), b.ax.end(), nb_float{0});
    std::fill(b.ay.begin(), b.ay.end(), nb_float{0});
    std::fill(b.az.begin(), b.az.end(), nb_float{0});
#ifdef ENABLE_OMP
    if (threads > 0) omp_set_num_threads(threads);
    
    // Choose scheduling based on problem size
    if (N < 1000) {
        // For small N, static scheduling with smaller chunks
#pragma omp parallel for schedule(static, 16)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
            nb_float aix = 0, aiy = 0, aiz = 0;
            const nb_float xi = b.x[i], yi = b.y[i], zi = b.z[i];
            
            // Manual loop unrolling hint and vectorization
#pragma GCC ivdep
            for (std::size_t j = 0; j < N; ++j) {
                if (static_cast<std::size_t>(i) == j) continue;
                const nb_float rx = b.x[j] - xi;
                const nb_float ry = b.y[j] - yi;
                const nb_float rz = b.z[j] - zi;
                const nb_float r2 = rx*rx + ry*ry + rz*rz + c.epsilon2;
                const nb_float inv_r = nb_float(1.0) / std::sqrt(static_cast<double>(r2));
                const nb_float inv_r3 = inv_r * inv_r * inv_r;
                const nb_float s = c.G * b.m[j] * inv_r3;
                aix += s * rx;
                aiy += s * ry;
                aiz += s * rz;
            }
            b.ax[i] = aix;
            b.ay[i] = aiy;
            b.az[i] = aiz;
        }
    } else {
        // For large N, dynamic scheduling for better load balancing
#pragma omp parallel for schedule(dynamic, 32)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
            nb_float aix = 0, aiy = 0, aiz = 0;
            const nb_float xi = b.x[i], yi = b.y[i], zi = b.z[i];
            
            // Prefetch next iteration data for better cache performance
            if (i + 1 < static_cast<std::ptrdiff_t>(N)) {
                __builtin_prefetch(&b.x[i + 1], 0, 3);
                __builtin_prefetch(&b.y[i + 1], 0, 3);
                __builtin_prefetch(&b.z[i + 1], 0, 3);
            }
            
            for (std::size_t j = 0; j < N; ++j) {
                if (static_cast<std::size_t>(i) == j) continue;
                const nb_float rx = b.x[j] - xi;
                const nb_float ry = b.y[j] - yi;
                const nb_float rz = b.z[j] - zi;
                const nb_float r2 = rx*rx + ry*ry + rz*rz + c.epsilon2;
                const nb_float inv_r = nb_float(1.0) / std::sqrt(static_cast<double>(r2));
                const nb_float inv_r3 = inv_r * inv_r * inv_r;
                const nb_float s = c.G * b.m[j] * inv_r3;
                aix += s * rx;
                aiy += s * ry;
                aiz += s * rz;
            }
            b.ax[i] = aix;
            b.ay[i] = aiy;
            b.az[i] = aiz;
        }
    }
#else
    // Fallback sequential implementation
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
        nb_float aix = 0, aiy = 0, aiz = 0;
        const nb_float xi = b.x[i], yi = b.y[i], zi = b.z[i];
        for (std::size_t j = 0; j < N; ++j) {
            if (static_cast<std::size_t>(i) == j) continue;
            const nb_float rx = b.x[j] - xi;
            const nb_float ry = b.y[j] - yi;
            const nb_float rz = b.z[j] - zi;
            const nb_float r2 = rx*rx + ry*ry + rz*rz + c.epsilon2;
            const nb_float inv_r = nb_float(1.0) / std::sqrt(static_cast<double>(r2));
            const nb_float inv_r3 = inv_r * inv_r * inv_r;
            const nb_float s = c.G * b.m[j] * inv_r3;
            aix += s * rx;
            aiy += s * ry;
            aiz += s * rz;
        }
        b.ax[i] = aix;
        b.ay[i] = aiy;
        b.az[i] = aiz;
    }
#endif
}

// step_velocity_verlet_omp
// Same behavior as sequential version but uses OMP acceleration kernel
// Optimized with parallel position and velocity updates
static void step_velocity_verlet_omp(Bodies& b, nb_float dt, const SimConstants& c, int threads) {
    const std::size_t N = b.x.size();
    compute_accelerations_omp(b, c, threads);
    const nb_float half_dt2 = nb_float(0.5) * dt * dt;
    
    // Parallelize position update
#ifdef ENABLE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
        b.x[i] += b.vx[i] * dt + b.ax[i] * half_dt2;
        b.y[i] += b.vy[i] * dt + b.ay[i] * half_dt2;
        b.z[i] += b.vz[i] * dt + b.az[i] * half_dt2;
    }
    
    std::vector<nb_float> ax_old = b.ax, ay_old = b.ay, az_old = b.az;
    compute_accelerations_omp(b, c, threads);
    const nb_float half_dt = nb_float(0.5) * dt;
    
    // Parallelize velocity update  
#ifdef ENABLE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
        b.vx[i] += (ax_old[i] + b.ax[i]) * half_dt;
        b.vy[i] += (ay_old[i] + b.ay[i]) * half_dt;
        b.vz[i] += (az_old[i] + b.az[i]) * half_dt;
    }
}

// run_sim_omp
// Arguments:
// - bodies, cfg, constants: same as sequential
// Behavior:
// - Uses OpenMP acceleration; returns false if non-finite values detected
// - Optimized with parallel finite value checking
bool run_sim_omp(Bodies& bodies, const SimConfig& cfg, const SimConstants& constants) {
    for (std::size_t s = 0; s < cfg.steps; ++s) {
        step_velocity_verlet_omp(bodies, cfg.dt, constants, cfg.threads);
        
        // Parallel finite check with early exit
        bool all_finite = true;
#ifdef ENABLE_OMP
#pragma omp parallel for schedule(static) reduction(&&:all_finite)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(bodies.x.size()); ++i) {
            if (!is_finite(bodies.x[i]) || !is_finite(bodies.y[i]) || !is_finite(bodies.z[i]) ||
                !is_finite(bodies.vx[i]) || !is_finite(bodies.vy[i]) || !is_finite(bodies.vz[i])) {
                all_finite = false;
            }
        }
        
        if (!all_finite) return false;
    }
    return true;
}