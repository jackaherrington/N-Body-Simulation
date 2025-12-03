#include "integrator.hpp"
#include <algorithm>

static inline nb_float inv_sqrt(nb_float x) {
    return nb_float(1.0) / std::sqrt(static_cast<double>(x));
}

void compute_accelerations_seq(Bodies& b, const SimConstants& c) {
    const std::size_t N = b.x.size();
    std::fill(b.ax.begin(), b.ax.end(), nb_float{0});
    std::fill(b.ay.begin(), b.ay.end(), nb_float{0});
    std::fill(b.az.begin(), b.az.end(), nb_float{0});
    for (std::size_t i = 0; i < N; ++i) {
        nb_float aix = 0, aiy = 0, aiz = 0;
        const nb_float xi = b.x[i], yi = b.y[i], zi = b.z[i];
        for (std::size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            const nb_float rx = b.x[j] - xi;
            const nb_float ry = b.y[j] - yi;
            const nb_float rz = b.z[j] - zi;
            const nb_float r2 = rx*rx + ry*ry + rz*rz + c.epsilon2;
            const nb_float inv_r = inv_sqrt(r2);      // 1/sqrt(r2)
            const nb_float inv_r3 = inv_r * inv_r * inv_r; // 1/(r^3)
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

void step_velocity_verlet_seq(Bodies& b, nb_float dt, const SimConstants& c) {
    const std::size_t N = b.x.size();
    // Accel at t
    compute_accelerations_seq(b, c);
    // Position update: x_{t+dt}
    const nb_float half_dt2 = nb_float(0.5) * dt * dt;
    for (std::size_t i = 0; i < N; ++i) {
        b.x[i] += b.vx[i] * dt + b.ax[i] * half_dt2;
        b.y[i] += b.vy[i] * dt + b.ay[i] * half_dt2;
        b.z[i] += b.vz[i] * dt + b.az[i] * half_dt2;
    }
    // Accel at t+dt
    std::vector<nb_float> ax_old = b.ax, ay_old = b.ay, az_old = b.az;
    compute_accelerations_seq(b, c);
    // Velocity update
    const nb_float half_dt = nb_float(0.5) * dt;
    for (std::size_t i = 0; i < N; ++i) {
        b.vx[i] += (ax_old[i] + b.ax[i]) * half_dt;
        b.vy[i] += (ay_old[i] + b.ay[i]) * half_dt;
        b.vz[i] += (az_old[i] + b.az[i]) * half_dt;
    }
}
