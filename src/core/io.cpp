#include "io.hpp"
#include <fstream>
#include <sstream>
#include <random>

bool load_bodies_from_csv(const std::string& path, Bodies& out) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;

    std::vector<nb_float> x, y, z, vx, vy, vz, m;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string item;
        std::vector<nb_float> vals;
        while (std::getline(ss, item, ',')) {
            try {
                nb_float v = static_cast<nb_float>(std::stod(item));
                if (!is_finite(v)) return false;
                vals.push_back(v);
            } catch (...) {
                return false;
            }
        }
        if (vals.size() != 7) return false; // x,y,z,vx,vy,vz,m
        x.push_back(vals[0]); y.push_back(vals[1]); z.push_back(vals[2]);
        vx.push_back(vals[3]); vy.push_back(vals[4]); vz.push_back(vals[5]);
        m.push_back(vals[6]);
    }

    std::size_t N = x.size();
    out.x = std::move(x); out.y = std::move(y); out.z = std::move(z);
    out.vx = std::move(vx); out.vy = std::move(vy); out.vz = std::move(vz);
    out.m = std::move(m);
    out.ax.assign(N, nb_float{0});
    out.ay.assign(N, nb_float{0});
    out.az.assign(N, nb_float{0});
    return true;
}

void generate_random_bodies(std::size_t N, nb_float mass, nb_float scale, Bodies& out) {
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-static_cast<double>(scale), static_cast<double>(scale));
    out.x.resize(N); out.y.resize(N); out.z.resize(N);
    out.vx.resize(N); out.vy.resize(N); out.vz.resize(N);
    out.ax.assign(N, nb_float{0}); out.ay.assign(N, nb_float{0}); out.az.assign(N, nb_float{0});
    out.m.resize(N);
    for (std::size_t i = 0; i < N; ++i) {
        out.x[i] = static_cast<nb_float>(dist(rng));
        out.y[i] = static_cast<nb_float>(dist(rng));
        out.z[i] = static_cast<nb_float>(dist(rng));
        out.vx[i] = nb_float{0};
        out.vy[i] = nb_float{0};
        out.vz[i] = nb_float{0};
        out.m[i] = mass;
    }
}
