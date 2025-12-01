#include "nbody_common.hpp"
#include "io.hpp"
#include "integrator.hpp"
#include <iostream>
#include <cstring>
#include <chrono>

// parse_args
// Arguments:
// - argc, argv: process command line arguments
// Behavior:
// - Supports: --backend, --threads, --block-size, --dt, --steps, --input, --N
// - Returns SimConfig populated with provided values
static SimConfig parse_args(int argc, char** argv) {
    SimConfig cfg;
    for (int i = 1; i < argc; ++i) {
        auto match = [&](const char* flag){ return std::strcmp(argv[i], flag) == 0; };
        if (match("--backend") && i+1 < argc) cfg.backend = argv[++i];
        else if (match("--threads") && i+1 < argc) cfg.threads = std::atoi(argv[++i]);
        else if (match("--block-size") && i+1 < argc) cfg.blockSize = std::atoi(argv[++i]);
        else if (match("--dt") && i+1 < argc) cfg.dt = static_cast<nb_float>(std::atof(argv[++i]));
        else if (match("--steps") && i+1 < argc) cfg.steps = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
        else if (match("--input") && i+1 < argc) cfg.inputPath = argv[++i];
        else if (match("--N") && i+1 < argc) cfg.N = static_cast<std::size_t>(std::strtoull(argv[++i], nullptr, 10));
    }
    return cfg;
}

// Logger callback used by timing utilities (future extension).
static void log_time(const std::string& name, double ms) {
    std::cerr << "[TIMER] " << name << ": " << ms << " ms\n";
}

// forward declaration from sequential backend
bool run_sim_seq(Bodies& bodies, const SimConfig& cfg, const SimConstants& constants);
// forward declaration from OpenMP backend
bool run_sim_omp(Bodies& bodies, const SimConfig& cfg, const SimConstants& constants);
#if defined(ENABLE_CUDA)
extern "C" bool run_sim_cuda(Bodies& bodies, const SimConfig& cfg, const SimConstants& constants);
#endif

int main(int argc, char** argv) {
    SimConfig cfg = parse_args(argc, argv);
    if (cfg.dt <= 0 || cfg.steps == 0) {
        std::cerr << "Error: --dt must be > 0 and --steps > 0\n";
        return 1;
    }

    Bodies bodies;
    if (!cfg.inputPath.empty()) {
        if (!load_bodies_from_csv(cfg.inputPath, bodies)) {
            std::cerr << "Error: failed to load input file: " << cfg.inputPath << "\n";
            return 1;
        }
    } else if (cfg.N > 0) {
        generate_random_bodies(cfg.N, nb_float{1.0}, nb_float{1.0}, bodies);
    } else {
        std::cerr << "Error: provide --input <file> or --N <count>\n";
        return 1;
    }

    SimConstants constants{ nb_float(6.67430e-11), nb_float(1e-9) }; // default units; epsilon^2 small

    const auto start = std::chrono::steady_clock::now();
    bool ok = false;
    if (cfg.backend == "seq") {
        ok = run_sim_seq(bodies, cfg, constants);
    } else if (cfg.backend == "omp") {
        ok = run_sim_omp(bodies, cfg, constants);
    } else if (cfg.backend == "cuda") {
#if defined(ENABLE_CUDA)
        ok = run_sim_cuda(bodies, cfg, constants);
#else
        std::cerr << "CUDA backend disabled or not available. Reconfigure with -DENABLE_CUDA=ON and ensure nvcc is installed.\n";
        return 3;
#endif
    } else {
        std::cerr << "Unknown backend: " << cfg.backend << "\n";
        return 4;
    }
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (!ok) {
        std::cerr << "Simulation aborted due to non-finite values.\n";
        return 5;
    }

    // Minimal result output: final position of body 0 and timing
    if (!bodies.x.empty()) {
        std::cout << "Final body[0]: x=" << bodies.x[0]
                  << ", y=" << bodies.y[0]
                  << ", z=" << bodies.z[0] << "\n";
    }
    std::cout << "Elapsed: " << ms << " ms for " << cfg.steps << " steps, N=" << bodies.x.size() << "\n";
    // CSV timing line for benchmarking: backend,N,threads,dt,steps,time_ms
    std::cout << "RESULT," << cfg.backend << "," << bodies.x.size() << "," << cfg.threads
              << "," << cfg.dt << "," << cfg.steps << "," << ms << "\n";
    return 0;
}
