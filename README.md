# Parallel Gravitational N-Body Simulation

## Overview
Complete N-body gravitational simulation with three high-performance backends:
- **Sequential**: Single-threaded CPU baseline implementation
- **OpenMP**: Multi-threaded CPU parallelization with thread scaling
- **CUDA**: GPU-accelerated computation with shared memory optimization

Features:
- Shared core (SoA data layout, CSV I/O, Velocity Verlet integrator)  
- Deterministic dataset generator for reproducible benchmarks
- Comprehensive benchmarking suite with visualization
- Production-ready CMake build system with conditional CUDA support

## Quick Start

### Build
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)
```

### Run Benchmarks
```bash
cd benchmarks
./run_benchmarks.sh    # Comprehensive performance analysis
```

### Individual Tests
```bash
# GPU test (100,000 particles)
./build/src/nbody_app --backend cuda --N 100000 --dt 0.01 --steps 3

# CPU comparison
./build/src/nbody_app --backend omp --N 100000 --dt 0.01 --steps 3
```

## Performance Results

| Problem Size | Sequential | OpenMP | CUDA | CUDA Speedup |
|-------------|------------|---------|------|-------------|
| 100         | 0.39ms     | 0.36ms  | 2,085ms | **0.00x** |
| 1,000       | 34ms       | 34ms    | 159ms | **0.22x** |
| 10,000      | 3,195ms    | 3,262ms | 2,787ms | **1.15x** 🎯 |
| 100,000     | 203s       | 200s    | 14s | **14.36x** 🚀 |

**Key Insights:**
- 🎯 **GPU crossover**: 10,000 particles
- 🚀 **Best performance**: 14x speedup at 100,000 particles
- 📈 **Scaling**: CUDA performance scales better than CPU for large problems

## Usage Examples

### Random Dataset Generation
```bash
./build/src/nbody_app --backend cuda --N 10000 --dt 0.01 --steps 5
```

All tests use deterministic random generation (seed=42) ensuring identical particle configurations across benchmark runs for fair performance comparison.

### With Custom Parameters
```bash
./build/src/nbody_app --backend omp --N 5000 --dt 0.001 --steps 20
```

## Project Structure
```
├── src/           # Source code (all backends)
├── include/       # Headers  
├── benchmarks/    # Performance testing suite
└── build/         # Build artifacts
```

## Benchmarking Suite

The `benchmarks/` directory contains comprehensive performance analysis tools:
- **Automated testing** across problem sizes and thread counts
- **Dual visualization**: Performance overview + detailed scaling analysis  
- **Comparative analysis** with speedup calculations and efficiency metrics

See `benchmarks/README.md` for detailed usage.

## Technical Notes
- **Physics**: Gravitational N-body with softening parameter (ε² = 1e-9)
- **Integration**: Velocity Verlet time stepping (symplectic)
- **Data Generation**: Deterministic random bodies (fixed seed=42) for reproducible benchmarks
- **Memory**: Structure-of-Arrays (SoA) layout for vectorization
- **GPU**: Shared memory tiling with coalesced access patterns
- **Precision**: Configurable float/double via `nb_float` typedef
