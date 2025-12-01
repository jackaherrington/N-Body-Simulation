# N-Body Gravitational Simulation

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/jackaherrington/N-Body-Simulation)
[![CUDA](https://img.shields.io/badge/CUDA-11.5+-blue)](https://developer.nvidia.com/cuda-downloads)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5+-green)](https://www.openmp.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

High-performance gravitational N-body simulation with CPU and GPU acceleration. Features sequential, OpenMP multi-threading, and CUDA GPU backends with comprehensive benchmarking suite.

## 🚀 Key Results

| Problem Size | Sequential | OpenMP | CUDA | **GPU Speedup** |
|-------------|------------|---------|------|-----------------|
| 1,000       | 34.40ms    | 34.25ms | 159ms | 0.22x |
| 10,000      | 3,195ms    | 3,262ms | 2,787ms | **1.15x** 🎯 |
| 100,000     | 203,199ms  | 199,666ms | 14,148ms | **14.36x** 🚀 |

**GPU achieves 14.36x speedup at large scale with RTX 3070 Mobile**

## 🎯 Features

- **Three Backends**: Sequential CPU, OpenMP multi-threading, CUDA GPU acceleration
- **Physics Accurate**: Velocity Verlet integrator with gravitational softening
- **Scalable**: 100 to 100,000+ particles with excellent GPU scaling
- **Comprehensive Benchmarks**: Automated testing with performance visualization
- **Reproducible**: Deterministic random generation (seed=42) for consistent results

## ⚡ Quick Start

### Prerequisites
- C++17 compatible compiler (GCC 7+)
- CMake 3.12+
- CUDA Toolkit 11.0+ (for GPU acceleration)
- NVIDIA GPU with compute capability 3.5+ (for GPU acceleration)

### Build
```bash
git clone https://github.com/jackaherrington/N-Body-Simulation.git
cd N-Body-Simulation
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)
```

### Run Benchmarks
```bash
cd ../benchmarks
./run_benchmarks.sh
```

### Individual Tests
```bash
# GPU acceleration (100K particles)
./build/src/nbody_app --backend cuda --N 100000 --dt 0.01 --steps 3

# CPU comparison  
./build/src/nbody_app --backend omp --N 100000 --dt 0.01 --steps 3
```

## 📊 Performance Analysis

The benchmarking suite generates comprehensive performance analysis:

- **performance_analysis.png**: 4-panel performance overview
- **scaling_analysis.png**: Detailed scaling behavior analysis  
- **benchmark_results.csv**: Raw performance data

### GPU Performance Characteristics
- **Crossover point**: ~10,000 particles where GPU becomes faster than CPU
- **Best speedup**: 14.36x at 100,000 particles  
- **Scalability**: Excellent performance scaling with problem size

## 🏗️ Architecture

```
├── src/
│   ├── core/           # Shared physics engine and I/O
│   ├── backends/       # Sequential, OpenMP, and CUDA implementations  
│   └── app/           # Main application with CLI
├── include/           # Public headers
├── benchmarks/        # Automated performance testing suite
└── build/            # Build artifacts
```

### Backend Implementations

**Sequential**: Single-threaded baseline with optimized memory layout  
**OpenMP**: Multi-threaded CPU parallelization across particles  
**CUDA**: GPU acceleration with shared memory tiling and coalesced access

## 🔬 Technical Details

- **Integration**: Velocity Verlet (symplectic, 2nd-order accurate)
- **Memory Layout**: Structure-of-Arrays for vectorization
- **GPU Optimization**: Shared memory tiling with 256-thread blocks
- **Softening**: ε² = 1×10⁻⁹ prevents numerical singularities
- **Precision**: Configurable float/double via `nb_float` typedef

## 📖 Documentation

- **[Final Report](FINAL_REPORT.md)**: Comprehensive project analysis and results
- **[Benchmark Guide](benchmarks/README.md)**: Detailed benchmarking documentation
- **[User Guide](#user-guide)**: Complete usage instructions

## 🛠️ Usage Examples

### Basic Simulation
```bash
# 10K particles, 5 timesteps with GPU
./build/src/nbody_app --backend cuda --N 10000 --dt 0.01 --steps 5

# Compare all backends
for backend in seq omp cuda; do
    echo "Testing $backend:"
    ./build/src/nbody_app --backend $backend --N 5000 --dt 0.01 --steps 3
done
```

### Performance Tuning
```bash
# Optimize OpenMP thread count
export OMP_NUM_THREADS=8
./build/src/nbody_app --backend omp --N 10000 --dt 0.01 --steps 5

# Adjust CUDA block size  
./build/src/nbody_app --backend cuda --N 10000 --block-size 256 --dt 0.01 --steps 5
```

## 🎯 Performance Guidelines

- **N < 1,000**: Use sequential or OpenMP
- **1,000 ≤ N < 10,000**: OpenMP typically fastest
- **N ≥ 10,000**: CUDA provides significant acceleration  
- **N ≥ 50,000**: CUDA strongly recommended (10x+ speedup)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **GPU Memory Optimization**: Keep data GPU-resident throughout simulation
- **Multi-GPU Support**: Scale across multiple GPUs
- **Advanced Algorithms**: Barnes-Hut tree methods for O(N log N) scaling
- **Visualization**: Real-time 3D particle visualization

## 📊 Benchmarking

The `benchmarks/` directory contains comprehensive testing tools:

```bash
cd benchmarks
./run_benchmarks.sh    # Full benchmark suite
cd results
python3 ../plot_results.py  # Generate plots from existing data
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **NVIDIA CUDA**: GPU computing platform and programming model
- **OpenMP**: Multi-platform shared-memory parallel programming
- **CMake**: Cross-platform build system generator

## 📧 Contact

**Author**: Jack Herrington  
**Project**: High-Performance N-Body Gravitational Simulation  
**Repository**: https://github.com/jackaherrington/N-Body-Simulation  

---

*Built with ❤️ for high-performance computing education*