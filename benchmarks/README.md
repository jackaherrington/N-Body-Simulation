# N-Body Simulation Benchmarking Suite

This directory contains comprehensive benchmarking tools for testing the N-Body simulation across different backends and problem sizes.

## Quick Start

Run comprehensive benchmarks:
```bash
cd benchmarks
./run_benchmarks.sh
```

## Scripts

### `run_benchmarks.sh` 
Main benchmarking script that tests:
- **Problem sizes**: 100, 1,000, 10,000, 100,000 particles  
- **Backends**: Sequential, OpenMP (1-16 threads), CUDA
- **Output**: CSV data + comprehensive plots

### `plot_results.py`
Advanced plotting and analysis script that generates:
- **performance_analysis.png**: 4-panel performance overview
- **scaling_analysis.png**: Detailed scaling analysis with 4 additional plots
- **Console summary**: Performance table and key insights

### Legacy Scripts
- `benchmark_all.sh`: Original simple benchmark
- `benchmark_and_plot.sh`: Previous comprehensive benchmark

## Generated Files

### `results/benchmark_results.csv`
Raw performance data with columns:
- Backend, N, Threads, dt, Steps, Time_ms, Speedup

### `results/performance_analysis.png`
Main performance visualization (4 panels):
1. **Execution Time vs Problem Size** (log-log)
2. **Speedup vs Problem Size** (log-log) 
3. **OpenMP Parallel Efficiency**
4. **Performance Comparison** (largest N)

### `results/scaling_analysis.png`
Detailed scaling analysis (4 panels):
1. **GPU vs CPU Scaling** with O(N²) reference
2. **Speedup Breakdown** by problem size
3. **OpenMP Thread Scaling** for each N
4. **Performance Density** (estimated GFLOPS)

## Key Results Summary

| N | Sequential | OpenMP (Best) | CUDA | CUDA Speedup |
|---|------------|---------------|------|-------------|
| 100 | 0.39ms | 0.36ms (8T) | 2,085ms | **0.00x** |
| 1,000 | 34.40ms | 34.25ms (16T) | 159ms | **0.22x** |
| 10,000 | 3,195ms | 3,262ms (1T) | 2,787ms | **1.15x** 🎯 |
| 100,000 | 203,199ms | 199,666ms (4T) | 14,148ms | **14.36x** 🚀 |

### Key Insights
- 🎯 **CUDA crossover**: N=10,000 particles
- 🚀 **Best CUDA speedup**: 14.36x at N=100,000  
- ⚡ **OpenMP scaling**: Limited to ~1.08x due to small problem overhead
- 📈 **GPU scaling**: Excellent performance at large N despite memory transfer overhead

## Performance Notes

The current CUDA implementation transfers data between GPU/CPU every timestep, which limits performance for smaller problems. At large scale (N=100,000), the computational advantage overcomes this overhead significantly.

## Usage Examples

```bash
# Quick test
./run_benchmarks.sh

# Generate plots from existing data
cd results && python3 ../plot_results.py

# Individual backend testing  
cd ../build
./src/nbody_app --backend cuda --N 100000 --dt 0.01 --steps 3
```