# Contributing to N-Body Gravitational Simulation

Thank you for your interest in contributing! This project welcomes contributions from the community.

## 🚀 Quick Start for Contributors

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-optimization`
3. Make your changes
4. Test thoroughly (see Testing section below)
5. Commit with clear messages: `git commit -m "Add GPU memory optimization"`
6. Push to your fork: `git push origin feature/amazing-optimization`
7. Create a Pull Request

## 🎯 Areas for Contribution

### High Impact
- **GPU Memory Optimization**: Implement GPU-resident simulation to eliminate host-device transfers
- **Barnes-Hut Algorithm**: Add O(N log N) tree-based force calculation
- **Multi-GPU Support**: Scale simulation across multiple GPUs
- **Memory Pool**: Implement custom memory allocators for better performance

### Medium Impact  
- **Visualization**: Real-time 3D particle rendering with OpenGL/Vulkan
- **File I/O**: Support for standard astronomical file formats (HDF5, etc.)
- **Advanced Integration**: Higher-order integrators (Leapfrog, RKN methods)
- **Profiling Tools**: Built-in performance profiling and analysis

### Easy Wins
- **Documentation**: Improve code comments and user documentation
- **Build System**: Additional platform support (Windows, macOS)
- **Testing**: Add unit tests and regression tests
- **Examples**: Create tutorial examples and use cases

## 🧪 Testing Your Changes

### Build and Test
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)

# Run basic functionality test
./src/nbody_app --backend seq --N 1000 --dt 0.01 --steps 5
./src/nbody_app --backend omp --N 1000 --dt 0.01 --steps 5
./src/nbody_app --backend cuda --N 1000 --dt 0.01 --steps 5
```

### Performance Validation
```bash
cd ../benchmarks
./run_benchmarks.sh
```

Ensure your changes don't regress performance significantly.

### Code Quality
- Follow existing code style (see Style Guidelines below)
- Add comments for complex algorithms
- Ensure no memory leaks with valgrind (if available)
- Verify CUDA code with cuda-memcheck

## 📝 Style Guidelines

### C++ Code Style
```cpp
// Use descriptive variable names
const nb_float gravitational_constant = 6.67430e-11;

// Prefer const references for parameters
void calculate_forces(const Bodies& bodies, const SimConstants& constants);

// Use structured bindings where appropriate (C++17)
const auto [x, y, z] = get_position(i);

// Comment complex algorithms
// Velocity Verlet integration: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
for (std::size_t i = 0; i < N; ++i) {
    bodies.x[i] += bodies.vx[i] * dt + half_dt2 * bodies.ax[i];
}
```

### CUDA Code Style
```cpp
// Use descriptive kernel names with template parameters
template<int TILE_SIZE>
__global__ void calculate_gravitational_forces_kernel(...);

// Document shared memory usage
extern __shared__ nb_float shared_memory[];
nb_float* shared_positions_x = shared_memory;
nb_float* shared_positions_y = shared_positions_x + TILE_SIZE;
```

### CMake Style
```cmake
# Use modern CMake practices
target_link_libraries(nbody_app PRIVATE nbody_core)
target_include_directories(nbody_app PRIVATE ${CMAKE_SOURCE_DIR}/include)
```

## 🐛 Bug Reports

Please include:
- **System information**: OS, GPU model, CUDA version
- **Build configuration**: CMake options used
- **Minimal reproduction**: Smallest example that shows the bug
- **Expected vs. actual behavior**
- **Error messages**: Full compilation/runtime errors

Example:
```
**System**: Ubuntu 22.04, RTX 3070, CUDA 11.5
**Build**: cmake .. -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
**Command**: ./src/nbody_app --backend cuda --N 50000 --dt 0.01 --steps 5
**Error**: CUDA out of memory error
**Expected**: Successful execution
```

## 🚀 Performance Optimization Guidelines

### GPU Optimizations
1. **Memory Coalescing**: Ensure adjacent threads access adjacent memory
2. **Shared Memory**: Use shared memory to reduce global memory access
3. **Occupancy**: Target 50-100% occupancy for compute-bound kernels
4. **Bank Conflicts**: Avoid shared memory bank conflicts

### CPU Optimizations  
1. **SIMD**: Use compiler auto-vectorization or explicit SIMD intrinsics
2. **Cache Optimization**: Minimize cache misses with good data locality
3. **Loop Unrolling**: Let compiler optimize hot loops
4. **Branch Prediction**: Minimize unpredictable branches in inner loops

### Algorithm Optimizations
1. **Complexity**: Consider algorithmic improvements (O(N²) → O(N log N))
2. **Numerical Stability**: Maintain accuracy while optimizing
3. **Memory Usage**: Minimize memory footprint for large problems

## 📊 Benchmarking New Features

When adding optimizations:

1. **Baseline**: Record performance before changes
2. **Validation**: Ensure results remain numerically consistent  
3. **Scaling**: Test across multiple problem sizes
4. **Documentation**: Update performance claims in README

```bash
# Before your changes
cd benchmarks && ./run_benchmarks.sh
cp results/benchmark_results.csv baseline_results.csv

# After your changes  
./run_benchmarks.sh
# Compare performance improvements
```

## 📄 Documentation Updates

When making changes:
- Update relevant README sections
- Add/update code comments
- Update FINAL_REPORT.md for significant algorithmic changes
- Add examples for new features

## ❓ Questions?

- **General Questions**: Open a GitHub Discussion
- **Bug Reports**: Create a GitHub Issue  
- **Feature Requests**: Open a GitHub Issue with enhancement label
- **Performance Questions**: Include benchmark results and system specs

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributor section
- Git commit history
- Major contributions may be mentioned in academic acknowledgments

Thank you for helping make this project better! 🚀