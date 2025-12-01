#!/bin/bash
# Comprehensive N-body benchmark script for all three backends

echo "Running comprehensive N-body simulation benchmarks..."
echo "Backend,N,Threads,dt,Steps,Time_ms" > benchmark_results.csv

cd build

# Test different problem sizes
for N in 100 500 1000 2500 5000; do
    echo "Testing N=$N..."
    
    # Sequential backend
    echo -n "  Sequential... "
    ./src/nbody_app --backend seq --N $N --dt 0.01 --steps 5 | tail -1 >> ../benchmark_results.csv
    echo "done"
    
    # OpenMP backend (multi-threaded)
    echo -n "  OpenMP... "
    ./src/nbody_app --backend omp --N $N --dt 0.01 --steps 5 | tail -1 >> ../benchmark_results.csv
    echo "done"
    
    # CUDA backend
    echo -n "  CUDA... "
    ./src/nbody_app --backend cuda --N $N --dt 0.01 --steps 5 | tail -1 >> ../benchmark_results.csv
    echo "done"
    
    echo ""
done

echo "Benchmark completed! Results saved to benchmark_results.csv"
echo ""
echo "Summary:"
cat ../benchmark_results.csv | column -t -s ','