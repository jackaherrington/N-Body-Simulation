#!/bin/bash
# Comprehensive N-body benchmark script with Python plotting
# Tests problem sizes: 100, 1000, 10000, 100000
# Tests thread counts for OpenMP scaling analysis

echo "Running comprehensive N-body simulation benchmarks..."
echo "Testing large-scale problems: 100, 1000, 10000, 100000 particles"
echo ""

# Create results directory
mkdir -p results
cd ../build

# CSV header
echo "Backend,N,Threads,dt,Steps,Time_ms,Speedup" > ../benchmarks/results/benchmark_results.csv

# Updated test parameters for larger scale testing
N_VALUES=(100 1000 10000 100000)
THREAD_VALUES=(1 2 4 8 16)
DT=0.01
STEPS=5  # Reduced steps for large N to keep runtime reasonable

echo "Testing ${#N_VALUES[@]} problem sizes: ${N_VALUES[*]}"
echo "Testing ${#THREAD_VALUES[@]} thread counts: ${THREAD_VALUES[*]}"
echo "Using dt=$DT, steps=$STEPS"
echo ""

total_tests=$((${#N_VALUES[@]} * (1 + ${#THREAD_VALUES[@]} + 1)))
current_test=0

for N in "${N_VALUES[@]}"; do
    echo "Testing N=$N..."
    
    # Adjust steps for very large problems to keep runtime reasonable
    if [ $N -ge 100000 ]; then
        CURRENT_STEPS=3
    else
        CURRENT_STEPS=$STEPS
    fi
    
    # Sequential backend (baseline)
    echo -n "  Sequential... "
    start_time=$(date +%s.%3N)
    result=$(./src/nbody_app --backend seq --N $N --dt $DT --steps $CURRENT_STEPS | tail -1)
    end_time=$(date +%s.%3N)
    runtime=$(echo "($end_time - $start_time) * 1000" | bc -l)
    
    seq_time=$(echo $result | cut -d',' -f6)
    echo "$result,1.0" >> ../benchmarks/results/benchmark_results.csv
    current_test=$((current_test + 1))
    echo "done (${runtime%.*}ms total) ($current_test/$total_tests)"
    
    # OpenMP backend with different thread counts
    for threads in "${THREAD_VALUES[@]}"; do
        echo -n "  OpenMP (${threads} threads)... "
        export OMP_NUM_THREADS=$threads
        start_time=$(date +%s.%3N)
        result=$(./src/nbody_app --backend omp --N $N --dt $DT --steps $CURRENT_STEPS | tail -1)
        end_time=$(date +%s.%3N)
        runtime=$(echo "($end_time - $start_time) * 1000" | bc -l)
        
        omp_time=$(echo $result | cut -d',' -f6)
        speedup=$(echo "scale=3; $seq_time / $omp_time" | bc -l)
        # Update the result line to include thread count and speedup
        modified_result=$(echo $result | sed "s/,1,/,$threads,/")
        echo "$modified_result,$speedup" >> ../benchmarks/results/benchmark_results.csv
        current_test=$((current_test + 1))
        echo "done (${runtime%.*}ms total) ($current_test/$total_tests)"
    done
    
    # CUDA backend
    echo -n "  CUDA... "
    start_time=$(date +%s.%3N)
    result=$(./src/nbody_app --backend cuda --N $N --dt $DT --steps $CURRENT_STEPS | tail -1)
    end_time=$(date +%s.%3N)
    runtime=$(echo "($end_time - $start_time) * 1000" | bc -l)
    
    cuda_time=$(echo $result | cut -d',' -f6)
    speedup=$(echo "scale=3; $seq_time / $cuda_time" | bc -l)
    echo "$result,$speedup" >> ../benchmarks/results/benchmark_results.csv
    current_test=$((current_test + 1))
    echo "done (${runtime%.*}ms total) ($current_test/$total_tests)"
    
    echo ""
done

cd ../benchmarks
echo "Benchmark completed! Results saved to results/benchmark_results.csv"
echo ""
echo "Generating plots and analysis..."

# Install required Python packages if not already installed
echo "Checking Python packages..."
python3 -c "import pandas, matplotlib, numpy" 2>/dev/null || {
    echo "Installing required packages..."
    python3 -m pip install pandas matplotlib numpy --quiet 2>/dev/null || {
        echo "Installing via apt..."
        sudo apt update >/dev/null && sudo apt install -y python3-pip python3-pandas python3-matplotlib python3-numpy >/dev/null
    }
}

# Run the Python plotting script
cd results
python3 ../plot_results.py

echo ""
echo "Benchmark and analysis complete!"
echo "Check the 'benchmarks/results/' directory for:"
echo "  - benchmark_results.csv (raw data)"
echo "  - performance_analysis.png (comprehensive plots)"
echo "  - scaling_analysis.png (detailed scaling plots)"

# Show quick summary
echo ""
echo "Quick Summary:"
tail -n +2 benchmark_results.csv | grep "seq\|cuda" | head -10