#!/bin/bash
# Simple N-body benchmark script
# Tests problem sizes: 100, 1000, 10000 particles

echo "Running N-body simulation benchmarks..."
echo ""

# Create results directory and file
mkdir -p results
echo "Backend,N,Threads,dt,Steps,Time_ms,Speedup" > results/benchmark_results.csv

# Test parameters
N_VALUES=(100 1000 10000 100000)
THREAD_VALUES=(1 2 4 8)
DT=0.01
STEPS=5

echo "Testing ${#N_VALUES[@]} problem sizes: ${N_VALUES[*]}"
echo ""

for N in "${N_VALUES[@]}"; do
    echo "=== N=$N ==="
    
    # Sequential baseline
    echo -n "Sequential: "
    seq_result=$(../build/src/nbody_app --backend seq --N $N --dt $DT --steps $STEPS --quiet 2>/dev/null | tail -1)
    if [[ $seq_result == *"RESULT"* ]]; then
        seq_time=$(echo $seq_result | cut -d',' -f7)
        seq_time_display=$(printf "%.0f" $seq_time)
        echo "${seq_time_display}ms"
        echo "$seq_result,1.0" >> results/benchmark_results.csv
    else
        echo "Failed"
        continue
    fi
    
    # OpenMP with different thread counts
    for threads in "${THREAD_VALUES[@]}"; do
        echo -n "OpenMP ${threads}t: "
        omp_result=$(../build/src/nbody_app --backend omp --N $N --dt $DT --steps $STEPS --threads $threads --quiet 2>/dev/null | tail -1)
        if [[ $omp_result == *"RESULT"* ]]; then
            omp_time=$(echo $omp_result | cut -d',' -f7)
            omp_time_display=$(printf "%.0f" $omp_time)
            # Use bc with here-string for reliable calculation
            speedup=$(bc -l <<< "scale=4; $seq_time / $omp_time")
            speedup_rounded=$(printf "%.2f" $speedup)
            echo "${omp_time_display}ms (${speedup_rounded}x)"
            
            # Update result to include correct thread count and speedup
            modified_result=$(echo $omp_result | sed "s/,1,/,$threads,/")
            echo "$modified_result,$speedup_rounded" >> results/benchmark_results.csv
        else
            echo "Failed"
        fi
    done
    
    # CUDA
    echo -n "CUDA: "
    cuda_result=$(../build/src/nbody_app --backend cuda --N $N --dt $DT --steps $STEPS --quiet 2>/dev/null | tail -1)
    if [[ $cuda_result == *"RESULT"* ]]; then
        cuda_time=$(echo $cuda_result | cut -d',' -f7)
        cuda_time_display=$(printf "%.0f" $cuda_time)
        # Use bc with here-string for reliable calculation
        speedup=$(bc -l <<< "scale=4; $seq_time / $cuda_time")
        speedup_rounded=$(printf "%.2f" $speedup)
        echo "${cuda_time_display}ms (${speedup_rounded}x)"
        echo "$cuda_result,$speedup_rounded" >> results/benchmark_results.csv
    else
        echo "CUDA not available"
    fi
    
    echo ""
done

echo "Benchmark complete! Results saved to results/benchmark_results.csv"
echo ""
echo "Summary:"
grep -E "(seq|omp.*,8,|cuda)" results/benchmark_results.csv
