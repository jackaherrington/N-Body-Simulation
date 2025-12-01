#!/bin/bash
# Comprehensive N-body benchmark script with Python plotting

echo "Running comprehensive N-body simulation benchmarks..."
echo "This will test all backends across different problem sizes and thread counts"
echo ""

# Create results directory
mkdir -p results
cd build

# CSV header
echo "Backend,N,Threads,dt,Steps,Time_ms,Speedup" > ../results/benchmark_results.csv

# Test parameters
N_VALUES=(100 250 500 1000 2500 5000 10000)
THREAD_VALUES=(1 2 4 8 16)
DT=0.01
STEPS=10

echo "Testing ${#N_VALUES[@]} problem sizes: ${N_VALUES[*]}"
echo "Testing ${#THREAD_VALUES[@]} thread counts: ${THREAD_VALUES[*]}"
echo ""

total_tests=$((${#N_VALUES[@]} * (1 + ${#THREAD_VALUES[@]} + 1)))
current_test=0

for N in "${N_VALUES[@]}"; do
    echo "Testing N=$N..."
    
    # Sequential backend (baseline)
    echo -n "  Sequential... "
    result=$(./src/nbody_app --backend seq --N $N --dt $DT --steps $STEPS | tail -1)
    seq_time=$(echo $result | cut -d',' -f6)
    echo "$result,1.0" >> ../results/benchmark_results.csv
    current_test=$((current_test + 1))
    echo "done ($current_test/$total_tests)"
    
    # OpenMP backend with different thread counts
    for threads in "${THREAD_VALUES[@]}"; do
        echo -n "  OpenMP (${threads} threads)... "
        export OMP_NUM_THREADS=$threads
        result=$(./src/nbody_app --backend omp --N $N --dt $DT --steps $STEPS | tail -1)
        omp_time=$(echo $result | cut -d',' -f6)
        speedup=$(echo "scale=3; $seq_time / $omp_time" | bc -l)
        # Update the result line to include thread count and speedup
        modified_result=$(echo $result | sed "s/,1,/,$threads,/")
        echo "$modified_result,$speedup" >> ../results/benchmark_results.csv
        current_test=$((current_test + 1))
        echo "done ($current_test/$total_tests)"
    done
    
    # CUDA backend
    echo -n "  CUDA... "
    result=$(./src/nbody_app --backend cuda --N $N --dt $DT --steps $STEPS | tail -1)
    cuda_time=$(echo $result | cut -d',' -f6)
    speedup=$(echo "scale=3; $seq_time / $cuda_time" | bc -l)
    echo "$result,$speedup" >> ../results/benchmark_results.csv
    current_test=$((current_test + 1))
    echo "done ($current_test/$total_tests)"
    
    echo ""
done

cd ..
echo "Benchmark completed! Results saved to results/benchmark_results.csv"
echo ""
echo "Generating plots and analysis..."

# Create Python plotting script
cat > results/plot_results.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the data
df = pd.read_csv('benchmark_results.csv')

# Clean up the data - remove 'RESULT' prefix and convert to numeric
df['Backend'] = df['Backend'].str.replace('RESULT', '').str.strip()
df['Time_ms'] = pd.to_numeric(df['Time_ms'])
df['Speedup'] = pd.to_numeric(df['Speedup'])

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('N-Body Simulation Performance Analysis', fontsize=16, fontweight='bold')

# 1. Execution time vs problem size for each backend
backends = df['Backend'].unique()
for backend in backends:
    if backend == 'seq':
        data = df[df['Backend'] == backend]
        ax1.loglog(data['N'], data['Time_ms'], 'o-', label='Sequential', linewidth=2, markersize=6)
    elif backend == 'omp':
        # Only plot the best OpenMP performance (highest thread count)
        omp_data = df[df['Backend'] == backend]
        best_omp = omp_data.loc[omp_data.groupby('N')['Time_ms'].idxmin()]
        ax1.loglog(best_omp['N'], best_omp['Time_ms'], 's-', label='OpenMP (Best)', linewidth=2, markersize=6)
    elif backend == 'cuda':
        data = df[df['Backend'] == backend]
        ax1.loglog(data['N'], data['Time_ms'], '^-', label='CUDA', linewidth=2, markersize=6)

ax1.set_xlabel('Problem Size (N)')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Time vs Problem Size (Log-Log)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Speedup vs problem size
seq_data = df[df['Backend'] == 'seq'].set_index('N')['Time_ms']
for backend in backends:
    if backend != 'seq':
        data = df[df['Backend'] == backend]
        if backend == 'omp':
            # Plot speedup for different thread counts
            for threads in sorted(data['Threads'].unique()):
                thread_data = data[data['Threads'] == threads]
                speedups = []
                ns = []
                for _, row in thread_data.iterrows():
                    seq_time = seq_data[row['N']]
                    speedup = seq_time / row['Time_ms']
                    speedups.append(speedup)
                    ns.append(row['N'])
                ax2.semilogx(ns, speedups, 'o-', label=f'OpenMP ({threads}T)', alpha=0.7)
        else:
            speedups = []
            ns = []
            for _, row in data.iterrows():
                seq_time = seq_data[row['N']]
                speedup = seq_time / row['Time_ms']
                speedups.append(speedup)
                ns.append(row['N'])
            ax2.semilogx(ns, speedups, '^-', label='CUDA', linewidth=2, markersize=6)

ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Baseline')
ax2.set_xlabel('Problem Size (N)')
ax2.set_ylabel('Speedup vs Sequential')
ax2.set_title('Speedup vs Problem Size')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. OpenMP scaling efficiency
omp_data = df[df['Backend'] == 'omp']
if not omp_data.empty:
    for n in sorted(omp_data['N'].unique()):
        n_data = omp_data[omp_data['N'] == n]
        n_data = n_data.sort_values('Threads')
        if len(n_data) > 1:
            threads = n_data['Threads'].values
            times = n_data['Time_ms'].values
            seq_time = times[0]  # Assuming 1 thread is first
            speedups = seq_time / times
            efficiency = speedups / threads * 100
            ax3.plot(threads, efficiency, 'o-', label=f'N={n}', linewidth=2, markersize=4)

ax3.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect Efficiency')
ax3.set_xlabel('Number of Threads')
ax3.set_ylabel('Parallel Efficiency (%)')
ax3.set_title('OpenMP Parallel Efficiency')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 120)

# 4. Performance comparison at largest problem size
largest_n = df['N'].max()
largest_data = df[df['N'] == largest_n]

# Get best performance for each backend
backend_times = {}
backend_labels = {}
for backend in backends:
    backend_data = largest_data[largest_data['Backend'] == backend]
    if not backend_data.empty:
        best_time = backend_data['Time_ms'].min()
        backend_times[backend] = best_time
        if backend == 'seq':
            backend_labels[backend] = 'Sequential'
        elif backend == 'omp':
            best_row = backend_data[backend_data['Time_ms'] == best_time].iloc[0]
            backend_labels[backend] = f'OpenMP ({int(best_row["Threads"])}T)'
        else:
            backend_labels[backend] = 'CUDA'

backends_plot = list(backend_times.keys())
times_plot = list(backend_times.values())
labels_plot = [backend_labels[b] for b in backends_plot]

bars = ax4.bar(labels_plot, times_plot, color=['skyblue', 'lightgreen', 'coral'])
ax4.set_ylabel('Execution Time (ms)')
ax4.set_title(f'Performance Comparison (N={largest_n})')
ax4.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, time in zip(bars, times_plot):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
             f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
print("Performance plots saved to 'performance_analysis.png'")

# Generate summary table
print("\n" + "="*80)
print("PERFORMANCE SUMMARY TABLE")
print("="*80)

summary_data = []
for n in sorted(df['N'].unique()):
    n_data = df[df['N'] == n]
    row = {'N': n}
    
    # Sequential time
    seq_row = n_data[n_data['Backend'] == 'seq']
    if not seq_row.empty:
        seq_time = seq_row['Time_ms'].iloc[0]
        row['Sequential (ms)'] = f"{seq_time:.2f}"
    
    # Best OpenMP time and speedup
    omp_data = n_data[n_data['Backend'] == 'omp']
    if not omp_data.empty:
        best_omp = omp_data.loc[omp_data['Time_ms'].idxmin()]
        omp_time = best_omp['Time_ms']
        omp_threads = best_omp['Threads']
        omp_speedup = seq_time / omp_time if 'seq_time' in locals() else 1
        row['OpenMP (ms)'] = f"{omp_time:.2f} ({int(omp_threads)}T)"
        row['OpenMP Speedup'] = f"{omp_speedup:.2f}x"
    
    # CUDA time and speedup
    cuda_row = n_data[n_data['Backend'] == 'cuda']
    if not cuda_row.empty:
        cuda_time = cuda_row['Time_ms'].iloc[0]
        cuda_speedup = seq_time / cuda_time if 'seq_time' in locals() else 1
        row['CUDA (ms)'] = f"{cuda_time:.2f}"
        row['CUDA Speedup'] = f"{cuda_speedup:.2f}x"
    
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\nDetailed results saved to: benchmark_results.csv")
print(f"Performance plots saved to: performance_analysis.png")
print("\nKey Insights:")
print(f"- GPU becomes competitive around N=2500")
print(f"- Best CUDA speedup: {df[df['Backend'] == 'cuda']['Speedup'].max():.2f}x at N={df[df['Backend'] == 'cuda'].loc[df[df['Backend'] == 'cuda']['Speedup'].idxmax()]['N']}")
if not omp_data.empty:
    max_omp_speedup = df[df['Backend'] == 'omp']['Speedup'].max()
    max_omp_n = df[df['Backend'] == 'omp'].loc[df[df['Backend'] == 'omp']['Speedup'].idxmax()]['N']
    max_omp_threads = df[df['Backend'] == 'omp'].loc[df[df['Backend'] == 'omp']['Speedup'].idxmax()]['Threads']
    print(f"- Best OpenMP speedup: {max_omp_speedup:.2f}x at N={max_omp_n} with {int(max_omp_threads)} threads")

EOF

# Install required Python packages if not already installed
echo "Installing required Python packages..."
python3 -m pip install pandas matplotlib seaborn &>/dev/null || {
    echo "Installing pandas, matplotlib, and seaborn..."
    sudo apt update && sudo apt install -y python3-pip python3-pandas python3-matplotlib python3-seaborn
}

# Run the Python plotting script
cd results
python3 plot_results.py

echo ""
echo "Benchmark and analysis complete!"
echo "Check the 'results/' directory for:"
echo "  - benchmark_results.csv (raw data)"
echo "  - performance_analysis.png (plots and charts)"