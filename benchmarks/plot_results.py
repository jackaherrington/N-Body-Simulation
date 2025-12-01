#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading benchmark data...")
# Read the data
df = pd.read_csv('results/benchmark_results.csv')

# Clean up the data - remove 'RESULT' prefix and convert to numeric
df['Backend'] = df['Backend'].str.replace('RESULT', '').str.strip()
df['Time_ms'] = pd.to_numeric(df['Time_ms'], errors='coerce')
df['Speedup'] = pd.to_numeric(df['Speedup'], errors='coerce')

print(f"Loaded {len(df)} benchmark results")

# Set up plotting
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['axes.grid'] = True

# Create main performance analysis figure
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig1.suptitle('N-Body Simulation Performance Analysis', fontsize=16, fontweight='bold')

# Colors and markers for consistency
colors = {'seq': '#1f77b4', 'omp': '#ff7f0e', 'cuda': '#d62728'}
markers = {'seq': 'o', 'omp': 's', 'cuda': '^'}

backends = df['Backend'].unique()

print("Generating execution time plot...")
# 1. Execution time vs problem size (log-log)
for backend in backends:
    data = df[df['Backend'] == backend]
    if backend == 'omp':
        # Only plot the best OpenMP performance
        best_data = data.loc[data.groupby('N')['Time_ms'].idxmin()]
        n_vals = best_data['N'].tolist()
        time_vals = best_data['Time_ms'].tolist()
        label = 'OpenMP (Best)'
    else:
        n_vals = data['N'].tolist()
        time_vals = data['Time_ms'].tolist()
        label = backend.upper() if backend == 'cuda' else 'Sequential'
    
    ax1.loglog(n_vals, time_vals, 
              color=colors.get(backend, 'black'), 
              marker=markers.get(backend, 'o'), 
              label=label, linewidth=2, markersize=8)

ax1.set_xlabel('Problem Size (N particles)')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('Execution Time vs Problem Size (Log-Log Scale)')
ax1.legend()

print("Generating speedup analysis...")
# 2. Speedup vs problem size
seq_data = df[df['Backend'] == 'seq']
seq_times = {row['N']: row['Time_ms'] for _, row in seq_data.iterrows()}

for backend in backends:
    if backend == 'seq':
        continue
    
    data = df[df['Backend'] == backend]
    
    if backend == 'omp':
        # Plot different thread counts
        thread_counts = sorted(data['Threads'].unique())
        for threads in thread_counts:
            thread_data = data[data['Threads'] == threads]
            n_vals = []
            speedup_vals = []
            for _, row in thread_data.iterrows():
                if row['N'] in seq_times:
                    n_vals.append(row['N'])
                    speedup = seq_times[row['N']] / row['Time_ms']
                    speedup_vals.append(speedup)
            if n_vals:
                alpha = 1.0 if threads == max(thread_counts) else 0.6
                linewidth = 2 if threads == max(thread_counts) else 1
                ax2.loglog(n_vals, speedup_vals, 'o-', 
                          label=f'OpenMP ({threads}T)', 
                          alpha=alpha, markersize=4, linewidth=linewidth)
    else:  # CUDA
        n_vals = []
        speedup_vals = []
        for _, row in data.iterrows():
            if row['N'] in seq_times:
                n_vals.append(row['N'])
                speedup = seq_times[row['N']] / row['Time_ms']
                speedup_vals.append(speedup)
        if n_vals:
            ax2.loglog(n_vals, speedup_vals, '^-', 
                      color='red', label='CUDA', linewidth=3, markersize=8)

ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No Speedup')
ax2.set_xlabel('Problem Size (N particles)')
ax2.set_ylabel('Speedup vs Sequential')
ax2.set_title('Speedup vs Problem Size (Log-Log Scale)')
ax2.legend()

print("Generating OpenMP efficiency analysis...")
# 3. OpenMP parallel efficiency
omp_data = df[df['Backend'] == 'omp']
problem_sizes = sorted(omp_data['N'].unique())

for n in problem_sizes:
    n_data = omp_data[omp_data['N'] == n].sort_values('Threads')
    threads = n_data['Threads'].tolist()
    times = n_data['Time_ms'].tolist()
    
    if len(threads) > 1:
        seq_time = times[0]  # 1 thread baseline
        speedups = [seq_time / t for t in times]
        efficiency = [s/t * 100 for s, t in zip(speedups, threads)]
        
        ax3.plot(threads, efficiency, 'o-', label=f'N={n:,}', linewidth=2, markersize=6)

ax3.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Perfect Efficiency')
ax3.axhline(y=50, color='red', linestyle=':', alpha=0.5, label='50% Efficiency')
ax3.set_xlabel('Number of Threads')
ax3.set_ylabel('Parallel Efficiency (%)')
ax3.set_title('OpenMP Parallel Efficiency')
ax3.legend()
ax3.set_ylim(0, 120)
ax3.set_xlim(0, 17)

print("Generating performance comparison...")
# 4. Performance comparison at largest problem size
largest_n = df['N'].max()
largest_data = df[df['N'] == largest_n]

backend_times = {}
backend_labels = {}
for backend in backends:
    backend_data = largest_data[largest_data['Backend'] == backend]
    if not backend_data.empty:
        if backend == 'omp':
            # Get best OpenMP performance
            best_time = backend_data['Time_ms'].min()
            best_mask = backend_data['Time_ms'] == best_time
            best_threads = backend_data[best_mask]['Threads'].iloc[0]
            backend_times[backend] = best_time
            backend_labels[backend] = f'OpenMP ({int(best_threads)}T)'
        else:
            best_time = backend_data['Time_ms'].min()
            backend_times[backend] = best_time
            backend_labels[backend] = 'Sequential' if backend == 'seq' else 'CUDA'

backends_plot = list(backend_times.keys())
times_plot = list(backend_times.values())
labels_plot = [backend_labels[b] for b in backends_plot]

bars = ax4.bar(range(len(labels_plot)), times_plot, 
               color=[colors.get(b, 'gray') for b in backends_plot])
ax4.set_xticks(range(len(labels_plot)))
ax4.set_xticklabels(labels_plot)
ax4.set_ylabel('Execution Time (ms)')
ax4.set_title(f'Performance Comparison (N={largest_n:,})')
ax4.set_yscale('log')

# Add value labels on bars
for bar, time in zip(bars, times_plot):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height * 1.1,
             f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
print("Main performance plots saved to 'performance_analysis.png'")

# Create detailed scaling analysis figure
fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Detailed Scaling Analysis', fontsize=16, fontweight='bold')

print("Generating detailed scaling plots...")

# 5. CUDA vs Sequential scaling
seq_data = df[df['Backend'] == 'seq']
cuda_data = df[df['Backend'] == 'cuda']

n_vals_seq = seq_data['N'].tolist()
times_seq = seq_data['Time_ms'].tolist()
n_vals_cuda = cuda_data['N'].tolist()
times_cuda = cuda_data['Time_ms'].tolist()

ax5.loglog(n_vals_seq, times_seq, 'o-', color='blue', label='Sequential', linewidth=2, markersize=8)
ax5.loglog(n_vals_cuda, times_cuda, '^-', color='red', label='CUDA', linewidth=2, markersize=8)

# Add theoretical O(N²) scaling reference
if n_vals_seq:
    n_ref = np.array(n_vals_seq)
    t_ref = times_seq[0] * (n_ref / n_vals_seq[0]) ** 2
    ax5.loglog(n_ref, t_ref, '--', color='gray', alpha=0.5, label='O(N²) reference')

ax5.set_xlabel('Problem Size (N particles)')
ax5.set_ylabel('Execution Time (ms)')
ax5.set_title('GPU vs CPU Scaling Comparison')
ax5.legend()

# 6. Speedup breakdown by problem size
problem_sizes = sorted(df['N'].unique())
cuda_speedups = []
best_omp_speedups = []

for n in problem_sizes:
    n_data = df[df['N'] == n]
    seq_time = n_data[n_data['Backend'] == 'seq']['Time_ms'].iloc[0]
    
    # CUDA speedup (if available)
    cuda_data = n_data[n_data['Backend'] == 'cuda']
    if not cuda_data.empty:
        cuda_time = cuda_data['Time_ms'].iloc[0]
        cuda_speedups.append(seq_time / cuda_time)
    else:
        cuda_speedups.append(0)  # No CUDA data available
    
    # Best OpenMP speedup
    omp_times = n_data[n_data['Backend'] == 'omp']['Time_ms']
    best_omp_time = omp_times.min()
    best_omp_speedups.append(seq_time / best_omp_time)

# Filter valid CUDA data
valid_cuda_indices = [i for i, s in enumerate(cuda_speedups) if s > 0]
valid_cuda_speedups = [cuda_speedups[i] for i in valid_cuda_indices]
valid_cuda_sizes = [problem_sizes[i] for i in valid_cuda_indices]

if valid_cuda_speedups:
    ax6.semilogx(valid_cuda_sizes, valid_cuda_speedups, '^-', color='red', label='CUDA', linewidth=2, markersize=8)
ax6.semilogx(problem_sizes, best_omp_speedups, 's-', color='orange', label='OpenMP (Best)', linewidth=2, markersize=8)
ax6.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No Speedup')
ax6.set_xlabel('Problem Size (N particles)')
ax6.set_ylabel('Speedup vs Sequential')
ax6.set_title('Speedup Comparison by Problem Size')
ax6.legend()

# 7. OpenMP thread scaling for each problem size
thread_counts = sorted(omp_data['Threads'].unique())
for n in problem_sizes:
    n_data = omp_data[omp_data['N'] == n]
    if len(n_data) > 1:
        speedups = []
        for t in thread_counts:
            thread_data = n_data[n_data['Threads'] == t]
            if not thread_data.empty:
                omp_time = thread_data['Time_ms'].iloc[0]
                seq_time = seq_times[n]
                speedups.append(seq_time / omp_time)
            else:
                speedups.append(np.nan)
        
        # Only plot if we have valid data
        valid_threads = []
        valid_speedups = []
        for t, s in zip(thread_counts, speedups):
            if not np.isnan(s):
                valid_threads.append(t)
                valid_speedups.append(s)
        
        if valid_threads:
            ax7.plot(valid_threads, valid_speedups, 'o-', label=f'N={n:,}', linewidth=2, markersize=6)

ax7.plot(thread_counts, thread_counts, '--', color='black', alpha=0.5, label='Ideal scaling')
ax7.set_xlabel('Number of Threads')
ax7.set_ylabel('Speedup vs Sequential')
ax7.set_title('OpenMP Thread Scaling by Problem Size')
ax7.legend()
ax7.set_xlim(0, 17)

# 8. Performance density (GFLOPS estimate)
# Rough estimate: N² force calculations per timestep, ~20 FLOPS per interaction
flops_per_step = {n: n * n * 20 for n in problem_sizes}

for backend in ['seq', 'omp', 'cuda']:
    data = df[df['Backend'] == backend]
    if backend == 'omp':
        # Use best OpenMP performance
        data = data.loc[data.groupby('N')['Time_ms'].idxmin()]
    
    n_vals = []
    gflops_vals = []
    for _, row in data.iterrows():
        n = row['N']
        time_ms = row['Time_ms']
        steps = 5 if n < 100000 else 3  # From our benchmark script
        total_flops = flops_per_step[n] * steps
        gflops = (total_flops / 1e9) / (time_ms / 1000)
        n_vals.append(n)
        gflops_vals.append(gflops)
    
    label = backend.upper() if backend == 'cuda' else ('OpenMP (Best)' if backend == 'omp' else 'Sequential')
    ax8.loglog(n_vals, gflops_vals, 
              marker=markers.get(backend, 'o'),
              color=colors.get(backend, 'black'),
              label=label, linewidth=2, markersize=8)

ax8.set_xlabel('Problem Size (N particles)')
ax8.set_ylabel('Performance (GFLOPS estimate)')
ax8.set_title('Computational Performance Comparison')
ax8.legend()

plt.tight_layout()
plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
print("Detailed scaling plots saved to 'scaling_analysis.png'")

# Generate enhanced summary table
print("\n" + "="*100)
print("COMPREHENSIVE PERFORMANCE SUMMARY")
print("="*100)

summary_data = []
for n in sorted(df['N'].unique()):
    n_data = df[df['N'] == n]
    row = [f"{n:,}"]
    
    # Sequential time
    seq_row = n_data[n_data['Backend'] == 'seq']
    if not seq_row.empty:
        seq_time = float(seq_row['Time_ms'].iloc[0])
        row.append(f"{seq_time:.2f}")
    else:
        row.append("N/A")
        seq_time = None
    
    # Best OpenMP performance
    omp_data_n = n_data[n_data['Backend'] == 'omp']
    if not omp_data_n.empty:
        omp_times = omp_data_n['Time_ms'].values
        omp_threads_list = omp_data_n['Threads'].values
        min_idx = np.argmin(omp_times)
        omp_time = float(omp_times[min_idx])
        omp_threads = int(omp_threads_list[min_idx])
        row.append(f"{omp_time:.2f} ({omp_threads}T)")
        if seq_time:
            omp_speedup = seq_time / omp_time
            row.append(f"{omp_speedup:.2f}x")
        else:
            row.append("N/A")
    else:
        row.extend(["N/A", "N/A"])
    
    # CUDA performance
    cuda_row = n_data[n_data['Backend'] == 'cuda']
    if not cuda_row.empty:
        cuda_time = float(cuda_row['Time_ms'].iloc[0])
        row.append(f"{cuda_time:.2f}")
        if seq_time:
            cuda_speedup = seq_time / cuda_time
            row.append(f"{cuda_speedup:.2f}x")
            # Performance improvement over OpenMP
            if not omp_data_n.empty:
                best_omp_time = float(omp_data_n['Time_ms'].min())
                cuda_vs_omp = best_omp_time / cuda_time
                row.append(f"{cuda_vs_omp:.2f}x")
            else:
                row.append("N/A")
        else:
            row.extend(["N/A", "N/A"])
    else:
        row.extend(["N/A", "N/A", "N/A"])
    
    summary_data.append(row)

# Print enhanced table
headers = ["N", "Sequential", "OpenMP (Best)", "OMP Speedup", "CUDA", "CUDA Speedup", "CUDA vs OMP"]
print(f"{'N':<10} {'Sequential':<12} {'OpenMP':<15} {'OMP':<12} {'CUDA':<12} {'CUDA':<12} {'CUDA vs':<12}")
print(f"{'particles':<10} {'(ms)':<12} {'(ms)':<15} {'Speedup':<12} {'(ms)':<12} {'Speedup':<12} {'OpenMP':<12}")
print("-" * 100)
for row in summary_data:
    print(f"{row[0]:<10} {row[1]:<12} {row[2]:<15} {row[3]:<12} {row[4]:<12} {row[5]:<12} {row[6] if len(row)>6 else 'N/A':<12}")

print(f"\nDetailed results: benchmark_results.csv")
print(f"Performance plots: performance_analysis.png")
print(f"Scaling analysis: scaling_analysis.png")

# Calculate and display key insights
cuda_data = df[df['Backend'] == 'cuda']
seq_data = df[df['Backend'] == 'seq']

if not cuda_data.empty and not seq_data.empty:
    print(f"\nKEY PERFORMANCE INSIGHTS:")
    print("-" * 40)
    
    # Calculate actual speedups vs sequential
    max_speedup = 0
    max_speedup_n = 0
    crossover_n = None
    
    for _, cuda_row in cuda_data.iterrows():
        n = cuda_row['N']
        cuda_time = cuda_row['Time_ms']
        seq_row = seq_data[seq_data['N'] == n]
        if not seq_row.empty:
            seq_time = seq_row['Time_ms'].iloc[0]
            speedup = seq_time / cuda_time
            
            if speedup > max_speedup:
                max_speedup = speedup
                max_speedup_n = n
                
            if speedup > 1.0 and crossover_n is None:
                crossover_n = n
    
    print(f"🚀 Best CUDA speedup: {max_speedup:.2f}x at N={int(max_speedup_n):,}")
    
    if crossover_n:
        print(f"🎯 CUDA becomes faster than sequential at N={int(crossover_n):,}")
    
    # OpenMP analysis
    omp_data = df[df['Backend'] == 'omp']
    if not omp_data.empty:
        max_omp_speedup = 0
        max_omp_n = 0
        max_omp_threads = 0
        
        for _, omp_row in omp_data.iterrows():
            n = omp_row['N']
            threads = omp_row['Threads']
            omp_time = omp_row['Time_ms']
            seq_row = seq_data[seq_data['N'] == n]
            if not seq_row.empty:
                seq_time = seq_row['Time_ms'].iloc[0]
                speedup = seq_time / omp_time
                
                if speedup > max_omp_speedup:
                    max_omp_speedup = speedup
                    max_omp_n = n
                    max_omp_threads = threads
        
        print(f"⚡ Best OpenMP speedup: {max_omp_speedup:.2f}x at N={int(max_omp_n):,} with {int(max_omp_threads)} threads")
    
    # Scaling analysis
    largest_n = df['N'].max()
    largest_cuda = cuda_data[cuda_data['N'] == largest_n]['Time_ms'].iloc[0]
    largest_seq = seq_data[seq_data['N'] == largest_n]['Time_ms'].iloc[0]
    print(f"📈 At N={int(largest_n):,}: CUDA is {largest_seq/largest_cuda:.2f}x faster than sequential")
    
    print(f"⚠️  Optimization opportunity: GPU memory transfers limit current performance")