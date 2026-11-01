import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for remote server
import matplotlib.pyplot as plt

def main():
    csv_path = "benchmarks/benchmark_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run 'make bench_dataset' first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}:")
    print(df)

    # Setup figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    threads = df['threads']

    # -------------------------------------------------------------
    # Plot 1: Execution Time Breakdown (Read vs. Norm vs. Total)
    # -------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(threads, df['read_ms'], marker='s', color='#e74c3c', label='File Read (fread, Serial)', linewidth=2)
    ax1.plot(threads, df['norm_ms'], marker='o', color='#3498db', label='Normalization (Math Loop)', linewidth=2)
    ax1.plot(threads, df['total_ms'], marker='^', color='#2c3e50', label='Total Execution Time', linewidth=2, linestyle='--')
    ax1.set_title('Execution Time Breakdown', fontsize=13, fontweight='bold')
    ax1.set_xlabel('OpenMP Threads', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_xticks(threads)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9)

    # -------------------------------------------------------------
    # Plot 2: Speedup (Total vs. Norm vs. Ideal)
    # -------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(threads, df['norm_speedup'], marker='o', color='#27ae60', label='Norm Math Speedup', linewidth=2)
    ax2.plot(threads, df['total_speedup'], marker='^', color='#8e44ad', label='Total Speedup (Amdahl Limited)', linewidth=2)
    ax2.plot(threads, threads, linestyle=':', color='#95a5a6', label='Ideal Linear Speedup (y = x)', linewidth=1.5)
    ax2.set_title('Speedup Analysis (Amdahl\'s Law)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('OpenMP Threads', fontsize=11)
    ax2.set_ylabel('Speedup Factor (x)', fontsize=11)
    ax2.set_xticks(threads)
    ax2.set_ylim(0, max(max(df['norm_speedup']) * 1.5, 5))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9)

    # -------------------------------------------------------------
    # Plot 3: Throughput (Mpix/s)
    # -------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(threads, df['throughput_mpix_s'], marker='D', color='#d35400', label='Throughput', linewidth=2)
    ax3.set_title('Throughput Scaling', fontsize=13, fontweight='bold')
    ax3.set_xlabel('OpenMP Threads', fontsize=11)
    ax3.set_ylabel('Throughput (Million Pixels / sec)', fontsize=11)
    ax3.set_xticks(threads)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(fontsize=9)

    plt.suptitle('MNIST Loader: OpenMP Thread Scaling Benchmark (Intel i7-13700K)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_image = "benchmarks/scaling_analysis.png"
    plt.savefig(out_image, bbox_inches='tight')
    print(f"\n? Successfully generated plot: {out_image}")

if __name__ == '__main__':
    main()
