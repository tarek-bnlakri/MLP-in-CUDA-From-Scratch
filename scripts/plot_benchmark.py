import os
import sys
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_cpu_info():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    name = line.split(":", 1)[1].strip()
                    return re.sub(r'\s+', ' ', name)
    except Exception:
        pass
    return "13th Gen Intel(R) Core(TM) i7-13700K"

def main():
    if len(sys.argv) > 1:
        raw_args = [a.strip() for a in sys.argv[1:] if a.strip()]
        arg_joined = "_".join(raw_args)
        csv_arg = raw_args[0]
    else:
        arg_joined = "benchmark_results"
        csv_arg = "benchmark_results.csv"

    candidates = [
        csv_arg,
        f"benchmarks/cpu_benchmarks/{csv_arg}",
        f"{csv_arg}.csv",
        f"benchmarks/cpu_benchmarks/{csv_arg}.csv",
        f"benchmarks/{csv_arg}",
        f"benchmarks/{csv_arg}.csv",
        arg_joined,
        f"benchmarks/cpu_benchmarks/{arg_joined}",
        f"{arg_joined}.csv",
        f"benchmarks/cpu_benchmarks/{arg_joined}.csv",
        f"benchmarks/{arg_joined}",
        f"benchmarks/{arg_joined}.csv"
    ]

    csv_path = next((c for c in candidates if os.path.exists(c)), None)
    if not csv_path:
        print(f"Error: CSV file not found (searched: {candidates}).")
        return

    experiment_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = os.path.dirname(csv_path) or "benchmarks/cpu_benchmarks"
    out_image = os.path.join(out_dir, f"{experiment_name}.png")

    df = pd.read_csv(csv_path)
    threads = df['threads']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    ax1 = axes[0]
    ax1.plot(threads, df['read_ms'], marker='s', color='#e74c3c', label='File Read (fread, Serial)', linewidth=2)
    ax1.plot(threads, df['norm_ms'], marker='o', color='#3498db', label='Normalization (OpenMP)', linewidth=2)
    ax1.plot(threads, df['total_ms'], marker='^', color='#2c3e50', label='Total Time', linewidth=2, linestyle='--')
    ax1.set_title('Execution Time Breakdown', fontsize=13, fontweight='bold')
    ax1.set_xlabel('OpenMP Threads', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_xticks(threads)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    ax2.plot(threads, df['norm_speedup'], marker='o', color='#27ae60', label='Normalization Speedup', linewidth=2)
    ax2.plot(threads, df['total_speedup'], marker='^', color='#8e44ad', label='Total Speedup (Amdahl)', linewidth=2)
    ax2.plot(threads, threads, linestyle=':', color='#95a5a6', label='Ideal Linear (y = x)', linewidth=1.5)
    ax2.set_title("Speedup Analysis (Amdahl's Law)", fontsize=13, fontweight='bold')
    ax2.set_xlabel('OpenMP Threads', fontsize=11)
    ax2.set_ylabel('Speedup Factor (x)', fontsize=11)
    ax2.set_xticks(threads)
    ax2.set_ylim(0, max(max(df['norm_speedup']) * 1.4, 5))
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9)

    ax3 = axes[2]
    ax3.plot(threads, df['throughput_mpix_s'], marker='D', color='#d35400', label='Throughput', linewidth=2)
    ax3.set_title('End-to-End Ingestion Throughput', fontsize=13, fontweight='bold')
    ax3.set_xlabel('OpenMP Threads', fontsize=11)
    ax3.set_ylabel('Throughput (Million Pixels / sec)', fontsize=11)
    ax3.set_xticks(threads)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(fontsize=9)

    cpu_name = get_cpu_info()
    plt.suptitle(f"{experiment_name} - {cpu_name} (16 Cores / 24 Threads)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_image, bbox_inches='tight')
    print(f"[OK] Generated plot: {out_image}")

if __name__ == '__main__':
    main()
