import os
import sys
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def get_gpu_info():
    try:
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        name, mem = out.split(",")
        mem_gb = round(float(mem.strip()) / 1024.0)
        return f"{name.strip()} ({mem_gb}GB)"
    except Exception:
        return "NVIDIA GeForce RTX 4070 Ti (12GB)"

def format_log2_xaxis(ax, batch_sizes):
    ax.set_xscale('log', base=2)
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.tick_params(axis='x', rotation=45, labelsize=8.5)

def main():
    if len(sys.argv) > 1:
        raw_args = [a.strip() for a in sys.argv[1:] if a.strip()]
        arg_joined = "_".join(raw_args)
        csv_arg = raw_args[0]
    else:
        arg_joined = "batch_scaling"
        csv_arg = "batch_scaling.csv"

    candidates = [
        csv_arg,
        f"benchmarks/gpu_benchmarks/{csv_arg}",
        f"{csv_arg}.csv",
        f"benchmarks/gpu_benchmarks/{csv_arg}.csv",
        f"benchmarks/{csv_arg}",
        f"benchmarks/{csv_arg}.csv",
        arg_joined,
        f"benchmarks/gpu_benchmarks/{arg_joined}",
        f"{arg_joined}.csv",
        f"benchmarks/gpu_benchmarks/{arg_joined}.csv",
        f"benchmarks/{arg_joined}",
        f"benchmarks/{arg_joined}.csv"
    ]

    csv_path = None
    for cand in candidates:
        if os.path.exists(cand):
            csv_path = cand
            break

    if not csv_path:
        print(f"Error: CSV file not found (searched: {candidates}). Run benchmark first.")
        sys.exit(1)

    base_file = os.path.basename(csv_path)
    experiment_name = os.path.splitext(base_file)[0]
    out_dir = os.path.dirname(csv_path) or "benchmarks/gpu_benchmarks"
    out_image = os.path.join(out_dir, f"{experiment_name}.png")

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from {csv_path}:")
    print(df[['batch_size', 'fwd_ms', 'bwd_ms', 'total_ms', 'throughput_img_s', 'throughput_speedup', 'tflops']].to_string(index=False))

    batch_sizes = df['batch_size'].tolist()
    has_l2_spill = max(batch_sizes) >= 8192

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4), dpi=150)

    ax1 = axes[0]
    tp_m = df['throughput_img_s'] / 1e6
    ax1.plot(batch_sizes, tp_m, marker='o', color='#2980b9', linewidth=2.5, label='Throughput (Million Img/s)')
    ax1.fill_between(batch_sizes, 0, tp_m, color='#3498db', alpha=0.14)
    if has_l2_spill:
        ax1.axvline(9500, color='#7f8c8d', linestyle='--', linewidth=1.3, alpha=0.85, label='48MB L2 Cache Limit')

    peak_idx = df['throughput_img_s'].idxmax()
    peak_b = int(df.loc[peak_idx, 'batch_size'])
    peak_tp = tp_m.loc[peak_idx]
    peak_sp = df.loc[peak_idx, 'throughput_speedup']
    ax1.scatter([peak_b], [peak_tp], color='#e74c3c', s=85, zorder=5)
    ax1.annotate(f'Peak: {peak_tp:.2f}M img/s\n({peak_sp:.1f}x vs B=1 @ B={peak_b})',
                 xy=(peak_b, peak_tp),
                 xytext=(max(batch_sizes[0], peak_b / 14.0), peak_tp * 0.52),
                 arrowprops=dict(facecolor='#e74c3c', shrink=0.08, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.9),
                 fontweight='bold', color='#c0392b', fontsize=9)

    ax1.set_title('Throughput Scaling (Images / sec)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Batch Size B (log2 scale)', fontsize=11)
    ax1.set_ylabel('Throughput (Million Images / sec)', fontsize=11)
    format_log2_xaxis(ax1, batch_sizes)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=8.5, loc='upper left')

    ax2 = axes[1]
    ax2.plot(batch_sizes, df['fwd_ms'], marker='s', color='#27ae60', linewidth=2, label='Forward Pass (ms)')
    ax2.plot(batch_sizes, df['bwd_ms'], marker='^', color='#e67e22', linewidth=2, label='Backward + SGD (ms)')
    ax2.plot(batch_sizes, df['total_ms'], marker='D', color='#8e44ad', linewidth=2.4, linestyle='--', label='Total Step Latency (ms)')
    if has_l2_spill:
        ax2.axvline(9500, color='#7f8c8d', linestyle='--', linewidth=1.3, alpha=0.85, label='48MB L2 Cache Limit')

    ax2.set_title('Step Latency Breakdown (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Batch Size B (log2 scale)', fontsize=11)
    ax2.set_ylabel('Execution Time (ms)', fontsize=11)
    format_log2_xaxis(ax2, batch_sizes)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=8.5, loc='upper left')

    ax3 = axes[2]
    ax3_r = ax3.twinx()

    l1 = ax3.plot(batch_sizes, df['tflops'], marker='o', color='#8e44ad', linewidth=2.5, label='Achieved FP32 (TFLOP/s)')
    l2 = ax3_r.plot(batch_sizes, df['throughput_speedup'], marker='h', color='#d35400', linewidth=1.8, linestyle=':', alpha=0.85, label='Speedup vs B=1 (x)')
    if has_l2_spill:
        l3 = ax3.axvline(9500, color='#7f8c8d', linestyle='--', linewidth=1.3, alpha=0.85, label='48MB L2 Cache Limit')

    peak_tf_idx = df['tflops'].idxmax()
    peak_tf_b = int(df.loc[peak_tf_idx, 'batch_size'])
    peak_tf = df.loc[peak_tf_idx, 'tflops']
    ax3.scatter([peak_tf_b], [peak_tf], color='#8e44ad', edgecolors='black', s=85, zorder=5)
    ax3.annotate(f'Peak: {peak_tf:.2f} TFLOP/s\n(@ B={peak_tf_b})',
                 xy=(peak_tf_b, peak_tf),
                 xytext=(max(batch_sizes[0], peak_tf_b / 12.0), peak_tf * 0.50),
                 arrowprops=dict(facecolor='#8e44ad', shrink=0.08, width=1.5, headwidth=6),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#8e44ad', alpha=0.9),
                 fontweight='bold', color='#6c3483', fontsize=9)

    ax3.set_title('Compute Throughput (TFLOP/s) & Speedup', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Batch Size B (log2 scale)', fontsize=11)
    ax3.set_ylabel('Achieved Compute (FP32 TFLOP/s)', fontsize=11, color='#6c3483')
    ax3_r.set_ylabel('Throughput Speedup vs B=1 (x)', fontsize=10.5, color='#b94a00')
    ax3.tick_params(axis='y', labelcolor='#6c3483')
    ax3_r.tick_params(axis='y', labelcolor='#b94a00')
    format_log2_xaxis(ax3, batch_sizes)
    ax3.grid(True, linestyle=':', alpha=0.6)

    lines = l1 + l2 + ([l3] if has_l2_spill else [])
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=8.5, loc='upper left')

    gpu_info = get_gpu_info()
    chart_title = f"{experiment_name} - {gpu_info}"
    plt.suptitle(chart_title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(out_image, bbox_inches='tight')
    print(f"\n[OK] Successfully generated GPU plot: {out_image}\n")

if __name__ == '__main__':
    main()
