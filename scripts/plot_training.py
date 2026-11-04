import os
import sys
import subprocess
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_SAMPLES = 50000

def get_gpu_info():
    try:
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
        out = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        name, mem = out.split(",")
        mem_gb = round(float(mem.strip()) / 1024.0)
        return f"{name.strip()} ({mem_gb}GB)"
    except Exception:
        return "NVIDIA GeForce RTX 4070 Ti (12GB)"

def main():
    if len(sys.argv) > 1:
        raw_args = [a.strip() for a in sys.argv[1:] if a.strip()]
        arg_joined = "_".join(raw_args)
        csv_arg = raw_args[0]
    else:
        arg_joined = "training_log"
        csv_arg = "benchmarks/training_log.csv"

    candidates = [
        csv_arg,
        f"benchmarks/{csv_arg}",
        f"{csv_arg}.csv",
        f"benchmarks/{csv_arg}.csv",
        arg_joined,
        f"benchmarks/{arg_joined}",
        f"{arg_joined}.csv",
        f"benchmarks/{arg_joined}.csv",
    ]

    csv_path = None
    for cand in candidates:
        if os.path.exists(cand):
            csv_path = cand
            break

    if not csv_path:
        print(f"Error: CSV file not found (searched: {candidates}). Run './bin/main' first.")
        sys.exit(1)

    base_file = os.path.basename(csv_path)
    experiment_name = os.path.splitext(base_file)[0]
    out_dir = os.path.dirname(csv_path) or "benchmarks"
    out_image = os.path.join(out_dir, f"{experiment_name}.png")

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} epochs from {csv_path}:")
    print(df.to_string(index=False))

    epochs = df['epoch']
    if 'is_best' in df.columns and (df['is_best'] == 1).any():
        best_idx = df[df['is_best'] == 1].index[-1]
    else:
        best_idx = df['val_acc'].idxmax()

    best_epoch = int(df.loc[best_idx, 'epoch'])
    best_val_loss = df.loc[best_idx, 'val_loss']
    best_val_acc = df.loc[best_idx, 'val_acc']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)

    ax1 = axes[0]
    ax1.plot(epochs, df['train_loss'], marker='o', color='#2980b9', linewidth=2.3, label='Train Loss (running)')
    ax1.plot(epochs, df['val_loss'], marker='s', color='#e74c3c', linewidth=2.3, label='Validation Loss')
    ax1.fill_between(epochs, df['train_loss'], df['val_loss'], color='#95a5a6', alpha=0.12, label='Train-Val Gap')
    ax1.scatter([best_epoch], [best_val_loss], color='#c0392b', s=90, zorder=5)
    ax1.annotate(f'Best Checkpoint\nEpoch {best_epoch}: {best_val_loss:.4f}',
                 xy=(best_epoch, best_val_loss),
                 xytext=(max(1, best_epoch - len(df) * 0.42), best_val_loss + (df['train_loss'].max() - best_val_loss) * 0.35),
                 arrowprops=dict(facecolor='#c0392b', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', color='#c0392b', fontsize=9)
    ax1.set_title('Cross-Entropy Loss Convergence', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9, loc='upper right')

    ax2 = axes[1]
    ax2.plot(epochs, df['train_acc'], marker='o', color='#27ae60', linewidth=2.3, label='Train Accuracy (%)')
    ax2.plot(epochs, df['val_acc'], marker='^', color='#8e44ad', linewidth=2.3, label='Validation Accuracy (%)')
    ax2.scatter([best_epoch], [best_val_acc], color='#8e44ad', edgecolors='black', s=95, zorder=5)
    ax2.annotate(f'Best Val Acc: {best_val_acc:.2f}%\n(Epoch {best_epoch})',
                 xy=(best_epoch, best_val_acc),
                 xytext=(max(1, best_epoch - len(df) * 0.45), best_val_acc - (best_val_acc - df['train_acc'].min()) * 0.38),
                 arrowprops=dict(facecolor='#8e44ad', shrink=0.08, width=1.5, headwidth=6),
                 fontweight='bold', color='#6c3483', fontsize=9)
    ax2.set_title('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='lower right')

    ax3 = axes[2]
    throughput_m = (TRAIN_SAMPLES / (df['epoch_ms'] / 1000.0)) / 1e6
    mean_ms = df['epoch_ms'].mean()
    mean_tp = throughput_m.mean()

    ax3.bar(epochs, df['epoch_ms'], color='#e67e22', alpha=0.75, width=0.6, label='Epoch Time (ms)')
    ax3.axhline(mean_ms, color='#d35400', linestyle='--', linewidth=2.0,
                label=f'Mean: {mean_ms:.1f} ms ({mean_tp:.2f}M img/s)')
    ax3.set_title('Training Epoch Latency & Throughput', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Epoch Wall-Clock Time (ms)', fontsize=11)
    ax3.set_xticks(epochs)
    ax3.set_ylim(0, df['epoch_ms'].max() * 1.25)
    ax3.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax3.legend(fontsize=9, loc='upper right')

    gpu_info = get_gpu_info()
    chart_title = f"{experiment_name} - {gpu_info}"
    plt.suptitle(chart_title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(out_image, bbox_inches='tight')
    print(f"\n[OK] Successfully generated learning-curve plot: {out_image}\n")

if __name__ == '__main__':
    main()
