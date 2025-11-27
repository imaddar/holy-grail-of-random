import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results():
    """Load benchmark results from JSON files"""
    with open("benchmark_results.json", "r") as f:
        mlx_results = json.load(f)

    with open("benchmark_comparison_results.json", "r") as f:
        comparison_results = json.load(f)

    return mlx_results, comparison_results


def plot_training_progression(mlx_results):
    """Plot training time progression over epochs"""
    # Check if detailed epoch data is available
    if (
        "epoch_losses" in mlx_results["training"]
        and "epoch_accuracies" in mlx_results["training"]
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(mlx_results["training"]["epoch_losses"]) + 1)
        losses = mlx_results["training"]["epoch_losses"]
        accuracies = mlx_results["training"]["epoch_accuracies"]

        # Plot loss
        ax1.plot(epochs, losses, "b-o", linewidth=2, markersize=6)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training Loss Over Epochs", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(epochs, accuracies, "g-o", linewidth=2, markersize=6)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Test Accuracy Over Epochs", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.8, 1.0])

        plt.tight_layout()
        plt.savefig("training_progression.png", dpi=300, bbox_inches="tight")
        print("Saved: training_progression.png")
        plt.close()
    else:
        print("Skipped: training_progression.png (detailed epoch data not available)")


def plot_inference_comparison(comparison_results):
    """Compare MLX vs PyTorch inference performance"""
    batch_sizes = [1, 32, 64, 128, 256, 512]

    mlx_times = []
    pytorch_times = []
    mlx_throughput = []
    pytorch_throughput = []

    for bs in batch_sizes:
        key = f"batch_{bs}"
        mlx_times.append(comparison_results["mlx"]["inference"][key]["avg_time"] * 1000)
        pytorch_times.append(
            comparison_results["pytorch"]["inference"][key]["avg_time"] * 1000
        )
        mlx_throughput.append(comparison_results["mlx"]["inference"][key]["throughput"])
        pytorch_throughput.append(
            comparison_results["pytorch"]["inference"][key]["throughput"]
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(batch_sizes))
    width = 0.35

    # Plot inference times
    bars1 = ax1.bar(x - width / 2, mlx_times, width, label="MLX", color="#2E86AB")
    bars2 = ax1.bar(
        x + width / 2, pytorch_times, width, label="PyTorch", color="#EE6352"
    )

    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Inference Time (ms)", fontsize=12)
    ax1.set_title("Inference Time Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot throughput
    ax2.plot(
        batch_sizes,
        np.array(mlx_throughput) / 1000,
        "o-",
        linewidth=2,
        markersize=8,
        label="MLX",
        color="#2E86AB",
    )
    ax2.plot(
        batch_sizes,
        np.array(pytorch_throughput) / 1000,
        "s-",
        linewidth=2,
        markersize=8,
        label="PyTorch",
        color="#EE6352",
    )

    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Throughput (K samples/sec)", fontsize=12)
    ax2.set_title("Inference Throughput Comparison", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig("inference_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved: inference_comparison.png")
    plt.close()


def plot_training_comparison(comparison_results):
    """Compare MLX vs PyTorch training performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Training time comparison
    frameworks = ["MLX", "PyTorch"]
    total_times = [
        comparison_results["mlx"]["training"]["total_time"],
        comparison_results["pytorch"]["training"]["total_time"],
    ]
    colors = ["#2E86AB", "#EE6352"]

    bars = ax1.bar(frameworks, total_times, color=colors, width=0.6)
    ax1.set_ylabel("Time (seconds)", fontsize=12)
    ax1.set_title("Total Training Time (10 Epochs)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Speedup visualization
    speedup = comparison_results["comparison"]["training_speedup"]
    ax2.barh(["Training"], [speedup], color="#A23B72", height=0.5)
    ax2.set_xlabel("Speedup Factor", fontsize=12)
    ax2.set_title("MLX Training Speedup vs PyTorch", fontsize=14, fontweight="bold")
    ax2.set_xlim([0, 3])
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.text(
        speedup + 0.1, 0, f"{speedup:.2f}x", va="center", fontsize=14, fontweight="bold"
    )

    # Throughput comparison
    throughputs = [
        comparison_results["mlx"]["training"]["samples_per_second"] / 1000,
        comparison_results["pytorch"]["training"]["samples_per_second"] / 1000,
    ]

    bars = ax3.bar(frameworks, throughputs, color=colors, width=0.6)
    ax3.set_ylabel("Throughput (K samples/sec)", fontsize=12)
    ax3.set_title("Training Throughput", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}K",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Accuracy comparison
    accuracies = [
        comparison_results["mlx"]["training"]["final_accuracy"] * 100,
        comparison_results["pytorch"]["training"]["final_accuracy"] * 100,
    ]

    bars = ax4.bar(frameworks, accuracies, color=colors, width=0.6)
    ax4.set_ylabel("Accuracy (%)", fontsize=12)
    ax4.set_title("Final Test Accuracy", fontsize=14, fontweight="bold")
    ax4.set_ylim([90, 100])
    ax4.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("training_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved: training_comparison.png")
    plt.close()


def plot_architecture_scaling(mlx_results):
    """Plot performance scaling across different architectures"""
    if "architectures" not in mlx_results:
        print("Skipped: architecture_scaling.png (architecture data not available)")
        return

    architectures = mlx_results["architectures"]

    configs = []
    params = []
    throughputs = []
    forward_times = []

    for name, data in architectures.items():
        configs.append(f"{data['num_layers']}L-{data['hidden_dim']}H")
        params.append(data["num_parameters"])
        throughputs.append(data["throughput"] / 1000)
        forward_times.append(data["avg_forward_time"] * 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput vs parameters
    ax1.scatter(
        params,
        throughputs,
        s=200,
        c=range(len(params)),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    for i, config in enumerate(configs):
        ax1.annotate(
            config,
            (params[i], throughputs[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax1.set_xlabel("Number of Parameters", fontsize=12)
    ax1.set_ylabel("Throughput (K samples/sec)", fontsize=12)
    ax1.set_title("Model Throughput vs Size", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    # Forward pass time comparison
    x = np.arange(len(configs))
    bars = ax2.bar(x, forward_times, color="#F18F01", width=0.6)
    ax2.set_xlabel("Architecture", fontsize=12)
    ax2.set_ylabel("Forward Pass Time (ms)", fontsize=12)
    ax2.set_title("Forward Pass Performance", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("architecture_scaling.png", dpi=300, bbox_inches="tight")
    print("Saved: architecture_scaling.png")
    plt.close()


def plot_batch_size_analysis(mlx_results):
    """Analyze performance across batch sizes"""
    if "inference" not in mlx_results:
        print("Skipped: batch_size_analysis.png (inference data not available)")
        return

    inference = mlx_results["inference"]

    batch_sizes = []
    avg_times = []
    throughputs = []
    accuracies = []

    for key in sorted(inference.keys(), key=lambda x: int(x.split("_")[1])):
        batch_size = int(key.split("_")[1])
        batch_sizes.append(batch_size)
        avg_times.append(inference[key]["avg_time"] * 1000)
        throughputs.append(inference[key]["throughput"] / 1000)
        if "avg_accuracy" in inference[key]:
            accuracies.append(inference[key]["avg_accuracy"] * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Latency
    ax1.plot(batch_sizes, avg_times, "o-", linewidth=2, markersize=8, color="#A23B72")
    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Inference Latency vs Batch Size", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Throughput with efficiency markers
    ax2.plot(batch_sizes, throughputs, "s-", linewidth=2, markersize=8, color="#2E86AB")
    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Throughput (K samples/sec)", fontsize=12)
    ax2.set_title("Inference Throughput vs Batch Size", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log", base=2)

    # Mark optimal batch size
    optimal_idx = throughputs.index(max(throughputs))
    ax2.scatter(
        [batch_sizes[optimal_idx]],
        [throughputs[optimal_idx]],
        s=300,
        c="red",
        marker="*",
        zorder=5,
        label=f"Optimal: {batch_sizes[optimal_idx]}",
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig("batch_size_analysis.png", dpi=300, bbox_inches="tight")
    print("Saved: batch_size_analysis.png")
    plt.close()


def create_summary_figure(comparison_results):
    """Create a comprehensive summary figure"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(
        "MLX Neural Network Benchmark Summary", fontsize=18, fontweight="bold", y=0.98
    )

    # 1. Training time comparison (large)
    ax1 = fig.add_subplot(gs[0, :2])
    frameworks = ["MLX", "PyTorch"]
    times = [
        comparison_results["mlx"]["training"]["total_time"],
        comparison_results["pytorch"]["training"]["total_time"],
    ]
    bars = ax1.barh(frameworks, times, color=["#2E86AB", "#EE6352"], height=0.5)
    ax1.set_xlabel("Training Time (seconds)", fontsize=11)
    ax1.set_title(
        "Training Performance (10 Epochs on MNIST)", fontsize=13, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="x")

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(
            width + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}s",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    # 2. Speedup metric
    ax2 = fig.add_subplot(gs[0, 2])
    speedup = comparison_results["comparison"]["training_speedup"]
    ax2.text(
        0.5,
        0.6,
        f"{speedup:.2f}×",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color="#A23B72",
    )
    ax2.text(0.5, 0.3, "MLX Speedup", ha="center", va="center", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    # 3. Throughput comparison
    ax3 = fig.add_subplot(gs[1, 0])
    throughputs = [
        comparison_results["mlx"]["training"]["samples_per_second"] / 1000,
        comparison_results["pytorch"]["training"]["samples_per_second"] / 1000,
    ]
    ax3.bar(frameworks, throughputs, color=["#2E86AB", "#EE6352"], width=0.6)
    ax3.set_ylabel("K samples/sec", fontsize=10)
    ax3.set_title("Training Throughput", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Accuracy comparison
    ax4 = fig.add_subplot(gs[1, 1])
    accuracies = [
        comparison_results["mlx"]["training"]["final_accuracy"] * 100,
        comparison_results["pytorch"]["training"]["final_accuracy"] * 100,
    ]
    ax4.bar(frameworks, accuracies, color=["#2E86AB", "#EE6352"], width=0.6)
    ax4.set_ylabel("Accuracy (%)", fontsize=10)
    ax4.set_title("Final Test Accuracy", fontsize=11, fontweight="bold")
    ax4.set_ylim([90, 100])
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Inference speedup across batches
    ax5 = fig.add_subplot(gs[1, 2])
    batch_sizes = [1, 32, 64, 128, 256, 512]
    speedups = []
    for bs in batch_sizes:
        key = f"batch_{bs}"
        mlx_time = comparison_results["mlx"]["inference"][key]["avg_time"]
        pytorch_time = comparison_results["pytorch"]["inference"][key]["avg_time"]
        speedups.append(pytorch_time / mlx_time)

    ax5.plot(
        range(len(batch_sizes)),
        speedups,
        "o-",
        linewidth=2,
        markersize=6,
        color="#A23B72",
    )
    ax5.set_xticks(range(len(batch_sizes)))
    ax5.set_xticklabels(batch_sizes, fontsize=9)
    ax5.set_xlabel("Batch Size", fontsize=10)
    ax5.set_ylabel("Speedup Factor", fontsize=10)
    ax5.set_title("Inference Speedup", fontsize=11, fontweight="bold")
    ax5.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax5.grid(True, alpha=0.3)

    # 6. MLX inference throughput
    ax6 = fig.add_subplot(gs[2, :])
    mlx_throughputs = []
    for bs in batch_sizes:
        key = f"batch_{bs}"
        mlx_throughputs.append(
            comparison_results["mlx"]["inference"][key]["throughput"] / 1000
        )

    ax6.plot(
        batch_sizes, mlx_throughputs, "o-", linewidth=2, markersize=8, color="#2E86AB"
    )
    ax6.set_xlabel("Batch Size", fontsize=11)
    ax6.set_ylabel("Throughput (K samples/sec)", fontsize=11)
    ax6.set_title("MLX Inference Throughput Scaling", fontsize=13, fontweight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale("log", base=2)

    # Highlight optimal batch size
    optimal_idx = mlx_throughputs.index(max(mlx_throughputs))
    ax6.scatter(
        [batch_sizes[optimal_idx]],
        [mlx_throughputs[optimal_idx]],
        s=300,
        c="red",
        marker="*",
        zorder=5,
    )
    ax6.annotate(
        f"Optimal: {batch_sizes[optimal_idx]}",
        xy=(batch_sizes[optimal_idx], mlx_throughputs[optimal_idx]),
        xytext=(20, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    plt.savefig("benchmark_summary.png", dpi=300, bbox_inches="tight")
    print("Saved: benchmark_summary.png")
    plt.close()


def main():
    print("=" * 70)
    print("Generating Benchmark Visualizations")
    print("=" * 70)

    # Check if result files exist
    has_mlx_results = Path("benchmark_results.json").exists()
    has_comparison = Path("benchmark_comparison_results.json").exists()

    if not has_mlx_results and not has_comparison:
        print("Error: No benchmark result files found.")
        print("Run mlx_nn_benchmark.py and/or benchmark_comparison.py first.")
        return

    print("\nLoading results...")
    mlx_results = None
    comparison_results = None

    if has_mlx_results:
        with open("benchmark_results.json", "r") as f:
            mlx_results = json.load(f)
        print("  ✓ Loaded benchmark_results.json")

    if has_comparison:
        with open("benchmark_comparison_results.json", "r") as f:
            comparison_results = json.load(f)
        print("  ✓ Loaded benchmark_comparison_results.json")

    print("\nGenerating visualizations...")
    print("-" * 70)

    generated_files = []

    # Generate plots based on available data
    if mlx_results:
        plot_training_progression(mlx_results)
        plot_architecture_scaling(mlx_results)
        plot_batch_size_analysis(mlx_results)

    if comparison_results:
        plot_inference_comparison(comparison_results)
        plot_training_comparison(comparison_results)
        create_summary_figure(comparison_results)

    print("-" * 70)
    print("\n✓ Visualization generation complete!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
