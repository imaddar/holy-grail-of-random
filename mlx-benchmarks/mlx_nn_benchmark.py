import json
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mnist


class MLP(nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.maximum(l(x), 0.0)
        return self.layers[-1](x)


def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def benchmark_training(
    model: nn.Module,
    train_images: mx.array,
    train_labels: mx.array,
    test_images: mx.array,
    test_labels: mx.array,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
) -> Dict:
    """Benchmark training performance"""

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    # Track metrics
    epoch_times = []
    epoch_losses = []
    epoch_accuracies = []
    batch_times = []

    total_start = time.perf_counter()

    for e in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        batch_count = 0

        for X, y in batch_iterate(batch_size, train_images, train_labels):
            batch_start = time.perf_counter()

            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            batch_end = time.perf_counter()
            batch_times.append(batch_end - batch_start)

            epoch_loss_sum += loss.item()
            batch_count += 1

        # Evaluate on test set
        test_accuracy = eval_fn(model, test_images, test_labels)
        mx.eval(test_accuracy)

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        epoch_losses.append(epoch_loss_sum / batch_count)
        epoch_accuracies.append(test_accuracy.item())

        print(
            f"Epoch {e + 1}/{num_epochs}: "
            f"Loss={epoch_losses[-1]:.4f}, "
            f"Accuracy={epoch_accuracies[-1]:.4f}, "
            f"Time={epoch_time:.3f}s"
        )

    total_end = time.perf_counter()
    total_time = total_end - total_start

    return {
        "total_time": total_time,
        "epoch_times": epoch_times,
        "avg_epoch_time": np.mean(epoch_times),
        "std_epoch_time": np.std(epoch_times),
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "final_accuracy": epoch_accuracies[-1],
        "batch_times": batch_times,
        "avg_batch_time": np.mean(batch_times),
        "std_batch_time": np.std(batch_times),
        "batches_per_second": len(batch_times) / total_time,
        "samples_per_second": len(train_images) * num_epochs / total_time,
    }


def benchmark_inference(
    model: nn.Module,
    test_images: mx.array,
    test_labels: mx.array,
    batch_sizes: List[int] = [1, 16, 32, 64, 128, 256, 512],
) -> Dict:
    """Benchmark inference performance with different batch sizes"""

    results = {}

    for batch_size in batch_sizes:
        times = []
        accuracies = []

        # Warm-up
        for _ in range(5):
            sample = test_images[:batch_size]
            _ = model(sample)
            mx.eval(model.parameters())

        # Actual benchmark
        num_iterations = min(100, len(test_images) // batch_size)

        for i in range(num_iterations):
            start_idx = (i * batch_size) % len(test_images)
            end_idx = start_idx + batch_size
            if end_idx > len(test_images):
                break

            X = test_images[start_idx:end_idx]
            y = test_labels[start_idx:end_idx]

            start = time.perf_counter()
            predictions = model(X)
            mx.eval(predictions)
            end = time.perf_counter()

            times.append(end - start)
            accuracy = mx.mean(mx.argmax(predictions, axis=1) == y)
            accuracies.append(accuracy.item())

        results[f"batch_{batch_size}"] = {
            "avg_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "throughput": batch_size / np.mean(times),  # samples/second
            "avg_accuracy": np.mean(accuracies),
        }

        print(
            f"Batch size {batch_size}: "
            f"{np.mean(times) * 1000:.2f}ms ± {np.std(times) * 1000:.2f}ms, "
            f"Throughput: {batch_size / np.mean(times):.1f} samples/sec"
        )

    return results


def benchmark_model_sizes(
    input_dim: int, output_dim: int, configurations: List[Tuple[int, int]]
) -> Dict:
    """Benchmark different model architectures"""

    results = {}

    for num_layers, hidden_dim in configurations:
        model_name = f"layers_{num_layers}_hidden_{hidden_dim}"
        print(f"\nBenchmarking {model_name}...")

        model = MLP(num_layers, input_dim, hidden_dim, output_dim)
        mx.eval(model.parameters())

        # Count parameters
        num_params = sum(
            p.size for layer in model.layers for p in [layer.weight, layer.bias]
        )

        # Quick forward pass benchmark
        test_input = mx.random.normal((256, input_dim))

        # Warm-up
        for _ in range(10):
            _ = model(test_input)
            mx.eval(model.parameters())

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            output = model(test_input)
            mx.eval(output)
            end = time.perf_counter()
            times.append(end - start)

        results[model_name] = {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "num_parameters": num_params,
            "avg_forward_time": np.mean(times),
            "std_forward_time": np.std(times),
            "throughput": 256 / np.mean(times),
        }

        print(f"  Parameters: {num_params:,}")
        print(
            f"  Forward pass: {np.mean(times) * 1000:.3f}ms ± {np.std(times) * 1000:.3f}ms"
        )

    return results


def main():
    print("=" * 70)
    print("MLX Neural Network Benchmark Suite")
    print("=" * 70)

    # Configuration
    num_layers = 2
    hidden_dim = 32
    num_classes = 10
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1

    print("\nConfiguration:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Load data
    print("\nLoading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())
    print(f"  Train images: {train_images.shape}")
    print(f"  Test images: {test_images.shape}")

    input_dim = train_images.shape[-1]

    # ========================================================================
    # Benchmark 1: Training Performance
    # ========================================================================
    print("\n" + "=" * 70)
    print("Benchmark 1: Training Performance")
    print("=" * 70)

    model = MLP(num_layers, input_dim, hidden_dim, num_classes)
    mx.eval(model.parameters())

    training_results = benchmark_training(
        model,
        train_images,
        train_labels,
        test_images,
        test_labels,
        batch_size,
        num_epochs,
        learning_rate,
    )

    print(f"\nTraining Summary:")
    print(f"  Total time: {training_results['total_time']:.2f}s")
    print(
        f"  Avg epoch time: {training_results['avg_epoch_time']:.2f}s ± {training_results['std_epoch_time']:.2f}s"
    )
    print(
        f"  Avg batch time: {training_results['avg_batch_time'] * 1000:.2f}ms ± {training_results['std_batch_time'] * 1000:.2f}ms"
    )
    print(f"  Batches/sec: {training_results['batches_per_second']:.2f}")
    print(f"  Samples/sec: {training_results['samples_per_second']:.2f}")
    print(f"  Final accuracy: {training_results['final_accuracy']:.4f}")

    # ========================================================================
    # Benchmark 2: Inference Performance
    # ========================================================================
    print("\n" + "=" * 70)
    print("Benchmark 2: Inference Performance (Different Batch Sizes)")
    print("=" * 70)

    inference_results = benchmark_inference(model, test_images, test_labels)

    # ========================================================================
    # Benchmark 3: Model Architecture Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("Benchmark 3: Model Architecture Comparison")
    print("=" * 70)

    architectures = [
        (1, 16),
        (2, 32),
        (3, 64),
        (4, 128),
        (5, 256),
    ]

    architecture_results = benchmark_model_sizes(input_dim, num_classes, architectures)

    # ========================================================================
    # Summary Report
    # ========================================================================
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    all_results = {
        "configuration": {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "dataset": "MNIST",
            "framework": "MLX",
        },
        "training": {
            "total_time_seconds": training_results["total_time"],
            "avg_epoch_time_seconds": training_results["avg_epoch_time"],
            "avg_batch_time_ms": training_results["avg_batch_time"] * 1000,
            "samples_per_second": training_results["samples_per_second"],
            "final_accuracy": training_results["final_accuracy"],
        },
        "inference": inference_results,
        "architectures": architecture_results,
    }

    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nBenchmark results saved to {output_file}")

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
