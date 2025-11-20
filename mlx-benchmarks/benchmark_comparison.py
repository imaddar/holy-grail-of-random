import json
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim
import numpy as np
import torch
import torch.nn as torch_nn
import torch.optim as torch_optim

import mnist


# ============================================================================
# MLX Implementation
# ============================================================================
class MLX_MLP(mlx_nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            mlx_nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = mx.maximum(l(x), 0.0)
        return self.layers[-1](x)


def mlx_loss_fn(model, X, y):
    return mx.mean(mlx_nn.losses.cross_entropy(model(X), y))


def mlx_eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def mlx_batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


# ============================================================================
# PyTorch Implementation
# ============================================================================
class PyTorch_MLP(torch_nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        layers = []
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for i in range(len(layer_sizes) - 1):
            layers.append(torch_nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(torch_nn.ReLU())
        self.network = torch_nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def torch_batch_iterate(batch_size, X, y):
    perm = torch.randperm(y.size(0))
    for s in range(0, y.size(0), batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


# ============================================================================
# Benchmark Functions
# ============================================================================
def benchmark_mlx_training(
    num_layers: int,
    hidden_dim: int,
    train_images: mx.array,
    train_labels: mx.array,
    test_images: mx.array,
    test_labels: mx.array,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
) -> Dict:
    """Benchmark MLX training"""
    input_dim = train_images.shape[-1]
    num_classes = 10

    model = MLX_MLP(num_layers, input_dim, hidden_dim, num_classes)
    mx.eval(model.parameters())

    loss_and_grad_fn = mlx_nn.value_and_grad(model, mlx_loss_fn)
    optimizer = mlx_optim.SGD(learning_rate=learning_rate)

    epoch_times = []
    epoch_losses = []
    epoch_accuracies = []

    total_start = time.perf_counter()

    for e in range(num_epochs):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        batch_count = 0

        for X, y in mlx_batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss_sum += loss.item()
            batch_count += 1

        test_accuracy = mlx_eval_fn(model, test_images, test_labels)
        mx.eval(test_accuracy)

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        epoch_losses.append(epoch_loss_sum / batch_count)
        epoch_accuracies.append(test_accuracy.item())

    total_end = time.perf_counter()
    total_time = total_end - total_start

    return {
        "total_time": total_time,
        "epoch_times": epoch_times,
        "avg_epoch_time": np.mean(epoch_times),
        "final_accuracy": epoch_accuracies[-1],
        "samples_per_second": len(train_images) * num_epochs / total_time,
    }


def benchmark_torch_training(
    num_layers: int,
    hidden_dim: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device: str = "mps",
) -> Dict:
    """Benchmark PyTorch training"""
    input_dim = train_images.shape[-1]
    num_classes = 10

    model = PyTorch_MLP(num_layers, input_dim, hidden_dim, num_classes).to(device)
    criterion = torch_nn.CrossEntropyLoss()
    optimizer = torch_optim.SGD(model.parameters(), lr=learning_rate)

    epoch_times = []
    epoch_losses = []
    epoch_accuracies = []

    total_start = time.perf_counter()

    for e in range(num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss_sum = 0.0
        batch_count = 0

        for X, y in torch_batch_iterate(batch_size, train_images, train_labels):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            batch_count += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_images)
            test_predictions = torch.argmax(test_outputs, dim=1)
            test_accuracy = (test_predictions == test_labels).float().mean().item()

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        epoch_losses.append(epoch_loss_sum / batch_count)
        epoch_accuracies.append(test_accuracy)

    total_end = time.perf_counter()
    total_time = total_end - total_start

    return {
        "total_time": total_time,
        "epoch_times": epoch_times,
        "avg_epoch_time": np.mean(epoch_times),
        "final_accuracy": epoch_accuracies[-1],
        "samples_per_second": len(train_images) * num_epochs / total_time,
    }


def benchmark_mlx_inference(
    model: MLX_MLP, test_images: mx.array, batch_size: int, num_iterations: int = 100
) -> Dict:
    """Benchmark MLX inference"""
    # Warm-up
    for _ in range(10):
        _ = model(test_images[:batch_size])
        mx.eval(model.parameters())

    times = []
    for i in range(num_iterations):
        start_idx = (i * batch_size) % len(test_images)
        end_idx = start_idx + batch_size
        if end_idx > len(test_images):
            break

        X = test_images[start_idx:end_idx]

        start = time.perf_counter()
        predictions = model(X)
        mx.eval(predictions)
        end = time.perf_counter()

        times.append(end - start)

    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "throughput": batch_size / np.mean(times),
    }


def benchmark_torch_inference(
    model: PyTorch_MLP,
    test_images: torch.Tensor,
    batch_size: int,
    device: str = "mps",
    num_iterations: int = 100,
) -> Dict:
    """Benchmark PyTorch inference"""
    model.eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_images[:batch_size])

    times = []
    with torch.no_grad():
        for i in range(num_iterations):
            start_idx = (i * batch_size) % len(test_images)
            end_idx = start_idx + batch_size
            if end_idx > len(test_images):
                break

            X = test_images[start_idx:end_idx]

            start = time.perf_counter()
            predictions = model(X)
            # Synchronize for accurate timing on MPS
            if device == "mps":
                torch.mps.synchronize()
            end = time.perf_counter()

            times.append(end - start)

    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "throughput": batch_size / np.mean(times),
    }


def main():
    print("=" * 80)
    print("MLX vs PyTorch Neural Network Benchmark Comparison")
    print("=" * 80)

    # Configuration
    num_layers = 2
    hidden_dim = 32
    batch_size = 256
    num_epochs = 10
    learning_rate = 1e-1
    device = "mps"  # Metal Performance Shaders for Apple Silicon

    print("\nConfiguration:")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")

    # Load data
    print("\nLoading MNIST dataset...")
    train_images_np, train_labels_np, test_images_np, test_labels_np = mnist.mnist()

    # Convert to MLX arrays
    train_images_mlx = mx.array(train_images_np)
    train_labels_mlx = mx.array(train_labels_np)
    test_images_mlx = mx.array(test_images_np)
    test_labels_mlx = mx.array(test_labels_np)

    # Convert to PyTorch tensors
    train_images_torch = torch.tensor(train_images_np, dtype=torch.float32).to(device)
    train_labels_torch = torch.tensor(train_labels_np, dtype=torch.long).to(device)
    test_images_torch = torch.tensor(test_images_np, dtype=torch.float32).to(device)
    test_labels_torch = torch.tensor(test_labels_np, dtype=torch.long).to(device)

    print(f"  Train images: {train_images_np.shape}")
    print(f"  Test images: {test_images_np.shape}")

    # ========================================================================
    # Benchmark 1: MLX Training
    # ========================================================================
    print("\n" + "=" * 80)
    print("Benchmark 1: MLX Training")
    print("=" * 80)

    mlx_training_results = benchmark_mlx_training(
        num_layers,
        hidden_dim,
        train_images_mlx,
        train_labels_mlx,
        test_images_mlx,
        test_labels_mlx,
        batch_size,
        num_epochs,
        learning_rate,
    )

    print(f"\nMLX Training Results:")
    print(f"  Total time: {mlx_training_results['total_time']:.3f}s")
    print(f"  Avg epoch time: {mlx_training_results['avg_epoch_time']:.3f}s")
    print(f"  Samples/sec: {mlx_training_results['samples_per_second']:.1f}")
    print(f"  Final accuracy: {mlx_training_results['final_accuracy']:.4f}")

    # ========================================================================
    # Benchmark 2: PyTorch Training
    # ========================================================================
    print("\n" + "=" * 80)
    print("Benchmark 2: PyTorch Training")
    print("=" * 80)

    torch_training_results = benchmark_torch_training(
        num_layers,
        hidden_dim,
        train_images_torch,
        train_labels_torch,
        test_images_torch,
        test_labels_torch,
        batch_size,
        num_epochs,
        learning_rate,
        device,
    )

    print(f"\nPyTorch Training Results:")
    print(f"  Total time: {torch_training_results['total_time']:.3f}s")
    print(f"  Avg epoch time: {torch_training_results['avg_epoch_time']:.3f}s")
    print(f"  Samples/sec: {torch_training_results['samples_per_second']:.1f}")
    print(f"  Final accuracy: {torch_training_results['final_accuracy']:.4f}")

    # ========================================================================
    # Benchmark 3: Inference Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("Benchmark 3: Inference Performance Comparison")
    print("=" * 80)

    # Create models for inference
    input_dim = train_images_np.shape[-1]
    num_classes = 10

    mlx_model = MLX_MLP(num_layers, input_dim, hidden_dim, num_classes)
    mx.eval(mlx_model.parameters())

    torch_model = PyTorch_MLP(num_layers, input_dim, hidden_dim, num_classes).to(device)

    inference_batch_sizes = [1, 32, 64, 128, 256, 512]
    mlx_inference_results = {}
    torch_inference_results = {}

    for bs in inference_batch_sizes:
        print(f"\nBatch size: {bs}")

        mlx_result = benchmark_mlx_inference(mlx_model, test_images_mlx, bs)
        torch_result = benchmark_torch_inference(
            torch_model, test_images_torch, bs, device
        )

        mlx_inference_results[f"batch_{bs}"] = mlx_result
        torch_inference_results[f"batch_{bs}"] = torch_result

        print(
            f"  MLX:     {mlx_result['avg_time'] * 1000:.2f}ms ± {mlx_result['std_time'] * 1000:.2f}ms "
            f"({mlx_result['throughput']:.0f} samples/sec)"
        )
        print(
            f"  PyTorch: {torch_result['avg_time'] * 1000:.2f}ms ± {torch_result['std_time'] * 1000:.2f}ms "
            f"({torch_result['throughput']:.0f} samples/sec)"
        )

        speedup = torch_result["avg_time"] / mlx_result["avg_time"]
        if speedup > 1:
            print(f"  → MLX is {speedup:.2f}x faster")
        else:
            print(f"  → PyTorch is {1 / speedup:.2f}x faster")

    # ========================================================================
    # Summary Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("Performance Comparison Summary")
    print("=" * 80)

    training_speedup = (
        torch_training_results["total_time"] / mlx_training_results["total_time"]
    )

    print("\nTraining Performance:")
    print(f"  MLX:     {mlx_training_results['total_time']:.3f}s")
    print(f"  PyTorch: {torch_training_results['total_time']:.3f}s")
    if training_speedup > 1:
        print(f"  → MLX is {training_speedup:.2f}x faster for training")
    else:
        print(f"  → PyTorch is {1 / training_speedup:.2f}x faster for training")

    print("\nThroughput (samples/sec):")
    print(f"  MLX:     {mlx_training_results['samples_per_second']:.1f}")
    print(f"  PyTorch: {torch_training_results['samples_per_second']:.1f}")

    print("\nFinal Accuracy:")
    print(f"  MLX:     {mlx_training_results['final_accuracy']:.4f}")
    print(f"  PyTorch: {torch_training_results['final_accuracy']:.4f}")

    # Save results
    all_results = {
        "configuration": {
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "device": device,
        },
        "mlx": {
            "training": mlx_training_results,
            "inference": mlx_inference_results,
        },
        "pytorch": {
            "training": torch_training_results,
            "inference": torch_inference_results,
        },
        "comparison": {
            "training_speedup": training_speedup,
            "framework_faster": "MLX" if training_speedup > 1 else "PyTorch",
        },
    }

    output_file = "benchmark_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
