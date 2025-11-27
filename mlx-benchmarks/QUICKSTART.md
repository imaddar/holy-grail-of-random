# MLX Neural Network Benchmark - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- macOS with Apple Silicon (M1, M2, M3+)
- Python 3.13+
- Internet connection (for downloading MNIST dataset)

### Step 1: Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install mlx numpy torch matplotlib
```

### Step 2: Run Your First Benchmark

```bash
# Simple MLX benchmark
python mlx_nn_benchmark.py
```

Expected output:
```
======================================================================
MLX Neural Network Benchmark Suite
======================================================================

Configuration:
  Layers: 2
  Hidden dimension: 32
  Batch size: 256
  Epochs: 10
  Learning rate: 0.1

Loading MNIST dataset...
  Train images: (60000, 784)
  Test images: (10000, 784)

======================================================================
Benchmark 1: Training Performance
======================================================================
Epoch 1/10: Loss=1.1024, Accuracy=0.8485, Time=0.630s
Epoch 2/10: Loss=0.3831, Accuracy=0.9038, Time=0.286s
...
```

**Time to complete**: ~1-2 minutes

### Step 3: Compare with PyTorch

```bash
python benchmark_comparison.py
```

This will run the same model in both MLX and PyTorch and show you:
- Training speed comparison
- Inference performance comparison
- Accuracy comparison

**Time to complete**: ~3-5 minutes

### Step 4: Generate Visualizations

```bash
uv run python visualize_results.py
```

This creates beautiful charts showing:
- Training vs inference performance
- MLX vs PyTorch speedup
- Batch size analysis
- Architecture scaling

**Time to complete**: ~30 seconds

## 📊 What You'll See

### MLX Benchmark Results
```
Training Summary:
  Total time: 3.23s
  Avg epoch time: 0.32s
  Samples/sec: 186,019
  Final accuracy: 0.9562

Inference Performance:
  Batch 1:   0.68ms (1,462 samples/sec)
  Batch 256: 0.44ms (577,264 samples/sec)
  Batch 512: 0.64ms (796,609 samples/sec)
```

### MLX vs PyTorch Comparison
```
Training Performance:
  MLX:     3.110s
  PyTorch: 8.296s
  → MLX is 2.67x faster

Inference (batch 256):
  MLX:     0.40ms (647K samples/sec)
  PyTorch: 0.63ms (409K samples/sec)
  → MLX is 1.58x faster
```

## 🎯 Key Takeaways

✅ **MLX is 2-3x faster** for training on Apple Silicon
✅ **MLX is 1.3-1.7x faster** for inference
✅ **Both achieve >94% accuracy** on MNIST
✅ **MLX has lower latency variance**

## 📁 Output Files

After running the benchmarks, you'll have:

```
benchmark_results.json                # Detailed MLX metrics
benchmark_comparison_results.json     # MLX vs PyTorch data
architecture_scaling.png              # Model size vs performance
batch_size_analysis.png               # Optimal batch size analysis
inference_comparison.png              # MLX vs PyTorch inference
training_comparison.png               # MLX vs PyTorch training
benchmark_summary.png                 # Comprehensive dashboard
```

## 🔧 Customization

### Change Model Architecture

Edit the configuration in the script:

```python
num_layers = 3      # Default: 2
hidden_dim = 64     # Default: 32
batch_size = 512    # Default: 256
num_epochs = 20     # Default: 10
```

### Benchmark Different Batch Sizes

In `mlx_nn_benchmark.py`, modify:

```python
batch_sizes = [1, 16, 32, 64, 128, 256, 512, 1024]
```

### Test More Architectures

In the benchmark script:

```python
architectures = [
    (1, 16),    # 1 layer, 16 hidden units
    (2, 32),    # 2 layers, 32 hidden units
    (3, 64),    # 3 layers, 64 hidden units
    (4, 128),   # 4 layers, 128 hidden units
    (5, 256),   # 5 layers, 256 hidden units
]
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mlx'"
```bash
# Solution: Install MLX
pip install mlx
# or
uv add mlx
```

### Issue: "MNIST dataset download fails"
```bash
# Solution: Check internet connection or use local MNIST
# The dataset will be downloaded to /tmp/ automatically
```

### Issue: "PyTorch MPS not available"
```bash
# Solution: Ensure you're on Apple Silicon with macOS 12.3+
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Issue: Visualizations not generating
```bash
# Solution: Install matplotlib
uv add matplotlib
# or
pip install matplotlib
```

## 📈 Understanding the Results

### Training Metrics
- **Total time**: End-to-end training duration
- **Epoch time**: Time per training epoch
- **Batch time**: Time to process one batch
- **Samples/sec**: Training throughput
- **Accuracy**: Test set accuracy

### Inference Metrics
- **Latency**: Time to process one batch
- **Throughput**: Samples processed per second
- **Std dev**: Timing variance (lower is better)
- **Speedup**: MLX performance vs PyTorch

## 🎓 Next Steps

1. **Read the full report**: Check `BENCHMARK_REPORT.md` for detailed analysis
2. **Explore the code**: See `mlx_nn.py` for the model implementation
3. **Try your own model**: Adapt the benchmark for your use case
4. **Share results**: Compare with community benchmarks

## 💡 Pro Tips

1. **Run benchmarks multiple times** for consistent results
2. **Close other applications** to reduce system noise
3. **Use consistent power settings** (plugged in, high performance)
4. **Monitor temperature** - thermal throttling affects results
5. **Check system activity** with Activity Monitor during runs

## 🆘 Need Help?

- Check `README.md` for comprehensive documentation
- Read `BENCHMARK_REPORT.md` for detailed analysis
- Review the source code for implementation details
- Open an issue for bugs or questions

## 📊 Typical Performance (M2 Pro)

| Task | MLX | PyTorch | Speedup |
|------|-----|---------|---------|
| Training (10 epochs) | ~3s | ~8s | 2.7x |
| Inference (256 batch) | 0.4ms | 0.6ms | 1.6x |
| Final Accuracy | 95.6% | 95.3% | - |

*Your results may vary based on hardware and system load*

## 🎉 Success!

You've now benchmarked MLX neural networks and compared them with PyTorch!

Key findings:
- ✅ MLX is significantly faster on Apple Silicon
- ✅ Both frameworks achieve similar accuracy
- ✅ MLX is ideal for Mac-based ML development
- ✅ Choose the right framework for your use case

Ready to dive deeper? Check out `README.md` for advanced usage!