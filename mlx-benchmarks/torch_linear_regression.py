import torch
import time

def linear_regression():
    num_features = 100
    num_examples = 1_000
    num_iters = 10_000
    lr = 0.01

    # True parameters
    w_star = torch.randn(num_features)

    # Input Examples
    X = torch.randn(num_examples, num_features)

    # Noisy labels
    eps = 1e-2
    y = X @ w_star + eps

    def loss_fn(w):
        return 0.5 * torch.mean((X @ w - y) ** 2)

    w = 1e-2 * torch.randn(num_features)

    for _ in range(num_iters):
        w.requires_grad_(True)
        loss = loss_fn(w)
        grad = torch.autograd.grad(loss, w)[0]
        with torch.no_grad():
            w = w - lr * grad
        w = w.detach()

    loss = loss_fn(w)
    error_norm = torch.sum((w - w_star) ** 2).item() ** 0.5

    print(
        f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f}, "
    )
    # Should print something close to: Loss 0.00005, |w-w*| = 0.00364

start = time.perf_counter()
linear_regression()
end = time.perf_counter()

print(f"Elapsed: {end - start:.6f} seconds")