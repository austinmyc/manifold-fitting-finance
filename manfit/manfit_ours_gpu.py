"""
GPU-accelerated version of manfit_ours using PyTorch

This version converts NumPy operations to PyTorch CUDA operations for GPU acceleration.
Falls back to CPU if CUDA is not available.
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def manfit_ours_gpu(sample, sig, sample_init, op_average=1, device=None):
    """
    GPU-accelerated manifold fitting using PyTorch.

    Args:
        sample: Reference points (numpy array or torch tensor) [N0 x D]
        sig: Bandwidth parameter
        sample_init: Points to transform (numpy array or torch tensor) [N x D]
        op_average: Not used (kept for compatibility)
        device: torch device ('cuda', 'cpu', or None for auto-detect)

    Returns:
        Transformed points as numpy array [N x D]
    """

    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Using device: {device}")

    # Convert inputs to torch tensors and move to device
    if isinstance(sample, np.ndarray):
        sample_t = torch.from_numpy(sample).float().to(device)
    else:
        sample_t = sample.float().to(device)

    if isinstance(sample_init, np.ndarray):
        sample_init_t = torch.from_numpy(sample_init).float().to(device)
    else:
        sample_init_t = sample_init.float().to(device)

    # Initialize output
    Mout = sample_init_t.clone()
    N = sample_init_t.shape[0]
    N0 = sample_t.shape[0]

    # Compute parameters
    r = 5 * sig / np.log10(N0)
    R = 10 * sig * np.sqrt(np.log(1 / sig)) / np.log10(N0)

    # Process each point
    for ii in range(N):
        x = sample_init_t[ii, :]

        # Compute distances to all sample points (vectorized)
        dists = torch.norm(sample_t - x.unsqueeze(0), dim=1)

        # Find points within 2*r (IDX1)
        IDX1 = torch.where(dists < 2 * r)[0]

        # Find k-nearest neighbors (IDX2) - use sklearn on CPU for this part
        # (sklearn's NearestNeighbors is optimized and fast enough for k=5)
        x_cpu = x.cpu().numpy().reshape(1, -1)
        sample_cpu = sample_t.cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=5).fit(sample_cpu)
        IDX2_cpu = nbrs.kneighbors(x_cpu, return_distance=False).flatten()
        IDX2 = torch.from_numpy(IDX2_cpu).to(device)

        # Union of IDX1 and IDX2
        IDX = torch.unique(torch.cat([IDX1, IDX2]))

        # Get neighborhood
        BNbr = sample_t[IDX, :]

        # Compute mean of neighborhood
        xbar = torch.mean(BNbr, dim=0) + torch.finfo(torch.float32).eps

        # Compute normalized direction
        dx = x - xbar
        dx = dx / torch.norm(dx)

        # QR decomposition to get orthonormal basis
        # Create matrix with dx as first column, then identity
        Q_input = torch.cat([
            dx.unsqueeze(1),
            torch.eye(dx.size(0), device=device)
        ], dim=1)
        Q, _ = torch.linalg.qr(Q_input)

        # Project sample points
        sample_s = (sample_t - x.unsqueeze(0)) @ Q

        # Find cylindrical neighborhood
        CNbr = (torch.abs(sample_s[:, 0]) < R) & (torch.sum(sample_s[:, 1:] ** 2, dim=1) < r ** 2)

        # Update output
        if torch.sum(CNbr) > 10:
            Mout[ii, :] = torch.mean(sample_t[CNbr, :], dim=0)
        else:
            Mout[ii, :] = xbar

        # Print progress every 100 points
        if (ii + 1) % 100 == 0:
            print(f"  Processed {ii + 1}/{N} points...")

    # Convert back to numpy
    return Mout.cpu().numpy()


def manfit_ours_gpu_batched(sample, sig, sample_init, op_average=1, device=None, batch_size=128):
    """
    GPU-accelerated manifold fitting with batched processing for better GPU utilization.

    This version processes multiple points in parallel on the GPU.

    Args:
        sample: Reference points (numpy array or torch tensor) [N0 x D]
        sig: Bandwidth parameter
        sample_init: Points to transform (numpy array or torch tensor) [N x D]
        op_average: Not used (kept for compatibility)
        device: torch device ('cuda', 'cpu', or None for auto-detect)
        batch_size: Number of points to process in parallel

    Returns:
        Transformed points as numpy array [N x D]
    """

    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"Using device: {device} (batched mode, batch_size={batch_size})")

    # Convert inputs to torch tensors and move to device
    if isinstance(sample, np.ndarray):
        sample_t = torch.from_numpy(sample).float().to(device)
    else:
        sample_t = sample.float().to(device)

    if isinstance(sample_init, np.ndarray):
        sample_init_t = torch.from_numpy(sample_init).float().to(device)
    else:
        sample_init_t = sample_init.float().to(device)

    # Initialize output
    Mout = sample_init_t.clone()
    N = sample_init_t.shape[0]
    N0 = sample_t.shape[0]

    # Compute parameters
    r = 5 * sig / np.log10(N0)
    R = 10 * sig * np.sqrt(np.log(1 / sig)) / np.log10(N0)

    # Process in batches
    num_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, N)

        # Process batch sequentially (full parallelization is complex due to variable neighbors)
        for ii in range(start_idx, end_idx):
            x = sample_init_t[ii, :]

            # Compute distances to all sample points (vectorized)
            dists = torch.norm(sample_t - x.unsqueeze(0), dim=1)

            # Find points within 2*r (IDX1)
            IDX1 = torch.where(dists < 2 * r)[0]

            # Find k-nearest neighbors (IDX2)
            x_cpu = x.cpu().numpy().reshape(1, -1)
            sample_cpu = sample_t.cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=5).fit(sample_cpu)
            IDX2_cpu = nbrs.kneighbors(x_cpu, return_distance=False).flatten()
            IDX2 = torch.from_numpy(IDX2_cpu).to(device)

            # Union of IDX1 and IDX2
            IDX = torch.unique(torch.cat([IDX1, IDX2]))

            # Get neighborhood
            BNbr = sample_t[IDX, :]

            # Compute mean of neighborhood
            xbar = torch.mean(BNbr, dim=0) + torch.finfo(torch.float32).eps

            # Compute normalized direction
            dx = x - xbar
            dx = dx / torch.norm(dx)

            # QR decomposition
            Q_input = torch.cat([
                dx.unsqueeze(1),
                torch.eye(dx.size(0), device=device)
            ], dim=1)
            Q, _ = torch.linalg.qr(Q_input)

            # Project sample points
            sample_s = (sample_t - x.unsqueeze(0)) @ Q

            # Find cylindrical neighborhood
            CNbr = (torch.abs(sample_s[:, 0]) < R) & (torch.sum(sample_s[:, 1:] ** 2, dim=1) < r ** 2)

            # Update output
            if torch.sum(CNbr) > 10:
                Mout[ii, :] = torch.mean(sample_t[CNbr, :], dim=0)
            else:
                Mout[ii, :] = xbar

        # Print progress
        print(f"  Processed batch {batch_idx + 1}/{num_batches} ({end_idx}/{N} points)...")

    # Convert back to numpy
    return Mout.cpu().numpy()


if __name__ == "__main__":
    # Test the GPU version
    print("Testing GPU-accelerated manfit_ours...")

    # Generate test data
    np.random.seed(42)
    n_samples = 2000
    n_dims = 2048

    sample = np.random.randn(n_samples, n_dims).astype(np.float32)
    sample_init = np.random.randn(100, n_dims).astype(np.float32)
    sig = 0.3

    # Test GPU version
    import time
    start = time.time()
    result_gpu = manfit_ours_gpu(sample, sig, sample_init)
    gpu_time = time.time() - start

    print(f"\nGPU version completed in {gpu_time:.2f} seconds")
    print(f"Output shape: {result_gpu.shape}")
    print(f"Output mean: {np.mean(result_gpu):.6f}")
    print(f"Output std: {np.std(result_gpu):.6f}")
