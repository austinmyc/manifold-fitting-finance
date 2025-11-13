import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from manfit.manfit_ours import manfit_ours
from manfit.noisy_data_generation import *

def quality_score(data, intrinsic_dim=2, k=10):
    """
    Combined quality metric for manifold denoising.
    Lower is better.

    Combines:
    1. Tangent space consistency (local linearity)
    2. Radius variance (manifold constraint for sphere)
    3. Diversity penalty (penalize collapse)
    """
    # 1. Check for collapse using pairwise distances
    nbrs = NearestNeighbors(n_neighbors=min(k, len(data))).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Average distance to k-nearest neighbors
    avg_nn_dist = np.mean(distances[:, 1:])  # Exclude self (distance 0)

    # If points are too close together, severe over-smoothing
    if avg_nn_dist < 1e-6:
        return 100.0  # Massive penalty for collapse

    # Penalize very small distances (over-smoothing)
    collapse_penalty = max(0, 1.0 - avg_nn_dist * 10)  # Want avg_nn_dist > 0.1

    # 2. Tangent space error
    tangent_errors = []
    for i in range(len(data)):
        neighborhood = data[indices[i]]

        # Skip degenerate neighborhoods
        if np.var(neighborhood) < 1e-10:
            tangent_errors.append(1.0)
            continue

        pca = PCA(n_components=min(intrinsic_dim, neighborhood.shape[0]-1))
        pca.fit(neighborhood)

        evr = pca.explained_variance_ratio_
        if len(evr) >= intrinsic_dim and not np.any(np.isnan(evr)):
            # Residual variance in higher dimensions
            residual = max(0.0, 1 - np.sum(evr[:intrinsic_dim]))
            tangent_errors.append(residual)
        else:
            tangent_errors.append(1.0)

    tangent_error = np.mean(tangent_errors)

    # 3. Radius variance (for sphere: all points should have same radius)
    radii = np.linalg.norm(data, axis=1)
    radius_var = np.var(radii)

    # Combined score (weighted sum)
    score = tangent_error + 0.5 * radius_var + 2.0 * collapse_penalty

    return score

clean, noisy = generate_noisy_manifold(manifold_type='sphere', n_samples=1000, noise_level=0.1, 
                           dim=3, radius=1.0, random_state=None)
noisy = np.array(noisy)

print(f"Noisy data shape: {noisy.shape}")
# Split data
train, val = train_test_split(noisy, test_size=0.2, random_state=42)

# Define search space
space = [Real(0.01, 1.0, name='sig')]

# Define objective function for Bayesian optimization
@use_named_args(space)
def objective(sig):
    """Objective function to minimize"""
    # Denoise validation set using training set as reference
    denoised_val = manfit_ours(sample=train, sample_init=val, sig=sig)

    # Compute quality score on validation
    score = quality_score(denoised_val, intrinsic_dim=2, k=10)

    print(f"sig={sig:.4f}: quality_score={score:.6f}")

    return score
'''
print("\nStarting Bayesian Optimization...")
print("=" * 50)

# Run Bayesian optimization
result = gp_minimize(
    objective,
    space,
    n_calls=20,           # Number of evaluations
    n_initial_points=5,   # Random exploration points
    random_state=42,
    verbose=False
)

best_sig = result.x[0]
best_score = result.fun

print("=" * 50)
print(f"\nBest sig: {best_sig:.4f}")
print(f"Best quality_score: {best_score:.6f}")

# Apply to full dataset
final_denoised = manfit_ours(noisy, best_sig, noisy)

print(final_denoised - noisy)
'''
