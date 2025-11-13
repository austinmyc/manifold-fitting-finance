"""
Step 2: Transform embeddings using manifold fitting.

This script applies manifold fitting transformation to the cached embeddings.
It includes hyperparameter tuning for sigma and applies the transformation
to both training and test data.

Usage:
    python transform_embeddings.py
    python transform_embeddings.py --sigma 0.15  # Use a specific sigma
    python transform_embeddings.py --tune-samples 1000  # Use more samples for tuning
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from manfit.manfit_ours_gpu import manfit_ours_gpu_batched as manfit_ours
from manfit.tuning import quality_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transform_embeddings.log'),
        logging.StreamHandler()
    ]
)


def tune_sigma(train_embeddings, tune_samples=1000, n_calls=15):
    """Tune sigma parameter using Bayesian optimization"""
    logging.info("="*70)
    logging.info("Tuning Sigma Parameter")
    logging.info("="*70)

    # Use subset for tuning to save time
    tuning_size = min(tune_samples, len(train_embeddings))
    tuning_embeddings = train_embeddings[:tuning_size]
    logging.info(f"Using {tuning_size} samples for hyperparameter tuning")

    # Split embeddings for tuning (80/20 split)
    train_emb, val_emb = train_test_split(tuning_embeddings, test_size=0.2, random_state=42)
    logging.info(f"Train embeddings: {train_emb.shape}")
    logging.info(f"Validation embeddings: {val_emb.shape}")

    # Define search space for sigma
    space = [Real(0.01, 1.0, name='sig')]

    # Define objective function
    @use_named_args(space)
    def objective(sig):
        """Objective function to minimize"""
        # Transform validation embeddings using training embeddings as reference
        transformed_val = manfit_ours(sample=train_emb, sample_init=val_emb, sig=sig)

        # Compute quality score
        score = quality_score(transformed_val, intrinsic_dim=2, k=5)

        logging.info(f"  sig={sig:.4f}: quality_score={score:.6f}")

        return score

    # Run Bayesian optimization
    logging.info("Running Bayesian optimization...")
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=5,
        random_state=42,
        verbose=False
    )

    best_sig = result.x[0]
    best_score = result.fun

    logging.info("\n" + "=" * 70)
    logging.info(f"Optimization complete!")
    logging.info(f"Best sigma: {best_sig:.4f}")
    logging.info(f"Best quality score: {best_score:.6f}")
    logging.info("=" * 70)

    return best_sig, best_score


def transform_embeddings(train_file="data/train_embeddings_raw.npz",
                         test_file="data/test_embeddings_raw.npz",
                         output_file="data/embeddings_transformed.npz",
                         sigma=None,
                         tune_samples=1000,
                         n_calls=15):
    """Transform embeddings using manifold fitting"""
    logging.info("="*70)
    logging.info("Manifold Fitting Transformation")
    logging.info("="*70)

    # Load raw embeddings
    logging.info(f"\nLoading raw embeddings...")
    logging.info(f"  Train: {train_file}")
    logging.info(f"  Test: {test_file}")

    train_data = np.load(train_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    train_embeddings = train_data['embeddings']
    train_labels = train_data['labels']
    train_sentences = train_data['sentences']

    test_embeddings = test_data['embeddings']
    test_labels = test_data['labels']
    test_sentences = test_data['sentences']

    logging.info(f"\n✓ Loaded data:")
    logging.info(f"  Train: {train_embeddings.shape}")
    logging.info(f"  Test: {test_embeddings.shape}")

    # Tune or use provided sigma
    if sigma is None:
        logging.info("\n" + "="*70)
        logging.info("No sigma provided. Starting hyperparameter tuning...")
        best_sigma, best_score = tune_sigma(train_embeddings, tune_samples, n_calls)
    else:
        logging.info(f"\nUsing provided sigma: {sigma:.4f}")
        best_sigma = sigma
        best_score = None

    # Transform training embeddings
    logging.info("\n" + "="*70)
    logging.info("Transforming training embeddings...")
    logging.info(f"Using sigma: {best_sigma:.4f}")

    train_embeddings_transformed = manfit_ours(
        sample=train_embeddings,
        sig=best_sigma,
        sample_init=train_embeddings
    )

    logging.info(f"✓ Transformed training embeddings: {train_embeddings_transformed.shape}")

    # Show statistics
    logging.info("\nTraining data statistics:")
    logging.info(f"  Original - Mean: {np.mean(train_embeddings):.6f}, Std: {np.std(train_embeddings):.6f}")
    logging.info(f"  Transformed - Mean: {np.mean(train_embeddings_transformed):.6f}, Std: {np.std(train_embeddings_transformed):.6f}")

    diff = train_embeddings_transformed - train_embeddings
    logging.info(f"  Difference - Mean: {np.mean(diff):.6f}, Std: {np.std(diff):.6f}")
    logging.info(f"  Max absolute change: {np.max(np.abs(diff)):.6f}")

    # Transform test embeddings
    logging.info("\n" + "="*70)
    logging.info("Transforming test embeddings...")
    logging.info(f"Using original training embeddings as reference manifold")

    test_embeddings_transformed = manfit_ours(
        sample=train_embeddings,  # Use ORIGINAL training as reference
        sig=best_sigma,
        sample_init=test_embeddings
    )

    logging.info(f"✓ Transformed test embeddings: {test_embeddings_transformed.shape}")

    # Show statistics
    logging.info("\nTest data statistics:")
    logging.info(f"  Original - Mean: {np.mean(test_embeddings):.6f}, Std: {np.std(test_embeddings):.6f}")
    logging.info(f"  Transformed - Mean: {np.mean(test_embeddings_transformed):.6f}, Std: {np.std(test_embeddings_transformed):.6f}")

    diff = test_embeddings_transformed - test_embeddings
    logging.info(f"  Difference - Mean: {np.mean(diff):.6f}, Std: {np.std(diff):.6f}")
    logging.info(f"  Max absolute change: {np.max(np.abs(diff)):.6f}")

    # Save transformed embeddings
    logging.info("\n" + "="*70)
    logging.info(f"Saving transformed embeddings to {output_file}...")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file,
             train_original=train_embeddings,
             train_transformed=train_embeddings_transformed,
             train_labels=train_labels,
             train_sentences=train_sentences,
             test_original=test_embeddings,
             test_transformed=test_embeddings_transformed,
             test_labels=test_labels,
             test_sentences=test_sentences,
             best_sigma=best_sigma,
             best_score=best_score if best_score is not None else np.nan)

    logging.info(f"✓ Saved transformed embeddings")
    logging.info(f"\nSaved arrays:")
    logging.info(f"  - train_original: {train_embeddings.shape}")
    logging.info(f"  - train_transformed: {train_embeddings_transformed.shape}")
    logging.info(f"  - test_original: {test_embeddings.shape}")
    logging.info(f"  - test_transformed: {test_embeddings_transformed.shape}")
    logging.info(f"  - best_sigma: {best_sigma:.4f}")

    logging.info("\n" + "="*70)
    logging.info("✓ Transformation complete!")
    logging.info("="*70)


def main():
    parser = argparse.ArgumentParser(description='Transform embeddings using manifold fitting')
    parser.add_argument('--train-file', type=str, default='data/train_embeddings_raw.npz',
                       help='Path to raw training embeddings')
    parser.add_argument('--test-file', type=str, default='data/test_embeddings_raw.npz',
                       help='Path to raw test embeddings')
    parser.add_argument('--output-file', type=str, default='data/embeddings_transformed.npz',
                       help='Path to save transformed embeddings')
    parser.add_argument('--sigma', type=float, default=None,
                       help='Sigma parameter (if not provided, will tune)')
    parser.add_argument('--tune-samples', type=int, default=1000,
                       help='Number of samples to use for tuning')
    parser.add_argument('--n-calls', type=int, default=15,
                       help='Number of Bayesian optimization calls')

    args = parser.parse_args()

    # Check if input files exist
    if not Path(args.train_file).exists():
        logging.error(f"Training embeddings not found: {args.train_file}")
        logging.error("Run: python download_embeddings.py --split train")
        return

    if not Path(args.test_file).exists():
        logging.error(f"Test embeddings not found: {args.test_file}")
        logging.error("Run: python download_embeddings.py --split test")
        return

    # Transform embeddings
    transform_embeddings(
        train_file=args.train_file,
        test_file=args.test_file,
        output_file=args.output_file,
        sigma=args.sigma,
        tune_samples=args.tune_samples,
        n_calls=args.n_calls
    )

    logging.info("\nNext step:")
    logging.info("  Run: python train_mlp.py")


if __name__ == "__main__":
    main()
