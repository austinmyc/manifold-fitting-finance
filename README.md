# Manifold Fitting Pipeline

This pipeline separates the embedding download, transformation, and model training into modular steps. This allows you to:
- Cache embeddings to avoid repeated API calls
- Experiment with different transformation parameters without re-downloading
- Train multiple models on the same embeddings
- Easily swap between original and transformed embeddings

## Pipeline Overview

```
1. download_embeddings.py  → Cache raw embeddings from API
2. transform_embeddings.py → Apply manifold fitting transformation
3. train_mlp.py           → Train and evaluate MLP models
```

## Quick Start

### 1. Download Embeddings (One-time)

Download training and test embeddings from the API and cache them:

```bash
# Download both train and test
python download_embeddings.py --split both

# Or download individually
python download_embeddings.py --split train
python download_embeddings.py --split test
```

**Output:**
- `data/train_embeddings_raw.npz` - Training embeddings and labels
- `data/test_embeddings_raw.npz` - Test embeddings and labels

**Note:** This step makes API calls. Use `--force` to re-download if needed.

### 2. Transform Embeddings

Apply manifold fitting transformation to the cached embeddings:

```bash
# Auto-tune sigma and transform
python transform_embeddings.py

# Use a specific sigma value
python transform_embeddings.py --sigma 0.15

# Use more samples for tuning (default: 1000)
python transform_embeddings.py --tune-samples 2000

# More optimization iterations (default: 15)
python transform_embeddings.py --n-calls 30
```

**Output:**
- `data/embeddings_transformed.npz` - All embeddings (original + transformed)

**Note:** This step does NOT make API calls. You can run it multiple times with different parameters.

### 3. Train MLP Models

Train and evaluate MLP classifiers:

```bash
# Train on both original and transformed embeddings
python train_mlp.py

# Train only on original embeddings
python train_mlp.py --mode original

# Train only on transformed embeddings
python train_mlp.py --mode transformed

# Customize hyperparameters
python train_mlp.py --hidden-dim 512 --epochs 100 --learning-rate 0.0005

# Add weight decay (L2 regularization)
python train_mlp.py --weight-decay 1e-4
```

**Output:**
- `results/mlp_model_original.pth` - Trained model on original embeddings
- `results/mlp_model_transformed.pth` - Trained model on transformed embeddings
- `results/mlp_results.npz` - Training history and metrics
- `results/mlp_comparison.png` - Comparison plots

**Note:** This step does NOT make API calls or transform embeddings. You can experiment with different model architectures and hyperparameters freely.

## Command-Line Options

### download_embeddings.py

```
--split {train,test,both}  Which split to download (default: both)
--data-dir PATH            Path to dataset directory
--force                    Force re-download even if files exist
```

### transform_embeddings.py

```
--train-file PATH          Path to raw training embeddings
--test-file PATH           Path to raw test embeddings
--output-file PATH         Path to save transformed embeddings
--sigma FLOAT              Sigma parameter (if not provided, will tune)
--tune-samples INT         Number of samples for tuning (default: 1000)
--n-calls INT              Number of Bayesian optimization calls (default: 15)
```

### train_mlp.py

```
--data-file PATH           Path to transformed embeddings file
--mode {original,transformed,both}  Which embeddings to train on
--hidden-dim INT           Hidden layer dimension (default: 256)
--batch-size INT           Batch size (default: 64)
--learning-rate FLOAT      Learning rate (default: 0.001)
--weight-decay FLOAT       Weight decay/L2 regularization (default: 1e-4)
--dropout FLOAT            Dropout rate (default: 0.3)
--epochs INT               Number of training epochs (default: 50)
--val-split FLOAT          Validation split ratio (default: 0.2)
--output-dir PATH          Output directory for results (default: results)
```

## Example Workflows

### Initial Setup
```bash
# 1. Download embeddings (one-time, makes API calls)
python download_embeddings.py --split both

# 2. Transform with auto-tuning
python transform_embeddings.py

# 3. Train models
python train_mlp.py
```

### Experiment with Different Sigma Values
```bash
# Try different sigma values without re-downloading
python transform_embeddings.py --sigma 0.1
python train_mlp.py --output-dir results_sigma_0.1

python transform_embeddings.py --sigma 0.2
python train_mlp.py --output-dir results_sigma_0.2

python transform_embeddings.py --sigma 0.3
python train_mlp.py --output-dir results_sigma_0.3
```

### Experiment with Different Model Architectures
```bash
# Small model
python train_mlp.py --hidden-dim 128 --output-dir results_small

# Large model
python train_mlp.py --hidden-dim 512 --output-dir results_large

# More regularization
python train_mlp.py --weight-decay 1e-3 --dropout 0.5 --output-dir results_regularized

# Longer training
python train_mlp.py --epochs 100 --output-dir results_long_training
```

### Quick Test (Original Embeddings Only)
```bash
# Train only on original embeddings (faster)
python train_mlp.py --mode original --epochs 20
```

## File Structure

```
data/
├── train_embeddings_raw.npz      # Cached training embeddings
├── test_embeddings_raw.npz       # Cached test embeddings
└── embeddings_transformed.npz    # Transformed embeddings

results/
├── mlp_model_original.pth        # Trained model (original)
├── mlp_model_transformed.pth     # Trained model (transformed)
├── mlp_results.npz               # Training history
└── mlp_comparison.png            # Comparison plots

*.log                              # Log files for each script
```

## Data Format

### train_embeddings_raw.npz / test_embeddings_raw.npz
```python
{
    'embeddings': np.array,  # Shape: (n_samples, 2048)
    'labels': np.array,      # Shape: (n_samples,)
    'sentences': np.array    # Shape: (n_samples,) - original text
}
```

### embeddings_transformed.npz
```python
{
    'train_original': np.array,      # Original training embeddings
    'train_transformed': np.array,   # Transformed training embeddings
    'train_labels': np.array,        # Training labels
    'train_sentences': np.array,     # Training sentences
    'test_original': np.array,       # Original test embeddings
    'test_transformed': np.array,    # Transformed test embeddings
    'test_labels': np.array,         # Test labels
    'test_sentences': np.array,      # Test sentences
    'best_sigma': float,             # Best sigma from tuning
    'best_score': float              # Best quality score
}
```

## Advantages of This Pipeline

1. **No Repeated API Calls**: Download embeddings once, experiment freely
2. **Modular Design**: Easy to modify individual components
3. **Fast Iteration**: Skip expensive steps when experimenting
4. **Easy Comparison**: Compare multiple transformation/model configurations
5. **Reproducible**: Cached data ensures consistent experiments
6. **Flexible**: Each step has multiple configuration options

## Legacy Scripts

The original scripts are still available:
- `process_finsent_embeddings.py` - Original embedding processing
- `compare_mlp_embeddings.py` - Original training script (downloads + transforms + trains)

These are now superseded by the modular pipeline above.
