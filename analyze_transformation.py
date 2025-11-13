"""
Analyze the transformation effects to understand why performance degrades.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = np.load('data/embeddings_transformed.npz')

train_orig = data['train_original']
train_trans = data['train_transformed']
train_labels = data['train_labels']

test_orig = data['test_original']
test_trans = data['test_transformed']
test_labels = data['test_labels']

best_sigma = float(data['best_sigma'])

print("="*70)
print("TRANSFORMATION ANALYSIS")
print("="*70)
print(f"\nBest sigma used: {best_sigma:.4f}")
print(f"Train samples: {train_orig.shape[0]}")
print(f"Test samples: {test_orig.shape[0]}")
print(f"Embedding dimension: {train_orig.shape[1]}")

# 1. Basic Statistics
print("\n" + "="*70)
print("1. BASIC STATISTICS")
print("="*70)

print("\nTraining data:")
print(f"  Original    - Mean: {np.mean(train_orig):.6f}, Std: {np.std(train_orig):.6f}")
print(f"  Transformed - Mean: {np.mean(train_trans):.6f}, Std: {np.std(train_trans):.6f}")

print("\nTest data:")
print(f"  Original    - Mean: {np.mean(test_orig):.6f}, Std: {np.std(test_orig):.6f}")
print(f"  Transformed - Mean: {np.mean(test_trans):.6f}, Std: {np.std(test_trans):.6f}")

# 2. Magnitude of Change
print("\n" + "="*70)
print("2. MAGNITUDE OF CHANGE")
print("="*70)

train_diff = train_trans - train_orig
test_diff = test_trans - test_orig

train_change_per_sample = np.linalg.norm(train_diff, axis=1)
test_change_per_sample = np.linalg.norm(test_diff, axis=1)

print("\nL2 norm of change per sample:")
print(f"  Train - Mean: {np.mean(train_change_per_sample):.6f}, Std: {np.std(train_change_per_sample):.6f}")
print(f"  Train - Min: {np.min(train_change_per_sample):.6f}, Max: {np.max(train_change_per_sample):.6f}")
print(f"  Test  - Mean: {np.mean(test_change_per_sample):.6f}, Std: {np.std(test_change_per_sample):.6f}")
print(f"  Test  - Min: {np.min(test_change_per_sample):.6f}, Max: {np.max(test_change_per_sample):.6f}")

# 3. Class Separability Analysis
print("\n" + "="*70)
print("3. CLASS SEPARABILITY ANALYSIS")
print("="*70)

def compute_class_separability(embeddings, labels):
    """Compute between-class vs within-class variance ratio"""
    unique_labels = np.unique(labels)

    # Overall mean
    overall_mean = np.mean(embeddings, axis=0)

    # Between-class variance
    between_var = 0
    for label in unique_labels:
        class_mask = labels == label
        class_mean = np.mean(embeddings[class_mask], axis=0)
        class_size = np.sum(class_mask)
        between_var += class_size * np.linalg.norm(class_mean - overall_mean)**2

    # Within-class variance
    within_var = 0
    for label in unique_labels:
        class_mask = labels == label
        class_mean = np.mean(embeddings[class_mask], axis=0)
        within_var += np.sum(np.linalg.norm(embeddings[class_mask] - class_mean, axis=1)**2)

    # Fisher ratio (higher is better for classification)
    fisher_ratio = between_var / (within_var + 1e-10)

    return fisher_ratio, between_var, within_var

print("\nFisher discriminant ratio (between-class / within-class variance):")
print("Higher ratio = better class separability")

fisher_train_orig, btw_train_orig, wtn_train_orig = compute_class_separability(train_orig, train_labels)
fisher_train_trans, btw_train_trans, wtn_train_trans = compute_class_separability(train_trans, train_labels)

print(f"\nTraining data:")
print(f"  Original:    {fisher_train_orig:.6f}")
print(f"  Transformed: {fisher_train_trans:.6f}")
print(f"  Change:      {fisher_train_trans - fisher_train_orig:+.6f} ({((fisher_train_trans/fisher_train_orig - 1)*100):+.2f}%)")

fisher_test_orig, btw_test_orig, wtn_test_orig = compute_class_separability(test_orig, test_labels)
fisher_test_trans, btw_test_trans, wtn_test_trans = compute_class_separability(test_trans, test_labels)

print(f"\nTest data:")
print(f"  Original:    {fisher_test_orig:.6f}")
print(f"  Transformed: {fisher_test_trans:.6f}")
print(f"  Change:      {fisher_test_trans - fisher_test_orig:+.6f} ({((fisher_test_trans/fisher_test_orig - 1)*100):+.2f}%)")

# 4. Per-class analysis
print("\n" + "="*70)
print("4. PER-CLASS MEAN DISTANCES")
print("="*70)

def class_mean_distances(embeddings, labels):
    """Compute distances between class means"""
    unique_labels = np.unique(labels)
    class_means = []
    for label in unique_labels:
        class_mask = labels == label
        class_mean = np.mean(embeddings[class_mask], axis=0)
        class_means.append(class_mean)

    class_means = np.array(class_means)

    # Compute pairwise distances
    distances = {}
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            dist = np.linalg.norm(class_means[i] - class_means[j])
            distances[f"Class {unique_labels[i]} vs {unique_labels[j]}"] = dist

    return distances

print("\nTraining data - distances between class means:")
dist_train_orig = class_mean_distances(train_orig, train_labels)
dist_train_trans = class_mean_distances(train_trans, train_labels)

for key in dist_train_orig.keys():
    print(f"  {key}:")
    print(f"    Original:    {dist_train_orig[key]:.6f}")
    print(f"    Transformed: {dist_train_trans[key]:.6f}")
    print(f"    Change:      {dist_train_trans[key] - dist_train_orig[key]:+.6f}")

# 5. Visualization
print("\n" + "="*70)
print("5. GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# PCA visualization of original
pca = PCA(n_components=2)
train_orig_pca = pca.fit_transform(train_orig)
test_orig_pca = pca.transform(test_orig)

for label in np.unique(train_labels):
    mask = train_labels == label
    axes[0, 0].scatter(train_orig_pca[mask, 0], train_orig_pca[mask, 1],
                       label=f'Class {label}', alpha=0.6, s=20)
axes[0, 0].set_title('Original Embeddings (PCA)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# PCA visualization of transformed
train_trans_pca = pca.fit_transform(train_trans)
test_trans_pca = pca.transform(test_trans)

for label in np.unique(train_labels):
    mask = train_labels == label
    axes[0, 1].scatter(train_trans_pca[mask, 0], train_trans_pca[mask, 1],
                       label=f'Class {label}', alpha=0.6, s=20)
axes[0, 1].set_title('Transformed Embeddings (PCA)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Change magnitude histogram
axes[0, 2].hist(train_change_per_sample, bins=50, alpha=0.7, label='Train')
axes[0, 2].hist(test_change_per_sample, bins=50, alpha=0.7, label='Test')
axes[0, 2].set_xlabel('L2 Norm of Change')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Magnitude of Transformation')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# LDA visualization of original
lda = LDA(n_components=2)
train_orig_lda = lda.fit_transform(train_orig, train_labels)

for label in np.unique(train_labels):
    mask = train_labels == label
    axes[1, 0].scatter(train_orig_lda[mask, 0], train_orig_lda[mask, 1],
                       label=f'Class {label}', alpha=0.6, s=20)
axes[1, 0].set_title('Original Embeddings (LDA)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# LDA visualization of transformed
train_trans_lda = lda.fit_transform(train_trans, train_labels)

for label in np.unique(train_labels):
    mask = train_labels == label
    axes[1, 1].scatter(train_trans_lda[mask, 0], train_trans_lda[mask, 1],
                       label=f'Class {label}', alpha=0.6, s=20)
axes[1, 1].set_title('Transformed Embeddings (LDA)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Fisher ratio comparison
categories = ['Original\nTrain', 'Transformed\nTrain', 'Original\nTest', 'Transformed\nTest']
fisher_ratios = [fisher_train_orig, fisher_train_trans, fisher_test_orig, fisher_test_trans]
colors = ['blue', 'red', 'blue', 'red']

axes[1, 2].bar(categories, fisher_ratios, color=colors, alpha=0.7)
axes[1, 2].set_ylabel('Fisher Ratio')
axes[1, 2].set_title('Class Separability (Higher = Better)')
axes[1, 2].grid(True, alpha=0.3, axis='y')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/transformation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to results/transformation_analysis.png")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
The transformation with sigma={best_sigma:.4f}:

1. Changes training embeddings by an average L2 norm of {np.mean(train_change_per_sample):.4f}
2. Changes test embeddings by an average L2 norm of {np.mean(test_change_per_sample):.4f}

3. Class separability (Fisher ratio):
   - Training: {fisher_train_orig:.4f} → {fisher_train_trans:.4f} ({((fisher_train_trans/fisher_train_orig - 1)*100):+.2f}%)
   - Test:     {fisher_test_orig:.4f} → {fisher_test_trans:.4f} ({((fisher_test_trans/fisher_test_orig - 1)*100):+.2f}%)

4. {"⚠️  WARNING: Transformation DECREASES class separability!" if fisher_train_trans < fisher_train_orig else "✓ Transformation increases class separability"}
   This explains why classification accuracy drops from 81% to 73%.

RECOMMENDATION:
{"- The transformation is HURTING classification performance" if fisher_train_trans < fisher_train_orig else "- The transformation should help classification"}
{"- Consider using a different sigma value or not using the transformation for classification" if fisher_train_trans < fisher_train_orig else "- The poor performance must be due to other factors (overfitting, etc.)"}
""")

print("="*70)
