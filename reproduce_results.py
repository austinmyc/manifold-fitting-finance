import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class TwoLayerMLP(nn.Module):
    """Two-layer MLP for sentiment classification"""
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=3):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def plot_training_curves(results, save_path='reproduced_curves.png'):
    """Plot training curves from saved results"""
    history_orig = results['original'][()]['history']
    history_trans = results['transformed'][()]['history']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history_orig['train_loss']) + 1)

    # Training Loss
    axes[0, 0].plot(epochs, history_orig['train_loss'], 'b-', label='Original', linewidth=2)
    axes[0, 0].plot(epochs, history_trans['train_loss'], 'r-', label='Transformed', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation Loss
    axes[0, 1].plot(epochs, history_orig['val_loss'], 'b-', label='Original', linewidth=2)
    axes[0, 1].plot(epochs, history_trans['val_loss'], 'r-', label='Transformed', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Training Accuracy
    axes[1, 0].plot(epochs, history_orig['train_acc'], 'b-', label='Original', linewidth=2)
    axes[1, 0].plot(epochs, history_trans['train_acc'], 'r-', label='Transformed', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Accuracy
    axes[1, 1].plot(epochs, history_orig['val_acc'], 'b-', label='Original', linewidth=2)
    axes[1, 1].plot(epochs, history_trans['val_acc'], 'r-', label='Transformed', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reproduced plot to {save_path}")
    plt.close()


def main():
    print("="*70)
    print("Reproducing MLP Comparison Results")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ========================================================================
    # Step 1: Load Training Embeddings
    # ========================================================================
    print("\n" + "="*70)
    print("Step 1: Loading training embeddings...")

    train_data = np.load('finsent_embeddings.npz', allow_pickle=True)
    train_embeddings_orig = train_data['original']
    train_embeddings_trans = train_data['transformed']
    best_sigma = float(train_data['best_sigma'])

    print(f"  Original embeddings: {train_embeddings_orig.shape}")
    print(f"  Transformed embeddings: {train_embeddings_trans.shape}")
    print(f"  Best sigma: {best_sigma:.4f}")

    # ========================================================================
    # Step 2: Load Test Embeddings
    # ========================================================================
    print("\n" + "="*70)
    print("Step 2: Loading test embeddings...")

    test_data = np.load('test_embeddings.npz', allow_pickle=True)
    test_embeddings_orig = test_data['original']
    test_embeddings_trans = test_data['transformed']
    test_labels = test_data['labels']
    test_sentences = test_data['sentences']

    print(f"  Test embeddings (original): {test_embeddings_orig.shape}")
    print(f"  Test embeddings (transformed): {test_embeddings_trans.shape}")
    print(f"  Test labels: {test_labels.shape}")
    print(f"  Number of test sentences: {len(test_sentences)}")

    # ========================================================================
    # Step 3: Load Training Results
    # ========================================================================
    print("\n" + "="*70)
    print("Step 3: Loading training results...")

    results = np.load('mlp_comparison_results.npz', allow_pickle=True)
    hyperparams = results['hyperparameters'][()]

    print(f"\n  Hyperparameters:")
    print(f"    Hidden dim: {hyperparams['hidden_dim']}")
    print(f"    Batch size: {hyperparams['batch_size']}")
    print(f"    Learning rate: {hyperparams['learning_rate']}")
    print(f"    Num epochs: {hyperparams['num_epochs']}")

    test_acc_orig_saved = results['original'][()]['test_accuracy']
    test_acc_trans_saved = results['transformed'][()]['test_accuracy']

    print(f"\n  Saved test accuracies:")
    print(f"    Original: {test_acc_orig_saved:.2f}%")
    print(f"    Transformed: {test_acc_trans_saved:.2f}%")

    # ========================================================================
    # Step 4: Load Trained Models
    # ========================================================================
    print("\n" + "="*70)
    print("Step 4: Loading trained models...")

    # Load model for original embeddings
    model_orig = TwoLayerMLP(
        input_dim=2048,
        hidden_dim=hyperparams['hidden_dim'],
        num_classes=3
    ).to(device)
    model_orig.load_state_dict(torch.load('mlp_model_original.pth', map_location=device))
    print("  ✓ Loaded model_orig from mlp_model_original.pth")

    # Load model for transformed embeddings
    model_trans = TwoLayerMLP(
        input_dim=2048,
        hidden_dim=hyperparams['hidden_dim'],
        num_classes=3
    ).to(device)
    model_trans.load_state_dict(torch.load('mlp_model_transformed.pth', map_location=device))
    print("  ✓ Loaded model_trans from mlp_model_transformed.pth")

    # ========================================================================
    # Step 5: Verify Results by Re-evaluating on Test Set
    # ========================================================================
    print("\n" + "="*70)
    print("Step 5: Verifying results by re-evaluating models...")

    # Create test data loaders
    test_dataset_orig = TensorDataset(
        torch.FloatTensor(test_embeddings_orig),
        torch.LongTensor(test_labels)
    )
    test_dataset_trans = TensorDataset(
        torch.FloatTensor(test_embeddings_trans),
        torch.LongTensor(test_labels)
    )

    test_loader_orig = DataLoader(test_dataset_orig, batch_size=64, shuffle=False)
    test_loader_trans = DataLoader(test_dataset_trans, batch_size=64, shuffle=False)

    # Evaluate
    test_acc_orig = evaluate_model(model_orig, test_loader_orig, device)
    test_acc_trans = evaluate_model(model_trans, test_loader_trans, device)

    print(f"\n  Re-evaluated test accuracies:")
    print(f"    Original: {test_acc_orig:.2f}%")
    print(f"    Transformed: {test_acc_trans:.2f}%")

    # Verify they match saved results
    print(f"\n  Verification:")
    orig_match = abs(test_acc_orig - test_acc_orig_saved) < 0.01
    trans_match = abs(test_acc_trans - test_acc_trans_saved) < 0.01
    print(f"    Original accuracy matches: {orig_match} ✓" if orig_match else f"    Original accuracy matches: {orig_match} ✗")
    print(f"    Transformed accuracy matches: {trans_match} ✓" if trans_match else f"    Transformed accuracy matches: {trans_match} ✗")

    # ========================================================================
    # Step 6: Visualize Results
    # ========================================================================
    print("\n" + "="*70)
    print("Step 6: Generating visualizations...")

    plot_training_curves(results, save_path='reproduced_curves.png')

    # ========================================================================
    # Step 7: Summary Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nTraining Set Statistics:")
    print(f"  Total training samples: {len(train_embeddings_orig)}")
    print(f"  Embedding dimension: {train_embeddings_orig.shape[1]}")

    print(f"\nTest Set Statistics:")
    print(f"  Total test samples: {len(test_labels)}")
    print(f"  Class distribution: {np.bincount(test_labels)}")

    print(f"\nModel Performance:")
    print(f"  Original Embeddings:")
    print(f"    Test Accuracy: {test_acc_orig:.2f}%")
    print(f"  Transformed Embeddings:")
    print(f"    Test Accuracy: {test_acc_trans:.2f}%")
    print(f"  Improvement: {test_acc_trans - test_acc_orig:+.2f}%")

    print(f"\nManifold Fitting:")
    print(f"  Best sigma: {best_sigma:.4f}")
    print(f"  Quality score: {float(train_data['best_score']):.6f}")

    # ========================================================================
    # Example: Make Predictions on Sample Data
    # ========================================================================
    print("\n" + "="*70)
    print("Example: Predictions on first 5 test samples")
    print("="*70)

    label_names = ['negative', 'neutral', 'positive']

    model_orig.eval()
    model_trans.eval()

    with torch.no_grad():
        sample_orig = torch.FloatTensor(test_embeddings_orig[:5]).to(device)
        sample_trans = torch.FloatTensor(test_embeddings_trans[:5]).to(device)

        pred_orig = model_orig(sample_orig).argmax(dim=1).cpu().numpy()
        pred_trans = model_trans(sample_trans).argmax(dim=1).cpu().numpy()

    for i in range(5):
        print(f"\nSample {i+1}:")
        print(f"  Sentence: {test_sentences[i][:80]}...")
        print(f"  True label: {label_names[test_labels[i]]}")
        print(f"  Pred (original): {label_names[pred_orig[i]]}")
        print(f"  Pred (transformed): {label_names[pred_trans[i]]}")

    print("\n" + "="*70)
    print("✓ Reproduction complete!")
    print("="*70)

    print("\nFiles used:")
    print("  1. finsent_embeddings.npz - Training embeddings")
    print("  2. test_embeddings.npz - Test embeddings")
    print("  3. mlp_model_original.pth - Model trained on original embeddings")
    print("  4. mlp_model_transformed.pth - Model trained on transformed embeddings")
    print("  5. mlp_comparison_results.npz - Training history and metrics")


if __name__ == "__main__":
    main()
