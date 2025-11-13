"""
Step 3: Train and evaluate MLP models on original and transformed embeddings.

This script trains MLP classifiers on cached embeddings and generates
comparison plots and metrics.

Usage:
    python train_mlp.py
    python train_mlp.py --hidden-dim 512 --epochs 100
    python train_mlp.py --mode original  # Only train on original embeddings
    python train_mlp.py --mode transformed  # Only train on transformed embeddings
"""

import numpy as np
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_mlp.log'),
        logging.StreamHandler()
    ]
)


class TwoLayerMLP(nn.Module):
    """Two-layer MLP for sentiment classification"""
    def __init__(self, input_dim=2048, hidden_dim=256, num_classes=3, dropout=0.3):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """Train the model and return training history"""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}] '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    return {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }


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


def plot_comparison(history_original, history_transformed, save_path='results/mlp_comparison.png'):
    """Plot training curves for both models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history_original['train_loss']) + 1)

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, history_original['train_loss'], 'b-', label='Original', linewidth=2)
    axes[0, 0].plot(epochs, history_transformed['train_loss'], 'r-', label='Transformed', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, history_original['val_loss'], 'b-', label='Original', linewidth=2)
    axes[0, 1].plot(epochs, history_transformed['val_loss'], 'r-', label='Transformed', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    axes[1, 0].plot(epochs, history_original['train_acc'], 'b-', label='Original', linewidth=2)
    axes[1, 0].plot(epochs, history_transformed['train_acc'], 'r-', label='Transformed', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    axes[1, 1].plot(epochs, history_original['val_acc'], 'b-', label='Original', linewidth=2)
    axes[1, 1].plot(epochs, history_transformed['val_acc'], 'r-', label='Transformed', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved comparison plot to {save_path}")
    plt.close()


def train_and_evaluate(data_file="data/embeddings_transformed.npz",
                       mode='both',
                       hidden_dim=256,
                       batch_size=64,
                       learning_rate=0.001,
                       weight_decay=1e-4,
                       num_epochs=50,
                       dropout=0.3,
                       val_split=0.2,
                       output_dir='results'):
    """Main training and evaluation function"""
    logging.info("="*70)
    logging.info("MLP Training and Evaluation")
    logging.info("="*70)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"\nUsing device: {device}")

    # Load data
    logging.info(f"\nLoading data from {data_file}...")
    data = np.load(data_file, allow_pickle=True)

    train_embeddings_original = data['train_original']
    train_embeddings_transformed = data['train_transformed']
    train_labels = data['train_labels']

    test_embeddings_original = data['test_original']
    test_embeddings_transformed = data['test_transformed']
    test_labels = data['test_labels']

    best_sigma = float(data['best_sigma'])

    logging.info(f"✓ Loaded data:")
    logging.info(f"  Train embeddings: {train_embeddings_original.shape}")
    logging.info(f"  Test embeddings: {test_embeddings_original.shape}")
    logging.info(f"  Best sigma: {best_sigma:.4f}")

    # Hyperparameters
    logging.info(f"\nHyperparameters:")
    logging.info(f"  Hidden dimension: {hidden_dim}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Weight decay: {weight_decay}")
    logging.info(f"  Dropout: {dropout}")
    logging.info(f"  Number of epochs: {num_epochs}")
    logging.info(f"  Validation split: {val_split}")

    # Split training data into train/val
    logging.info(f"\nSplitting training data ({int((1-val_split)*100)}/{int(val_split*100)})...")
    indices = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(indices, test_size=val_split, random_state=42, stratify=train_labels)

    logging.info(f"  Training samples: {len(train_idx)}")
    logging.info(f"  Validation samples: {len(val_idx)}")

    results = {}
    criterion = nn.CrossEntropyLoss()

    # ============================================================================
    # Train on Original Embeddings
    # ============================================================================
    if mode in ['original', 'both']:
        logging.info("\n" + "="*70)
        logging.info("EXPERIMENT 1: Training MLP on Original Embeddings")
        logging.info("="*70)

        # Prepare data loaders
        train_dataset_orig = TensorDataset(
            torch.FloatTensor(train_embeddings_original[train_idx]),
            torch.LongTensor(train_labels[train_idx])
        )
        val_dataset_orig = TensorDataset(
            torch.FloatTensor(train_embeddings_original[val_idx]),
            torch.LongTensor(train_labels[val_idx])
        )
        test_dataset_orig = TensorDataset(
            torch.FloatTensor(test_embeddings_original),
            torch.LongTensor(test_labels)
        )

        train_loader_orig = DataLoader(train_dataset_orig, batch_size=batch_size, shuffle=True)
        val_loader_orig = DataLoader(val_dataset_orig, batch_size=batch_size, shuffle=False)
        test_loader_orig = DataLoader(test_dataset_orig, batch_size=batch_size, shuffle=False)

        # Initialize model
        model_orig = TwoLayerMLP(input_dim=2048, hidden_dim=hidden_dim,
                                 num_classes=3, dropout=dropout).to(device)
        optimizer_orig = optim.Adam(model_orig.parameters(), lr=learning_rate,
                                   weight_decay=weight_decay)

        # Train
        logging.info("Training model on original embeddings...")
        history_orig = train_model(model_orig, train_loader_orig, val_loader_orig,
                                   criterion, optimizer_orig, num_epochs, device)

        # Test
        test_acc_orig = evaluate_model(model_orig, test_loader_orig, device)
        logging.info(f"\nOriginal Embeddings - Test Accuracy: {test_acc_orig:.2f}%")

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model_orig.state_dict(), output_path / 'mlp_model_original.pth')
        logging.info(f"Saved model to {output_path / 'mlp_model_original.pth'}")

        results['original'] = {
            'history': history_orig,
            'test_accuracy': test_acc_orig
        }

    # ============================================================================
    # Train on Transformed Embeddings
    # ============================================================================
    if mode in ['transformed', 'both']:
        logging.info("\n" + "="*70)
        logging.info("EXPERIMENT 2: Training MLP on Transformed Embeddings")
        logging.info("="*70)

        # Prepare data loaders
        train_dataset_trans = TensorDataset(
            torch.FloatTensor(train_embeddings_transformed[train_idx]),
            torch.LongTensor(train_labels[train_idx])
        )
        val_dataset_trans = TensorDataset(
            torch.FloatTensor(train_embeddings_transformed[val_idx]),
            torch.LongTensor(train_labels[val_idx])
        )
        test_dataset_trans = TensorDataset(
            torch.FloatTensor(test_embeddings_transformed),
            torch.LongTensor(test_labels)
        )

        train_loader_trans = DataLoader(train_dataset_trans, batch_size=batch_size, shuffle=True)
        val_loader_trans = DataLoader(val_dataset_trans, batch_size=batch_size, shuffle=False)
        test_loader_trans = DataLoader(test_dataset_trans, batch_size=batch_size, shuffle=False)

        # Initialize model
        model_trans = TwoLayerMLP(input_dim=2048, hidden_dim=hidden_dim,
                                  num_classes=3, dropout=dropout).to(device)
        optimizer_trans = optim.Adam(model_trans.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

        # Train
        logging.info("Training model on transformed embeddings...")
        history_trans = train_model(model_trans, train_loader_trans, val_loader_trans,
                                    criterion, optimizer_trans, num_epochs, device)

        # Test
        test_acc_trans = evaluate_model(model_trans, test_loader_trans, device)
        logging.info(f"\nTransformed Embeddings - Test Accuracy: {test_acc_trans:.2f}%")

        # Save model
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model_trans.state_dict(), output_path / 'mlp_model_transformed.pth')
        logging.info(f"Saved model to {output_path / 'mlp_model_transformed.pth'}")

        results['transformed'] = {
            'history': history_trans,
            'test_accuracy': test_acc_trans
        }

    # ============================================================================
    # Results Summary and Visualization
    # ============================================================================
    logging.info("\n" + "="*70)
    logging.info("RESULTS SUMMARY")
    logging.info("="*70)

    if 'original' in results:
        logging.info(f"\nOriginal Embeddings:")
        logging.info(f"  Final Train Accuracy: {results['original']['history']['train_acc'][-1]:.2f}%")
        logging.info(f"  Final Val Accuracy: {results['original']['history']['val_acc'][-1]:.2f}%")
        logging.info(f"  Test Accuracy: {results['original']['test_accuracy']:.2f}%")

    if 'transformed' in results:
        logging.info(f"\nTransformed Embeddings:")
        logging.info(f"  Final Train Accuracy: {results['transformed']['history']['train_acc'][-1]:.2f}%")
        logging.info(f"  Final Val Accuracy: {results['transformed']['history']['val_acc'][-1]:.2f}%")
        logging.info(f"  Test Accuracy: {results['transformed']['test_accuracy']:.2f}%")

    if mode == 'both':
        improvement = results['transformed']['test_accuracy'] - results['original']['test_accuracy']
        logging.info(f"\nTest Accuracy Improvement: {improvement:+.2f}%")

        # Plot comparison
        logging.info("\n" + "="*70)
        logging.info("Generating comparison plots...")
        plot_comparison(results['original']['history'],
                       results['transformed']['history'],
                       save_path=f'{output_dir}/mlp_comparison.png')

    # Save results
    logging.info("\nSaving results...")
    results_dict = {
        'hyperparameters': {
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout': dropout,
            'num_epochs': num_epochs,
            'val_split': val_split
        },
        'best_sigma': best_sigma
    }
    results_dict.update(results)

    np.savez(f'{output_dir}/mlp_results.npz', **results_dict)
    logging.info(f"Saved training results to {output_dir}/mlp_results.npz")

    logging.info("\n" + "="*70)
    logging.info("✓ Training complete!")
    logging.info("="*70)


def main():
    parser = argparse.ArgumentParser(description='Train MLP on embeddings')
    parser.add_argument('--data-file', type=str, default='data/embeddings_transformed.npz',
                       help='Path to transformed embeddings file')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['original', 'transformed', 'both'],
                       help='Which embeddings to train on')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Check if data file exists
    if not Path(args.data_file).exists():
        logging.error(f"Data file not found: {args.data_file}")
        logging.error("Run: python transform_embeddings.py")
        return

    # Train models
    train_and_evaluate(
        data_file=args.data_file,
        mode=args.mode,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        dropout=args.dropout,
        val_split=args.val_split,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
