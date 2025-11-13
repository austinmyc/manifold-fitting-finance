import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from utils.zhipu_embedding import get_zhipu_embedding
from manfit.manfit_ours_gpu import manfit_ours_gpu_batched as manfit_ours
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compare_mlp.log'),
        logging.StreamHandler()
    ]
)


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


def get_embeddings_in_batches(sentences, batch_size=20):
    """Get embeddings for sentences in batches"""
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    all_embeddings = []

    for batch_idx in tqdm(range(num_batches), desc="Getting embeddings"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sentences))
        batch_sentences = sentences[start_idx:end_idx]

        try:
            response = get_zhipu_embedding(model="embedding-3", input=batch_sentences)
            batch_embeddings = np.array([item.embedding for item in response.data])
            all_embeddings.append(batch_embeddings)

            # Rate limiting
            if batch_idx < num_batches - 1:
                time.sleep(0.5)

        except Exception as e:
            logging.error(f"Error processing batch {batch_idx+1}: {e}")
            raise

    return np.vstack(all_embeddings)


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


def plot_comparison(history_original, history_transformed, save_path='mlp_comparison.png'):
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved comparison plot to {save_path}")
    plt.close()


def main():
    logging.info("="*70)
    logging.info("MLP Comparison: Original vs Transformed Embeddings")
    logging.info("="*70)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Hyperparameters
    HIDDEN_DIM = 256
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50

    logging.info(f"\nHyperparameters:")
    logging.info(f"  Hidden dimension: {HIDDEN_DIM}")
    logging.info(f"  Batch size: {BATCH_SIZE}")
    logging.info(f"  Learning rate: {LEARNING_RATE}")
    logging.info(f"  Number of epochs: {NUM_EPOCHS}")

    # Step 1: Load saved training embeddings
    logging.info("\n" + "="*70)
    logging.info("Step 1: Loading saved training embeddings...")
    data = np.load('finsent_embeddings.npz', allow_pickle=True)
    train_embeddings_original = data['original']
    train_embeddings_transformed = data['transformed']
    best_sigma = float(data['best_sigma'])
    logging.info(f"Loaded training embeddings: {train_embeddings_original.shape}")
    logging.info(f"Best sigma from training: {best_sigma:.4f}")

    # Step 2: Load test data and get embeddings
    logging.info("\n" + "="*70)
    logging.info("Step 2: Loading test data and getting embeddings...")
    dataset = load_from_disk("data/finsent")
    train_data = dataset['train']
    test_data = dataset['test']

    train_labels = np.array([train_data[i]['label'] for i in range(len(train_data))])
    test_sentences = [test_data[i]['text'] for i in range(len(test_data))]
    test_labels = np.array([test_data[i]['label'] for i in range(len(test_data))])

    logging.info(f"Train samples: {len(train_labels)}")
    logging.info(f"Test samples: {len(test_labels)}")

    logging.info("\nGetting test embeddings from ZhiPu...")
    test_embeddings_original = get_embeddings_in_batches(test_sentences, batch_size=20)
    logging.info(f"Test embeddings shape: {test_embeddings_original.shape}")

    # Step 3: Transform test embeddings using transformed training data
    logging.info("\n" + "="*70)
    logging.info("Step 3: Transforming test embeddings using fitted manifold...")
    logging.info(f"Using transformed training embeddings as reference with sigma={best_sigma:.4f}")
    test_embeddings_transformed = manfit_ours(
        sample=train_embeddings_transformed,  # Use transformed training as reference
        sig=best_sigma,
        sample_init=test_embeddings_original
    )
    logging.info(f"Transformed test embeddings shape: {test_embeddings_transformed.shape}")

    # Step 4: Split training data into train/val
    logging.info("\n" + "="*70)
    logging.info("Step 4: Splitting training data into train/val (80/20)...")
    indices = np.arange(len(train_labels))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=train_labels)

    logging.info(f"Training samples: {len(train_idx)}")
    logging.info(f"Validation samples: {len(val_idx)}")

    # ============================================================================
    # EXPERIMENT 1: Original Embeddings
    # ============================================================================
    logging.info("\n" + "="*70)
    logging.info("EXPERIMENT 1: Training MLP on Original Embeddings")
    logging.info("="*70)

    # Prepare data loaders for original embeddings
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

    train_loader_orig = DataLoader(train_dataset_orig, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_orig = DataLoader(val_dataset_orig, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_orig = DataLoader(test_dataset_orig, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss, optimizer
    model_orig = TwoLayerMLP(input_dim=2048, hidden_dim=HIDDEN_DIM, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_orig = optim.Adam(model_orig.parameters(), lr=LEARNING_RATE)

    # Train
    logging.info("Training model on original embeddings...")
    history_orig = train_model(model_orig, train_loader_orig, val_loader_orig,
                               criterion, optimizer_orig, NUM_EPOCHS, device)

    # Test
    test_acc_orig = evaluate_model(model_orig, test_loader_orig, device)
    logging.info(f"\nOriginal Embeddings - Test Accuracy: {test_acc_orig:.2f}%")

    # Save model
    torch.save(model_orig.state_dict(), 'mlp_model_original.pth')
    logging.info("Saved model to mlp_model_original.pth")

    # ============================================================================
    # EXPERIMENT 2: Transformed Embeddings
    # ============================================================================
    logging.info("\n" + "="*70)
    logging.info("EXPERIMENT 2: Training MLP on Transformed Embeddings")
    logging.info("="*70)

    # Prepare data loaders for transformed embeddings
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

    train_loader_trans = DataLoader(train_dataset_trans, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_trans = DataLoader(val_dataset_trans, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_trans = DataLoader(test_dataset_trans, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model with same architecture
    model_trans = TwoLayerMLP(input_dim=2048, hidden_dim=HIDDEN_DIM, num_classes=3).to(device)
    optimizer_trans = optim.Adam(model_trans.parameters(), lr=LEARNING_RATE)

    # Train
    logging.info("Training model on transformed embeddings...")
    history_trans = train_model(model_trans, train_loader_trans, val_loader_trans,
                                criterion, optimizer_trans, NUM_EPOCHS, device)

    # Test
    test_acc_trans = evaluate_model(model_trans, test_loader_trans, device)
    logging.info(f"\nTransformed Embeddings - Test Accuracy: {test_acc_trans:.2f}%")

    # Save model
    torch.save(model_trans.state_dict(), 'mlp_model_transformed.pth')
    logging.info("Saved model to mlp_model_transformed.pth")

    # ============================================================================
    # Results Summary and Visualization
    # ============================================================================
    logging.info("\n" + "="*70)
    logging.info("RESULTS SUMMARY")
    logging.info("="*70)
    logging.info(f"\nOriginal Embeddings:")
    logging.info(f"  Final Train Accuracy: {history_orig['train_acc'][-1]:.2f}%")
    logging.info(f"  Final Val Accuracy: {history_orig['val_acc'][-1]:.2f}%")
    logging.info(f"  Test Accuracy: {test_acc_orig:.2f}%")

    logging.info(f"\nTransformed Embeddings:")
    logging.info(f"  Final Train Accuracy: {history_trans['train_acc'][-1]:.2f}%")
    logging.info(f"  Final Val Accuracy: {history_trans['val_acc'][-1]:.2f}%")
    logging.info(f"  Test Accuracy: {test_acc_trans:.2f}%")

    improvement = test_acc_trans - test_acc_orig
    logging.info(f"\nTest Accuracy Improvement: {improvement:+.2f}%")

    # Plot comparison
    logging.info("\n" + "="*70)
    logging.info("Generating comparison plots...")
    plot_comparison(history_orig, history_trans, save_path='mlp_comparison.png')

    # Save test embeddings
    logging.info("\nSaving test embeddings...")
    np.savez('test_embeddings.npz',
             original=test_embeddings_original,
             transformed=test_embeddings_transformed,
             labels=test_labels,
             sentences=test_sentences)
    logging.info("Saved test embeddings to test_embeddings.npz")

    # Save all results
    logging.info("\nSaving complete results...")
    results = {
        'hyperparameters': {
            'hidden_dim': HIDDEN_DIM,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS
        },
        'original': {
            'history': history_orig,
            'test_accuracy': test_acc_orig
        },
        'transformed': {
            'history': history_trans,
            'test_accuracy': test_acc_trans
        },
        'best_sigma': best_sigma
    }

    np.savez('mlp_comparison_results.npz', **results)
    logging.info("Saved training results to mlp_comparison_results.npz")

    # Save summary
    logging.info("\n" + "="*70)
    logging.info("Files saved:")
    logging.info("  1. test_embeddings.npz - Test embeddings (original & transformed)")
    logging.info("  2. mlp_model_original.pth - Trained MLP on original embeddings")
    logging.info("  3. mlp_model_transformed.pth - Trained MLP on transformed embeddings")
    logging.info("  4. mlp_comparison_results.npz - Training history and metrics")
    logging.info("  5. mlp_comparison.png - Comparison plots")

    logging.info("\n" + "="*70)
    logging.info("✓ Comparison completed successfully!")
    logging.info("="*70)


if __name__ == "__main__":
    main()
