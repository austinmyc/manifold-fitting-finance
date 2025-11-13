"""
Test different sigma values to see which works best for classification.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Import the MLP and training functions
from train_mlp import TwoLayerMLP, evaluate_model

# Load original data
data = np.load('data/embeddings_transformed.npz', allow_pickle=True)
train_orig = data['train_original']
train_labels = data['train_labels']
test_orig = data['test_original']
test_labels = data['test_labels']

logging.info("="*70)
logging.info("Testing Original Embeddings (Baseline)")
logging.info("="*70)

# Train/val split
indices = np.arange(len(train_labels))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=train_labels)

# Create datasets
train_dataset = TensorDataset(
    torch.FloatTensor(train_orig[train_idx]),
    torch.LongTensor(train_labels[train_idx])
)
val_dataset = TensorDataset(
    torch.FloatTensor(train_orig[val_idx]),
    torch.LongTensor(train_labels[val_idx])
)
test_dataset = TensorDataset(
    torch.FloatTensor(test_orig),
    torch.LongTensor(test_labels)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Quick training (30 epochs for speed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TwoLayerMLP(input_dim=2048, hidden_dim=256, num_classes=3, dropout=0.3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

logging.info(f"Training for 30 epochs on device: {device}")
best_val_acc = 0
best_epoch = 0

for epoch in range(30):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    val_acc = 100. * correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        best_model_state = model.state_dict().copy()

# Restore best and test
model.load_state_dict(best_model_state)
test_acc_orig = evaluate_model(model, test_loader, device)

logging.info(f"\nOriginal Embeddings:")
logging.info(f"  Best Val Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
logging.info(f"  Test Accuracy: {test_acc_orig:.2f}%")

# Now test transformed
logging.info("\n" + "="*70)
logging.info("Testing Transformed Embeddings (sigma=0.1)")
logging.info("="*70)

train_trans = data['train_transformed']
test_trans = data['test_transformed']

train_dataset_trans = TensorDataset(
    torch.FloatTensor(train_trans[train_idx]),
    torch.LongTensor(train_labels[train_idx])
)
val_dataset_trans = TensorDataset(
    torch.FloatTensor(train_trans[val_idx]),
    torch.LongTensor(train_labels[val_idx])
)
test_dataset_trans = TensorDataset(
    torch.FloatTensor(test_trans),
    torch.LongTensor(test_labels)
)

train_loader_trans = DataLoader(train_dataset_trans, batch_size=64, shuffle=True)
val_loader_trans = DataLoader(val_dataset_trans, batch_size=64, shuffle=False)
test_loader_trans = DataLoader(test_dataset_trans, batch_size=64, shuffle=False)

model_trans = TwoLayerMLP(input_dim=2048, hidden_dim=256, num_classes=3, dropout=0.3).to(device)
optimizer_trans = optim.Adam(model_trans.parameters(), lr=0.001, weight_decay=1e-4)

best_val_acc_trans = 0
best_epoch_trans = 0

for epoch in range(30):
    model_trans.train()
    for batch_X, batch_y in train_loader_trans:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer_trans.zero_grad()
        outputs = model_trans(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_trans.step()

    # Validation
    model_trans.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader_trans:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model_trans(batch_X)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    val_acc = 100. * correct / total
    if val_acc > best_val_acc_trans:
        best_val_acc_trans = val_acc
        best_epoch_trans = epoch
        best_model_state_trans = model_trans.state_dict().copy()

model_trans.load_state_dict(best_model_state_trans)
test_acc_trans = evaluate_model(model_trans, test_loader_trans, device)

logging.info(f"\nTransformed Embeddings (sigma=0.1):")
logging.info(f"  Best Val Accuracy: {best_val_acc_trans:.2f}% (epoch {best_epoch_trans})")
logging.info(f"  Test Accuracy: {test_acc_trans:.2f}%")

# Summary
logging.info("\n" + "="*70)
logging.info("SUMMARY")
logging.info("="*70)
logging.info(f"Original:    Val={best_val_acc:.2f}%, Test={test_acc_orig:.2f}%")
logging.info(f"Transformed: Val={best_val_acc_trans:.2f}%, Test={test_acc_trans:.2f}%")
logging.info(f"Difference:  Val={best_val_acc_trans-best_val_acc:+.2f}%, Test={test_acc_trans-test_acc_orig:+.2f}%")

if best_epoch_trans < 10:
    logging.info(f"\n⚠️  WARNING: Transformed model peaked at epoch {best_epoch_trans} (very early)")
    logging.info("   This suggests severe overfitting or loss of information")
    logging.info("   RECOMMENDATION: Try larger sigma values (0.5, 1.0, 2.0)")
