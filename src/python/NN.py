"""
German Credit Risk Classification using PyTorch Neural Network
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_curve, auc
from data_prep import load_and_preprocess_data
from ROC_gen import ROC_Generator
from conf_matrix import Matrix_Display

X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 100
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2

# Early stopping parameters
PATIENCE = 3
VALIDATION_SPLIT = 0.2

# Convert to PyTorch tensors and move to appropriate device
X_train_tensor = torch.tensor(X_train.values.astype(np.float32)).to(device)
X_test_tensor = torch.tensor(X_test.values.astype(np.float32)).to(device)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).unsqueeze(1).to(device)

class CreditRiskClassifier(nn.Module):
    """
    Neural network for binary credit risk classification.

    Architecture:
    - Input layer: matches number of features
    - Hidden layer 1: 64 neurons with ReLU activation and dropout
    - Hidden layer 2: 32 neurons with ReLU activation and dropout
    - Output layer: 1 neuron with sigmoid activation for binary classification
    """

    def __init__(self, input_dim):
        super(CreditRiskClassifier, self).__init__()
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_1),

            # Second hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE_2),

            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.model(x)


input_dimension = X_train_tensor.shape[1]
model = CreditRiskClassifier(input_dimension).to(device)

criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create validation split from training data
validation_size = int(len(X_train_tensor) * VALIDATION_SPLIT)
training_size = len(X_train_tensor) - validation_size

# Create dataset and split
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset,
    [training_size, validation_size]
)

# Create data loaders for batching
train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_subset,
    batch_size=BATCH_SIZE
)

print("\nStarting training with early stopping...")
print(f"Early stopping patience: {PATIENCE} epochs")

# Initialize early stopping variables
best_validation_loss = float('inf')
epochs_without_improvement = 0

# Training loop
for epoch in range(MAX_EPOCHS):
    model.train()
    training_losses = []

    for batch_features, batch_labels in train_loader:
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_features)
        loss = criterion(predictions, batch_labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())

    model.eval()
    validation_losses = []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            validation_losses.append(loss.item())

    # Calculate average losses
    avg_train_loss = np.mean(training_losses)
    avg_val_loss = np.mean(validation_losses)

    print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} - "
        f"Train loss: {avg_train_loss:.4f} - "
        f"Val loss: {avg_val_loss:.4f}")

    # Early stopping logic
    if avg_val_loss < best_validation_loss:
        best_validation_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  → New best model saved (val_loss: {avg_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_validation_loss:.4f}")
            break

print("\nEvaluating model on test set...")

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

with torch.no_grad():
    y_probabilities = model(X_test_tensor).cpu().numpy().ravel()

    y_predictions = (y_probabilities >= 0.5).astype(int)

    y_true_labels = y_test_tensor.cpu().numpy()



print("\n" + "="*50)
print("CLASSIFICATION REPORT:")
print("="*50)
print(classification_report(
    y_true_labels,
    y_predictions,
    target_names=['Bad Credit (0)', 'Good Credit (1)']
))

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true_labels, y_probabilities)
roc_auc = auc(fpr, tpr)
ROC_Generator(fpr,tpr,roc_auc)
Matrix_Display(y_true_labels, y_predictions)

