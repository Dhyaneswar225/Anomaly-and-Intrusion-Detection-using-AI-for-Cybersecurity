# src/autoencoder_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score
import os

# ============================= CONFIG =============================
DATA_PROCESSED = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================= MODEL =============================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================= TRAINING & EVALUATION =============================
def train_autoencoder():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    # Safety check
    required_col = "label_binary"
    if required_col not in train_df.columns or required_col not in test_df.columns:
        raise ValueError(f"Both datasets must contain '{required_col}' column!")

    # Define label columns to drop
    label_columns = ["label", "label_attack", "label_binary"]

    # Extract features (drop all label columns)
    feature_cols = [col for col in train_df.columns if col not in label_columns]
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Binary labels: normal=0, attack=1
    y_train = (train_df["label_binary"] == "attack").astype(int).values
    y_test = (test_df["label_binary"] == "attack").astype(int).values

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Train samples: {len(X_train):,} (Normal: {sum(y_train == 0):,}, Attack: {sum(y_train == 1):,})")
    print(f"Test samples : {len(X_test):,} (Normal: {sum(y_test == 0):,}, Attack: {sum(y_test == 1):,})")

    # Use only NORMAL samples for training the autoencoder
    normal_mask_train = (y_train == 0)
    X_normal = X_train[normal_mask_train]
    print(f"Training Autoencoder on {len(X_normal):,} normal samples only")

    # DataLoader
    dataset = TensorDataset(torch.tensor(X_normal, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

    # Model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, encoding_dim=32).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Training loop
    print("Starting training...")
    epochs = 50
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            x = batch[0].to(DEVICE)
            recon = model(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        avg_loss = epoch_loss / len(X_normal)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")

    # Save Model
    model_path = f"{MODELS_DIR}/autoencoder.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'feature_columns': feature_cols
    }, model_path)
    print(f"Model saved: {model_path}")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss", color="tab:blue")
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/ae_training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ============================= EVALUATION =============================
    print("Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        # Reconstruction error on normal training data (for threshold)
        X_normal_tensor = torch.tensor(X_normal, dtype=torch.float32).to(DEVICE)
        recon_train = model(X_normal_tensor)
        train_errors = torch.mean((recon_train - X_normal_tensor) ** 2, dim=1).cpu().numpy()

        # Reconstruction error on test data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        recon_test = model(X_test_tensor)
        test_errors = torch.mean((recon_test - X_test_tensor) ** 2, dim=1).cpu().numpy()

    # Threshold: 95th percentile of reconstruction error on normal training data
    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th percentile of normal train errors): {threshold:.6f}")

    # Predictions based on threshold
    y_pred = (test_errors > threshold).astype(int)

    # ============================= METRICS =============================
    auc_roc = roc_auc_score(y_test, test_errors)
    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    auc_pr = auc(recall, precision)

    # Precision @ Top 10%
    k = max(1, int(0.1 * len(test_errors)))
    top_k_idx = np.argsort(test_errors)[-k:]
    precision_at_10 = np.mean(y_test[top_k_idx])

    # Standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAutoencoder Results:")
    print(f"  ROC-AUC         : {auc_roc:.6f}")
    print(f"  PR-AUC          : {auc_pr:.6f}")
    print(f"  Precision@10%   : {precision_at_10:.6f}")
    print(f"  Accuracy        : {accuracy:.6f}")
    print(f"  F1 Score        : {f1:.6f}")
    print(f"  Threshold used  : {threshold:.6f}")

    # Save detailed scores
    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": y_test,
        "predicted_label": y_pred
    })
    scores_df.to_csv(f"{RESULTS_DIR}/autoencoder_scores.csv", index=False)

    # Save metrics summary
    metrics_df = pd.DataFrame({
        "Model": ["Autoencoder"],
        "ROC_AUC": [auc_roc],
        "PR_AUC": [auc_pr],
        "Precision@10%": [precision_at_10],
        "Accuracy": [accuracy],
        "F1_Score": [f1],
        "Threshold": [threshold]
    })
    metrics_df.to_csv(f"{RESULTS_DIR}/autoencoder_metrics.csv", index=False)

    print(f"\nResults saved in '{RESULTS_DIR}'")

    # ============================= FEATURE IMPORTANCE (Reconstruction Error) =============================
    with torch.no_grad():
        feature_errors = torch.abs(recon_test - X_test_tensor).cpu().numpy()
        feature_mse = np.mean(feature_errors, axis=0)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'avg_recon_error': feature_mse
    }).sort_values(by='avg_recon_error', ascending=False)

    feature_importance.to_csv(f"{RESULTS_DIR}/autoencoder_feature_errors.csv", index=False)

    plt.figure(figsize=(10, 8))
    top_n = min(15, len(feature_cols))
    sns.barplot(data=feature_importance.head(top_n),
                y='feature', x='avg_recon_error', palette="viridis")
    plt.title(f"Top {top_n} Features by Average Reconstruction Error")
    plt.xlabel("Mean Absolute Reconstruction Error")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ae_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Top 10 most anomalous features:")
    print(feature_importance.head(10)[['feature', 'avg_recon_error']])

    print("\nAutoencoder training & evaluation complete!")
    print(f"Model → {model_path}")
    print(f"All results saved in '{RESULTS_DIR}' folder\n")


# RUN
if __name__ == "__main__":
    train_autoencoder()