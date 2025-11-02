# src/autoencoder_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
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
    test_df  = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    # Extract features and labels
    X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
    y_train = (train_df["label"] == "attack").astype(int).values
    X_test  = test_df.drop(columns=["label"]).values.astype(np.float32)
    y_test  = (test_df["label"] == "attack").astype(int).values

    # Use only NORMAL samples for training
    normal_mask_train = (y_train == 0)
    X_normal = X_train[normal_mask_train]
    print(f"Training on {len(X_normal):,} normal samples")

    # DataLoader
    dataset = TensorDataset(torch.tensor(X_normal, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, encoding_dim=32).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")

    # Save model
    model_path = f"{MODELS_DIR}/autoencoder.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss", color="#1f77b4")
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/ae_training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ============================= EVALUATION =============================
    print("Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        # Reconstruct TRAIN normal data
        X_normal_tensor = torch.tensor(X_normal, dtype=torch.float32).to(DEVICE)
        recon_train = model(X_normal_tensor)
        train_errors = torch.mean((recon_train - X_normal_tensor) ** 2, dim=1).cpu().numpy()

        # Reconstruct TEST data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        recon_test = model(X_test_tensor)
        test_errors = torch.mean((recon_test - X_test_tensor) ** 2, dim=1).cpu().numpy()

    # === THRESHOLD: 95th percentile of TRAIN normal errors ===
    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th %ile of normal train errors): {threshold:.6f}")

    # === METRICS ===
    auc_roc = roc_auc_score(y_test, test_errors)
    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    auc_pr = auc(recall, precision)

    k = int(0.1 * len(test_errors))
    top_k_idx = np.argsort(test_errors)[-k:]
    precision_at_10 = np.mean(y_test[top_k_idx])

    print(f"\nAutoencoder Results:")
    print(f"  ROC-AUC     : {auc_roc:.6f}")
    print(f"  PR-AUC      : {auc_pr:.6f}")
    print(f"  Precision@10%: {precision_at_10:.6f}")

    # === NORMALIZED ERROR PLOT (ROBUST & INTERPRETABLE) ===
    base_error = np.percentile(train_errors, 95)  # Use 95th %ile as scale
    norm_errors = test_errors / base_error
    norm_threshold = 1.0  # Threshold = 95th %ile → 1.0

    plt.figure(figsize=(10, 6))
    sns.histplot(norm_errors[y_test == 0], label="Normal", color="#1f77b4", alpha=0.7, bins=50)
    sns.histplot(norm_errors[y_test == 1], label="Attack", color="#d62728", alpha=0.7, bins=50)
    plt.axvline(norm_threshold, color='black', linestyle='--', linewidth=2, label='Threshold = 1.0')
    plt.xlabel("Normalized Reconstruction Error\n(× 95th percentile of normal)")
    plt.ylabel("Count")
    plt.title("Autoencoder – Normalized Error Distribution")
    plt.legend()
    plt.xlim(0, 4)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ae_error_distribution_normalized.png", dpi=150)
    plt.close()

    # === RAW ERROR PLOT (FOR REFERENCE) ===
    plt.figure(figsize=(10, 6))
    sns.histplot(test_errors[y_test == 0], label="Normal", color="#1f77b4", alpha=0.7, bins=50)
    sns.histplot(test_errors[y_test == 1], label="Attack", color="#d62728", alpha=0.7, bins=50)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.title("Autoencoder – Raw MSE Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ae_error_distribution_raw.png", dpi=150)
    plt.close()

    # === SAVE SCORES ===
    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "normalized_error": norm_errors,
        "true_label": y_test,
        "predicted_label": (test_errors > threshold).astype(int)
    })
    scores_df.to_csv(f"{RESULTS_DIR}/autoencoder_scores.csv", index=False)

    print(f"\nAll results saved in: {RESULTS_DIR}/")
    print(f"Model saved in: {MODELS_DIR}/")

    # === FEATURE-LEVEL ERROR ANALYSIS ===
    print("\nComputing feature-level reconstruction errors...")

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        recon_test = model(X_test_tensor)
        feature_errors = torch.abs(recon_test - X_test_tensor).cpu().numpy()  # [n_samples, n_features]
        feature_mse = np.mean(feature_errors, axis=0)  # Average error per feature

    # Get feature names
    feature_names = train_df.drop(columns=["label"]).columns

    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'avg_recon_error': feature_mse
    }).sort_values(by='avg_recon_error', ascending=False)

    # Save
    feature_importance.to_csv(f"{RESULTS_DIR}/autoencoder_feature_errors.csv", index=False)

    # Plot top 10
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    sns.barplot(data=top_features, x='avg_recon_error', y='feature', palette='viridis')
    plt.title("Top 10 Features with Highest Reconstruction Error")
    plt.xlabel("Mean Absolute Reconstruction Error")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ae_feature_importance.png", dpi=150)
    plt.close()

    print("Top 5 features with highest reconstruction error:")
    print(feature_importance.head(5)[['feature', 'avg_recon_error']])

    print("Autoencoder training and evaluation completed successfully!")

# ============================= RUN =============================
if __name__ == "__main__":
    train_autoencoder()