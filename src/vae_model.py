# src/vae_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    accuracy_score, f1_score
)
import os

# ============================= CONFIG =============================
DATA_PROCESSED = "data/processed"
RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================= VAE MODEL =============================
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ============================= VAE LOSS (FIXED) =============================
def vae_loss(recon_x, x, mu, logvar, epoch, total_epochs=50, beta=1.0):
    """
    Stable VAE loss:
    - Mean reconstruction loss
    - Mean KL divergence
    - KL annealing
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    kl_weight = min(1.0, (epoch + 1) / (0.8 * total_epochs)) * beta
    return recon_loss + kl_weight * kld


# ============================= TRAINING & EVALUATION =============================
def train_vae():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df  = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    label_cols = ["label", "label_attack", "label_binary"]
    feature_cols = [c for c in train_df.columns if c not in label_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)

    y_train = (train_df["label_binary"] == "attack").astype(int).values
    y_test  = (test_df["label_binary"] == "attack").astype(int).values

    print(f"Features: {len(feature_cols)} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Normal (train): {sum(y_train == 0):,}")

    # ============================= NO SCALING HERE =============================
    X_train_scaled = X_train
    X_test_scaled  = X_test

    # Train only on normal samples
    X_normal = X_train_scaled[y_train == 0]
    print(f"Training VAE on {len(X_normal):,} normal samples")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_normal)),
        batch_size=256,
        shuffle=True
    )

    # ============================= MODEL =============================
    model = VAE(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    print("Starting VAE training...")
    epochs = 50
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for (x,) in train_loader:
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar, epoch, epochs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), f"{MODELS_DIR}/vae.pth")
    print(f"Model saved → {MODELS_DIR}/vae.pth")

    # ============================= EVALUATION =============================
    model.eval()

    # Test reconstruction errors
    test_errors = []
    with torch.no_grad():
        for x in torch.tensor(X_test_scaled).split(256):
            x = x.to(DEVICE)
            recon, _, _ = model(x)
            err = torch.mean((recon - x) ** 2, dim=1)
            test_errors.extend(err.cpu().numpy())

    test_errors = np.array(test_errors)

    # Train reconstruction errors (normal only)
    train_errors = []
    with torch.no_grad():
        for x in torch.tensor(X_normal).split(256):
            x = x.to(DEVICE)
            recon, _, _ = model(x)
            err = torch.mean((recon - x) ** 2, dim=1)
            train_errors.extend(err.cpu().numpy())

    threshold = np.percentile(train_errors, 95)
    preds = (test_errors > threshold).astype(int)

    # ============================= METRICS =============================
    roc_auc = roc_auc_score(y_test, test_errors)
    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    pr_auc = auc(recall, precision)

    precision_at_10 = np.mean(
        y_test[np.argsort(test_errors)[-int(0.1 * len(test_errors)) :]]
    )

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\nVAE Results")
    print("=" * 55)
    print(f"ROC-AUC       : {roc_auc:.6f}")
    print(f"PR-AUC        : {pr_auc:.6f}")
    print(f"Precision@10% : {precision_at_10:.6f}")
    print(f"Accuracy      : {acc:.6f}")
    print(f"F1 Score      : {f1:.6f}")
    print(f"Threshold     : {threshold:.6f}")
    print("=" * 55)

    print("VAE training and evaluation completed successfully!")


if __name__ == "__main__":
    train_vae()
