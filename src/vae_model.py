# src/vae_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
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
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Optional: for bounded outputs

    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.fc4(h)  # Linear output (scaled data)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# VAE loss with **slow** KL annealing
def vae_loss(recon_x, x, mu, logvar, epoch, total_epochs=50, beta=1.0):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # **SLOWER KL RAMP-UP** — over 80% of epochs
    kl_weight = min(1.0, epoch / (0.8 * total_epochs)) * beta
    
    return MSE + kl_weight * KLD, MSE.item()

# ============================= TRAINING & EVALUATION =============================
def train_vae():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df  = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
    y_train = (train_df["label"] == "attack").astype(int).values
    X_test  = test_df.drop(columns=["label"]).values.astype(np.float32)
    y_test  = (test_df["label"] == "attack").astype(int).values

    # === SCALE FEATURES ===
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODELS_DIR}/vae_scaler.pkl")

    # Use only NORMAL samples
    normal_mask_train = (y_train == 0)
    X_normal = X_train_scaled[normal_mask_train]
    print(f"Training on {len(X_normal):,} normal samples")

    # DataLoader
    dataset = TensorDataset(torch.tensor(X_normal))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Model (larger capacity)
    input_dim = X_train.shape[1]
    model = VAE(input_dim, hidden_dim=256, latent_dim=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # Training loop — **NO EARLY STOPPING**
    print("Starting training (50 epochs, no early stopping)...")
    epochs = 50
    total_loss_history = []
    mse_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        for batch in loader:
            x = batch[0].to(DEVICE)
            recon, mu, logvar = model(x)
            loss, mse_batch = vae_loss(recon, x, mu, logvar, epoch, epochs)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent explosion
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_batch

        avg_loss = epoch_loss / len(X_normal)
        avg_mse = epoch_mse / len(X_normal)
        total_loss_history.append(avg_loss)
        mse_history.append(avg_mse)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d} | Total Loss: {avg_loss:8.4f} | MSE: {avg_mse:8.6f} | KL Weight: {min(1.0, (epoch+1)/(0.8*epochs)):.3f}")

    # Save final model
    torch.save(model.state_dict(), f"{MODELS_DIR}/vae.pth")
    print(f"Final model saved: {MODELS_DIR}/vae.pth")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_loss_history, label="Total Loss", color="#1f77b4")
    plt.title("VAE Total Loss (ELBO)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(mse_history, label="Reconstruction MSE", color="#ff7f0e")
    plt.title("VAE Reconstruction MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/vae_training_curves.png", dpi=150)
    plt.close()

    # ============================= EVALUATION =============================
    print("Evaluating on test set...")
    model.eval()

    test_dataset = TensorDataset(torch.tensor(X_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    test_errors = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(DEVICE)
            recon, _, _ = model(x)
            mse = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            test_errors.extend(mse)
    test_errors = np.array(test_errors)

    # Train errors for threshold
    train_errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            recon, _, _ = model(x)
            mse = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            train_errors.extend(mse)
    train_errors = np.array(train_errors)

    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th %ile of normal): {threshold:.6f}")

    # Metrics
    auc_roc = roc_auc_score(y_test, test_errors)
    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    auc_pr = auc(recall, precision)
    k = int(0.1 * len(test_errors))
    top_k_idx = np.argsort(test_errors)[-k:]
    precision_at_10 = np.mean(y_test[top_k_idx])

    print(f"\nVAE Results:")
    print(f"  ROC-AUC     : {auc_roc:.6f}")
    print(f"  PR-AUC      : {auc_pr:.6f}")
    print(f"  Precision@10%: {precision_at_10:.6f}")

    # Save scores
    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": y_test,
        "predicted_label": (test_errors > threshold).astype(int)
    })
    scores_df.to_csv(f"{RESULTS_DIR}/vae_scores.csv", index=False)

    # Distribution plots
    plt.figure(figsize=(10, 6))
    sns.histplot(test_errors[y_test == 0], label="Normal", color="#1f77b4", alpha=0.7, bins=50)
    sns.histplot(test_errors[y_test == 1], label="Attack", color="#d62728", alpha=0.7, bins=50)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    plt.title("VAE – Reconstruction Error Distribution")
    plt.xlabel("MSE"); plt.ylabel("Count"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/vae_error_distribution.png", dpi=150)
    plt.close()

    # Normalized
    norm_errors = test_errors / threshold
    plt.figure(figsize=(10, 6))
    sns.histplot(norm_errors[y_test == 0], label="Normal", color="#1f77b4", alpha=0.7, bins=50)
    sns.histplot(norm_errors[y_test == 1], label="Attack", color="#d62728", alpha=0.7, bins=50)
    plt.axvline(1.0, color='black', linestyle='--', linewidth=2, label='Threshold = 1.0')
    plt.xlabel("Normalized Error"); plt.title("VAE – Normalized Error")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/vae_error_distribution_normalized.png", dpi=150)
    plt.close()

    print(f"\nAll results saved in: {RESULTS_DIR}/")
    print(f"Model and scaler saved in: {MODELS_DIR}/")
    print("VAE training completed successfully!")

# ============================= RUN =============================
if __name__ == "__main__":
    train_vae()