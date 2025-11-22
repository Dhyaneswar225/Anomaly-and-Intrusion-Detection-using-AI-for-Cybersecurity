# src/vae_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score, confusion_matrix
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

# ============================= VAE MODEL =============================
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
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, epoch, total_epochs=50, beta=1.0):
    """VAE loss with KL annealing"""
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Anneal KL weight (ramp up over 80% of training)
    kl_weight = min(1.0, (epoch + 1) / (0.8 * total_epochs)) * beta

    total_loss = mse + kl_weight * kld
    return total_loss, mse.item()


# ============================= TRAINING & EVALUATION =============================
def train_vae():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df  = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    # Safety check
    if "label_binary" not in train_df.columns:
        raise ValueError("Column 'label_binary' not found in dataset!")

    # Define label columns to drop
    label_columns = ["label", "label_attack", "label_binary"]
    feature_columns = [col for col in train_df.columns if col not in label_columns]

    # Extract features and binary labels
    X_train = train_df[feature_columns].values.astype(np.float32)
    X_test  = test_df[feature_columns].values.astype(np.float32)

    y_train = (train_df["label_binary"] == "attack").astype(int).values
    y_test  = (test_df["label_binary"]  == "attack").astype(int).values

    print(f"Features: {len(feature_columns)} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Normal (train): {sum(y_train == 0):,} | Attack (train): {sum(y_train == 1):,}")

    # ============================= SCALING =============================
    print("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, f"{MODELS_DIR}/vae_scaler.pkl")
    print(f"Scaler saved → {MODELS_DIR}/vae_scaler.pkl")

    # Use only normal samples for training
    normal_mask = (y_train == 0)
    X_normal_train = X_train_scaled[normal_mask]
    print(f"Training VAE on {len(X_normal_train):,} normal samples only")

    # DataLoader
    dataset = TensorDataset(torch.tensor(X_normal_train))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=False)

    # ============================= MODEL =============================
    input_dim = X_train.shape[1]
    model = VAE(input_dim=input_dim, hidden_dim=256, latent_dim=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    print("Starting VAE training...")
    epochs = 50
    total_losses = []
    mse_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_mse_loss = 0.0

        for batch in loader:
            x = batch[0].to(DEVICE)
            recon, mu, logvar = model(x)

            loss, mse_batch = vae_loss(recon, x, mu, logvar, epoch, epochs, beta=1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_mse_loss += mse_batch

        avg_total = epoch_total_loss / len(X_normal_train)
        avg_mse = epoch_mse_loss / len(X_normal_train)
        total_losses.append(avg_total)
        mse_losses.append(avg_mse)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d} | Total Loss: {avg_total:.6f} | MSE: {avg_mse:.6f}")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'feature_columns': feature_columns,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_
    }, f"{MODELS_DIR}/vae.pth")
    print(f"VAE model saved → {MODELS_DIR}/vae.pth")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_losses, label="Total Loss (MSE + KL)")
    plt.title("VAE Total Loss"); plt.grid(True); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(mse_losses, label="Reconstruction MSE", color="orange")
    plt.title("Reconstruction Loss"); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/vae_training_loss.png", dpi=150, bbox_inches='tight')
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

    # Threshold from normal training data
    train_errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            recon, _, _ = model(x)
            mse = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            train_errors.extend(mse)
    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th percentile on normal): {threshold:.6f}")

    pred_labels = (test_errors > threshold).astype(int)

    # ============================= METRICS =============================
    roc_auc = roc_auc_score(y_test, test_errors)
    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    pr_auc = auc(recall, precision)
    k = max(1, int(0.1 * len(test_errors)))
    precision_at_10 = np.mean(y_test[np.argsort(test_errors)[-k:]])

    accuracy = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_labels).ravel()

    print("\n" + "="*60)
    print("VAE Anomaly Detection Results")
    print("="*60)
    print(f"  ROC-AUC           : {roc_auc:.6f}")
    print(f"  PR-AUC            : {pr_auc:.6f}")
    print(f"  Precision@10%     : {precision_at_10:.6f}")
    print(f"  Accuracy          : {accuracy:.6f}")
    print(f"  F1 Score          : {f1:.6f}")
    print(f"  Threshold         : {threshold:.6f}")
    print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print("="*60)

    # Save results
    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": y_test,
        "predicted_label": pred_labels
    })
    scores_df.to_csv(f"{RESULTS_DIR}/vae_scores.csv", index=False)

    metrics_df = pd.DataFrame({
        "Model": ["VAE"],
        "ROC_AUC": [roc_auc],
        "PR_AUC": [pr_auc],
        "Precision@10%": [precision_at_10],
        "Accuracy": [accuracy],
        "F1_Score": [f1],
        "Threshold": [threshold]
    })
    metrics_df.to_csv(f"{RESULTS_DIR}/vae_metrics.csv", index=False)

    print(f"\nAll results saved in '{RESULTS_DIR}'")
    print(f"Model + scaler saved in '{MODELS_DIR}'")
    print("VAE training and evaluation completed successfully!")


# ============================= RUN =============================
if __name__ == "__main__":
    train_vae()