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

# ============================= MODEL =============================
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=64):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

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
    mse = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    kl_weight = min(1.0, epoch / (0.8 * total_epochs)) * beta

    return mse + kl_weight * kld, mse.item()

# ============================= TRAINING & EVALUATION =============================
def train_vae():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df  = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
    y_train = (train_df["label"] == "attack").astype(int).values
    X_test  = test_df.drop(columns=["label"]).values.astype(np.float32)
    y_test  = (test_df["label"] == "attack").astype(int).values

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODELS_DIR}/vae_scaler.pkl")

    normal_mask_train = (y_train == 0)
    X_normal = X_train_scaled[normal_mask_train]
    print(f"Training on {len(X_normal):,} normal samples")

    dataset = TensorDataset(torch.tensor(X_normal))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    input_dim = X_train.shape[1]
    model = VAE(input_dim, hidden_dim=256, latent_dim=64).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    print("Starting training...")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_batch

        avg_loss = epoch_loss / len(X_normal)
        avg_mse = epoch_mse / len(X_normal)
        total_loss_history.append(avg_loss)
        mse_history.append(avg_mse)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f}")

    torch.save(model.state_dict(), f"{MODELS_DIR}/vae.pth")
    print(f"Model saved: {MODELS_DIR}/vae.pth")

    # ============================= EVALUATION =============================
    print("Evaluating model...")
    model.eval()

    test_dataset = TensorDataset(torch.tensor(X_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    test_errors = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(DEVICE)
            recon, _, _ = model(x)
            mse = torch.mean((recon - x)**2, dim=1).cpu().numpy()
            test_errors.extend(mse)

    test_errors = np.array(test_errors)

    train_errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            recon, _, _ = model(x)
            mse = torch.mean((recon - x)**2, dim=1).cpu().numpy()
            train_errors.extend(mse)

    train_errors = np.array(train_errors)
    threshold = np.percentile(train_errors, 95)

    pred_labels = (test_errors > threshold).astype(int)

    roc = roc_auc_score(y_test, test_errors)
    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    pr_auc = auc(recall, precision)
    top_k = int(0.1 * len(test_errors))
    precision_at_10 = np.mean(y_test[np.argsort(test_errors)[-top_k:]])

    # === NEW METRICS ===
    accuracy = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_labels).ravel()

    print("\n📌 VAE Results:")
    print(f"ROC-AUC       : {roc:.6f}")
    print(f"PR-AUC        : {pr_auc:.6f}")
    print(f"Precision@10% : {precision_at_10:.6f}")
    print(f"Accuracy      : {accuracy:.6f}")
    print(f"F1 Score      : {f1:.6f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": y_test,
        "predicted_label": pred_labels
    })
    scores_df.to_csv(f"{RESULTS_DIR}/vae_scores.csv", index=False)

    print("\nAll results saved in:", RESULTS_DIR)
    print("Model and scaler saved in:", MODELS_DIR)
    print("VAE training + evaluation completed successfully! 🚀")


if __name__ == "__main__":
    train_vae()
