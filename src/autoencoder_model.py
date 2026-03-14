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
import random

# ============================= REPRODUCIBILITY =============================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

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


# ============================= TRAINING FUNCTION =============================
def train_autoencoder():

    print("Loading processed data...")

    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    if "label_binary" not in train_df.columns:
        raise ValueError("label_binary column missing")

    label_columns = ["label", "label_attack", "label_binary"]

    feature_cols = [c for c in train_df.columns if c not in label_columns]

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    y_train = (train_df["label_binary"] == "attack").astype(int).values
    y_test = (test_df["label_binary"] == "attack").astype(int).values

    print(f"Feature dimension: {X_train.shape[1]}")
    print(f"Train samples: {len(X_train):,} (Normal: {sum(y_train==0):,}, Attack: {sum(y_train==1):,})")
    print(f"Test samples : {len(X_test):,} (Normal: {sum(y_test==0):,}, Attack: {sum(y_test==1):,})")

    # ============================= NORMAL DATA ONLY =============================
    X_normal = X_train[y_train == 0]

    print(f"Training Autoencoder on {len(X_normal):,} normal samples only")

    dataset = TensorDataset(torch.tensor(X_normal))

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        drop_last=False,
        generator=generator
    )

    # ============================= MODEL =============================
    input_dim = X_train.shape[1]

    model = Autoencoder(input_dim).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    # ============================= TRAINING =============================
    print("Starting training...")

    epochs = 50
    losses = []

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0

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
            print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")

    # ============================= SAVE MODEL =============================
    model_path = f"{MODELS_DIR}/autoencoder.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "feature_columns": feature_cols
    }, model_path)

    print(f"Model saved: {model_path}")

    # ============================= LOSS PLOT =============================
    plt.figure(figsize=(8,5))
    plt.plot(losses)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plt.savefig(f"{RESULTS_DIR}/ae_training_loss.png")
    plt.close()

    # ============================= EVALUATION =============================
    print("Evaluating on test set...")

    model.eval()

    with torch.no_grad():

        X_normal_tensor = torch.tensor(X_normal).to(DEVICE)
        recon_train = model(X_normal_tensor)

        train_errors = torch.mean(
            (recon_train - X_normal_tensor) ** 2,
            dim=1
        ).cpu().numpy()

        X_test_tensor = torch.tensor(X_test).to(DEVICE)

        recon_test = model(X_test_tensor)

        test_errors = torch.mean(
            (recon_test - X_test_tensor) ** 2,
            dim=1
        ).cpu().numpy()

    # ============================= THRESHOLD =============================
    threshold = np.percentile(train_errors, 95)

    print(f"Threshold (95th percentile): {threshold:.6f}")

    y_pred = (test_errors > threshold).astype(int)

    # ============================= METRICS =============================
    auc_roc = roc_auc_score(y_test, test_errors)

    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    auc_pr = auc(recall, precision)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    k = int(0.1 * len(test_errors))
    top_k_idx = np.argsort(test_errors)[-k:]

    precision_at_10 = np.mean(y_test[top_k_idx])

    print("\nAutoencoder Results:")
    print(f"ROC-AUC       : {auc_roc:.6f}")
    print(f"PR-AUC        : {auc_pr:.6f}")
    print(f"Precision@10% : {precision_at_10:.6f}")
    print(f"Accuracy      : {accuracy:.6f}")
    print(f"F1 Score      : {f1:.6f}")

    # ============================= SAVE SCORES =============================
    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": y_test,
        "predicted_label": y_pred
    })

    scores_df.to_csv(f"{RESULTS_DIR}/autoencoder_scores.csv", index=False)

    # ============================= FEATURE IMPORTANCE =============================
    with torch.no_grad():

        feature_errors = torch.abs(
            recon_test - X_test_tensor
        ).cpu().numpy()

        feature_mse = np.mean(feature_errors, axis=0)

    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "avg_recon_error": feature_mse
    }).sort_values("avg_recon_error", ascending=False)

    feature_importance.to_csv(
        f"{RESULTS_DIR}/autoencoder_feature_errors.csv",
        index=False
    )

    # ============================= FEATURE PLOT =============================
    top_n = min(15, len(feature_cols))

    plt.figure(figsize=(10,8))

    sns.barplot(
        data=feature_importance.head(top_n),
        y="feature",
        x="avg_recon_error",
        hue="feature",
        palette="viridis",
        legend=False
    )

    plt.title("Top Anomalous Features")
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR}/ae_feature_importance.png")
    plt.close()

    print("\nTop 10 anomalous features:")
    print(feature_importance.head(10))

    print("\nAutoencoder training & evaluation complete.")
    print(f"Results saved in: {RESULTS_DIR}")


# ============================= RUN =============================
if __name__ == "__main__":
    train_autoencoder()