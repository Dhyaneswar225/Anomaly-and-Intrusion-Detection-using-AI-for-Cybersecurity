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

    # Extract features and labels
    X_train = train_df.drop(columns=["label"]).values.astype(np.float32)
    y_train = (train_df["label"] == "attack").astype(int).values
    X_test = test_df.drop(columns=["label"]).values.astype(np.float32)
    y_test = (test_df["label"] == "attack").astype(int).values

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

    # Save Model
    model_path = f"{MODELS_DIR}/autoencoder.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
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
        X_normal_tensor = torch.tensor(X_normal, dtype=torch.float32).to(DEVICE)
        recon_train = model(X_normal_tensor)
        train_errors = torch.mean((recon_train - X_normal_tensor) ** 2, dim=1).cpu().numpy()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        recon_test = model(X_test_tensor)
        test_errors = torch.mean((recon_test - X_test_tensor) ** 2, dim=1).cpu().numpy()

    # Threshold: 95th percentile of normal train errors
    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th %ile): {threshold:.6f}")

    # Predictions
    y_pred = (test_errors > threshold).astype(int)

    # ============================= METRICS =============================
    auc_roc = roc_auc_score(y_test, test_errors)
    precision, recall, _ = precision_recall_curve(y_test, test_errors)
    auc_pr = auc(recall, precision)

    k = int(0.1 * len(test_errors))
    top_k_idx = np.argsort(test_errors)[-k:]
    precision_at_10 = np.mean(y_test[top_k_idx])

    # New Metrics Added ✔️
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAutoencoder Results:")
    print(f"  ROC-AUC      : {auc_roc:.6f}")
    print(f"  PR-AUC       : {auc_pr:.6f}")
    print(f"  Precision@10%: {precision_at_10:.6f}")
    print(f"  Accuracy     : {accuracy:.6f}")
    print(f"  F1 Score     : {f1:.6f}")

    # Save Scores
    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": y_test,
        "predicted_label": y_pred
    })
    scores_df.to_csv(f"{RESULTS_DIR}/autoencoder_scores.csv", index=False)

    # Save Metrics
    metrics_df = pd.DataFrame({
        "ROC_AUC": [auc_roc],
        "PR_AUC": [auc_pr],
        "Precision@10%": [precision_at_10],
        "Accuracy": [accuracy],
        "F1_Score": [f1]
    })
    metrics_df.to_csv(f"{RESULTS_DIR}/autoencoder_metrics.csv", index=False)

    print("\nMetrics saved as autoencoder_metrics.csv 🎯")
    print("Feature-level analysis starting...")

    # ============================= FEATURE-LEVEL ERROR =============================
    with torch.no_grad():
        feature_errors = torch.abs(recon_test - X_test_tensor).cpu().numpy()
        feature_mse = np.mean(feature_errors, axis=0)

    feature_names = train_df.drop(columns=["label"]).columns
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'avg_recon_error': feature_mse
    }).sort_values(by='avg_recon_error', ascending=False)

    feature_importance.to_csv(f"{RESULTS_DIR}/autoencoder_feature_errors.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='avg_recon_error', y='feature')
    plt.title("Top 10 Features by Reconstruction Error")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ae_feature_importance.png", dpi=150)
    plt.close()

    print("Top 5 most anomalous features:")
    print(feature_importance.head(5))

    print("\n🎉 Autoencoder training & evaluation complete!\n")
    print(f"Results saved in '{RESULTS_DIR}' and model in '{MODELS_DIR}'")

# RUN
if __name__ == "__main__":
    train_autoencoder()
