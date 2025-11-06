# src/lstm_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

SEQ_LEN = 10  # Sequence length

# ============================= DATASET =============================
class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.X[idx:idx + self.seq_len]
        target = self.X[idx + self.seq_len - 1]  # Last timestep as target
        return torch.tensor(seq), torch.tensor(target)

# ============================= MODEL =============================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: input → hidden
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Decoder: hidden → hidden
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        # Final projection
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Encoder
        _, (h, c) = self.encoder(x)
        h = h[-1]  # [batch, hidden_dim]
        
        # Repeat latent state across sequence
        latent = h.unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, hidden_dim]
        
        # Decoder
        out, _ = self.decoder(latent)
        recon = self.fc(out)  # [batch, seq_len, input_dim]
        
        return recon

# ============================= TRAINING & EVALUATION =============================
def train_lstm_autoencoder():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df  = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    # Extract features and labels
    X_train = train_df.drop(columns=["label"]).values
    y_train = (train_df["label"] == "attack").astype(int).values
    X_test  = test_df.drop(columns=["label"]).values
    y_test  = (test_df["label"] == "attack").astype(int).values

    # Use only NORMAL samples for training
    normal_mask_train = (y_train == 0)
    X_normal = X_train[normal_mask_train]
    print(f"Training on {len(X_normal):,} normal samples")

    # Dataset and DataLoader
    train_dataset = SequenceDataset(X_normal, y_train[normal_mask_train])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Model
    input_dim = X_train.shape[1]
    model = LSTMAutoencoder(input_dim, hidden_dim=64, num_layers=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("Starting training...")
    epochs = 50
    losses = []
    best_loss = float('inf')
    patience = 0
    patience_limit = 5

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for seq, target in train_loader:
            seq = seq.to(DEVICE)
            target = target.to(DEVICE)

            recon = model(seq)
            loss = criterion(recon[:, -1, :], target)  # Reconstruct last timestep

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_autoencoder_best.pth")
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")

    # Load best model
    model.load_state_dict(torch.load(f"{MODELS_DIR}/lstm_autoencoder_best.pth"))
    print(f"Best model loaded from epoch with loss: {best_loss:.6f}")

    # Save final model
    torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_autoencoder.pth")
    print(f"Final model saved: {MODELS_DIR}/lstm_autoencoder.pth")

    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss", color="#1f77b4")
    plt.title("LSTM Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/lstm_training_loss.png", dpi=150)
    plt.close()

    # ============================= EVALUATION =============================
    print("Evaluating on test set...")
    model.eval()

    # Test dataset
    test_dataset = SequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Compute test errors
    test_errors = []
    true_labels = []
    with torch.no_grad():
        for i, (seq, target) in enumerate(test_loader):
            seq = seq.to(DEVICE)
            recon = model(seq)
            error = torch.mean((recon[:, -1, :] - target.to(DEVICE)) ** 2, dim=1).cpu().numpy()
            test_errors.extend(error)
            # Align labels: each sequence ends at index i + seq_len - 1
            start_idx = i * 256
            end_idx = start_idx + len(seq)
            labels = y_test[test_dataset.seq_len - 1 + np.arange(start_idx, end_idx)]
            true_labels.extend(labels)

    test_errors = np.array(test_errors)
    true_labels = np.array(true_labels)

    # Compute train errors for threshold
    train_errors = []
    with torch.no_grad():
        for seq, target in train_loader:
            seq = seq.to(DEVICE)
            recon = model(seq)
            error = torch.mean((recon[:, -1, :] - target.to(DEVICE)) ** 2, dim=1).cpu().numpy()
            train_errors.extend(error)
    train_errors = np.array(train_errors)

    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th %ile of normal train errors): {threshold:.6f}")

    # Metrics
    auc_roc = roc_auc_score(true_labels, test_errors)
    precision, recall, _ = precision_recall_curve(true_labels, test_errors)
    auc_pr = auc(recall, precision)

    k = int(0.1 * len(test_errors))
    top_k_idx = np.argsort(test_errors)[-k:]
    precision_at_10 = np.mean(true_labels[top_k_idx])

    print(f"\nLSTM Autoencoder Results:")
    print(f"  ROC-AUC     : {auc_roc:.6f}")
    print(f"  PR-AUC      : {auc_pr:.6f}")
    print(f"  Precision@10%: {precision_at_10:.6f}")

    # Predicted labels
    pred_labels = (test_errors > threshold).astype(int)

    # Save scores
    scores_df = pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": true_labels,
        "predicted_label": pred_labels
    })
    scores_df.to_csv(f"{RESULTS_DIR}/lstm_scores.csv", index=False)

    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=scores_df,
        x="reconstruction_error",
        hue="true_label",
        bins=60,
        kde=True,
        palette=["#1f77b4", "#d62728"],
        alpha=0.7
    )
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    plt.title("LSTM Autoencoder – Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.legend(title="Label", labels=["Normal", "Attack", "Threshold"])
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/lstm_error_distribution.png", dpi=150)
    plt.close()

    # Normalized plot
    base_error = np.percentile(train_errors, 95)
    norm_errors = test_errors / base_error
    norm_threshold = 1.0

    plt.figure(figsize=(10, 6))
    sns.histplot(
        norm_errors[true_labels == 0], label="Normal", color="#1f77b4", alpha=0.7, bins=50
    )
    sns.histplot(
        norm_errors[true_labels == 1], label="Attack", color="#d62728", alpha=0.7, bins=50
    )
    plt.axvline(norm_threshold, color='black', linestyle='--', linewidth=2, label='Threshold = 1.0')
    plt.xlabel("Normalized Reconstruction Error (×95th %ile)")
    plt.ylabel("Count")
    plt.title("LSTM Autoencoder – Normalized Error Distribution")
    plt.legend()
    plt.xlim(0, 4)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/lstm_error_distribution_normalized.png", dpi=150)
    plt.close()

    print(f"\nAll results saved in: {RESULTS_DIR}/")
    print(f"Model saved in: {MODELS_DIR}/")
    print("LSTM Autoencoder training and evaluation completed successfully!")

# ============================= RUN =============================
if __name__ == "__main__":
    train_lstm_autoencoder()