# src/lstm_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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

SEQ_LEN = 10

# ============================= DATASET =============================
class SequenceDataset(Dataset):
    def __init__(self, X, y=None, seq_len=SEQ_LEN):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32) if y is not None else None
        self.seq_len = seq_len

        if len(self.X) < seq_len:
            raise ValueError(f"Data has only {len(self.X)} samples, need at least {seq_len}")

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        seq = self.X[idx:idx + self.seq_len]
        target = self.X[idx + self.seq_len - 1]          # reconstruct last timestep
        label = self.y[idx + self.seq_len - 1] if self.y is not None else -1
        return torch.tensor(seq), torch.tensor(target), torch.tensor(label)


# ============================= MODEL =============================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]                                           # last layer hidden state
        latent = h.unsqueeze(1).repeat(1, x.size(1), 1)       # (B, T, H)
        out, _ = self.decoder(latent)
        return self.fc(out)                                   # (B, T, input_dim)


# ============================= MAIN =============================
def train_lstm_autoencoder():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df  = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    if "label_binary" not in train_df.columns:
        raise ValueError("Column 'label_binary' missing!")

    label_cols = ["label", "label_attack", "label_binary"]
    feature_cols = [c for c in train_df.columns if c not in label_cols]

    X_train = train_df[feature_cols].values
    X_test  = test_df[feature_cols].values
    y_train = (train_df["label_binary"] == "attack").astype(int).values
    y_test  = (test_df["label_binary"]  == "attack").astype(int).values

    print(f"Features: {len(feature_cols)} | Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ---------- Train only on normal traffic ----------
    normal_mask = (y_train == 0)
    X_normal = X_train[normal_mask]
    print(f"Training LSTM Autoencoder on {len(X_normal):,} normal samples")

    train_dataset = SequenceDataset(X_normal, seq_len=SEQ_LEN)                    # no labels needed
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset  = SequenceDataset(X_test, y_test, seq_len=SEQ_LEN)              # labels included
    test_loader   = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # ---------- Model ----------
    model = LSTMAutoencoder(input_dim=X_train.shape[1], hidden_dim=64, num_layers=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # ---------- Training ----------
    print("Starting training...")
    best_loss = float('inf')
    patience = 0
    max_patience = 7
    losses = []

    for epoch in range(1, 51):
        model.train()
        epoch_loss = 0.0
        for seq, target, _ in train_loader:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            recon = model(seq)
            loss = criterion(recon[:, -1, :], target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_autoencoder_best.pth")
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.6f}{' (Best)' if avg_loss == best_loss else ''}")

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(f"{MODELS_DIR}/lstm_autoencoder_best.pth"))
    torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_autoencoder.pth")
    print(f"Best model saved (loss: {best_loss:.6f})")

    # Plot loss
    plt.figure(figsize=(9,5))
    plt.plot(losses, label="Training Loss")
    plt.title("LSTM Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{RESULTS_DIR}/lstm_training_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ---------- Evaluation ----------
    print("Evaluating on test set...")
    model.eval()
    test_errors = []
    true_labels = []

    with torch.no_grad():
        for seq, target, label in test_loader:
            seq = seq.to(DEVICE)
            recon = model(seq)
            errors = torch.mean((recon[:, -1, :] - target.to(DEVICE)) ** 2, dim=1)
            test_errors.extend(errors.cpu().numpy())
            true_labels.extend(label.numpy())

    test_errors = np.array(test_errors)
    true_labels = np.array(true_labels)

    # Threshold from normal training data
    train_errors = []
    with torch.no_grad():
        for seq, target, _ in train_loader:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            recon = model(seq)
            errors = torch.mean((recon[:, -1, :] - target) ** 2, dim=1)
            train_errors.extend(errors.cpu().numpy())
    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th percentile on normal): {threshold:.6f}")

    pred_labels = (test_errors > threshold).astype(int)

    # ---------- Metrics ----------
    auc_roc = roc_auc_score(true_labels, test_errors)
    prec, rec, _ = precision_recall_curve(true_labels, test_errors)
    auc_pr = auc(rec, prec)
    k = max(1, int(0.1 * len(test_errors)))
    precision_at_10 = np.mean(true_labels[np.argsort(test_errors)[-k:]])

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print("\n" + "="*55)
    print("LSTM Autoencoder Final Results (Binary Anomaly Detection)")
    print("="*55)
    print(f"  Samples evaluated : {len(test_errors):,}")
    print(f"  ROC-AUC           : {auc_roc:.6f}")
    print(f"  PR-AUC            : {auc_pr:.6f}")
    print(f"  Precision@10%     : {precision_at_10:.6f}")
    print(f"  Accuracy          : {accuracy:.6f}")
    print(f"  F1 Score          : {f1:.6f}")
    print(f"  Threshold         : {threshold:.6f}")
    print("="*55)

    # Save results
    pd.DataFrame({
        "reconstruction_error": test_errors,
        "true_label": true_labels,
        "predicted_label": pred_labels
    }).to_csv(f"{RESULTS_DIR}/lstm_scores.csv", index=False)

    pd.DataFrame({
        "Model": ["LSTM_Autoencoder"],
        "ROC_AUC": [auc_roc],
        "PR_AUC": [auc_pr],
        "Precision@10%": [precision_at_10],
        "Accuracy": [accuracy],
        "F1_Score": [f1],
        "Threshold": [threshold]
    }).to_csv(f"{RESULTS_DIR}/lstm_metrics.csv", index=False)

    print(f"\nAll results saved in '{RESULTS_DIR}'")
    print("LSTM Autoencoder finished successfully!")

if __name__ == "__main__":
    train_lstm_autoencoder()