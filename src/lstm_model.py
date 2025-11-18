# src/lstm_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
        target = self.X[idx + self.seq_len - 1]
        return torch.tensor(seq), torch.tensor(target)

# ============================= MODEL =============================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1]
        latent = h.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(latent)
        return self.fc(out)

# ============================= TRAINING & EVALUATION =============================
def train_lstm_autoencoder():
    print("Loading processed data...")
    train_df = pd.read_csv(f"{DATA_PROCESSED}/train_processed.csv")
    test_df = pd.read_csv(f"{DATA_PROCESSED}/test_processed.csv")

    X_train = train_df.drop(columns=["label"]).values
    y_train = (train_df["label"] == "attack").astype(int).values
    X_test = test_df.drop(columns=["label"]).values
    y_test = (test_df["label"] == "attack").astype(int).values

    normal_mask_train = (y_train == 0)
    X_normal = X_train[normal_mask_train]
    print(f"Training on {len(X_normal):,} normal samples")

    train_dataset = SequenceDataset(X_normal, y_train[normal_mask_train])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    input_dim = X_train.shape[1]
    model = LSTMAutoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    epochs = 50
    best_loss = float('inf')
    patience = 0
    patience_limit = 5

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for seq, target in train_loader:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            recon = model(seq)
            loss = criterion(recon[:, -1, :], target)
            optimizer.zero_grad()
            loss.backward()
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
            if patience >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.6f}")

    model.load_state_dict(torch.load(f"{MODELS_DIR}/lstm_autoencoder_best.pth"))
    print(f"Best model loaded. Best loss: {best_loss:.6f}")

    torch.save(model.state_dict(), f"{MODELS_DIR}/lstm_autoencoder.pth")
    print("Final model saved.")

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("LSTM Autoencoder Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/lstm_training_loss.png", dpi=150)
    plt.close()

    # ===================== Evaluation =====================
    print("Evaluating on test set...")
    model.eval()

    test_dataset = SequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    test_errors, true_labels = [], []
    with torch.no_grad():
        for i, (seq, target) in enumerate(test_loader):
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            recon = model(seq)
            errors = torch.mean((recon[:, -1, :] - target) ** 2, dim=1).cpu().numpy()
            test_errors.extend(errors)

            start_idx = i * 256
            end_idx = start_idx + len(seq)
            labels = y_test[test_dataset.seq_len - 1 + np.arange(start_idx, end_idx)]
            true_labels.extend(labels)

    test_errors, true_labels = np.array(test_errors), np.array(true_labels)

    # Compute threshold from normal training errors
    train_errors = []
    with torch.no_grad():
        for seq, target in train_loader:
            seq, target = seq.to(DEVICE), target.to(DEVICE)
            errors = torch.mean((model(seq)[:, -1, :] - target) ** 2, dim=1).cpu().numpy()
            train_errors.extend(errors)
    train_errors = np.array(train_errors)

    threshold = np.percentile(train_errors, 95)
    print(f"Threshold (95th %ile): {threshold:.6f}")

    pred_labels = (test_errors > threshold).astype(int)

    # Metrics
    auc_roc = roc_auc_score(true_labels, test_errors)
    precision, recall, _ = precision_recall_curve(true_labels, test_errors)
    auc_pr = auc(recall, precision)
    k = int(0.1 * len(test_errors))
    precision_at_10 = np.mean(true_labels[np.argsort(test_errors)[-k:]])

    # Newly Added Metrics ✔️
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    print("\nLSTM Autoencoder Results:")
    print(f"  ROC-AUC      : {auc_roc:.6f}")
    print(f"  PR-AUC       : {auc_pr:.6f}")
    print(f"  Precision@10%: {precision_at_10:.6f}")
    print(f"  Accuracy     : {accuracy:.6f}")
    print(f"  F1 Score     : {f1:.6f}")

    metrics_df = pd.DataFrame({
        "ROC_AUC": [auc_roc],
        "PR_AUC": [auc_pr],
        "Precision@10%": [precision_at_10],
        "Accuracy": [accuracy],
        "F1_Score": [f1]
    })
    metrics_df.to_csv(f"{RESULTS_DIR}/lstm_metrics.csv", index=False)

    print("\nMetrics saved! ✔️")
    print("LSTM Autoencoder evaluation completed successfully.")

# Run
if __name__ == "__main__":
    train_lstm_autoencoder()
