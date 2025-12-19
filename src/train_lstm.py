# src/train_lstm.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from lstm_model import LSTMAutoencoder

# ================= CONFIG =================
BASE_DIR = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 10
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3

# ================= LOAD DATA =================
train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
test_df  = pd.read_csv(DATA_DIR / "test_processed.csv")

scaler = joblib.load(DATA_DIR / "standard_scaler.pkl")
FEATURE_NAMES = list(scaler.feature_names_in_)

X_train = train_df[FEATURE_NAMES].values.astype(np.float32)
X_test  = test_df[FEATURE_NAMES].values.astype(np.float32)

y_test = (test_df["label_binary"] == "attack").astype(int).values

# ================= SEQUENCES =================
def create_sequences(X, seq_len):
    return np.stack([X[i:i+seq_len] for i in range(len(X)-seq_len+1)])

X_train_seq = create_sequences(X_train, SEQ_LEN)
X_test_seq  = create_sequences(X_test, SEQ_LEN)

# only NORMAL for training
X_train_seq = X_train_seq[:len(train_df[train_df["label_binary"]=="normal"])]

# ================= MODEL =================
model = LSTMAutoencoder(input_dim=len(FEATURE_NAMES)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ================= TRAIN =================
print("Training LSTM Autoencoder...")
for epoch in range(1, EPOCHS+1):
    model.train()
    losses = []

    for i in range(0, len(X_train_seq), BATCH_SIZE):
        batch = torch.tensor(X_train_seq[i:i+BATCH_SIZE]).to(DEVICE)
        recon = model(batch)
        loss = criterion(recon, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.6f}")

MODEL_DIR.mkdir(exist_ok=True)
torch.save(model.state_dict(), MODEL_DIR / "lstm_autoencoder.pth")
print("Model saved.")

# ================= EVAL =================
model.eval()
errors = []

with torch.no_grad():
    for seq in X_test_seq:
        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)
        recon = model(x)
        err = torch.mean((recon[:, -1] - x[:, -1])**2).item()
        errors.append(err)

errors = np.array(errors)
threshold = np.percentile(errors[y_test[:len(errors)]==0], 95)
preds = (errors > threshold).astype(int)

print("ROC-AUC:", roc_auc_score(y_test[:len(preds)], errors))
print("F1:", f1_score(y_test[:len(preds)], preds))
print("Threshold:", threshold)
