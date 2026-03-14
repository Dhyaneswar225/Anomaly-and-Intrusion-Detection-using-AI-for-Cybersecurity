# src/train_lstm.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import random
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from lstm_model import LSTMAutoencoder


# ================= REPRODUCIBILITY =================
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

y_train = (train_df["label_binary"] == "attack").astype(int).values
y_test  = (test_df["label_binary"] == "attack").astype(int).values


# ================= SEQUENCE CREATION =================

def create_sequences(data, seq_len):

    seqs = []

    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i+seq_len])

    return np.array(seqs)


# only NORMAL samples for training

train_normal = train_df[train_df["label_binary"] == "normal"]

X_train_normal = train_normal[FEATURE_NAMES].values.astype(np.float32)

X_train_seq = create_sequences(X_train_normal, SEQ_LEN)

X_test_seq  = create_sequences(X_test, SEQ_LEN)


# ================= MODEL =================

model = LSTMAutoencoder(input_dim=len(FEATURE_NAMES)).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

criterion = nn.MSELoss()

print("Training LSTM Autoencoder...")


# ================= TRAIN =================

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

torch.save(model.state_dict(), MODEL_DIR / "lstm_autoencoder_best.pth")

print("Model saved.")


# ================= TRAIN RECON ERROR =================

model.eval()

train_errors = []

with torch.no_grad():

    for seq in X_train_seq:

        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        recon = model(x)

        err = torch.mean((recon[:, -1] - x[:, -1])**2).item()

        train_errors.append(err)

train_errors = np.array(train_errors)

threshold = np.percentile(train_errors, 95)


# ================= TEST RECON ERROR =================

errors = []

with torch.no_grad():

    for seq in X_test_seq:

        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        recon = model(x)

        err = torch.mean((recon[:, -1] - x[:, -1])**2).item()

        errors.append(err)

errors = np.array(errors)

preds = (errors > threshold).astype(int)

print("ROC-AUC:", roc_auc_score(y_test[:len(preds)], errors))
print("F1:", f1_score(y_test[:len(preds)], preds))
print("Threshold:", threshold)