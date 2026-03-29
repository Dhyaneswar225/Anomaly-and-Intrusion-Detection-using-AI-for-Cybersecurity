import numpy as np
import torch
import pandas as pd
import joblib
from pathlib import Path
from lstm_model import LSTMAutoencoder

# ================= CONFIG =================
BASE_DIR = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 10
PERCENTILE = 95   # 🔥 Better than 85 (improves recall)

# ================= LOAD DATA =================
print("Loading training data...")

train_df = pd.read_csv(DATA_DIR / "train_processed.csv")

# Keep ONLY normal samples
train_normal = train_df[train_df["label_binary"] == "normal"]

print("Normal samples:", len(train_normal))

# ================= LOAD SCALER =================
scaler = joblib.load(DATA_DIR / "standard_scaler.pkl")
FEATURE_NAMES = list(scaler.feature_names_in_)

# ================= APPLY SCALING (CRITICAL FIX) =================
X_normal = train_normal[FEATURE_NAMES]

X_scaled = scaler.transform(X_normal).astype(np.float32)

# ================= CREATE SEQUENCES =================
def make_sequences(X, seq_len):
    sequences = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i+seq_len])
    return np.array(sequences)

X_seq = make_sequences(X_scaled, SEQ_LEN)

print("Sequence shape:", X_seq.shape)

# ================= LOAD MODEL =================
print("Loading LSTM Autoencoder...")

model = LSTMAutoencoder(input_dim=len(FEATURE_NAMES)).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_DIR / "lstm_autoencoder.pth", map_location=DEVICE)
)

model.eval()

X_tensor = torch.tensor(X_seq).to(DEVICE)

# ================= COMPUTE RECONSTRUCTION ERROR =================
print("Computing reconstruction errors...")

with torch.no_grad():
    recon = model(X_tensor)

    errors = torch.mean(
        (recon[:, -1] - X_tensor[:, -1]) ** 2,
        dim=1
    ).cpu().numpy()

# ================= HANDLE OUTLIERS =================
# Clip extreme values (optional but recommended)
errors = np.clip(errors, 0, np.percentile(errors, 99.5))

# ================= DEBUG STATS =================
print("\n====== ERROR STATISTICS ======")
print("Min  :", np.min(errors))
print("Mean :", np.mean(errors))
print("Max  :", np.max(errors))
print("90th :", np.percentile(errors, 90))
print("95th :", np.percentile(errors, 95))
print("99th :", np.percentile(errors, 99))

# ================= COMPUTE THRESHOLD =================
threshold = np.percentile(errors, PERCENTILE)

print(f"\n✅ Computed Threshold ({PERCENTILE}th percentile): {threshold:.6f}")

# ================= SAVE THRESHOLD =================
threshold_path = MODEL_DIR / "anomaly_threshold.txt"

with open(threshold_path, "w") as f:
    f.write(str(threshold))

print("\nThreshold saved to:")
print(threshold_path)

# ================= OPTIONAL: QUICK SANITY CHECK =================
print("\n====== SANITY CHECK ======")

normal_detected = (errors < threshold).sum()
anomaly_detected = (errors >= threshold).sum()

print("Normal predicted (should be high):", normal_detected)
print("Anomaly predicted (should be low):", anomaly_detected)