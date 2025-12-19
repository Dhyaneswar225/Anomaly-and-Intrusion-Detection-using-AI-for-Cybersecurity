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
PERCENTILE = 95  # change to 97 or 99 if needed

# ================= LOAD DATA =================
train_df = pd.read_csv(DATA_DIR / "train_processed.csv")

# Keep ONLY normal samples
train_normal = train_df[train_df["label_binary"] == "normal"]

# Features used by the model
scaler = joblib.load(DATA_DIR / "standard_scaler.pkl")
FEATURE_NAMES = list(scaler.feature_names_in_)

X_normal = train_normal[FEATURE_NAMES].values.astype(np.float32)

# Create LSTM sequences
def make_sequences(X, seq_len):
    return np.repeat(X.reshape(-1, 1, X.shape[1]), seq_len, axis=1)

X_seq = make_sequences(X_normal, SEQ_LEN)

# ================= LOAD MODEL =================
model = LSTMAutoencoder(input_dim=len(FEATURE_NAMES)).to(DEVICE)
model.load_state_dict(
    torch.load(MODEL_DIR / "lstm_autoencoder.pth", map_location=DEVICE)
)
model.eval()

X_tensor = torch.tensor(X_seq).to(DEVICE)

# ================= CALCULATE RECONSTRUCTION ERRORS =================
with torch.no_grad():
    recon = model(X_tensor)
    errors = torch.mean((recon - X_tensor) ** 2, dim=(1, 2)).cpu().numpy()

# ================= COMPUTE THRESHOLD =================
threshold = np.percentile(errors, PERCENTILE)

print(f"✅ Threshold ({PERCENTILE}th percentile): {threshold:.6f}")

# Optional: save threshold
with open(MODEL_DIR / "anomaly_threshold.txt", "w") as f:
    f.write(str(threshold))
