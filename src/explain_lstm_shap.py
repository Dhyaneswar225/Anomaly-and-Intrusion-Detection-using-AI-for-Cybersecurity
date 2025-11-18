# src/explain_lstm_shap.py
import os
import torch
import shap
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch import nn
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------
# Load and preprocess test data
# ------------------------------
print("Loading processed test data...")
test_data_path = "data/processed/test_processed.csv"
df = pd.read_csv(test_data_path)

# Sample to reduce SHAP compute cost
df = df.sample(2000, random_state=42)

# Keep only numeric columns (same as training: drop 'label' earlier)
numeric_df = df.select_dtypes(include=[np.number])
feature_cols = numeric_df.columns.tolist()
print("Numeric feature columns:", feature_cols)

X_test = numeric_df.values
print(f"Test sample shape: {X_test.shape}")

# Normalize features (StandardScaler)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# ------------------------------
# Convert to LSTM sequence format
# ------------------------------
SEQ_LEN = 10

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):  # matches training dataset length
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)

X_seq = create_sequences(X_test, SEQ_LEN)
print(f"Sequence dataset shape: {X_seq.shape}")

# Subsample for SHAP
X_shap = X_seq[:150]
print(f"SHAP sample shape: {X_shap.shape}")

# ------------------------------
# Rebuild original LSTM Autoencoder (same as training)
# ------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h, _) = self.encoder(x)
        h = h[-1]  # (batch, hidden_dim)
        latent = h.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, hidden_dim)
        out, _ = self.decoder(latent)  # (batch, seq_len, hidden_dim)
        return self.fc(out)  # (batch, seq_len, input_dim)

# ------------------------------
# Load trained model
# ------------------------------
n_features = X_seq.shape[2]
print(f"Detected feature count: {n_features}")

model = LSTMAutoencoder(input_dim=n_features, hidden_dim=64).to(DEVICE)

ckpt_path = "models/lstm_autoencoder_best.pth"
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
print(f"Loading checkpoint: {ckpt_path}")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model.eval()  # keep eval mode (ok for gradients), batchnorm/dropout not used anyway
print("Model loaded successfully ✔️")

# ------------------------------
# Wrapper to compute anomaly score (per-sample scalar) - returns (batch,1)
# ------------------------------
class WrappedModel(nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.m = autoencoder

    def forward(self, x):
        # Expect x as torch tensor: (batch, seq_len, features)
        recon = self.m(x)
        last_recon = recon[:, -1, :]   # (batch, features)
        last_input = x[:, -1, :]       # (batch, features)
        mse = torch.mean((last_recon - last_input) ** 2, dim=1)  # (batch,)
        return mse.unsqueeze(1)  # (batch, 1)

wrapped_model = WrappedModel(model).to(DEVICE)
print("Wrapper model ready ✔️")

# ------------------------------
# SHAP explainability
# ------------------------------
print("Preparing SHAP inputs (torch tensors on same device as model)...")
# Use torch tensors for background and inputs so model sees tensors
background_t = torch.tensor(X_seq[:50]).float().to(DEVICE)  # (B, seq_len, features)
shap_input_t = torch.tensor(X_shap).float().to(DEVICE)      # (N, seq_len, features)

print("Computing SHAP values...⏳ (this may take a while on CPU)")

# Initialize DeepExplainer with torch tensors
explainer = shap.DeepExplainer(wrapped_model, background_t)

# Compute SHAP values
# Note: shap.DeepExplainer accepts torch tensors; returns list (per output).
shap_values_list = explainer.shap_values(shap_input_t, check_additivity=False)

# Convert returned object to numpy for consistent handling
raw_sv = shap_values_list[0]
sv = np.array(raw_sv)
print("Raw SHAP array shape (as returned):", sv.shape)

# ------------------------------
# Normalize SHAP output to (N, seq_len, features)
# ------------------------------
N = shap_input_t.shape[0]
S = SEQ_LEN
F = n_features

def normalize_shap_array(sv, N, S, F):
    # If already correct:
    if sv.ndim == 3 and sv.shape == (N, S, F):
        return sv
    # common torch-returned shape: (S, F, N) -> transpose to (N, S, F)
    if sv.ndim == 3 and sv.shape == (S, F, N):
        return sv.transpose(2, 0, 1)
    # shape: (S, F, 1) -> broadcast to N
    if sv.ndim == 3 and sv.shape == (S, F, 1):
        sv2 = sv.transpose(2, 0, 1)  # (1, S, F)
        return np.tile(sv2, (N, 1, 1))
    # shape: (N, S, F, 1) -> squeeze
    if sv.ndim == 4 and sv.shape[0] == N and sv.shape[1] == S and sv.shape[2] == F and sv.shape[3] == 1:
        return sv.reshape((N, S, F))
    # shape: (S, F) -> tile across N
    if sv.ndim == 2 and sv.shape == (S, F):
        return np.tile(sv[np.newaxis, :, :], (N, 1, 1))
    # try permutations to find (N,S,F)
    from itertools import permutations
    for perm in permutations(range(sv.ndim)):
        try:
            cand = np.transpose(sv, perm)
            if cand.ndim == 3 and cand.shape == (N, S, F):
                return cand
        except Exception:
            pass
    raise ValueError(f"Unable to normalize SHAP array of shape {sv.shape} to (N,S,F)=({N},{S},{F}).")

shap_arr = normalize_shap_array(sv, N, S, F)
print("Normalized SHAP shape (N, seq_len, features):", shap_arr.shape)

# ------------------------------
# Collapse time axis -> produce (N, features)
# ------------------------------
shap_feature_mean = shap_arr.mean(axis=1)   # (N, features)
X_shap_mean = X_shap.mean(axis=1)           # (N, features) original numpy data

print("Final SHAP feature matrix shape:", shap_feature_mean.shape)
print("Final feature data matrix shape:", X_shap_mean.shape)

# ------------------------------
# Plot summary
# ------------------------------
print("Generating summary plot...")
shap.summary_plot(shap_feature_mean, X_shap_mean, feature_names=feature_cols,show=False)
plt.savefig("results/shap_summary_plot.png", dpi=150, bbox_inches="tight")
plt.close()
np.save("results/shap_values.npy", shap_arr)
np.save("results/shap_inputs.npy", X_shap)
print("SHAP values and inputs saved ✔️")
shap_abs_mean = np.abs(shap_feature_mean).mean(axis=0)
plt.figure(figsize=(12,6))
plt.barh(feature_cols, shap_abs_mean)
plt.xlabel("Mean |SHAP value|")
plt.title("Feature importance (mean absolute SHAP)")
plt.tight_layout()
plt.savefig("results/shap_feature_importance.png", dpi=150)
plt.close()
shap_df = pd.DataFrame(shap_feature_mean, columns=feature_cols)
shap_df.to_csv("results/shap_feature_values.csv", index=False)
print("Done 🎯")
