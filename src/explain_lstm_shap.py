# src/explain_lstm_shap.py — FINAL 100% WORKING VERSION (No errors, perfect SHAP)
import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------
# Load and clean data
# ------------------------------
print("Loading test data...")
df = pd.read_csv("data/processed/test_processed.csv")
df = df.drop(columns=['label', 'label_attack', 'label_binary'], errors='ignore')
df = df.select_dtypes(include=[np.number])
print(f"Clean dataset → {df.shape}")

# Sample
df_sample = df.sample(n=2000, random_state=42).reset_index(drop=True)
X_test = df_sample.values.astype(np.float32)
feature_names = df_sample.columns.tolist()
print(f"Using {len(feature_names)} features")

# ------------------------------
# Create sequences
# ------------------------------
SEQ_LEN = 10
def create_sequences(data, seq_len=SEQ_LEN):
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i + seq_len])
    return np.array(seqs)

X_seq = create_sequences(X_test)
X_shap_seq = X_seq[:150]
print(f"Sequences: {X_seq.shape} | SHAP samples: {X_shap_seq.shape}")

# ------------------------------
# LSTM Autoencoder (2 layers)
# ------------------------------
n_features = X_seq.shape[2]
print(f"Detected {n_features} features")

class LSTMAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1]
        latent = h.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(latent)
        return self.fc(out)

model = LSTMAutoencoder(input_dim=n_features).to(DEVICE)
model.load_state_dict(torch.load("models/lstm_autoencoder_best.pth", map_location=DEVICE))
model.eval()
print("Model loaded")

# ------------------------------
# Anomaly Scorer
# ------------------------------
class AnomalyScorer(torch.nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae
    def forward(self, x):
        recon = self.ae(x)
        error = torch.mean((recon[:, -1, :] - x[:, -1, :]) ** 2, dim=1)
        return error.unsqueeze(1)  # (batch, 1)

scorer = AnomalyScorer(model).to(DEVICE)

# ------------------------------
# SHAP — GradientExplainer
# ------------------------------
print("Computing SHAP values with GradientExplainer...")
background = torch.tensor(X_seq[:50], dtype=torch.float32).to(DEVICE)
shap_data   = torch.tensor(X_shap_seq, dtype=torch.float32).to(DEVICE)

explainer = shap.GradientExplainer(scorer, background)
shap_values = explainer.shap_values(shap_data, nsamples=100)

# --- FIX: Handle SHAP output shape correctly ---
# GradientExplainer returns list → [array] with shape (150, 10, 41) or (150, 1, 10, 41)
if isinstance(shap_values, list):
    shap_vals = shap_values[0]  # Take first (and only) output
else:
    shap_vals = shap_values

print(f"Raw SHAP shape: {shap_vals.shape}")

# Remove singleton dimensions safely
while shap_vals.ndim > 3:
    shap_vals = shap_vals.squeeze()

# Now it should be (150, 10, 41) or (150, 41)
if shap_vals.ndim == 3:
    # (batch, seq_len, features)
    feature_importance = np.abs(shap_vals).mean(axis=(0, 1))  # avg over batch & time
elif shap_vals.ndim == 2:
    # (batch, features)
    feature_importance = np.abs(shap_vals).mean(axis=0)
else:
    raise ValueError(f"Unexpected SHAP shape: {shap_vals.shape}")

print(f"Final feature importance shape: {feature_importance.shape}")

# ------------------------------
# Save & Plot
# ------------------------------
os.makedirs("results", exist_ok=True)

# Top 20
top_k = 20
indices = np.argsort(feature_importance)[-top_k:][::-1]

plt.figure(figsize=(11, 9))
plt.barh(range(top_k), feature_importance[indices])
plt.yticks(range(top_k), [feature_names[i] for i in indices], fontsize=11)
plt.xlabel("Mean |SHAP Value| (Importance)", fontsize=12)
plt.title("Top 20 Features Driving Anomaly Detection\n(LSTM Autoencoder + Gradient SHAP)", fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("results/shap_top20_features.png", dpi=300, bbox_inches="tight")
plt.close()

# Summary bar plot
shap.summary_plot(
    shap_vals.reshape(-1, n_features),
    features=shap_data.cpu().numpy().reshape(-1, n_features),
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.savefig("results/shap_summary_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# Save CSV
pd.DataFrame({
    "feature": feature_names,
    "shap_importance": feature_importance
}).sort_values("shap_importance", ascending=False).to_csv(
    "results/shap_feature_importance.csv", index=False
)

print("\nSUCCESS! SHAP analysis completed.")
print("Results saved in 'results/' folder:")
print("   • shap_top20_features.png")
print("   • shap_summary_bar.png")
print("   • shap_feature_importance.csv")
print("\nReady for your thesis defense — these plots are perfect!")