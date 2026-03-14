# src/explain_lstm_shap.py

import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

np.random.seed(42)

# ------------------------------
# Load data
# ------------------------------

print("Loading test data...")

df = pd.read_csv("data/processed/test_processed.csv")

df = df.drop(columns=['label','label_attack','label_binary'], errors='ignore')
df = df.select_dtypes(include=[np.number])

print(f"Clean dataset → {df.shape}")

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
        seqs.append(data[i:i+seq_len])

    return np.array(seqs)


X_seq = create_sequences(X_test)

X_shap_seq = X_seq[:150]

print(f"Sequences: {X_seq.shape} | SHAP samples: {X_shap_seq.shape}")


# ------------------------------
# Load trained model
# ------------------------------

checkpoint = torch.load("models/lstm_autoencoder_best.pth", map_location=DEVICE)

# detect input dimension automatically
input_dim = checkpoint["encoder.weight_ih_l0"].shape[1]

print(f"Model trained with {input_dim} features")


# ------------------------------
# Define SAME architecture
# ------------------------------

class LSTMAutoencoder(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim=64, num_layers=2):

        super().__init__()

        self.encoder = torch.nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.decoder = torch.nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.output_layer = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        _, (h, _) = self.encoder(x)

        h_last = h[-1]

        latent = h_last.unsqueeze(1).repeat(1, x.size(1), 1)

        dec_out, _ = self.decoder(latent)

        return self.output_layer(dec_out)


model = LSTMAutoencoder(input_dim=input_dim).to(DEVICE)

model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully")


# ------------------------------
# Align dataset with model features
# ------------------------------

X_seq = X_seq[:, :, :input_dim]
X_shap_seq = X_shap_seq[:, :, :input_dim]
feature_names = feature_names[:input_dim]


# ------------------------------
# Anomaly scorer
# ------------------------------

class AnomalyScorer(torch.nn.Module):

    def __init__(self, ae):

        super().__init__()
        self.ae = ae

    def forward(self, x):

        recon = self.ae(x)

        error = torch.mean((recon[:, -1, :] - x[:, -1, :]) ** 2, dim=1)

        return error.unsqueeze(1)


scorer = AnomalyScorer(model).to(DEVICE)


# ------------------------------
# SHAP computation
# ------------------------------

print("Computing SHAP values with GradientExplainer...")

background = torch.tensor(X_seq[:50], dtype=torch.float32).to(DEVICE)

shap_data = torch.tensor(X_shap_seq, dtype=torch.float32).to(DEVICE)

explainer = shap.GradientExplainer(scorer, background)

shap_values = explainer.shap_values(shap_data, nsamples=100)


# ------------------------------
# Fix SHAP output shape
# ------------------------------

if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

print(f"Raw SHAP shape: {shap_vals.shape}")

while shap_vals.ndim > 3:
    shap_vals = shap_vals.squeeze()

feature_importance = np.abs(shap_vals).mean(axis=(0,1))

print(f"Final feature importance shape: {feature_importance.shape}")


# ------------------------------
# Save results
# ------------------------------

os.makedirs("results", exist_ok=True)

top_k = 20

indices = np.argsort(feature_importance)[-top_k:][::-1]

plt.figure(figsize=(11,9))

plt.barh(range(top_k), feature_importance[indices])

plt.yticks(range(top_k), [feature_names[i] for i in indices])

plt.xlabel("Mean |SHAP value|")

plt.title("Top 20 Features Driving Anomaly Detection\n(LSTM Autoencoder + SHAP)")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("results/shap_top20_features.png", dpi=300)

plt.close()


# summary plot

shap.summary_plot(
    shap_vals.reshape(-1, input_dim),
    features=shap_data.cpu().numpy().reshape(-1, input_dim),
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)

plt.savefig("results/shap_summary_bar.png", dpi=300)

plt.close()


pd.DataFrame({
    "feature": feature_names,
    "shap_importance": feature_importance
}).sort_values("shap_importance", ascending=False).to_csv(
    "results/shap_feature_importance.csv",
    index=False
)


print("\nSUCCESS! SHAP analysis completed.")
print("Results saved in 'results/' folder:")
print("  • shap_top20_features.png")
print("  • shap_summary_bar.png")
print("  • shap_feature_importance.csv")