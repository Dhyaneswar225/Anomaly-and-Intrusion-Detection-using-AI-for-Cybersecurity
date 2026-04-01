# src/explain_dense_shap.py

import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DenseAutoEncoderModel import DenseAutoencoder   # Make sure this file is in your path
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

np.random.seed(42)
torch.manual_seed(42)

# ------------------------------
# Load data
# ------------------------------

print("Loading test data...")

df = pd.read_csv("data/processed/test_processed.csv")

# Drop label columns and keep only numeric features
df = df.drop(columns=['label', 'label_attack', 'label_binary'], errors='ignore')
df = df.select_dtypes(include=[np.number])

print(f"Clean dataset → {df.shape}")

# Sample for SHAP (GradientExplainer can be slow on very large data)
df_sample = df.sample(n=2000, random_state=42).reset_index(drop=True)

X_test = df_sample.values.astype(np.float32)
feature_names = df_sample.columns.tolist()

print(f"Using {len(feature_names)} features")

# ------------------------------
# Load trained Dense Autoencoder
# ------------------------------

checkpoint_path = "models/dense_autoencoder_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Detect input dimension
input_dim = checkpoint["encoder.0.weight"].shape[1]   # First Linear layer: (out_features, in_features)

print(f"Model trained with {input_dim} features")

# ------------------------------
# Define / Load DenseAutoencoder
# ------------------------------

model = DenseAutoencoder(
    input_dim=input_dim,
    bottleneck=32,      # Make sure this matches your training config
    dropout=0.2
).to(DEVICE)

model.load_state_dict(checkpoint)
model.eval()

print("Dense Autoencoder loaded successfully")

# ------------------------------
# Align dataset with model (in case of mismatch)
# ------------------------------

X_test = X_test[:, :input_dim]
feature_names = feature_names[:input_dim]

# ------------------------------
# Anomaly Scorer (for SHAP)
# ------------------------------

class AnomalyScorer(torch.nn.Module):
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def forward(self, x):
        # x shape: (batch, features)
        recon = self.ae(x)
        # Mean squared error per sample (across all features)
        error = torch.mean((recon - x) ** 2, dim=1)
        return error.unsqueeze(1)   # shape: (batch, 1)  → required by SHAP


scorer = AnomalyScorer(model).to(DEVICE)

# ------------------------------
# SHAP computation with GradientExplainer
# ------------------------------

print("Computing SHAP values with GradientExplainer...")

# Use a small background set (important for GradientExplainer)
background = torch.tensor(X_test[:50], dtype=torch.float32).to(DEVICE)

# Compute SHAP on more samples (but keep reasonable for speed)
shap_data = torch.tensor(X_test[:300], dtype=torch.float32).to(DEVICE)   # Adjust if too slow

explainer = shap.GradientExplainer(scorer, background)

shap_values = explainer.shap_values(shap_data, nsamples=100)

# ------------------------------
# Process SHAP output
# ------------------------------

if isinstance(shap_values, list):
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

print(f"Raw SHAP shape: {shap_vals.shape}")

# For Dense AE + GradientExplainer, shape should be (samples, features)
# Remove extra dimensions if any
while shap_vals.ndim > 2:
    shap_vals = shap_vals.squeeze()

feature_importance = np.abs(shap_vals).mean(axis=0)   # Mean absolute SHAP across samples

print(f"Final feature importance shape: {feature_importance.shape}")

# ------------------------------
# Save results
# ------------------------------

os.makedirs("results", exist_ok=True)

top_k = 20
indices = np.argsort(feature_importance)[-top_k:][::-1]

plt.figure(figsize=(11, 9))
plt.barh(range(top_k), feature_importance[indices])
plt.yticks(range(top_k), [feature_names[i] for i in indices])
plt.xlabel("Mean |SHAP value|")
plt.title("Top 20 Features Driving Anomaly Detection\n(Dense Autoencoder + SHAP)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("results/shap_dense_top20_features.png", dpi=300)
plt.close()

# Summary bar plot
shap.summary_plot(
    shap_vals,
    features=shap_data.cpu().numpy(),
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.savefig("results/shap_dense_summary_bar.png", dpi=300)
plt.close()

# Save CSV
pd.DataFrame({
    "feature": feature_names,
    "shap_importance": feature_importance
}).sort_values("shap_importance", ascending=False).to_csv(
    "results/shap_dense_feature_importance.csv",
    index=False
)

print("\nSUCCESS! SHAP analysis for Dense Autoencoder completed.")
print("Results saved in 'results/' folder:")
print("  • shap_dense_top20_features.png")
print("  • shap_dense_summary_bar.png")
print("  • shap_dense_feature_importance.csv")