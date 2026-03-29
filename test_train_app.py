"""
DenseAETest.py
──────────────
Inference / evaluation script using the Dense Autoencoder.

Changes vs original LSTM script:
  - Uses DenseAutoencoder instead of LSTMAutoencoder
  - No sequence creation (X_seq / repeat trick) — rows fed directly
  - Threshold loaded from saved .npy file (auto-selected p80 during training)
  - Cleaner preprocessing pipeline
"""

import pandas as pd
import numpy as np
import torch
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score
)

from src.DenseAutoEncoderModel import DenseAutoencoder   # ← same folder as this script


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR  = Path("F:/Master Thesis/anomaly-ids")
RAW_PATH  = BASE_DIR / "data/raw/nsl-kdd/KDDTrain+.txt"
DATA_DIR  = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "data/generated"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading artefacts...")

scaler        = joblib.load(DATA_DIR / "standard_scaler.pkl")
FEATURE_NAMES = list(scaler.feature_names_in_)

# threshold saved by training script (best percentile = p80)
ANOMALY_THRESHOLD = float(np.load(MODEL_DIR / "dense_ae_threshold.npy"))

print(f"✅ Features          : {len(FEATURE_NAMES)}")
print(f"✅ Anomaly threshold : {ANOMALY_THRESHOLD:.6f}")

with open(DATA_DIR / "label_mappings.json") as f:
    mappings = json.load(f)

# ── Dense Autoencoder ────────────────────────────────────────────────────────
model = DenseAutoencoder(
    input_dim  = len(FEATURE_NAMES),
    bottleneck = 32,
    dropout    = 0.2
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_DIR / "dense_autoencoder_best.pth", map_location=DEVICE)
)
model.eval()
print("✅ Dense Autoencoder loaded")

# ── Attack classifier (XGBoost) — kept for multi-class labelling ─────────────
clf = joblib.load(MODEL_DIR / "attack_classifier_xgb.pkl")
le  = joblib.load(MODEL_DIR / "attack_label_encoder.pkl")
print("✅ Attack classifier loaded")


# ══════════════════════════════════════════════════════════════════════════════
# NSL-KDD COLUMN NAMES
# ══════════════════════════════════════════════════════════════════════════════

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate',
    'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
    'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','label','difficulty'
]


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREPROCESS RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading {RAW_PATH.name} ...")

df = pd.read_csv(RAW_PATH, names=COLUMNS)
df["label_attack"] = df["label"]

print(f"Total rows : {len(df)}")
print(f"Label dist :\n{df['label_attack'].value_counts().head(10)}\n")

# ── Encode categoricals using saved mappings ─────────────────────────────────
df["protocol_type"] = df["protocol_type"].map(mappings["protocol_type"])
df["service"]       = df["service"].map(mappings["service"])
df["flag"]          = df["flag"].map(mappings["flag"])
df = df.fillna(0)

# ── Scale features ───────────────────────────────────────────────────────────
X        = df[FEATURE_NAMES].values.astype(np.float32)
X_scaled = scaler.transform(X)


# ══════════════════════════════════════════════════════════════════════════════
# DENSE AE — RECONSTRUCTION ERRORS
# (no sequences needed — each row is independent)
# ══════════════════════════════════════════════════════════════════════════════

print("Running Dense AE anomaly detection...")

X_tensor = torch.from_numpy(X_scaled).to(DEVICE)

errors = []
BATCH  = 1024   # process in batches to avoid OOM on large files

with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH):
        batch = X_tensor[i:i+BATCH]
        recon = model(batch)
        err   = torch.mean((recon - batch) ** 2, dim=1)   # per-sample MSE
        errors.extend(err.cpu().numpy())

errors = np.array(errors)
print(f"Reconstruction error  mean={errors.mean():.4f}  std={errors.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS
# Normal  → reconstruction error below threshold
# Attack  → above threshold → XGBoost classifies attack type
# ══════════════════════════════════════════════════════════════════════════════

print("Generating predictions...")

# binary anomaly flags
is_anomaly = errors > ANOMALY_THRESHOLD   # boolean array, shape (N,)

# XGBoost predictions only for flagged rows (much faster than row-by-row loop)
attack_rows   = X_scaled[is_anomaly]
attack_ids    = clf.predict(attack_rows) if len(attack_rows) > 0 else []
attack_labels = le.inverse_transform(attack_ids) if len(attack_ids) > 0 else []

# build predicted_label column
predicted_labels              = np.where(is_anomaly, "attack_placeholder", "normal")
predicted_labels[is_anomaly]  = attack_labels   # fill in specific attack types

results_df = pd.DataFrame({
    "actual_label"         : df["label_attack"].values,
    "predicted_label"      : predicted_labels,
    "reconstruction_error" : errors,
    "flagged_as_anomaly"   : is_anomaly.astype(int)
})


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "train_ids_results_dense_ae.csv"
results_df.to_csv(out_path, index=False)
print(f"\nResults saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — BINARY (normal vs attack)
# ══════════════════════════════════════════════════════════════════════════════

y_true_bin = (results_df["actual_label"] != "normal").astype(int).values
y_pred_bin = (results_df["predicted_label"] != "normal").astype(int).values
tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
fpr = fp / (fp + tn)

print("\n" + "=" * 45)
print("BINARY METRICS  (normal vs attack)")
print("=" * 45)
print(f"Accuracy  : {accuracy_score(y_true_bin, y_pred_bin):.4f}")
print(f"Precision : {precision_score(y_true_bin, y_pred_bin):.4f}")
print(f"Recall    : {recall_score(y_true_bin, y_pred_bin):.4f}")
print(f"F1 Score  : {f1_score(y_true_bin, y_pred_bin):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_true_bin, errors):.4f}")

print("\n" + classification_report(
    y_true_bin, y_pred_bin,
    target_names=["Normal", "Attack"]
))


# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX COUNTS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 45)
print("CONFUSION MATRIX")
print("=" * 45)
print(f"True  Positives (attacks caught)  : {tp:>6}")
print(f"True  Negatives (normal correct)  : {tn:>6}")
print(f"False Positives (false alarms)    : {fp:>6}")
print(f"False Negatives (missed attacks)  : {fn:>6}")


# ══════════════════════════════════════════════════════════════════════════════
# EXACT LABEL MATCH  (normal / specific attack type)
# ══════════════════════════════════════════════════════════════════════════════

exact_correct = (results_df["actual_label"] == results_df["predicted_label"]).sum()
exact_wrong   = len(results_df) - exact_correct

print("\n" + "=" * 45)
print("EXACT LABEL MATCH  (multi-class)")
print("=" * 45)
print(f"Correct : {exact_correct}  ({exact_correct/len(results_df):.2%})")
print(f"Wrong   : {exact_wrong}  ({exact_wrong/len(results_df):.2%})")


# ══════════════════════════════════════════════════════════════════════════════
# PER ATTACK TYPE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 45)
print("PER ATTACK TYPE DETECTION RATE")
print("=" * 45)

attack_types = results_df[results_df["actual_label"] != "normal"]["actual_label"].unique()

for attack in sorted(attack_types):
    mask      = results_df["actual_label"] == attack
    total     = mask.sum()
    detected  = (results_df.loc[mask, "flagged_as_anomaly"] == 1).sum()
    rate      = detected / total if total > 0 else 0
    print(f"  {attack:<20} detected {detected:>5}/{total:<5}  ({rate:.2%})")

print("\nDone.")