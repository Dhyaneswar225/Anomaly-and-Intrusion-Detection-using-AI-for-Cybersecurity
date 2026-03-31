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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.DenseAutoEncoderModel import DenseAutoencoder
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

BASE_DIR  = Path("F:/Master Thesis/anomaly-ids")

TRAIN_PATH = BASE_DIR / "data/raw/nsl-kdd/KDDTrain+.txt"
TEST_PATH  = BASE_DIR / "data/raw/nsl-kdd/KDDTest+.txt"

DATA_DIR  = BASE_DIR / "data/processed"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "data/generated"
RES_DIR = BASE_DIR / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════

print("\nLoading artifacts...")

scaler = joblib.load(DATA_DIR / "standard_scaler.pkl")
FEATURE_NAMES = list(scaler.feature_names_in_)

ANOMALY_THRESHOLD = float(np.load(MODEL_DIR / "dense_ae_threshold.npy"))
print(f"✅ Features          : {len(FEATURE_NAMES)}")
print(f"✅ Anomaly threshold : {ANOMALY_THRESHOLD:.6f}")

with open(DATA_DIR / "label_mappings.json") as f:
    mappings = json.load(f)

# Load Autoencoder
model = DenseAutoencoder(
    input_dim=len(FEATURE_NAMES),
    bottleneck=32,
    dropout=0.2
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_DIR / "dense_autoencoder_best.pth", map_location=DEVICE)
)
model.eval()
print("✅ Dense Autoencoder loaded")

# Load classifier
clf = joblib.load(MODEL_DIR / "attack_classifier_full_xgb.pkl")
le  = joblib.load(MODEL_DIR / "attack_label_encoder_full.pkl")
print("✅ Attack classifier loaded")

# ═══════════════════════════════════════════════════════════════
# COLUMNS
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# LOAD + MERGE DATA
# ═══════════════════════════════════════════════════════════════

print("\nLoading Train + Selected Test Attacks...")

train_df = pd.read_csv(TRAIN_PATH, names=COLUMNS)
test_df  = pd.read_csv(TEST_PATH, names=COLUMNS)

# ✅ Use FULL test dataset for evaluation
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
df["label_attack"] = df["label"]

print(f"Train rows: {len(train_df)}")
print(f"Test rows : {len(test_df)}")
print(f"Total rows: {len(df)}")

# ═══════════════════════════════════════════════════════════════
# PREPROCESS
# ═══════════════════════════════════════════════════════════════

df["protocol_type"] = df["protocol_type"].map(mappings["protocol_type"])
df["service"]       = df["service"].map(mappings["service"])
df["flag"]          = df["flag"].map(mappings["flag"])

df = df.fillna(0)

# 🔥 FIX: Ensure float32
X_scaled = scaler.transform(df[FEATURE_NAMES]).astype(np.float32)

# ═══════════════════════════════════════════════════════════════
# AUTOENCODER
# ═══════════════════════════════════════════════════════════════

print("Running Dense AE...")

X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)

errors = []
BATCH = 1024

with torch.no_grad():
    for i in range(0, len(X_tensor), BATCH):
        batch = X_tensor[i:i+BATCH]
        recon = model(batch)
        err = torch.mean((recon - batch) ** 2, dim=1)
        errors.extend(err.cpu().numpy())

errors = np.array(errors)

print(f"Reconstruction error mean={errors.mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# PREDICTIONS
# ═══════════════════════════════════════════════════════════════

print("Generating predictions...")

is_anomaly = errors > ANOMALY_THRESHOLD

attack_rows = X_scaled[is_anomaly]

if len(attack_rows) > 0:
    attack_ids = clf.predict(attack_rows)
    attack_labels = le.inverse_transform(attack_ids)
else:
    attack_labels = []

predicted_labels = np.where(is_anomaly, "attack", "normal")
predicted_labels[is_anomaly] = attack_labels

results_df = pd.DataFrame({
    "actual_label": df["label_attack"].values,
    "predicted_label": predicted_labels,
    "error": errors,
    "is_anomaly": is_anomaly.astype(int)
})

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "ids_results_full.csv"
results_df.to_csv(out_path, index=False)

print(f"\n✅ Results saved → {out_path}")

# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

y_true = (results_df["actual_label"] != "normal").astype(int)
y_pred = (results_df["predicted_label"] != "normal").astype(int)

print("\n=== METRICS ===")
print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision : {precision_score(y_true, y_pred):.4f}")
print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_true, errors):.4f}")

print("\n" + classification_report(y_true, y_pred))

# ----------- 1. BINARY CONFUSION MATRIX -----------
print("\n=== BINARY CONFUSION MATRIX ===")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

# Plot Binary CM
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Binary Confusion Matrix")
plt.tight_layout()
plt.savefig(RES_DIR / "confusion_matrix_binary_train_test_app.png")


# ----------- 2. MULTI-CLASS CONFUSION MATRIX -----------
print("\n=== MULTI-CLASS CONFUSION MATRIX ===")

labels = sorted(results_df["actual_label"].unique())

cm_multi = confusion_matrix(
    results_df["actual_label"],
    results_df["predicted_label"],
    labels=labels
)

print("Matrix shape:", cm_multi.shape)

# Plot Multi-class CM
plt.figure(figsize=(14,12))
sns.heatmap(cm_multi,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multi-Class Confusion Matrix (All Attacks)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(RES_DIR / "confusion_matrix_multiclass_train_test_app.png")


# ----------- 3. SAVE CONFUSION MATRIX -----------
OUT_DIR.mkdir(parents=True, exist_ok=True)

cm_df = pd.DataFrame(cm_multi, index=labels, columns=labels)
cm_df.to_csv(RES_DIR / "confusion_matrix_38_classes_train_test_app.csv")

print("✅ Confusion matrix saved to CSV")


# ═══════════════════════════════════════════════════════════════
# PER ATTACK DETECTION
# ═══════════════════════════════════════════════════════════════

print("\n=== PER ATTACK DETECTION ===")

attack_types = results_df[results_df["actual_label"] != "normal"]["actual_label"].unique()

for attack in sorted(attack_types):
    mask = results_df["actual_label"] == attack
    total = mask.sum()
    detected = (results_df.loc[mask, "is_anomaly"] == 1).sum()
    rate = detected / total if total > 0 else 0
    print(f"{attack:<20} {detected}/{total} ({rate:.2%})")

print("\n✅ Done.")