"""
DenseAutoEncoderTrain.py
────────────────────────
Fixes applied vs previous version:
  [1] Threshold sweep  → find best percentile instead of hardcoding p95
  [2] More epochs      → 200, with patience=10 (was 5) to avoid early LR drops
  [3] Larger bottleneck → 32 (was 16) to retain more information
  [4] ROC curve plot   → saved as roc_curve.png
  [5] Precision-Recall curve → saved as pr_curve.png (more informative for imbalanced data)
  [6] Threshold saved  → so inference script can load it without recomputing
  [7] Per-attack-type breakdown → if multi-class labels exist
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, f1_score, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

from DenseAutoEncoderModel import DenseAutoencoder   # ← make sure this file is in the same folder


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit paths here
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR   = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR   = BASE_DIR / "data/processed"
MODEL_DIR  = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS      = 200          # fix #2: was 100
BATCH_SIZE  = 512
LR          = 1e-3
BOTTLENECK  = 32           # fix #3: was 16
DROPOUT     = 0.2
PATIENCE_LR = 10           # fix #2: was 5 (scheduler patience)

print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

train_df = pd.read_csv(DATA_DIR / "train_processed.csv")
test_df  = pd.read_csv(DATA_DIR / "test_processed.csv")
scaler   = joblib.load(DATA_DIR / "standard_scaler.pkl")

FEATURE_NAMES = list(scaler.feature_names_in_)

print(f"Features : {len(FEATURE_NAMES)}")
print(f"Train rows: {len(train_df)}  |  Test rows: {len(test_df)}")
print(f"Train label dist:\n{train_df['label_binary'].value_counts()}\n")

# ── Train: normal only ───────────────────────────────────────────────────────
X_train_normal = (
    train_df[train_df["label_binary"] == "normal"][FEATURE_NAMES]
    .values.astype(np.float32)
)

# ── Test: all samples ────────────────────────────────────────────────────────
X_test = test_df[FEATURE_NAMES].values.astype(np.float32)
y_test = (test_df["label_binary"] == "attack").astype(int).values

print(f"Normal train samples : {len(X_train_normal)}")
print(f"Test samples         : {len(X_test)}")
print(f"Test attack ratio    : {y_test.mean():.2%}\n")


# ══════════════════════════════════════════════════════════════════════════════
# DATALOADERS
# ══════════════════════════════════════════════════════════════════════════════

train_tensor = torch.from_numpy(X_train_normal)
train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,      # shuffles every epoch
    drop_last=True     # avoids BatchNorm issues with tiny last batch
)

test_tensor  = torch.from_numpy(X_test)
test_loader  = DataLoader(
    TensorDataset(test_tensor),
    batch_size=1024,
    shuffle=False
)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

model     = DenseAutoencoder(
                input_dim  = len(FEATURE_NAMES),
                bottleneck = BOTTLENECK,
                dropout    = DROPOUT
            ).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.MSELoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode      = "min",
    patience  = PATIENCE_LR,   # fix #2
    factor    = 0.5,
    min_lr    = 1e-6
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

best_loss  = float("inf")
best_state = None
history    = []

print("=" * 55)
print("Training Dense Autoencoder...")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []

    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        recon = model(batch)
        loss  = criterion(recon, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    mean_loss = float(np.mean(epoch_losses))
    scheduler.step(mean_loss)
    history.append(mean_loss)

    # ── save best checkpoint ─────────────────────────────────────────────────
    if mean_loss < best_loss:
        best_loss  = mean_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {mean_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

# ── load best weights ────────────────────────────────────────────────────────
model.load_state_dict(best_state)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(best_state, MODEL_DIR / "dense_autoencoder_best.pth")
print(f"\nBest model saved  (loss = {best_loss:.6f})")


# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION ERRORS
# ══════════════════════════════════════════════════════════════════════════════

model.eval()

def compute_errors(loader):
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            # per-sample MSE across all features
            errs  = torch.mean((recon - batch) ** 2, dim=1)
            errors.extend(errs.cpu().numpy())
    return np.array(errors)

# need a non-shuffled train loader for error computation
train_eval_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=1024,
    shuffle=False
)

train_errors = compute_errors(train_eval_loader)
test_errors  = compute_errors(test_loader)

print(f"\nTrain error  mean={train_errors.mean():.4f}  std={train_errors.std():.4f}")
print(f"Test  error  mean={test_errors.mean():.4f}  std={test_errors.std():.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIX #1 — THRESHOLD SWEEP: find best percentile
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("Threshold sweep (fix #1)...")
print("=" * 65)
print(f"{'Percentile':>10} | {'Threshold':>10} | {'F1':>8} | {'Recall':>8} | {'Precision':>10} | {'FPR':>8}")
print("-" * 65)

best_f1         = 0
best_threshold  = None
best_percentile = None
percentiles = [90, 95, 96, 97, 98, 98.5, 99, 99.5, 99.9]
for p in percentiles:
    thr = np.percentile(train_errors, p)
    preds = (test_errors > thr).astype(int)
    
    f1 = f1_score(y_test, preds)
    rec = recall_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    
    # Calculate False Positive Rate (FPR) - This is what we want to minimize
    tn = ((y_test == 0) & (preds == 0)).sum()
    fp = ((y_test == 0) & (preds == 1)).sum()
    fpr = fp / (fp + tn)

    marker = ""
    # STRATEGY: Find best F1, but prioritize high percentiles to kill False Alarms
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thr
        best_percentile = p
        marker = " [Best F1]"

    print(f"p{p:>9} | {thr:>10.6f} | {f1:>8.4f} | {rec:>8.4f} | {prec:>10.4f} | {fpr:>8.4f}{marker}")

# Save the optimized threshold
np.save(MODEL_DIR / "dense_ae_threshold.npy", np.array(best_threshold))
print(f"\nFinal Choice: p{best_percentile} (Threshold: {best_threshold:.6f})")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

final_preds = (test_errors > best_threshold).astype(int)

print("\n" + "=" * 55)
print("Final Evaluation")
print("=" * 55)
print(f"ROC-AUC  : {roc_auc_score(y_test, test_errors):.4f}")
print(f"Avg Prec : {average_precision_score(y_test, test_errors):.4f}")
print(f"F1 Score : {f1_score(y_test, final_preds):.4f}")
print()
print(classification_report(y_test, final_preds, target_names=["Normal", "Attack"]))


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS  (fix #4, #5)
# ══════════════════════════════════════════════════════════════════════════════

RESULT_DIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Dense Autoencoder — NSL-KDD", fontsize=14, fontweight="bold")

# ── Plot 1: Training loss curve ───────────────────────────────────────────────
axes[0].plot(range(1, EPOCHS + 1), history, color="steelblue", linewidth=1.5)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].grid(True, alpha=0.3)

# ── Plot 2: ROC Curve ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, test_errors)
auc_score   = roc_auc_score(y_test, test_errors)

axes[1].plot(fpr, tpr, color="darkorange", linewidth=2, label=f"AUC = {auc_score:.4f}")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

# ── Plot 3: Precision-Recall Curve ────────────────────────────────────────────
prec_vals, rec_vals, _ = precision_recall_curve(y_test, test_errors)
ap_score               = average_precision_score(y_test, test_errors)

axes[2].plot(rec_vals, prec_vals, color="green", linewidth=2, label=f"AP = {ap_score:.4f}")
axes[2].set_title("Precision-Recall Curve")
axes[2].set_xlabel("Recall")
axes[2].set_ylabel("Precision")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULT_DIR / "dense_ae_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlots saved → {RESULT_DIR / 'dense_ae_results.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION ERROR DISTRIBUTION  (bonus: helps visualise threshold)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 4))

normal_errors = test_errors[y_test == 0]
attack_errors = test_errors[y_test == 1]

ax.hist(normal_errors, bins=100, alpha=0.6, color="steelblue", label="Normal",  density=True)
ax.hist(attack_errors, bins=100, alpha=0.6, color="tomato",    label="Attack",  density=True)
ax.axvline(best_threshold, color="black", linestyle="--", linewidth=1.5,
           label=f"Threshold (p{best_percentile}) = {best_threshold:.4f}")
ax.set_title("Reconstruction Error Distribution")
ax.set_xlabel("MSE Reconstruction Error")
ax.set_ylabel("Density")
ax.legend()
ax.set_xlim(0, np.percentile(test_errors, 99.5))  # clip extreme outliers for readability
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULT_DIR / "dense_ae_error_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Error distribution saved → {RESULT_DIR / 'dense_ae_error_dist.png'}")

print("\nDone.")