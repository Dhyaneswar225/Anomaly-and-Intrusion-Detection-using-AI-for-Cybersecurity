# src/baseline_models.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Setup directories
# ===========================
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===========================
# Load processed data
# ===========================
train = pd.read_csv("data/processed/train_processed.csv")
test  = pd.read_csv("data/processed/test_processed.csv")

# ---- safety checks -------------------------------------------------
if "label_binary" not in train.columns or "label_binary" not in test.columns:
    raise ValueError("Both files must contain a 'label_binary' column.")

# ---- map labels ----------------------------------------------------
y_train = train["label_binary"].map({"normal": 0, "attack": 1})
y_test  = test["label_binary"].map({"normal": 0, "attack": 1})

X_train_full = train.drop(columns=["label", "label_attack", "label_binary"])
X_test_full  = test.drop(columns=["label", "label_attack", "label_binary"])

# ---- keep **only common** columns ----------------------------------
common_cols = X_train_full.columns.intersection(X_test_full.columns)
if len(common_cols) == 0:
    raise ValueError("No overlapping feature columns between train and test!")

X_train = X_train_full[common_cols]
X_test  = X_test_full[common_cols]

# ---- CONVERT TO NUMPY TO AVOID FEATURE NAME WARNINGS ----
X_train = X_train.to_numpy()
X_test  = X_test.to_numpy()

print(f"Using {len(common_cols)} common features "
      f"(train had {len(X_train_full.columns)}, test had {len(X_test_full.columns)})")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train normal/attack: {sum(y_train==0)} / {sum(y_train==1)}")
print(f"Test  normal/attack: {sum(y_test==0)}  / {sum(y_test==1)}")

# --------------------------------------------------------------------
results = []  # one dict per model

# =========================== SUPERVISED ===========================
print("\n=== SUPERVISED MODELS ===")

def evaluate_supervised(model, name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = (model.predict_proba(X_test)[:, 1]
             if hasattr(model, "predict_proba") else preds)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    # ---- confusion matrix plot ----
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} – Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"results/{name}_confusion_matrix.png")
    plt.close()

    # ---- save model ----
    joblib.dump(model, f"models/{name}.pkl")

    # ---- store results ----
    results.append({
        "Model"       : name,
        "Type"        : "Supervised",
        "Accuracy"    : round(acc, 6),
        "F1"          : round(f1, 6),
        "ROC-AUC"     : round(roc, 6),
        "AUC"         : round(roc, 6),
        "Precision@K" : None
    })

# RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_supervised(rf, "RandomForest")
print("Random Forest Done")

# XGBoost
xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss"
)
evaluate_supervised(xgb, "XGBoost")
print("XGBoost Done")

# ========================= UNSUPERVISED ==========================
print("\n=== UNSUPERVISED MODELS ===")

def evaluate_unsupervised(model, name, contamination=0.1, novelty=False):
    """
    novelty=True  → fit on normal train data → predict on test
    novelty=False → fit_predict on test (only for LOF classic mode)
    """
    if name == "LocalOutlierFactor" and not novelty:
        preds  = model.fit_predict(X_test)
        scores = -model.negative_outlier_factor_
    else:
        normal_train = X_train[y_train == 0]
        if len(normal_train) == 0:
            raise ValueError("No normal training samples.")
        model.fit(normal_train)

        if hasattr(model, "decision_function"):
            scores = -model.decision_function(X_test)
        else:
            scores = -model.score_samples(X_test)

        preds = model.predict(X_test)

    # ------------------------------------------------------------------
    # 1. NORMALISE SCORES TO [0, 1]  (Min-Max scaling)
    # ------------------------------------------------------------------
    #   * IsolationForest  → already roughly -0.5 … +0.5  (optional)
    #   * OneClassSVM      → raw distance (0 … thousands)
    #   * LOF              → density ratio (≈1 … millions)
    # ------------------------------------------------------------------
    if name in ("OneClassSVM", "LocalOutlierFactor"):
        # Min-Max scaling → 0 … 1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    # (IsolationForest is left untouched – its scores are already bounded)

    # ------------------------------------------------------------------
    # 2. Convert binary predictions (-1 / +1) → 0 / 1
    # ------------------------------------------------------------------
    preds = np.where(preds == -1, 1, 0)          # -1 → attack (1), +1 → normal (0)

    # ------------------------------------------------------------------
    # 3. METRICS (use the *same* scores for ranking)
    # ------------------------------------------------------------------
    auc_score = roc_auc_score(y_test, scores)
    k = int(0.1 * len(scores))
    top_k_idx = np.argsort(scores)[-k:]          # highest scores = most anomalous
    prec_k = np.mean(y_test.iloc[top_k_idx])

    # ------------------------------------------------------------------
    # 4. PLOT – now all histograms are on the same 0-1 axis
    # ------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    sns.histplot(scores, bins=50, kde=True, color="#9467bd")
    plt.title(f"{name} – Anomaly Score Distribution (normalised)")
    plt.xlabel("Score (0 = normal, 1 = most anomalous)")
    plt.tight_layout()
    plt.savefig(f"results/{name}_anomaly_score_distribution.png")
    plt.close()

    # ------------------------------------------------------------------
    # 5. SAVE MODEL
    # ------------------------------------------------------------------
    joblib.dump(model, f"models/{name}.pkl")
    print(f"Model saved: models/{name}.pkl")

    # ------------------------------------------------------------------
    # 6. STORE RESULTS
    # ------------------------------------------------------------------
    results.append({
        "Model"       : name,
        "Type"        : "Unsupervised",
        "Accuracy"    : None,
        "F1"          : None,
        "ROC-AUC"     : None,
        "AUC"         : round(auc_score, 6),
        "Precision@K" : round(prec_k, 6)
    })

# ----------------------------------------------------------------------
#  Run the three unsupervised models
# ----------------------------------------------------------------------
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
evaluate_unsupervised(iso, "IsolationForest")
print("Isolation Forest Done")

ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
evaluate_unsupervised(ocsvm, "OneClassSVM")
print("One-Class SVM Done")

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
evaluate_unsupervised(lof, "LocalOutlierFactor", novelty=True)
print("Local Outlier Factor Done")

# ========================= SUMMARY ==============================
results_df = pd.DataFrame(results)
cols_order = ["Model", "Type", "Accuracy", "F1", "ROC-AUC", "AUC", "Precision@K"]
results_df = results_df[cols_order]

# Replace None with NaN for clean CSV
results_df = results_df.fillna(pd.NA)

results_df.to_csv("results/metrics_summary.csv", index=False,na_rep="None")
print("\nMetrics saved → results/metrics_summary.csv")

# Pretty print
pd.set_option('display.float_format', '{:.6f}'.format)
print("\n=== FINAL METRICS TABLE ===")
print(results_df.to_string(index=False, na_rep="None"))