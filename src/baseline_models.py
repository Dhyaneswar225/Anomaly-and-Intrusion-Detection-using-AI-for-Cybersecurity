# src/baseline_models.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Setup
# ===========================
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===========================
# Load data
# ===========================
train = pd.read_csv("data/processed/train_processed.csv")
test  = pd.read_csv("data/processed/test_processed.csv")

y_train = train["label_binary"].map({"normal": 0, "attack": 1})
y_test  = test["label_binary"].map({"normal": 0, "attack": 1})

X_train_full = train.drop(columns=["label", "label_attack", "label_binary"])
X_test_full  = test.drop(columns=["label", "label_attack", "label_binary"])

common_cols = X_train_full.columns.intersection(X_test_full.columns)

X_train = X_train_full[common_cols].to_numpy()
X_test  = X_test_full[common_cols].to_numpy()

print(f"\nUsing {len(common_cols)} features")

results = []

# ===========================
# Utility Functions
# ===========================

def evaluate_supervised(name, y_true, preds, probs):
    acc = accuracy_score(y_true, preds)
    f1  = f1_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec  = recall_score(y_true, preds)
    roc  = roc_auc_score(y_true, probs)

    print(f"\n{name}")
    print("Accuracy :", round(acc,4))
    print("F1 Score :", round(f1,4))
    print("Precision:", round(prec,4))
    print("Recall   :", round(rec,4))
    print("ROC-AUC  :", round(roc,4))

    results.append({
        "Model": name,
        "Type": "Supervised",
        "Accuracy": acc,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC-AUC": roc
    })


def evaluate_unsupervised(name, scores):

    auc = roc_auc_score(y_test, scores)

    k = int(0.1 * len(scores))
    top_k = np.argsort(scores)[-k:]
    prec_k = np.mean(y_test.iloc[top_k])

    print(f"\n{name}")
    print("AUC        :", round(auc,4))
    print("Precision@K:", round(prec_k,4))

    results.append({
        "Model": name,
        "Type": "Unsupervised",
        "Accuracy": None,
        "F1": None,
        "Precision": None,
        "Recall": None,
        "ROC-AUC": None,
        "AUC": auc,
        "Precision@K": prec_k
    })


# ===========================
# Random Forest
# ===========================
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:,1]

evaluate_supervised("RandomForest", y_test, rf_preds, rf_probs)
joblib.dump(rf, "models/RandomForest.pkl")


# ===========================
# XGBoost
# ===========================
scale_pos_weight = sum(y_train==0)/sum(y_train==1)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)

probs = xgb.predict_proba(X_test)[:,1]
preds = (probs > 0.2).astype(int)

evaluate_supervised("XGBoost", y_test, preds, probs)
joblib.dump(xgb, "models/XGBoost.pkl")


# ===========================
# Isolation Forest
# ===========================
iso = IsolationForest(n_estimators=150, contamination=0.1, random_state=42)
iso.fit(X_train[y_train==0])

iso_scores = -iso.decision_function(X_test)

iso_scaler = MinMaxScaler()
iso_scores = iso_scaler.fit_transform(iso_scores.reshape(-1,1)).flatten()

evaluate_unsupervised("IsolationForest", iso_scores)
joblib.dump(iso, "models/IsolationForest.pkl")


# ===========================
# One-Class SVM
# ===========================
ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
ocsvm.fit(X_train[y_train==0])

ocsvm_scores = -ocsvm.decision_function(X_test)
ocsvm_scores = MinMaxScaler().fit_transform(ocsvm_scores.reshape(-1,1)).flatten()

evaluate_unsupervised("OneClassSVM", ocsvm_scores)
joblib.dump(ocsvm, "models/OneClassSVM.pkl")


# ===========================
# Local Outlier Factor
# ===========================
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X_train[y_train==0])

lof_scores = -lof.decision_function(X_test)
lof_scores = MinMaxScaler().fit_transform(lof_scores.reshape(-1,1)).flatten()

evaluate_unsupervised("LocalOutlierFactor", lof_scores)
joblib.dump(lof, "models/LocalOutlierFactor.pkl")


# ===========================
# HYBRID MODEL (BEST)
# ===========================
print("\nTraining HYBRID MODEL...")

X_train_hybrid = np.hstack([
    X_train,
    MinMaxScaler().fit_transform(-iso.decision_function(X_train).reshape(-1,1))
])

X_test_hybrid = np.hstack([
    X_test,
    iso_scores.reshape(-1,1)
])

hybrid = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

hybrid.fit(X_train_hybrid, y_train)

hybrid_probs = hybrid.predict_proba(X_test_hybrid)[:,1]
hybrid_preds = (hybrid_probs > 0.2).astype(int)

evaluate_supervised("Hybrid IDS", y_test, hybrid_preds, hybrid_probs)
joblib.dump(hybrid, "models/Hybrid_XGBoost.pkl")


# ===========================
# Save Results
# ===========================
results_df = pd.DataFrame(results)
results_df.to_csv("results/final_results.csv", index=False)

print("\n=== FINAL RESULTS ===")
print(results_df)