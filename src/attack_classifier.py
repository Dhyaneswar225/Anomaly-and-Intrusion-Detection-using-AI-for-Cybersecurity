import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import os

# ============================
# Load processed dataset
# ============================
train = pd.read_csv("data/processed/train_processed.csv")

# ============================
# Extract true attack labels
# ============================
if "label_attack" not in train.columns:
    raise ValueError("train_processed.csv must contain label_attack column.")

# Remove useless columns
drop_cols = ["label", "label_binary"]
features = [c for c in train.columns if c not in drop_cols + ["label_attack"]]

X = train[features].values.astype(np.float32)
y = train["label_attack"]

# Encode attack names
le = LabelEncoder()
y_enc = le.fit_transform(y)

print("Attack classes:", list(le.classes_))
print("Total classes:", len(le.classes_))

# ============================
# Train classifier
# ============================
clf = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42
)

clf.fit(X, y_enc)

# ============================
# Save model and encoder
# ============================
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/attack_classifier_xgb.pkl")
joblib.dump(le, "models/attack_label_encoder.pkl")

print("Saved attack classifier + label encoder.")
