import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ====================== 1. LOAD YOUR DATA ======================
columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

BASE_DIR   = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR = BASE_DIR / "data/raw/nsl-kdd"
MODEL_DIR  = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"
df = pd.read_csv(DATA_DIR / 'KDDTest+.txt', header=None, names=columns)

# ====================== 2. TARGET CREATION ======================
df['is_attack'] = (df['label'] != 'normal').astype(int)

# ====================== 3. NUMERIC FEATURES ======================
numeric_cols = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'difficulty'
]

# ====================== 4. REMOVE CONSTANT COLUMNS ======================
std_dev = df[numeric_cols].std()
valid_numeric_cols = std_dev[std_dev > 0].index.tolist()

removed_cols = set(numeric_cols) - set(valid_numeric_cols)
if removed_cols:
    print(f"⚠️ Removed constant columns: {removed_cols}")
# ====================== REMOVE DATA LEAKAGE ======================
if 'difficulty' in valid_numeric_cols:
    valid_numeric_cols.remove('difficulty')
    print("⚠️ Removed 'difficulty' (data leakage)")

# ====================== 5. CORRELATION ======================
corr = df[valid_numeric_cols].corrwith(df['is_attack'])

# Remove NaNs safely
corr = corr.replace([np.inf, -np.inf], np.nan).dropna()

corr = corr.abs().sort_values(ascending=False)

print("\n=== TOP 15 FEATURES BY ABSOLUTE CORRELATION ===")
print(corr.head(15))

# ====================== 6. RANDOM FOREST ======================
cat_cols = ['protocol_type', 'service', 'flag']

# Encode categorical features safely
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Final feature set
X = df[valid_numeric_cols + cat_cols]
y = df['is_attack']

# ====================== 7. TRAIN RF ======================
rf = RandomForestClassifier(
    n_estimators=200,     # increased for stability
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\n=== TOP 15 FEATURES BY RANDOM FOREST IMPORTANCE ===")
print(importances.head(15))

# ====================== 8. BONUS: SAVE RESULTS ======================
corr.to_csv(RESULT_DIR / "correlation_features.csv")
importances.to_csv(RESULT_DIR / "rf_feature_importance.csv")

print("\n✅ Results saved to:", RESULT_DIR)