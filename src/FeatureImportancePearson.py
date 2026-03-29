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
BASE_DIR = Path("F:/Master Thesis/anomaly-ids")
DATA_DIR = BASE_DIR / "data/raw/nsl-kdd"
df = pd.read_csv(DATA_DIR / 'KDDTest+.txt', header=None, names=columns)

# Binary target: 0 = normal, 1 = attack
df['is_attack'] = (df['label'] != 'normal').astype(int)

# ====================== 2. NUMERIC FEATURES ONLY (CORRELATION) ======================
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

corr = df[numeric_cols].corrwith(df['is_attack']).abs().sort_values(ascending=False)
print("=== TOP 15 FEATURES BY ABSOLUTE CORRELATION ===")
print(corr.head(15))

# ====================== 3. RANDOM FOREST FEATURE IMPORTANCE (FULL PROOF) ======================
# Encode categorical features
cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[numeric_cols + cat_cols]
y = df['is_attack']

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== TOP 15 FEATURES BY RANDOM FOREST IMPORTANCE ===")
print(importances.head(15))