from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # remove if running interactively
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
import seaborn as sns
from matplotlib.patches import Patch

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load & prepare the dataset
# ══════════════════════════════════════════════════════════════
cols = [
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
df = pd.read_csv(DATA_DIR / 'KDDTest+.txt', header=None, names=cols)
df = df.drop(columns=['difficulty'])

# Binary label: 0 = normal, 1 = attack
df['binary_label'] = (df['label'] != 'normal').astype(int)

# Encode categorical features
cat_cols = ['protocol_type', 'service', 'flag']
le = LabelEncoder()
for c in cat_cols:
    df[c] = le.fit_transform(df[c])

feature_cols = [c for c in df.columns if c not in ['label', 'binary_label']]
X = df[feature_cols]
y = df['binary_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Dataset: {df.shape[0]} records  |  41 features  |  Attack ratio: {y.mean():.2%}")
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}\n")

# ══════════════════════════════════════════════════════════════
# METHOD 1 — Random Forest (Gini/MDI importance)
# ══════════════════════════════════════════════════════════════
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"  Accuracy: {rf_acc:.4f}")

# ══════════════════════════════════════════════════════════════
# METHOD 2 — Decision Tree importance
# ══════════════════════════════════════════════════════════════
print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
dt_imp = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"  Accuracy: {dt_acc:.4f}")

# ══════════════════════════════════════════════════════════════
# METHOD 3 — Gradient Boosting importance
# ══════════════════════════════════════════════════════════════
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
gb_imp = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"  Accuracy: {gb_acc:.4f}")

# ══════════════════════════════════════════════════════════════
# METHOD 4 — Permutation Importance (model-agnostic)
# ══════════════════════════════════════════════════════════════
print("Computing Permutation Importance...")
perm = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)

# ══════════════════════════════════════════════════════════════
# METHOD 5 — Pearson Correlation with binary label
# ══════════════════════════════════════════════════════════════
corr_abs = df[feature_cols + ['binary_label']].corr()['binary_label'].drop('binary_label').abs().sort_values(ascending=False)

# ══════════════════════════════════════════════════════════════
# AGGREGATE — average rank across all 5 methods
# ══════════════════════════════════════════════════════════════
methods = {
    'RandomForest': rf_imp,
    'DecisionTree': dt_imp,
    'GradBoost':    gb_imp,
    'Permutation':  perm_imp,
    'Correlation':  corr_abs,
}
rank_df = pd.DataFrame({name: s.rank(ascending=False) for name, s in methods.items()})
rank_df['avg_rank'] = rank_df.mean(axis=1)
rank_df = rank_df.sort_values('avg_rank')

# ── thesis features to validate ──────────────────────────────
thesis_features = [
    'protocol_type','flag','src_bytes','logged_in','num_failed_logins',
    'serror_rate','rerror_rate','diff_srv_rate',
    'dst_host_count','dst_host_srv_count','service'
]

print("\n════════════════════════════════════════════════════════")
print("  THESIS FEATURE VALIDATION  —  Rankings across 5 methods")
print("  (lower rank number = more important)")
print("════════════════════════════════════════════════════════")
print(f"{'Feature':<28} {'RF':>5} {'DT':>5} {'GB':>5} {'Perm':>6} {'Corr':>6} {'Avg':>7} {'Overall':>8}")
print("─"*73)
for f in thesis_features:
    row  = rank_df.loc[f]
    pos  = list(rank_df.index).index(f) + 1
    flag = " ✓" if pos <= 20 else "  "
    print(f"{f:<28} {row['RandomForest']:>5.0f} {row['DecisionTree']:>5.0f} "
          f"{row['GradBoost']:>5.0f} {row['Permutation']:>6.0f} {row['Correlation']:>6.0f} "
          f"{row['avg_rank']:>7.1f} #{pos:>5}{flag}")

print("\n  ✓ = feature ranks in top-20 across all 41 features")

# ══════════════════════════════════════════════════════════════
# PLOT — 6-panel figure for thesis
# ══════════════════════════════════════════════════════════════
highlight = set(thesis_features)
CLR_HI    = '#E24B4A'   # red  — thesis features
CLR_LO    = '#B4B2A9'   # gray — others
TOP_N     = 20

def bar_plot(ax, series, title, highlight_set, xlabel='Score'):
    top    = series.head(TOP_N)
    colors = [CLR_HI if f in highlight_set else CLR_LO for f in top.index]
    bars   = ax.barh(range(len(top)), top.values, color=colors, edgecolor='none', height=0.65)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    mx = max(top.values)
    for i, (_, val) in enumerate(zip(bars, top.values)):
        ax.text(val + mx*0.01, i, f'{val:.4f}', va='center', fontsize=7, color='#555')

fig = plt.figure(figsize=(18, 22), facecolor='#FAFAF8')
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0]); bar_plot(ax1, rf_imp,   'Method 1 — Random Forest (MDI)',      highlight)
ax2 = fig.add_subplot(gs[0, 1]); bar_plot(ax2, gb_imp,   'Method 3 — Gradient Boosting',         highlight)
ax3 = fig.add_subplot(gs[1, 0]); bar_plot(ax3, dt_imp,   'Method 2 — Decision Tree',             highlight)
ax4 = fig.add_subplot(gs[1, 1]); bar_plot(ax4, perm_imp, 'Method 4 — Permutation Importance',    highlight)
ax5 = fig.add_subplot(gs[2, 0]); bar_plot(ax5, corr_abs, 'Method 5 — Pearson |Correlation|',     highlight, xlabel='|Correlation|')

# heatmap — thesis features × 5 methods
ax6  = fig.add_subplot(gs[2, 1])
hm   = pd.DataFrame({name: methods[name][thesis_features] for name in methods})
hm_n = hm.apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-9))
sns.heatmap(hm_n, ax=ax6, cmap='RdYlGn', linewidths=0.5, linecolor='white',
            annot=True, fmt='.2f', annot_kws={'size': 7},
            cbar_kws={'shrink': 0.6})
ax6.set_title('Method 6 — Normalised score heatmap\n(thesis features across all 5 methods)',
              fontsize=10.5, fontweight='bold', pad=6)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=30, ha='right', fontsize=8)
ax6.set_yticklabels(ax6.get_yticklabels(), rotation=0, fontsize=8)

legend_elements = [
    Patch(facecolor=CLR_HI, label='Thesis target features (11)'),
    Patch(facecolor=CLR_LO, label='Other features'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, 0.998), frameon=False)

fig.suptitle(
    'Feature Importance Validation — KDD Cup Network Intrusion Dataset\n'
    '5-Method Proof: RF · Decision Tree · Gradient Boosting · Permutation · Pearson Correlation\n'
    f'Model accuracies — RF: {rf_acc:.4f}  |  DT: {dt_acc:.4f}  |  GB: {gb_acc:.4f}',
    fontsize=12.5, fontweight='bold', y=1.015
)

plt.savefig('results/feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#FAFAF8')
print("\nFigure saved → results/feature_importance.png")