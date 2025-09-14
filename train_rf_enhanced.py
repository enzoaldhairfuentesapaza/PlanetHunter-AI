
import os
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    accuracy_score,
)

# ------------------------------
# 0. Paths
# ------------------------------
BASE = Path(__file__).resolve().parent
DATA_RAW = BASE / "data" / "raw"
FIG_DIR = BASE / "data_science" / "figures"
RPT_DIR = BASE / "data_science" / "reports"
MODEL_DIR = BASE / "data_science" / "models"

for p in [FIG_DIR, RPT_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------
# 1. Load latest CSV (prefer raw to avoid parquet engine issues)
# ------------------------------
csvs = sorted(DATA_RAW.glob("*.csv"))
if not csvs:
    raise FileNotFoundError(f"No CSVs found in {DATA_RAW}")
csv_path = csvs[-1]
print(f"[INFO] Using dataset: {csv_path.name}")

df = pd.read_csv(csv_path)

# Muestreo opcional para ejecuciones rápidas
sample_n = os.getenv('SAMPLE_N')
if sample_n:
    sample_n = int(sample_n)
    if sample_n < len(df):
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampling to {sample_n} rows for fast run")


# ------------------------------
# 2. Labeling
#   - Binary: 1 = CONFIRMED, 0 = FALSE POSITIVE
#   - Drop CANDIDATE for supervision
# ------------------------------
df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)

# ------------------------------
# 3. Feature selection
# ------------------------------
drop_cols = {
    "label",
    "koi_disposition", "koi_pdisposition", "koi_disp_prov",
    "kepoi_name", "kepler_name",
    "ra_str", "dec_str",
}
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df["label"].copy()

# Identify numeric / categorical
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# ------------------------------
# 4. Preprocess & split
# ------------------------------
RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
)

n_est = int(os.getenv('N_EST', '500'))
rf = RandomForestClassifier(
    n_estimators=n_est,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)

pipe = Pipeline(steps=[("preprocess", preprocess), ("rf", rf)])

# ------------------------------
# 5. Train
# ------------------------------
pipe.fit(X_train, y_train)

# ------------------------------
# 6. Evaluate
# ------------------------------
y_proba = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
ap = average_precision_score(y_test, y_proba)  # PR-AUC
try:
    roc = roc_auc_score(y_test, y_proba)
except Exception:
    roc = float("nan")

report = classification_report(y_test, y_pred, digits=3)
cm = confusion_matrix(y_test, y_pred)

print("[METRICS]")
print(f"Accuracy: {acc:.4f}")
print(f"PR-AUC (Average Precision): {ap:.4f}")
print(f"ROC-AUC: {roc:.4f}")
print(report)

# Save report
(RPT_DIR / "classification_report.txt").write_text(report)
(RPT_DIR / "version.txt").write_text(f"RF enhanced {datetime.now():%Y-%m-%d %H:%M:%S}\n")

# ------------------------------
# 7. Precision-Recall curve & recall@P=0.99
# ------------------------------
prec, rec, thr = precision_recall_curve(y_test, y_proba)
# thr has len n-1 vs prec/rec len n

target_precision = 0.99
recall_at_099 = 0.0
threshold_at_099 = 1.0

for p, r, t in zip(prec[:-1], rec[:-1], thr):
    if p >= target_precision and r > recall_at_099:
        recall_at_099 = r
        threshold_at_099 = t

metrics_dict = {
    "accuracy": acc,
    "pr_auc": ap,
    "roc_auc": roc,
    "recall_at_precision_0.99": recall_at_099,
    "threshold_at_precision_0.99": threshold_at_099,
    "n_test": int(y_test.shape[0]),
    "pos_rate_test": float(y_test.mean()),
}
(RPT_DIR / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

# Plot PR curve
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (RF Enhanced)")
plt.grid(True)
plt.savefig(FIG_DIR / "pr_curve_rf_enhanced.png")
plt.close()

# Confusion matrix plot
import seaborn as sns
plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (RF Enhanced)")
plt.savefig(FIG_DIR / "confusion_matrix_rf_enhanced.png")
plt.close()

# ------------------------------
# 8. Feature importances (top 20)
# ------------------------------
# Get feature names after preprocessing
ohe = pipe.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
num_names = num_cols
cat_names = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
all_names = num_names + cat_names

importances = pipe.named_steps["rf"].feature_importances_

# Safeguard different lengths
k = min(len(importances), len(all_names))
imp = pd.DataFrame({
    "feature": all_names[:k],
    "importance": importances[:k],
}).sort_values("importance", ascending=False).head(20)

imp.to_csv(RPT_DIR / "feature_importance_top20.csv", index=False)

plt.figure(figsize=(8, 6))
plt.barh(imp["feature"][::-1], imp["importance"][::-1])
plt.xlabel("Importance")
plt.title("Top 20 Features — RF Enhanced")
plt.tight_layout()
plt.savefig(FIG_DIR / "top20_features_rf_enhanced.png")
plt.close()

# ------------------------------
# 9. Save model & operating point
# ------------------------------
import joblib
model_path = MODEL_DIR / "random_forest_enhanced.pkl"
joblib.dump(pipe, model_path)
op_path = RPT_DIR / "operating_point_0.99.json"
op_path.write_text(json.dumps({
    "threshold": threshold_at_099,
    "recall_at_precision_0.99": recall_at_099
}, indent=2))

print(f"[SAVED] Model -> {model_path}")
print(f"[SAVED] Reports -> {RPT_DIR}")
print(f"[SAVED] Figures -> {FIG_DIR}")
print(f"[SAVED] Operating point (0.99) -> {op_path}")
