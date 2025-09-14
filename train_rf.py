import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

# ------------------------------
# 1. Rutas y carpetas
# ------------------------------
raw_path = "data/raw/"
csv_files = glob.glob(os.path.join(raw_path, "cumulative_koi_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No se encontraron archivos CSV en {raw_path}")

latest_csv = max(csv_files, key=os.path.getctime)
print(f"Archivo más reciente: {latest_csv}")

# Carpetas de salida
os.makedirs("data_science/models", exist_ok=True)
os.makedirs("data_science/figures", exist_ok=True)
os.makedirs("data_science/reports", exist_ok=True)

# Rutas de archivos
model_path = "data_science/models/random_forest_model.pkl"
report_path = "data_science/reports/classification_report.txt"
features_path = "data_science/reports/selected_features.csv"
version_txt = "data_science/reports/version.txt"

# ------------------------------
# 2. Cargar y preparar datos
# ------------------------------
df = pd.read_csv(latest_csv)
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
df['target'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

# Seleccionar features numéricas y eliminar identificadores
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
drop_cols = ['kepid', 'rowid', 'target']
numeric_features = [col for col in numeric_features if col not in drop_cols]
numeric_features = [col for col in numeric_features if df[col].notna().mean() > 0.5]

X = df[numeric_features].fillna(0)
y = df['target']

# ------------------------------
# 3. Train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 4. Entrenar Random Forest
# ------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ------------------------------
# 5. Evaluación
# ------------------------------
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, digits=3)
accuracy = accuracy_score(y_test, y_pred)
print(report)
print(f"Accuracy: {accuracy:.3f}")

# Guardar reporte
with open(report_path, "w") as f:
    f.write(report + f"\nAccuracy: {accuracy:.3f}\n")

# ------------------------------
# 6. Importancia de features
# ------------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top15 = importances.head(15)

plt.figure(figsize=(10,6))
top15.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 features más importantes")
plt.savefig("data_science/figures/top15_features.png")
plt.close()

# ------------------------------
# 7. Eliminación de correlación alta
# ------------------------------
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
X_filtered = X.drop(columns=to_drop)
print(f"Eliminadas {len(to_drop)} features por correlación alta: {to_drop}")

# ------------------------------
# 8. RFECV
# ------------------------------
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy', n_jobs=-1)
rfecv.fit(X_filtered, y)

selected_features = X_filtered.columns[rfecv.support_]
print(f"Mejor número de características: {rfecv.n_features_}")
print("Características seleccionadas:", selected_features.tolist())

# Guardar features seleccionadas
selected_features.to_series().to_csv(features_path, index=False)

# Gráfico de RFECV
plt.figure(figsize=(12,6))
scores = rfecv.cv_results_['mean_test_score']
plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.xlabel("Número de características seleccionadas")
plt.ylabel("Accuracy en CV")
plt.title("Optimización de features con RFECV")
plt.savefig("data_science/figures/rfecv_scores.png")
plt.close()

# ------------------------------
# 9. Guardar modelo
# ------------------------------
joblib.dump(rf, model_path)
print(f"Modelo guardado en {model_path}")

# ------------------------------
# 10. Guardar versión con fecha
# ------------------------------
with open(version_txt, "w") as f:
    f.write(f"Random Forest versión: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
