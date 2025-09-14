import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------
# 1. Definir rutas y carpetas
# -------------------------
raw_path = "PlanetHunter-AI/data/raw/"
data_science_path = "PlanetHunter-AI/data_science"
models_path = os.path.join(data_science_path, "models")
figures_path = os.path.join(data_science_path, "figures")
reports_path = os.path.join(data_science_path, "reports")

# Crear carpetas si no existen
for path in [data_science_path, models_path, figures_path, reports_path]:
    os.makedirs(path, exist_ok=True)

# -------------------------
# 2. Cargar CSV más reciente
# -------------------------
csv_files = glob.glob(os.path.join(raw_path, "cumulative_koi_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No se encontraron archivos CSV en {raw_path}")
latest_csv = max(csv_files, key=os.path.getctime)
print(f"Archivo más reciente: {latest_csv}")

df = pd.read_csv(latest_csv)
print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# -------------------------
# 3. Filtrar y preparar target
# -------------------------
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
df['target'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

# -------------------------
# 4. Seleccionar features numéricas
# -------------------------
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
drop_cols = ['kepid', 'rowid', 'target']
numeric_features = [col for col in numeric_features if col not in drop_cols]

# Filtrar columnas con al menos 50% de datos no nulos
numeric_features = [col for col in numeric_features if df[col].notna().mean() > 0.5]
print(f"Columnas numéricas seleccionadas: {len(numeric_features)}")

X = df[numeric_features].fillna(0)
y = df['target']

# -------------------------
# 5. Split train/test
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 6. Entrenar Random Forest
# -------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# -------------------------
# 7. Evaluación y reporte
# -------------------------
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, digits=3)
accuracy = accuracy_score(y_test, y_pred)
print("\nReporte de clasificación:\n", report)
print(f"Accuracy: {accuracy:.3f}")

# Guardar reporte
report_file = os.path.join(reports_path, "classification_report.txt")
with open(report_file, "w") as f:
    f.write(report + f"\nAccuracy: {accuracy:.3f}\n")
print(f"✅ Reporte guardado en {report_file}")

# -------------------------
# 8. Importancia de features
# -------------------------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top15 = importances.head(15)
print("\nTop 15 features más importantes:\n", top15)

# Gráfico top 15
plt.figure(figsize=(10,6))
top15.plot(kind="barh")
plt.title("Top 15 Features (Random Forest)")
plt.gca().invert_yaxis()
fig_file = os.path.join(figures_path, "top15_features.png")
plt.savefig(fig_file)
plt.close()
print(f"✅ Gráfico de importancia guardado en {fig_file}")

# -------------------------
# 9. Eliminar features altamente correlacionadas
# -------------------------
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
X_filtered = X.drop(columns=to_drop)
print(f"Eliminadas {len(to_drop)} features por correlación alta.")

# -------------------------
# 10. Selección de features con RFECV
# -------------------------
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy', n_jobs=-1)
rfecv.fit(X_filtered, y)

selected_features = X_filtered.columns[rfecv.support_]
print("Features seleccionadas por RFECV:", selected_features.tolist())

# Guardar features seleccionadas
features_file = os.path.join(data_science_path, "selected_features.csv")
pd.DataFrame(selected_features, columns=["feature"]).to_csv(features_file, index=False)
print(f"✅ Features seleccionadas guardadas en {features_file}")

# Gráfico performance RFECV
plt.figure(figsize=(12,6))
scores = rfecv.cv_results_['mean_test_score']
plt.plot(range(1, len(scores)+1), scores, marker='o')
plt.xlabel("Número de características seleccionadas")
plt.ylabel("Accuracy en CV")
plt.title("Optimización de features con RFECV")
fig_rfecv = os.path.join(figures_path, "rfecv_performance.png")
plt.savefig(fig_rfecv)
plt.close()
print(f"✅ Gráfico RFECV guardado en {fig_rfecv}")

# -------------------------
# 11. Guardar modelo final
# -------------------------
model_file = os.path.join(models_path, "random_forest_model.pkl")
joblib.dump(rf, model_file)
print(f"✅ Modelo guardado en {model_file}")
