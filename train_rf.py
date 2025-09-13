import pandas as pd
import numpy as np
import os, glob, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ===========================
# 1) Cargar dataset más reciente
# ===========================
raw_path = "data/raw/"
csv_files = glob.glob(os.path.join(raw_path, "cumulative_koi_*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No se encontraron archivos CSV en {raw_path}")
latest_csv = max(csv_files, key=os.path.getctime)
print(f"📂 Usando dataset: {latest_csv}")

df = pd.read_csv(latest_csv)
print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ===========================
# 2) Preprocesamiento básico
# ===========================
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
df['target'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [col for col in numeric_features if col not in ['kepid', 'rowid', 'target']]

# Solo columnas con >50% datos válidos
numeric_features = [col for col in numeric_features if df[col].notna().mean() > 0.5]

X = df[numeric_features].fillna(0)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# 3) Entrenar Random Forest
# ===========================
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ===========================
# 4) Guardar métricas
# ===========================
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

os.makedirs("results/metrics", exist_ok=True)
with open("results/metrics/rf_metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "report": report}, f, indent=4)

os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/feature_importance.png", bbox_inches="tight")

with open("results/metrics/rf_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))
    
print(f"📊 Accuracy: {accuracy:.4f}")
print("📁 Métricas guardadas en reports/metrics/rf_metrics.json")

# ===========================
# 5) Guardar gráfico de importancia de features
# ===========================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]  # top 20


plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(numeric_features)[indices])
plt.title("Top 20 Features - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Feature")
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/feature_importance.png", bbox_inches="tight")
plt.close()

print("📈 Gráfico guardado en reports/figures/feature_importance.png")

# ===========================
# 6) Guardar modelo
# ===========================
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/random_forest.pkl")
print("🤖 Modelo guardado en models/random_forest.pkl")
