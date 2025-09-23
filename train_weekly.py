import os
import requests
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# ============================
# 1. Descargar dataset NASA
# ============================
DATA_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
DATA_FILE = "exoplanets.csv"

print("Descargando dataset de la NASA...")
response = requests.get(DATA_URL)
with open(DATA_FILE, "wb") as f:
    f.write(response.content)
print(f"✅ Dataset descargado: {DATA_FILE}")

# ============================
# 2. Cargar y preprocesar datos
# ============================
df = pd.read_csv(DATA_FILE)

# Convertir etiquetas: Confirmado (1) vs Falso Positivo (0)
if "koi_disposition" in df.columns:
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    df["target"] = df["koi_disposition"].map({"CONFIRMED": 1, "FALSE POSITIVE": 0})

# Selección de features
selected_features = [
    "koi_count", "koi_prad", "koi_prad_err1",
    "koi_steff_err1", "koi_steff_err2",
    "koi_smet_err1", "koi_smet_err2",
    "koi_fwm_stat_sig", "koi_dicco_msky",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co",
    "koi_score"
]

# Filtrar solo columnas que existan en el dataset
features = [col for col in selected_features if col in df.columns]

# Imputación con mediana
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)
y = df["target"]

# ============================
# 3. Entrenar modelo
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("✅ Modelo entrenado.")

# ============================
# 4. Guardar modelo
# ============================
MODEL_FILE = "random_forest_exoplanets.pkl"
joblib.dump(rf, MODEL_FILE)
print(f"✅ Modelo guardado en {MODEL_FILE}")
