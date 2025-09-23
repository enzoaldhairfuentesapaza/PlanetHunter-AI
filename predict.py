import argparse
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

# ============================
# 1. Argumentos de entrada
# ============================
parser = argparse.ArgumentParser(description="Predecir exoplanetas con modelo RandomForest")
parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
parser.add_argument("--output", required=True, help="Ruta al CSV de salida")
parser.add_argument("--model", default="models/random_forest_exoplanets.pkl", help="Ruta al modelo entrenado")
args = parser.parse_args()

# ============================
# 2. Cargar modelo
# ============================
print("Cargando modelo...")
model = joblib.load(args.model)

# ============================
# 3. Features esperadas
# ============================
selected_features = [
    "koi_count", "koi_prad", "koi_prad_err1",
    "koi_steff_err1", "koi_steff_err2",
    "koi_smet_err1", "koi_smet_err2",
    "koi_fwm_stat_sig", "koi_dicco_msky",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co",
    "koi_score"
]

# ============================
# 4. Cargar y limpiar datos nuevos
# ============================
print(f"Leyendo datos de {args.input}...")
df_new = pd.read_csv(args.input)

# Mantener solo las 13 columnas seleccionadas
df_new = df_new[selected_features]

# Imputar valores faltantes con mediana
imputer = SimpleImputer(strategy="median")
X_new = pd.DataFrame(imputer.fit_transform(df_new), columns=selected_features)

# ============================
# 5. Predicción
# ============================
print("Generando predicciones...")
predictions = model.predict(X_new)

# Crear dataframe final con solo features + predicción
df_output = X_new.copy()
df_output["prediction"] = predictions

# ============================
# 6. Guardar salida
# ============================
df_output.to_csv(args.output, index=False)
print(f"✅ Resultados guardados en {args.output}")
