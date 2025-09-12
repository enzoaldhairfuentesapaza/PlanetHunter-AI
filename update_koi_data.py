import os
import pandas as pd
from datetime import datetime
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

# ================================
# 1. Configuración de carpetas
# ================================
RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
META_FILE = "data/metadata/last_update.txt"

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

# ================================
# 2. Función para obtener última fecha guardada
# ================================
def get_last_update():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            return f.read().strip()
    return None

# ================================
# 3. Descargar datos desde NASA
# ================================
def download_latest():
    print("🔎 Descargando datos más recientes de NASA Exoplanet Archive...")
    koi = NasaExoplanetArchive.query_criteria(table="cumulative")
    df = koi.to_pandas()
    print(f"✅ Descargado: {len(df)} registros, {len(df.columns)} columnas.")
    return df

# ================================
# 4. Guardar datos en disco
# ================================
def save_data(df):
    today = datetime.today().strftime("%Y-%m-%d")

    raw_file = os.path.join(RAW_PATH, f"cumulative_koi_{today}.csv")
    processed_file = os.path.join(PROCESSED_PATH, f"cumulative_koi_{today}.parquet")

    df.to_csv(raw_file, index=False)
    df.to_parquet(processed_file, index=False)

    with open(META_FILE, "w") as f:
        f.write(today)

    print(f"💾 Guardado en:\n - {raw_file}\n - {processed_file}")
    print(f"📅 Fecha registrada: {today}")

# ================================
# 5. Pipeline de actualización
# ================================
def update_pipeline():
    last_update = get_last_update()
    today = datetime.today().strftime("%Y-%m-%d")

    if last_update == today:
        print(f"⚠️ Ya existe una versión descargada hoy ({today}). No se descarga de nuevo.")
    else:
        df = download_latest()
        save_data(df)

# ================================
# 6. Ejecutar
# ================================
if __name__ == "__main__":
    update_pipeline()
