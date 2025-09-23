# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="PlanetHunter-AI", layout="wide")

# ======================
# Header
# ======================
st.title("🌌 PlanetHunter-AI")
st.markdown("Un sistema de IA para clasificar exoplanetas con datos de la NASA")
st.markdown("**Última evaluación:** 22 de Septiembre, 2025")

# ======================
# Comparación de modelos
# ======================
st.header("📊 Comparación de Modelos")

metrics = pd.DataFrame({
    "Modelo": ["RandomForest", "XGBoost", "TabNet"],
    "Accuracy": [0.991, 0.993, 0.992],
    "Precision": [0.990, 0.991, 0.990],
    "Recall": [0.980, 0.990, 0.987],
    "F1": [0.985, 0.990, 0.988],
    "AUC": [0.9997, 0.9997, 0.9996]
})

st.dataframe(metrics.set_index("Modelo"))

# Gráfico de barras de AUC
st.subheader("AUC por Modelo")
fig, ax = plt.subplots()
ax.bar(metrics["Modelo"], metrics["AUC"], color=["blue", "orange", "green"])
ax.set_ylabel("AUC")
ax.set_ylim(0.99, 1.0)
st.pyplot(fig)

# ======================
# Importancia de features (ejemplo estático)
# ======================
st.header("🔎 Importancia de Características (RandomForest)")

feature_importance = pd.Series({
    "koi_score": 0.46,
    "koi_smet_err2": 0.11,
    "koi_dicco_msky": 0.09,
    "koi_prad": 0.08,
    "koi_fpflag_nt": 0.07,
    "koi_fpflag_co": 0.05,
    "koi_fpflag_ss": 0.04,
    "koi_smet_err1": 0.03,
    "koi_steff_err1": 0.02,
    "koi_steff_err2": 0.02,
    "koi_fwm_stat_sig": 0.02,
    "koi_prad_err1": 0.01,
    "koi_count": 0.01
})

fig2, ax2 = plt.subplots(figsize=(8,6))
feature_importance.sort_values().plot.barh(ax=ax2, color="purple")
ax2.set_title("Importancia de Features en RandomForest")
st.pyplot(fig2)
