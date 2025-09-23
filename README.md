# PlanetHunter-AI

### 🌌 Descripción del proyecto
PlanetHunter-AI es un sistema de aprendizaje automático diseñado para **clasificar exoplanetas** usando datos públicos de la NASA (Kepler, K2 y TESS).  
El modelo principal está basado en **RandomForest**, entrenado únicamente con las **13 características más relevantes**, lo que le permite lograr un alto rendimiento con menor complejidad y mejor interpretabilidad.

Además, el proyecto incluye un plan para desarrollar una **interfaz web interactiva**, que permitirá a investigadores y usuarios visualizar cómo el modelo analiza los datos, mostrar estadísticas de precisión y facilitar la exploración de los exoplanetas descubiertos.

---

### 🎯 Objetivos
1. **Actualizar semanalmente los datos** de exoplanetas desde los repositorios de la NASA.  
2. **Entrenar automáticamente el modelo** RandomForest con las 13 características seleccionadas.  
3. **Generar predicciones** que clasifiquen cada objeto como:  
   - Confirmado  
   - Candidato planetario  
   - Falso positivo  
4. **Proveer una interfaz web** donde los usuarios puedan:  
   - Explorar los exoplanetas y sus características.  
   - Ver cómo evoluciona el modelo con cada actualización.  
   - Consultar estadísticas de precisión del modelo.  

---

### ⚙️ Flujo del sistema
1. **Descarga de datos:** un script en Python obtiene semanalmente la última versión de la base de datos de exoplanetas de la NASA.  
2. **Preprocesamiento:** limpieza de datos, imputación de valores faltantes con la **mediana** y selección de las 13 características principales.  
3. **Entrenamiento:** el modelo RandomForest se reentrena automáticamente con los datos actualizados.  
4. **Almacenamiento:** se guarda el modelo entrenado (`.pkl`) para ser usado en predicciones.  
5. **Interfaz web:** permite visualizar exoplanetas, métricas del modelo y resultados de clasificación.  

---

### 📊 Características seleccionadas para el modelo
- `koi_count`  
- `koi_prad`  
- `koi_prad_err1`  
- `koi_steff_err1`  
- `koi_steff_err2`  
- `koi_smet_err1`  
- `koi_smet_err2`  
- `koi_fwm_stat_sig`  
- `koi_dicco_msky`  
- `koi_fpflag_nt`  
- `koi_fpflag_ss`  
- `koi_fpflag_co`  
- `koi_score`  

---

### 🚀 Próximos pasos
- Implementar el script de **actualización automática semanal**.  
- Diseñar la **API/backend** para exponer predicciones.  
- Crear la **interfaz web** (Flask/Django/Streamlit) para visualizar resultados y métricas.  
- (Opcional) permitir que el usuario suba sus propios datos para probar el modelo.  
