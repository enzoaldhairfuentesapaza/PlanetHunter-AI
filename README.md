# PlanetHunter-AI
ExoFormer es un proyecto de inteligencia artificial aplicado a la astronomía, cuyo objetivo es detectar y clasificar exoplanetas a partir de datos públicos de la NASA (Kepler y TESS) utilizando modelos avanzados como Transformers especializados en datos tabulares y series temporales.

# 🌌 ExoFormer: Descubriendo exoplanetas con IA

ExoFormer es un proyecto de investigación aplicada que combina astronomía e inteligencia artificial para detectar y clasificar exoplanetas a partir de datos públicos de la NASA, como las tablas de objetos de interés de Kepler (KOI), candidatos de TESS y curvas de luz de estrellas.

El objetivo principal es construir un sistema end-to-end capaz de:

Procesar y limpiar grandes volúmenes de datos astronómicos.

Entrenar modelos de aprendizaje automático y profundo (desde Random Forest hasta Transformers especializados en datos tabulares y series temporales).

Diferenciar entre planetas confirmados, candidatos y falsos positivos.

Evaluar el rendimiento científico con métricas robustas (PR-AUC, sensibilidad, tasa de falsos positivos).

Integrar interpretabilidad (atención, SHAP, saliency maps) para que los resultados sean útiles a la comunidad científica.

# 🚀 Características clave

📡 Datos abiertos de la NASA: KOI, TCE y curvas de luz de Kepler/TESS.

🧠 Modelos de IA avanzados: TabTransformer, Time-Series Transformer, CNNs 1D y ensembles multimodales.

📊 Pipeline reproducible: ETL, entrenamiento, validación y comparación de modelos.

🔍 Enfoque científico: validación con injection-recovery tests y métricas usadas en la astrofísica.

🌍 Visión futura: despliegue como aplicación web para que la comunidad pueda explorar candidatos y contribuir al proceso de verificación.

# 📑 Plan de trabajo

ETL y limpieza de datos (KOI, TCE, curvas de luz).

Análisis exploratorio y baseline (XGBoost, Random Forest).

Modelado avanzado con arquitecturas Transformer adaptadas a datos tabulares y series temporales.

Evaluación científica con métricas robustas y pruebas de generalización entre misiones.

Interpretabilidad y visualización de resultados.

Despliegue como API y plataforma web colaborativa.

# 🤝 Contribuciones

Este proyecto está abierto a la comunidad.
Si te interesa la astronomía, el machine learning o simplemente explorar el cosmos con datos reales, ¡eres bienvenido a contribuir!
