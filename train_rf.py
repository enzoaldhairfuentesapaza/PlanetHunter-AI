import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Cargar dataset procesado
df = pd.read_parquet("data/processed/cumulative_koi_2025-09-13.parquet")

# 2. Preparar features y target
df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
df['target'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

X = df.drop(columns=['kepid', 'rowid', 'koi_disposition', 'target'], errors="ignore")
y = df['target']

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Entrenar modelo
rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 5. Evaluación
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Guardar modelo entrenado
joblib.dump(rf, "data/metadata/random_forest_model.pkl")
print("✅ Modelo guardado en data/metadata/random_forest_model.pkl")
