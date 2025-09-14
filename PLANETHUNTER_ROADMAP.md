# PlanetHunter-AI — Roadmap hacia un Multimodal Transformer (NASA Challenge)

> Objetivo: pasar del baseline RandomForest a un **Multimodal Transformer** que combine **tablas (KOI/TCE)** + **curvas de luz** y desplegar un **dashboard estilo Mission Control**.

## Fase 0 — Endurecer baseline (1–2 días)
- [ ] Fijar semillas (`random_state=42`) en *train/test split*, `RandomForestClassifier`, y cualquier CV.
- [ ] Añadir `class_weight="balanced"` al RF y reporte de **matriz de confusión** + **PR-AUC**.
- [ ] Guardar **umbral de decisión** para operar a **99% de precisión** y reportar *recall* a ese punto.
- [ ] Registrar versiones de datos y artefactos (MLflow/W&B o CSV en `data_science/reports/`).
- [ ] Añadir tests básicos con `pytest` (lectura KOI, split estratificado, shape de features).

## Fase 1 — Datos y features (2–4 días)
- [ ] **Tablas (KOI/TCE):** normalizar, imputar NaNs, *label encode* de categorías, *feature store* en Parquet.
- [ ] **Curvas de luz:** descargar de Kepler/TESS; *detrending* (Savitzky–Golay, co-trending basis vectors), *folding* por periodo si aplica, *windowing* en eclipses.
- [ ] **Split sin fuga:** estratificar por etiqueta y **bloquear por estrella** (no mezclar el mismo objeto entre train/test).
- [ ] **Datasets balanceados:** *undersampling* controlado o *class weights*; opcional *focal loss* en DL.

## Fase 2 — Unimodales (5–7 días)
- **TabTransformer (tabular)**
  - [ ] Encoder de columnas categóricas → **embeddings** + bloques Transformer; numéricas → *MLP/LayerNorm* y fusión.
  - [ ] Head MLP binaria con *BCEWithLogitsLoss* y *class weights*.
- **Time Series Transformer (curvas de luz)**
  - [ ] Tokens = muestras temporales; *positional encodings* + *padding mask* para longitudes variables.
  - [ ] Augmentations: *jittering*, *random cropping*, *time warping* leves.

> Entregar: dos modelos con **PR-AUC** y curvas precision–recall; *recall @ 99% precision*.

## Fase 3 — Multimodal (7–10 días)
Estrategias de fusión (empezar simple → complejo):
1. **Late fusion:** combinar *logits* (promedio ponderado o *stacking* con un metaclassifier).
2. **Mid fusion (concat):** concatenar *[CLS]* del TS-Transformer + *[CLS]* del TabTransformer → head MLP.
3. **Cross-attention:** usar *cross-modal attention* (el vector tabular consulta a la serie o viceversa).

- [ ] Entrenar las tres y comparar; elegir por **recall @ 99% precision** y **PR-AUC**.
- [ ] Calibración de probabilidades (Platt/Isotonic).

## Fase 4 — Evaluación científica (3–5 días)
- [ ] **Injection–recovery tests**: inyectar tránsitos sintéticos de diferente SNR, periodo y duración; medir tasa de recuperación.
- [ ] **Generalización entre misiones:** entrenar en Kepler, validar en TESS (2m y/o 30m).
- [ ] **Errores típicos:** reporte de *top FN* y *top FP* con *light curve snippets*.
- [ ] **Interpretabilidad:**
  - Tabular: **SHAP** global/local, *permutation importance*.
  - Series: **Integrated Gradients** / *attention rollout* sobre ventanas de tránsito.

## Fase 5 — Dashboard estilo Mission Control (2–4 días)
**Stack sugerido:** Streamlit o Dash + Plotly.
- [ ] Panel de **KPIs**: Accuracy, Precision, Recall, F1, **PR-AUC**, y **recall @ P=0.99**.
- [ ] **Curvas de luz** con overlay del tránsito y zonas de mayor atención/IG.
- [ ] Tabla de candidatos con filtros (SNR, periodo, probabilidad, flags).
- [ ] Vista de **caso**: features tabulares + explicación (SHAP) + señal cruda/procesada.
- [ ] **Animaciones**: 
  - Simulación de “señal de luz” pasando por el pipeline (frames/Plotly frames).
  - Evolución del *training* (loss/PR-AUC vs epoch).

## Fase 6 — MLOps / CI-CD (2–3 días)
- [ ] **DVC** o Git LFS para datos; *make targets* o `tox` para jobs (`make update-data`, `make train`, `make eval`).
- [ ] **GitHub Actions**: jobs separados para datos, train, tests y lint.
- [ ] Versionado de modelos (`data_science/models/`) + *model registry* simple (CSV + hash + fecha + métricas).
- [ ] Semillas y `torch.backends.cudnn.deterministic=True` cuando aplique.

---

## Esqueleto de Modelo Multimodal (PyTorch)

```python
class TabEncoder(nn.Module):
    def __init__(self, emb_info, d_model=128, n_layers=2, n_heads=4):
        super().__init__()
        # emb_info: dict {col: (cardinality, emb_dim)}
        self.embs = nn.ModuleDict({c: nn.Embedding(card, dim) for c,(card,dim) in emb_info.items()})
        self.num_proj = nn.Sequential(nn.Linear(num_num_feats, d_model), nn.LayerNorm(d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.cls = nn.Parameter(torch.randn(1,1,d_model))
    def forward(self, x_cat, x_num):
        tokens = [emb(x_cat[c]) for c,emb in self.embs.items()]  # [B, Lc, d]
        num = self.num_proj(x_num).unsqueeze(1)                  # [B, 1, d]
        seq = torch.cat(tokens + [num], dim=1)                   # [B, Lt, d]
        cls = self.cls.expand(len(x_num), -1, -1)
        seq = torch.cat([cls, seq], dim=1)
        h = self.encoder(seq)[:,0]  # [CLS]
        return h

class TSencoder(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=8):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
    def forward(self, x_ts, mask=None):          # x_ts: [B, T, 1]
        z = self.pos(self.proj(x_ts))
        h = self.encoder(z, src_key_padding_mask=mask)[:,0]  # usar primer token como [CLS]
        return h

class MultiModal(nn.Module):
    def __init__(self, tab_enc, ts_enc, d_model=128):
        super().__init__()
        self.tab = tab_enc
        self.ts  = ts_enc
        self.head = nn.Sequential(nn.Linear(2*d_model, d_model),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(d_model, 1))
    def forward(self, x_cat, x_num, x_ts, mask=None):
        h_tab = self.tab(x_cat, x_num)
        h_ts  = self.ts(x_ts, mask)
        h = torch.cat([h_tab, h_ts], dim=-1)
        return self.head(h).squeeze(-1)
```

> Entrenar con `BCEWithLogitsLoss(pos_weight=...)`, *scheduler* tipo OneCycle o Cosine, y evaluación en cada epoch con **PR-AUC** + **recall @ P=0.99**.

---

## Métricas obligatorias a reportar
- Accuracy, Precision, Recall, F1 (macro/weighted).
- **PR-AUC** y **ROC-AUC**.
- **Recall @ 99% Precision** (operating point para minimizar FP).
- *Confusion matrix* y *Top FP/FN* con ejemplos.

---

## Próximos pasos inmediatos (sugeridos)
- [ ] Añadir `random_state` y `class_weight` al RF + matriz de confusión y PR-AUC.
- [ ] Definir `feature_spec.json` (categóricas con cardinalidad, numéricas con normalización).
- [ ] Prototipar el **TabTransformer** en un notebook y guardar baseline (artefacto + métricas).
- [ ] Prototipar el **Time Series Transformer** con *folding* sobre tránsitos.
- [ ] Implementar **late fusion** y comparar.
- [ ] Bosquejar **Dashboard (Streamlit)** con KPIs y vista de caso.

¡Éxitos! 🚀
