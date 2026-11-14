"""
verificar_metricas.py — Verifica las métricas de la tesis SIN reentrenar.

Carga el dataset analítico publicado (data/dataset_tesis_clean.parquet) y los
modelos exactos de la tesis (models/*.cbm), predice sobre el conjunto de prueba
2022 y compara contra los valores reportados.

Uso (parado en la raíz del repo, ~1 minuto en CPU):
    pip install -r requirements.txt
    python verificar_metricas.py

Qué verifica:
  1. Modelo final CatBoost v3 (target: litigio evitable) — tesis §4.10.2 y §5.2:
     AUC-ROC, PR-AUC, KS y la matriz de confusión con τ = 0,7084.
  2. El mismo modelo con el umbral elegido en validación 2021 (τ = 0,6903, §5.3).
  3. Concentración por decil (§5.6, Tabla 38).
  4. Modelo optimizado sobre el target original (§4.15): AUC-ROC y PR-AUC.

Nota sobre tipos de datos: los modelos se entrenaron leyendo el CSV
(raw_data/dataset_tesis_clean.csv). Al pasar por CSV, los códigos WCIO e
industry_code se leen como números ("9.0") y no como texto ("09"). Para que las
categorías coincidan con las del entrenamiento, este script reproduce en memoria
ese mismo paso Parquet -> CSV -> pandas.read_csv antes de predecir.
"""
import hashlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "data", "dataset_tesis_clean.parquet")
M_V3    = os.path.join(BASE, "models", "catboost_v3_full.cbm")
M_ORIG  = os.path.join(BASE, "models", "best_catboost_target_original.cbm")

SHA256 = {
    M_V3:   "6cf055c1b061cafd5e275659db6c41eb098420d9f864467d4b4294bd5e24fe9e",
    M_ORIG: "90dd51d10ab07cc7f119e6e134c8fedd725597ee5f588cac484b54c599e043a9",
}

AVOIDABLE = {"2. NON-COMP", "3. MED ONLY", "4. TEMPORARY"}
CAT_FEATURES = [
    "gender", "accident_type", "occupational_disease",
    "county_of_injury", "medical_fee_region",
    "wcio_cause_code", "wcio_nature_code", "wcio_body_code",
    "carrier_type", "district_name", "industry_code", "industry_desc",
]
NUM_FEATURES = [
    "days_to_assembly", "days_C2_to_accident", "days_C3_to_accident",
    "age_at_injury", "aww",
    "has_C2", "has_C3", "has_ANCR_early",
    "accident_year", "accident_month", "accident_dow",
]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
# Los umbrales salen de la grilla de best_f1_threshold (08_retrain_v3_gpu.py):
# 150 valores entre 0,05 y 0,95. La tesis los informa redondeados (0,7084 y 0,6903).
_GRID = np.linspace(0.05, 0.95, 150)
TAU_TEST = float(_GRID[np.argmin(np.abs(_GRID - 0.7084))])   # 0,70839…
TAU_VAL  = float(_GRID[np.argmin(np.abs(_GRID - 0.6903))])   # 0,69027…

checks = []   # (descripción, obtenido, esperado, ok)

def check(desc, got, expected, tol=0.0):
    ok = abs(got - expected) <= tol
    checks.append((desc, got, expected, ok))

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def confusion(y, s, tau):
    pred = s >= tau
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return tp, fp, fn, tn

def ks(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))

# ── 0. Archivos ──────────────────────────────────────────────────────────────
for path in (DATA, M_V3, M_ORIG):
    if not os.path.exists(path):
        sys.exit(f"No encuentro {path}. ¿Clonaste el repo completo?")
for path, expected in SHA256.items():
    got = sha256(path)
    if got != expected:
        sys.exit(f"El modelo {os.path.basename(path)} no es el de la tesis (SHA-256 {got[:12]}…, esperado {expected[:12]}…).")
print("Modelos: SHA-256 verificado.")

# ── 1. Datos: validación 2021 y prueba 2022, con los tipos que ve read_csv ───
tabla = pq.read_table(DATA, filters=[("accident_year", ">=", 2021)])
buf = io.StringIO()
tabla.to_pandas().to_csv(buf, index=False)
buf.seek(0)
raw = pd.read_csv(buf, low_memory=False)
test_raw = raw[raw["accident_year"] == 2022]
print(f"Prueba 2022: {len(test_raw):,} reclamos")
check("Reclamos en prueba 2022", len(test_raw), 260_156)

# ── 2. Modelo final v3 — litigio evitable ────────────────────────────────────
df = raw.copy()
df["y"] = (df["target"].eq(1) & df["claim_injury_type_REF"].isin(AVOIDABLE)).astype(int)
for c in CAT_FEATURES:
    df[c] = df[c].fillna("UNKNOWN").astype(str)
for c in NUM_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1)
test = df[df["accident_year"] == 2022]
y = test["y"].values

model = CatBoostClassifier(); model.load_model(M_V3)
cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]
p = model.predict_proba(Pool(test[ALL_FEATURES], cat_features=cat_idx))[:, 1]

check("Litigios evitables en prueba 2022", int(y.sum()), 41_148)
check("v3 · AUC-ROC", round(roc_auc_score(y, p), 4), 0.8833, 5e-5)
check("v3 · PR-AUC", round(average_precision_score(y, p), 4), 0.5981, 5e-5)
check("v3 · KS", round(ks(y, p), 4), 0.6202, 5e-5)
tp, fp, fn, tn = confusion(y, p, TAU_TEST)
for name, got, exp in (("TP", tp, 27_903), ("FP", fp, 25_868), ("FN", fn, 13_245), ("TN", tn, 193_140)):
    check(f"v3 · τ=0,7084 · {name}", got, exp)
check("v3 · τ=0,7084 · recall", round(tp / (tp + fn), 3), 0.678, 5e-4)
check("v3 · τ=0,7084 · precisión", round(tp / (tp + fp), 3), 0.519, 5e-4)

# Umbral fijado en validación 2021 (máximo F1 en 2021), aplicado a prueba 2022
tp2, fp2, fn2, tn2 = confusion(y, p, TAU_VAL)
check("v3 · τ=0,6903 (validación) · recall", round(tp2 / (tp2 + fn2), 3), 0.701, 5e-4)
check("v3 · τ=0,6903 (validación) · precisión", round(tp2 / (tp2 + fp2), 3), 0.506, 5e-4)

# Concentración por decil (Tabla 38)
orden = np.argsort(-p, kind="stable")
decil = np.arange(len(p)) * 10 // len(p)
pos_decil = np.bincount(decil, weights=y[orden], minlength=10)
acum = np.cumsum(pos_decil) / y.sum()
check("v3 · captura del primer decil (%)", round(100 * acum[0], 1), 41.7, 0.05)
check("v3 · captura de los primeros cuatro deciles (%)", round(100 * acum[3], 1), 90.1, 0.05)

# ── 3. Modelo optimizado sobre el target original ────────────────────────────
do = test_raw.copy()
for c in CAT_FEATURES:
    do[c] = do[c].astype(object).where(do[c].notna(), "MISSING").astype(str)
for c in NUM_FEATURES:
    do[c] = pd.to_numeric(do[c], errors="coerce").fillna(-1)
y0 = do["target"].values
m0 = CatBoostClassifier(); m0.load_model(M_ORIG)
p0 = m0.predict_proba(Pool(do[ALL_FEATURES], cat_features=cat_idx))[:, 1]
check("Judicializados (target original) en prueba 2022", int(y0.sum()), 55_383)
check("Target original · AUC-ROC", round(roc_auc_score(y0, p0), 4), 0.9121, 5e-5)
check("Target original · PR-AUC", round(average_precision_score(y0, p0), 4), 0.7408, 5e-5)

# ── Resumen ──────────────────────────────────────────────────────────────────
print()
print(f"{'Verificación':<50} {'Obtenido':>12} {'Tesis':>12}")
print("-" * 78)
for desc, got, exp, ok in checks:
    fmt = (lambda v: f"{v:,.4f}" if isinstance(v, float) else f"{v:,}")
    print(f"{desc:<50} {fmt(got):>12} {fmt(exp):>12}  {'OK' if ok else 'DIFIERE'}")
fallas = [c for c in checks if not c[3]]
print("-" * 78)
if fallas:
    sys.exit(f"{len(fallas)} de {len(checks)} verificaciones difieren de la tesis.")
print(f"Las {len(checks)} verificaciones coinciden con la tesis.")
