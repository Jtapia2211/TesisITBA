"""
milliman_comparison.py
─────────────────────
Replica el análisis de concentración de Milliman Nodal (2018) sobre el
modelo CatBoost de esta tesis en test-2022.

Milliman Nodal:
  - Targets top 10% of claims
  - That 10% historically represents 87% of ultimate claim payments
  - Identified 19/20 largest claims within 30 days of reporting

Nuestro análogo:
  - Top 10% por probabilidad predicha de litigación
  - ¿Qué % de todos los casos litigados captura ese 10%?
  - ¿Qué % del AWW total "en riesgo" captura ese 10%?
"""

import pandas as pd
import numpy as np
import json
import os
from catboost import CatBoostClassifier, Pool

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE   = "/sessions/epic-intelligent-hawking/mnt/Tesis_ML"
DATA   = f"{BASE}/raw_data/dataset_tesis_clean.csv"
MODEL  = f"{BASE}/codigo/model_v3/catboost_v3_full.cbm"
OUT    = f"{BASE}/codigo/model_v3/milliman_comparison_results.json"

# ─── Target definition (igual que fairness_audit.py) ──────────────────────────
AVOIDABLE = [
    "CANCELLED", "CONTROVERTED", "NO LOST TIME",
    "MED ONLY", "NON-COMP"
]
# Orden exacto de fairness_audit.py — NUM primero, luego CAT
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
FEATURES = NUM_FEATURES + CAT_FEATURES

print("Cargando dataset...")
df_all = pd.read_csv(DATA)
df_test = df_all[df_all['accident_year'] == 2022].copy()
df_test['target'] = (
    df_test['target'].eq(1) &
    ~df_test['claim_injury_type_REF'].isin(AVOIDABLE)
).astype(int)
print(f"  Test-2022: {len(df_test):,} reclamos | positivos: {df_test['target'].sum():,} ({df_test['target'].mean()*100:.1f}%)")

# ─── Cargar modelo y predecir ─────────────────────────────────────────────────
print("Cargando modelo CatBoost v3...")
model = CatBoostClassifier()
model.load_model(MODEL)

feats   = [c for c in FEATURES if c in df_test.columns]
cat_idx = [feats.index(c) for c in CAT_FEATURES if c in feats]

X_test = df_test[feats].copy()
for col in CAT_FEATURES:
    if col in X_test.columns:
        X_test[col] = X_test[col].astype(str).fillna("MISSING")

pool_test = Pool(X_test, cat_features=cat_idx)
print("Prediciendo probabilidades...")
probs = model.predict_proba(pool_test)[:, 1]
df_test = df_test.copy()
df_test['prob'] = probs

# ─── Análisis por decil (estilo Milliman) ────────────────────────────────────
print("\nCalculando análisis por decil...")
df_test['decile'] = pd.qcut(df_test['prob'], q=10, labels=False, duplicates='drop')
# Decil 9 = top 10% (mayor probabilidad)
df_test['decile_label'] = 10 - df_test['decile']  # 1=top, 10=bottom

# AWW como proxy del valor económico del reclamo (igual que Milliman usa costos)
# Usamos AWW > 0 para el análisis de concentración económica
aww_positive = df_test[df_test['aww'] > 0]['aww'].sum()
total_positives = df_test['target'].sum()

decile_stats = []
cumulative_positives = 0
cumulative_aww = 0

for d in sorted(df_test['decile_label'].unique()):
    subset = df_test[df_test['decile_label'] == d]
    n = len(subset)
    n_pos = subset['target'].sum()
    aww_sum = subset[subset['aww'] > 0]['aww'].sum()
    precision = n_pos / n if n > 0 else 0
    recall = n_pos / total_positives if total_positives > 0 else 0
    aww_share = aww_sum / aww_positive if aww_positive > 0 else 0
    cumulative_positives += n_pos
    cumulative_aww += aww_sum
    cum_recall = cumulative_positives / total_positives
    cum_aww_share = cumulative_aww / aww_positive

    decile_stats.append({
        "decil": int(d),
        "n_claims": int(n),
        "n_litigados": int(n_pos),
        "precision": round(float(precision), 4),
        "recall_incremental": round(float(recall), 4),
        "recall_acumulado": round(float(cum_recall), 4),
        "aww_share_incremental": round(float(aww_share), 4),
        "aww_share_acumulado": round(float(cum_aww_share), 4),
        "prob_min": round(float(subset['prob'].min()), 4),
        "prob_max": round(float(subset['prob'].max()), 4),
    })

# ─── Métricas clave top-10% ───────────────────────────────────────────────────
top10 = df_test[df_test['decile_label'] == 1]
n_top10 = len(top10)
pos_top10 = top10['target'].sum()
aww_top10 = top10[top10['aww'] > 0]['aww'].sum()

# Comparación con top-20 casos de mayor AWW (análogo a los 20 casos más costosos de Milliman)
top20_by_aww = df_test.nlargest(20, 'aww')
top20_in_top10pct = top20_by_aww[top20_by_aww['decile_label'] == 1]
top20_flagged = len(top20_in_top10pct)

# Baseline aleatorio esperado
random_baseline_recall = 0.10   # un random flagging del 10% capturaría ~10% de positivos
random_baseline_aww    = 0.10

# ─── Lift por decil ──────────────────────────────────────────────────────────
overall_rate = df_test['target'].mean()
top10_rate   = top10['target'].mean()
lift_top10   = top10_rate / overall_rate

results = {
    "descripcion": "Comparación estilo Milliman Nodal (2018) — CatBoost NYSWCB test-2022",
    "dataset": {
        "total_claims": int(len(df_test)),
        "litigados": int(total_positives),
        "prevalencia_pct": round(df_test['target'].mean() * 100, 2),
        "aww_total_positivo": round(float(aww_positive), 0),
    },
    "top_10pct": {
        "n_claims_flagged": int(n_top10),
        "n_litigados_capturados": int(pos_top10),
        "recall_pct": round(float(pos_top10 / total_positives * 100), 1),
        "precision_pct": round(float(pos_top10 / n_top10 * 100), 1),
        "aww_share_pct": round(float(aww_top10 / aww_positive * 100), 1),
        "lift_sobre_baseline": round(float(lift_top10), 2),
        "prob_umbral_minimo": round(float(top10['prob'].min()), 4),
    },
    "top20_casos_mayor_aww": {
        "capturados_en_top10pct": int(top20_flagged),
        "de_20_totales": 20,
        "pct_captura": round(top20_flagged / 20 * 100, 1),
        "milliman_referencia_pct": 95.0,
    },
    "baseline_aleatorio": {
        "recall_esperado_pct": round(random_baseline_recall * 100, 1),
        "aww_share_esperado_pct": round(random_baseline_aww * 100, 1),
    },
    "tabla_deciles": decile_stats
}

# ─── Imprimir resumen ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ANÁLISIS DE CONCENTRACIÓN — ESTILO MILLIMAN NODAL")
print("="*60)
print(f"Dataset test-2022: {len(df_test):,} reclamos | {total_positives:,} litigados ({df_test['target'].mean()*100:.1f}%)")
print()
print("TOP 10% (mayor probabilidad predicha):")
print(f"  Reclamos flaggeados:          {n_top10:,}")
print(f"  Litigados capturados:         {pos_top10:,} de {total_positives:,}")
print(f"  Recall del top-10%:           {pos_top10/total_positives*100:.1f}%   (Milliman análogo: 87% pagos)")
print(f"  Precisión del top-10%:        {pos_top10/n_top10*100:.1f}%")
print(f"  Concentración AWW:            {aww_top10/aww_positive*100:.1f}%  (Milliman: 87% pagos en top-10%)")
print(f"  Lift sobre baseline aleatorio: {lift_top10:.2f}x")
print()
print(f"TOP-20 CASOS POR AWW MÁXIMO:")
print(f"  Capturados en top-10%:        {top20_flagged}/20 ({top20_flagged/20*100:.0f}%)")
print(f"  Milliman referencia:          19/20 (95%)")
print()
print("TABLA POR DECIL (1=top riesgo, 10=menor riesgo):")
print(f"{'Decil':>6} {'N':>7} {'Positivos':>10} {'Precision':>10} {'Recall Acum':>12} {'AWW Acum%':>10} {'Lift':>6}")
print("-"*65)
for s in decile_stats:
    lift_d = s['precision'] / overall_rate
    print(f"  {s['decil']:>3}   {s['n_claims']:>7,}   {s['n_litigados']:>8,}   {s['precision']*100:>8.1f}%   {s['recall_acumulado']*100:>9.1f}%   {s['aww_share_acumulado']*100:>8.1f}%   {lift_d:>4.1f}x")

# ─── Guardar resultados ───────────────────────────────────────────────────────
with open(OUT, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✓ Resultados guardados en {OUT}")
