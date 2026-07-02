"""
fnr_economic_analysis.py
────────────────────────
Análisis de impacto económico diferencial del FNR por quintil de AWW.

Lógica:
  El modelo de costos del §8.1 usa C_j = $13,000 como costo plano por caso no
  detectado. Pero los trabajadores de quintiles altos tienen mayor AWW, lo que
  implica beneficios semanales más elevados y, por ende, montos litigados mayores.
  Este script cuantifica el diferencial usando AWW como multiplicador de costo.

Metodología:
  C_j_ajustado(q) = C_j × (AWW_media_q / AWW_media_global)
  Costo_FN(q)     = FN_q × C_j_ajustado(q)

Compara baseline (τ=0.708 global) vs calibrado (τ_q per-quintile).
"""

import pandas as pd
import numpy as np
import json

BASE   = "/sessions/epic-intelligent-hawking/mnt/Tesis_ML"
DATA   = f"{BASE}/raw_data/dataset_tesis_clean.csv"
CAL    = f"{BASE}/codigo/model_v3/calibration_results.json"
OUT    = f"{BASE}/codigo/model_v3/fnr_economic_results.json"
C_J    = 13_000   # costo base por litigio no detectado (§8.1)

AVOIDABLE = ["CANCELLED","CONTROVERTED","NO LOST TIME","MED ONLY","NON-COMP"]

# ─── Cargar datos ─────────────────────────────────────────────────────────────
print("Cargando datos...")
with open(CAL) as f:
    cal = json.load(f)

df_all = pd.read_csv(DATA)
df = df_all[df_all['accident_year'] == 2022].copy()
df['target'] = (
    df['target'].eq(1) & ~df['claim_injury_type_REF'].isin(AVOIDABLE)
).astype(int)

# ─── Quintiles por pd.qcut solo en AWW > 0 (igual que fairness_calibration.py)
Q_LABELS = ['Q1','Q2','Q3','Q4','Q5']
df_aww = df[df['aww'] > 0].copy()
df_aww['aww_quintile'] = pd.qcut(df_aww['aww'], q=5, labels=Q_LABELS, duplicates='drop')

# ─── AWW media por quintil ────────────────────────────────────────────────────
aww_by_q = df_aww.groupby('aww_quintile', observed=True)['aww'].mean()
aww_global_mean = df_aww['aww'].mean()

print(f"\nAWW media global (AWW>0): ${aww_global_mean:,.0f}")
print("AWW media por quintil:")
for q, v in aww_by_q.items():
    print(f"  {q}: ${v:,.0f}  (multiplicador: {v/aww_global_mean:.2f}×)")

# ─── Resultados de calibración ────────────────────────────────────────────────
q_labels = ['Q1 (bajo)', 'Q2', 'Q3', 'Q4', 'Q5 (alto)']
q_short   = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

rows = []
total_fn_base = 0
total_fn_cal  = 0
total_cost_base = 0
total_cost_cal  = 0

print("\n" + "="*72)
print(f"{'Quintil':>10} {'AWW media':>10} {'Mult.':>6} | {'FN base':>8} {'FN cal':>8} {'ΔFN':>6} | {'Costo base $K':>14} {'Costo cal $K':>12} {'Ahorro $K':>10}")
print("-"*72)

for ql, qs in zip(q_labels, q_short):
    aww_q = aww_by_q.get(qs, aww_global_mean)
    mult  = aww_q / aww_global_mean
    cj_q  = C_J * mult

    fn_base = cal['baseline'][ql]['FN']
    fn_cal  = cal['calibrated'][ql]['FN']
    d_fn    = fn_base - fn_cal

    cost_base = fn_base * cj_q
    cost_cal  = fn_cal  * cj_q
    saving    = cost_base - cost_cal

    total_fn_base   += fn_base
    total_fn_cal    += fn_cal
    total_cost_base += cost_base
    total_cost_cal  += cost_cal

    print(f"  {qs:>6}     ${aww_q:>8,.0f}  {mult:>5.2f}×  | {fn_base:>8,} {fn_cal:>8,} {d_fn:>+6,} | ${cost_base/1e3:>12,.0f}K  ${cost_cal/1e3:>10,.0f}K  ${saving/1e3:>8,.0f}K")

    rows.append({
        "quintil": qs, "aww_media": round(aww_q, 0),
        "multiplicador_costo": round(mult, 3),
        "cj_ajustado": round(cj_q, 0),
        "fn_baseline": fn_base, "fn_calibrado": fn_cal,
        "delta_fn": int(fn_base - fn_cal),
        "costo_fn_base_usd": round(cost_base, 0),
        "costo_fn_cal_usd":  round(cost_cal, 0),
        "ahorro_diferencial_usd": round(saving, 0),
    })

print("-"*72)
total_saving = total_cost_base - total_cost_cal
d_fn_total   = total_fn_base - total_fn_cal
print(f"  {'TOTAL':>6}     ${aww_global_mean:>8,.0f}  {'1.00':>5}×  | {total_fn_base:>8,} {total_fn_cal:>8,} {d_fn_total:>+6,} | ${total_cost_base/1e3:>12,.0f}K  ${total_cost_cal/1e3:>10,.0f}K  ${total_saving/1e3:>8,.0f}K")

# ─── Resumen del ahorro diferencial ───────────────────────────────────────────
print("\n" + "="*72)
print("RESUMEN:")
print(f"  Reducción total de FN:             {d_fn_total:,} casos")
print(f"  De esos, en Q5 (mayor costo):      {rows[4]['delta_fn']:,} casos ({rows[4]['delta_fn']/d_fn_total*100:.0f}% del total)")
print(f"  Ahorro diferencial AWW-ajustado:   ${total_saving/1e6:.2f}M")
print(f"  Ahorro solo Q5 (alto AWW):         ${rows[4]['ahorro_diferencial_usd']/1e6:.2f}M")
print(f"  % del ahorro total que viene de Q5: {rows[4]['ahorro_diferencial_usd']/total_saving*100:.0f}%")
print(f"  Costo plano equivalente (§8.1):    ${d_fn_total * C_J / 1e6:.2f}M  (sin ajuste AWW)")

# ─── Guardar resultados ───────────────────────────────────────────────────────
results = {
    "descripcion": "Impacto económico diferencial del FNR por quintil AWW — test-2022",
    "parametros": {
        "C_j_base": C_J,
        "ajuste": "C_j × (AWW_media_quintil / AWW_media_global)",
        "aww_media_global": round(aww_global_mean, 0),
    },
    "por_quintil": rows,
    "totales": {
        "fn_baseline_total": total_fn_base,
        "fn_calibrado_total": total_fn_cal,
        "delta_fn_total": int(d_fn_total),
        "costo_fn_base_usd": round(total_cost_base, 0),
        "costo_fn_cal_usd": round(total_cost_cal, 0),
        "ahorro_diferencial_usd": round(total_saving, 0),
        "ahorro_diferencial_M": round(total_saving / 1e6, 2),
        "pct_ahorro_en_Q5": round(rows[4]['ahorro_diferencial_usd'] / total_saving * 100, 1),
        "ahorro_costo_plano_M": round(d_fn_total * C_J / 1e6, 2),
    }
}

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✓ Resultados guardados en {OUT}")
