"""
Simulación Monte Carlo del impacto económico (tesis §5.4, Tabla 35, Figura 14).
Incluye la efectividad de la mediación (p_éxito) como variable aleatoria.

Resultados esperados (seed 42, N = 50.000):
  Referencia (mediación privada, p_éxito 40–60 %): P50 ≈ USD 72 M  [P5 ≈ 23 M – P95 ≈ 125 M]
  Cota superior (efectividad plena):               P50 ≈ USD 254 M
  Mediación pública (misma efectividad):           P50 ≈ USD 134 M

Nota: las cifras de la tesis se calcularon con la matriz al umbral redondeado
τ = 0,7084 (TP = 27.900, FP = 25.866); este script usa la matriz del umbral exacto
(TP = 27.903, FP = 25.868; Tabla de §5.2). La diferencia es menor a USD 0,03 M:
el P95 da 125,51 aquí y 125,50 en la tesis, que lo informa como 125.

Generates:
  fig5_montecarlo_distribution.png  — histogram comparing upper-bound (p=1) vs. full MC
  fig6_montecarlo_tornado.png       — tornado chart (now with p_success as 4th driver)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from pathlib import Path
from scipy import stats

# ── Seed and size ─────────────────────────────────────────────────────────────
SEED = 42
N = 50_000

PLOTS_DIR = Path(__file__).resolve().parent.parent / "codigo" / "model_plots8"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH  = PLOTS_DIR / "economic_summary.json"

# ── Confusion matrix (modelo v3, litigio evitable, test 2022, τ=0.708) ────────
TP =  27_903
FP =  25_868
FN =  13_245
TN = 193_140

# ── Distribuciones triangulares de los parámetros (Tabla 35 de la tesis) ─────
# (mínimo, moda, máximo). Escenario de referencia: mediación privada.
C_j_params  = dict(low=9_000,  mode=13_000, high=17_000)   # costo de litigio, ±30% sobre el valor base (Tabla 30)
C_p_params  = dict(low=2_100,  mode=3_400,  high=4_800)    # mediación privada, por componentes (Tabla 31)
C_fp_params = dict(low=350,    mode=500,    high=650)      # falsa alarma, ±30%
p_success_params = dict(low=0.40, mode=0.50, high=0.60)    # tasa de resolución de la mediación

# Escenarios alternativos de la Figura 14
C_p_public_params   = dict(low=850,  mode=1_200, high=1_600)   # mediación pública
p_success_low_params = dict(low=0.10, mode=0.20, high=0.35)    # efectividad baja

def tri(rng, params, n):
    return rng.triangular(params['low'], params['mode'], params['high'], n)

# ── Sampling (mismo generador y orden de extracción que los resultados de §5.4) ──
rng = np.random.default_rng(SEED)
C_j     = tri(rng, C_j_params,  N)
C_fp    = tri(rng, C_fp_params, N)
C_p     = tri(rng, C_p_params,  N)
C_p_pub = tri(rng, C_p_public_params, N)
p_suc   = tri(rng, p_success_params,  N)
p_low   = tri(rng, p_success_low_params, N)

# ── Ahorro neto ──────────────────────────────────────────────────────────────
# Sin modelo: todos los positivos litigan  -> (TP + FN) * C_j
# Con modelo: TP pagan C_p y una fracción p evita el litigio; FP pagan C_fp;
#             FN litigan igual.
# => ahorro_neto = TP*(p*C_j - C_p) - FP*C_fp
def net_savings(p, cp):
    return (TP*(p*C_j - cp) - FP*C_fp) / 1e6          # millones USD

def investment(cp):
    return (TP*cp + FP*C_fp) / 1e6

net_savings_full = net_savings(p_suc, C_p)             # referencia: privada, p ~ Tri[0,40; 0,50; 0,60]
net_savings_ub   = net_savings(1.0,   C_p)             # cota superior: efectividad plena
net_savings_pub  = net_savings(p_suc, C_p_pub)         # mediación pública, misma efectividad
net_savings_low  = net_savings(p_low, C_p)             # efectividad baja
roi50_f = float(np.median(net_savings_full / investment(C_p)))
inv50_f = float(np.median(investment(C_p)))

# ── Stats — full (primary) ────────────────────────────────────────────────────
def pct(arr, q): return float(np.percentile(arr, q))

p5_f  = pct(net_savings_full, 5)
p25_f = pct(net_savings_full, 25)
p50_f = pct(net_savings_full, 50)
p75_f = pct(net_savings_full, 75)
p95_f = pct(net_savings_full, 95)
mean_f = float(np.mean(net_savings_full))
std_f  = float(np.std(net_savings_full))
prob_pos_f   = float(np.mean(net_savings_full > 0) * 100)
prob_500_f   = float(np.mean(net_savings_full > 500) * 100)
prob_200_f   = float(np.mean(net_savings_full > 200) * 100)

# Stats — upper bound (p_success=1)
p50_pub = pct(net_savings_pub, 50)
p50_low = pct(net_savings_low, 50)
p5_ub   = pct(net_savings_ub, 5)
p50_ub  = pct(net_savings_ub, 50)
p95_ub  = pct(net_savings_ub, 95)
mean_ub = float(np.mean(net_savings_ub))

print("── Monte Carlo Results (FULL: including p_success uncertainty) ──────────")
print(f"  P5   = ${p5_f:,.1f}M")
print(f"  P25  = ${p25_f:,.0f}M")
print(f"  Mean = ${mean_f:,.0f}M")
print(f"  P50  = ${p50_f:,.1f}M")
print(f"  P75  = ${p75_f:,.0f}M")
print(f"  P95  = ${p95_f:,.1f}M")
print(f"  Std  = ${std_f:,.0f}M")
print(f"  P(>0)    = {prob_pos_f:.1f}%")
print(f"  P(>200M) = {prob_200_f:.1f}%")
print(f"  P(>500M) = {prob_500_f:.1f}%")
print()
print("── Upper-bound (p_success=1, cota superior) ─────────────────────────────")
print(f"  P5={p5_ub:,.0f}M  P50={p50_ub:,.0f}M  P95={p95_ub:,.0f}M  Mean={mean_ub:,.0f}M")
print()
print("── Escenarios alternativos ──────────────────────────────────────────────")
print(f"  Mediación pública (misma efectividad): P50 = ${p50_pub:,.0f}M")
print(f"  Efectividad baja (10%–35%):            P50 = ${p50_low:,.0f}M")
print(f"  Referencia: inversión mediana = ${inv50_f:,.0f}M  |  ROI mediano = {roi50_f:.1f}x")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Dual distribution: upper-bound vs. full MC
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

# Common bin range
x_min = min(net_savings_full.min(), net_savings_ub.min()) - 20
x_max = max(net_savings_full.max(), net_savings_ub.max()) + 20
bins = np.linspace(x_min, x_max, 90)

# Upper-bound histogram (ghost)
ax.hist(net_savings_ub,   bins=bins, color='#AABDD6', alpha=0.45,
        label=f'Cota superior (p_éxito = 100%)  —  P50 = USD {p50_ub:,.0f}M',
        edgecolor='white', linewidth=0.2)

# Full MC histogram (primary)
ax.hist(net_savings_full, bins=bins, color='#2166AC', alpha=0.75,
        label=f'Referencia: mediación privada (p_éxito ~ Triang[0.40, 0.50, 0.60])  —  P50 = USD {p50_f:,.0f}M',
        edgecolor='white', linewidth=0.2)

# KDE — full
kde_f = stats.gaussian_kde(net_savings_full, bw_method=0.12)
x_kde = np.linspace(x_min, x_max, 500)
scale = N * (bins[1] - bins[0])
ax.plot(x_kde, kde_f(x_kde) * scale, color='#1a1a2e', lw=2.0, zorder=6)

# Vertical lines — full MC
for val, col, lbl, ls in [
    (p5_f,   '#D73027', f'P5 = USD {p5_f:,.0f}M',  'dashed'),
    (p50_f,  '#222222', f'P50 = USD {p50_f:,.0f}M', 'solid'),
    (p95_f,  '#1A9641', f'P95 = USD {p95_f:,.0f}M', 'dashed'),
]:
    ax.axvline(val, color=col, lw=1.8, linestyle=ls, label=lbl, zorder=7)

ax.axvline(mean_f, color='#7B2D8B', lw=1.4, linestyle='dotted',
           label=f'Media = USD {mean_f:,.0f}M', zorder=7)

# Zero line
ax.axvline(0, color='black', lw=1.2, linestyle='solid', alpha=0.4)

ax.set_xlabel('Ahorro Neto (millones USD)', fontsize=12)
ax.set_ylabel('Frecuencia (simulaciones)', fontsize=12)
ax.set_title(
    f'Distribución del Ahorro Neto — Simulación Monte Carlo (N = {N:,})\n'
    f'P(ahorro > 0) = {prob_pos_f:.1f}%   |   P(ahorro > USD 200M) = {prob_200_f:.1f}%   |   Desv. estándar = USD {std_f:,.0f}M',
    fontsize=10.5, pad=10
)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
plt.tight_layout()

out5 = PLOTS_DIR / "fig5_montecarlo_distribution.png"
fig.savefig(out5, dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"\n  ✓ Saved {out5.name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Tornado chart (one-at-a-time, now with p_success)
# ══════════════════════════════════════════════════════════════════════════════
def calc_savings_full(cj, cp, cfp, ps):
    base  = (TP + FN) * cj
    model = TP*cp + TP*(1-ps)*cj + FP*cfp + FN*cj
    return (base - model) / 1e6

# Base point (all modes)
base_sav = calc_savings_full(
    C_j_params['mode'], C_p_params['mode'],
    C_fp_params['mode'], p_success_params['mode']
)

tornado_data = []
configs = [
    ('p_éxito (efectividad mediación)',  p_success_params, 'ps',  '#D73027'),
    ('C_j (costo litigación)',           C_j_params,       'C_j', '#2166AC'),
    ('C_p (costo prevención)',           C_p_params,       'C_p', '#4DAC26'),
    ('C_fp (costo falsa alarma)',        C_fp_params,      'C_fp','#F4A82E'),
]

for label, params, var, color in configs:
    def _s(v, which):
        return calc_savings_full(
            params[which] if var == 'C_j'  else C_j_params['mode'],
            params[which] if var == 'C_p'  else C_p_params['mode'],
            params[which] if var == 'C_fp' else C_fp_params['mode'],
            params[which] if var == 'ps'   else p_success_params['mode'],
        )
    lo_sav = _s(params['low'],  'low')
    hi_sav = _s(params['high'], 'high')
    swing  = abs(hi_sav - lo_sav)
    tornado_data.append((label, lo_sav, hi_sav, swing, color))

tornado_data.sort(key=lambda x: x[3], reverse=True)

fig2, ax2 = plt.subplots(figsize=(9.5, 4.2))
bar_h = 0.48

for i, (label, lo_sav, hi_sav, swing, color) in enumerate(tornado_data):
    left  = min(lo_sav, hi_sav)
    right = max(lo_sav, hi_sav)
    ax2.barh(i, right - left, left=left, height=bar_h,
             color=color, alpha=0.82, edgecolor='white', linewidth=0.5)
    ax2.text(left  - 10, i, f'${lo_sav:,.0f}M', va='center', ha='right', fontsize=8.5, color='#333')
    ax2.text(right + 10, i, f'${hi_sav:,.0f}M', va='center', ha='left',  fontsize=8.5, color='#333')
    ax2.text((left+right)/2, i, f'Δ${swing:,.0f}M', va='center', ha='center',
             fontsize=8, color='white', fontweight='bold')

ax2.axvline(base_sav, color='black', lw=1.5, linestyle='--',
            label=f'Base = ${base_sav:,.0f}M')
ax2.axvline(0, color='red', lw=0.8, linestyle=':', alpha=0.6, label='Cero (sin ahorro)')
ax2.set_yticks(range(len(tornado_data)))
ax2.set_yticklabels([d[0] for d in tornado_data], fontsize=10)
ax2.set_xlabel('Ahorro Neto (millones USD)', fontsize=11)
ax2.set_title('Análisis de Sensibilidad (Tornado) — Impacto de Cada Parámetro sobre el Ahorro Neto',
              fontsize=10, pad=10)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}M'))
ax2.legend(fontsize=9, loc='lower right')
all_vals = [d[1] for d in tornado_data] + [d[2] for d in tornado_data]
ax2.set_xlim(min(all_vals) - 70, max(all_vals) + 70)
plt.tight_layout()

out6 = PLOTS_DIR / "fig6_montecarlo_tornado.png"
fig2.savefig(out6, dpi=160, bbox_inches='tight')
plt.close(fig2)
print(f"  ✓ Saved {out6.name}")

# ── Update economic_summary.json ──────────────────────────────────────────────
if JSON_PATH.exists():
    with open(JSON_PATH) as f:
        summary = json.load(f)
else:
    summary = {}

summary['montecarlo'] = {
    'N': N,
    # Full MC (primary — includes p_success uncertainty)
    'p5_M':  round(p5_f, 1),
    'p25_M': round(p25_f, 1),
    'mean_M': round(mean_f, 1),
    'p50_M': round(p50_f, 1),
    'p75_M': round(p75_f, 1),
    'p95_M': round(p95_f, 1),
    'std_M': round(std_f, 1),
    'prob_positive_pct':    round(prob_pos_f, 1),
    'prob_above_200M_pct':  round(prob_200_f, 1),
    'prob_above_500M_pct':  round(prob_500_f, 1),
    # Upper bound (p_success=1)
    'ub_p5_M':  round(p5_ub, 1),
    'ub_p50_M': round(p50_ub, 1),
    'ub_p95_M': round(p95_ub, 1),
    'ub_mean_M': round(mean_ub, 1),
    # Escenarios alternativos
    'public_p50_M': round(p50_pub, 1),
    'low_effectiveness_p50_M': round(p50_low, 1),
    'roi_p50': round(roi50_f, 2),
    'investment_p50_M': round(inv50_f, 1),
    # Parameter distributions
    'C_j_params':       C_j_params,
    'C_p_params':       C_p_params,
    'C_fp_params':      C_fp_params,
    'p_success_params': p_success_params,
    'C_p_public_params': C_p_public_params,
    'p_success_low_params': p_success_low_params,
    'seed': SEED,
}

with open(JSON_PATH, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Updated economic_summary.json")
print("✅ Monte Carlo (full) complete.")
