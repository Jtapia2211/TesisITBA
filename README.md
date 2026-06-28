# Anticipación del Conflicto Legal en Siniestros Laborales — Modelo Predictivo (NYSWCB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jtapia2211/TesisITBA/blob/main/Tesis_NYSWCB_Colab.ipynb)

Código fuente de la tesis de Maestría en Ciencia de Datos (ITBA, 2026).
Repositorio: https://github.com/Jtapia2211/TesisITBA

**Autor:** Julián Tapia · **Director:** Ariel Aizemberg · Instituto Tecnológico de Buenos Aires
**Modelo de producción:** CatBoost v3 (target: *litigio evitable*)

| Métrica (test 2022, n = 260.156) | Valor |
|---|---|
| AUC-ROC | 0,8833 |
| PR-AUC | 0,5981 |
| Recall @ τ=0,708 | 67,8 % |
| Precisión @ τ=0,708 | 51,9 % |
| KS | 0,6202 |
| Ahorro estimado (cota superior) | $252 M |
| Ahorro central (Monte Carlo, P50) | $100 M [IC 90 %: $39 M – $164 M] |

---

## Dataset

**Fuente:** [NYSWCB — *Assembled Workers' Compensation Claims: Beginning 2000*](https://data.ny.gov/Government-Finance/Assembled-Workers-Compensation-Claims-Beginning-20/jshw-gkgu) (NY Open Data, dataset público).

El **dataset analítico final** (1.592.919 reclamos, 23 features + target, 2017–2022) viene incluido en el repo, comprimido en Parquet (~25 MB):

```python
import pandas as pd
df = pd.read_parquet("data/dataset_tesis_clean.parquet")   # 1.592.919 × 25
```

El notebook de Colab lo descarga solo desde el repo. Los scripts de `src/` que leen `dataset_tesis_clean.csv` pueden apuntar a este Parquet cambiando `pd.read_csv(...)` por `pd.read_parquet("data/dataset_tesis_clean.parquet")`.

**Reproducir el dataset desde cero** (opcional): el crudo (≈5,4 M registros, 54 variables) **no** se versiona por tamaño. Descargalo del [portal NYSWCB](https://data.ny.gov/Government-Finance/Assembled-Workers-Compensation-Claims-Beginning-20/jshw-gkgu), dejalo en `raw_data/nyswcb_claims.csv` y corré `src/build_dataset.py`.

> Las rutas se controlan con la variable de entorno `TESIS_BASE_DIR` (por defecto, la raíz del repo). Ej.: `export TESIS_BASE_DIR=/ruta/al/repo`.

---

## Instalación

```bash
pip install -r requirements.txt
```

Python 3.10+. El benchmark y el reentrenamiento usan GPU (NVIDIA, CatBoost `task_type=GPU`); funcionan en CPU con mayor tiempo de cómputo.

---

## Estructura y correspondencia con la tesis

Los nombres de archivo coinciden con los citados en el documento, para trazar cada resultado a su código.

| Orden | Script (`src/`) | Capítulo / Sección | Qué hace |
|---|---|---|---|
| 1 | `build_dataset.py` | Cap. 3 (§3.2–3.4) | Filtros de alcance, definición del target (original y refinado *litigio evitable*) y protocolo anti-leakage. |
| 2 | `eda_script.py` | Cap. 4 (§4.1–4.14) | Análisis exploratorio: distribuciones, tasas por segmento, correlaciones. |
| 3 | `benchmark_gpu_windows.py` | Cap. 4 (§4.15–4.19) | Benchmark de los 9 modelos bajo validación temporal (2017-20 / 2021 / 2022). |
| 3b | `benchmark_lgbm_xgb.py`, `benchmark_mlp.py`, `benchmark_catboost.py` | Cap. 4 (§4.16) | Entrenamiento por familia de modelos. |
| 4 | `cap6_tuning_catboost.py` | Cap. 4 (§4.20–4.27) | Optimización bayesiana con Optuna (TPE, 60 trials) + análisis fANOVA. |
| 5 | `retrain_v3_gpu.py` | Cap. 4 (§4.19.2) | Modelo final CatBoost v3 sobre target refinado; umbral por máximo F1 (`best_f1_threshold`). |
| 6 | `depth_ext_experiment.py` | Cap. 4 (§4.23) | Verificación de robustez de la profundidad óptima. |
| 7 | `cap7_shap.py` | Cap. 4 (§4.28–4.35) | Interpretabilidad SHAP (TreeSHAP): importancia global, beeswarm, dependencia, casos. |
| 8 | `montecarlo_cap8.py` | Cap. 5 (§5.5) | Simulación Monte Carlo del impacto económico (incertidumbre en costos y efectividad). |
| 9 | `milliman_comparison.py` | Cap. 5 (§5.8) | Análisis de concentración por decil y comparación con Milliman Nodal. |
| 10 | `fnr_economic_analysis.py` | Cap. 5 (§5.9) | Impacto económico diferencial del FNR por quintil de salario (AWW). |
| 11 | `fairness_audit.py` | Cap. 6 (§6.3.1) | Auditoría de equidad (Equal Opportunity / Predictive Parity, regla EEOC 4/5). |
| 12 | `fairness_calibration.py` | Cap. 6 (§6.3.1) | Calibración diferenciada del umbral por quintil de AWW. |

### Notebooks (`notebooks/`)

Serie final del análisis exploratorio, benchmark e interpretabilidad (versión "Sovereign"), más el script de inferencia:
`301`–`304` EDA · `305` feature engineering + selección · `306` benchmark · `307` XAI/SHAP · `308` inferencia y cálculo de probabilidades.

---

## Reproducción end-to-end

```bash
export TESIS_BASE_DIR=$(pwd)
python src/build_dataset.py            # 1. dataset analítico
python src/benchmark_gpu_windows.py    # 2. benchmark 9 modelos
python src/cap6_tuning_catboost.py     # 3. Optuna
python src/retrain_v3_gpu.py           # 4. modelo final v3
python src/cap7_shap.py                # 5. SHAP
python src/montecarlo_cap8.py          # 6. impacto económico
python src/fairness_audit.py           # 7. equidad
python src/fairness_calibration.py     # 8. calibración por quintil
```

---

## Nota

Las proyecciones económicas son estimaciones bajo supuestos explícitos (no validadas en producción). El dataset del NYSWCB se actualiza periódicamente; los conteos pueden variar según la fecha de descarga.
