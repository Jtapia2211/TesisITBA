# Anticipación del Conflicto Legal en Siniestros Laborales — Modelo Predictivo (NYSWCB)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jtapia2211/TesisITBA/blob/c1c6955ef45e5c31b60282e8b39b47306b7a5575/notebooks/Tesis_NYSWCB_Colab.ipynb)

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
| Ahorro con efectividad plena de la mediación (cota superior, §5.2) | $255 M |
| Ahorro mediano, Monte Carlo de referencia (§5.4, Tabla 35) | $72 M [IC 90 %: $23 M – $125 M] |

El umbral τ = 0,708 es el de máximo F1 sobre el propio conjunto de prueba 2022 (§5.3). Con el umbral elegido en validación 2021 (τ = 0,690), el mismo modelo da en prueba 2022 un recall de 70,1 % y una precisión de 50,6 %.

---

## Verificar las cifras sin reentrenar

**Primer paso recomendado para revisar el trabajo.** El repo incluye los modelos exactos de la tesis en `models/` y un script que recalcula las métricas desde el Parquet (alrededor de un minuto en CPU):

```bash
pip install -r requirements.txt
python verificar_metricas.py
```

Comprueba, contra los valores de la tesis: AUC-ROC, PR-AUC, KS y la matriz de confusión del modelo final; el resultado con el umbral de validación; la concentración por decil de la Tabla 38; y el AUC-ROC y PR-AUC del modelo optimizado sobre el target original.

| Modelo | Archivo | SHA-256 (inicio) |
|---|---|---|
| Final, CatBoost v3, litigio evitable (§4.10.2) | `models/catboost_v3_full.cbm` | `6cf055c1b061` |
| Optimizado, target original (§4.15) | `models/best_catboost_target_original.cbm` | `90dd51d10ab0` |

Reentrenar con `src/08_retrain_v3_gpu.py` produce un modelo equivalente pero no idéntico bit a bit (las cifras cambian en el cuarto decimal), por eso se publican los archivos originales.

---

## Dataset

**Fuente:** [NYSWCB — *Assembled Workers' Compensation Claims: Beginning 2000*](https://data.ny.gov/Government-Finance/Assembled-Workers-Compensation-Claims-Beginning-20/jshw-gkgu) (NY Open Data, dataset público).

El **dataset analítico final** (1.592.919 reclamos, 23 features + target, 2017–2022) viene incluido en el repo, comprimido en Parquet (~25 MB):

```python
import pandas as pd
df = pd.read_parquet("data/dataset_tesis_clean.parquet")   # 1.592.919 × 25
```

El notebook de Colab lo descarga solo desde el repo. Los scripts de `src/` leen `raw_data/dataset_tesis_clean.csv`; para generarlo desde el Parquet sin descargar el crudo:

```bash
python preparar_datos_local.py   # Parquet -> raw_data/dataset_tesis_clean.csv, y copia el modelo de la tesis a codigo/model_v3/
```

> **No reemplazar `pd.read_csv` por `pd.read_parquet` en los scripts.** Los modelos se entrenaron con los tipos que infiere `read_csv`: los códigos WCIO y de industria quedan como números (`9.0`). Leídos directo del Parquet quedan como texto (`09`), las categorías dejan de coincidir con las del entrenamiento y el AUC-ROC del modelo final baja de 0,8833 a 0,871.

**Reproducir el dataset desde cero** (opcional): el crudo (≈5,4 M registros, 54 variables) **no** se versiona por tamaño, pero se reconstruye automáticamente:

```bash
python reconstruir_crudo.py      # descarga el crudo comprimido (~128 MB) desde Google Drive
                                  # y reconstruye raw_data/nyswcb_claims.csv (~1,9 GB)
python src/01_build_dataset.py   # genera raw_data/dataset_tesis_clean.csv (1.592.919 filas)
```

> Si `reconstruir_crudo.py` falla con un error de Google Drive del tipo `Cannot retrieve the public link of the file`, esperar unos minutos y reintentar — suele deberse a un límite temporal de cuota de descargas de Drive, no a un problema de permisos.

> Las rutas se controlan con la variable de entorno `TESIS_BASE_DIR` (por defecto, la raíz del repo). Ej.: `export TESIS_BASE_DIR=/ruta/al/repo`.

---

## Instalación

```bash
pip install -r requirements.txt
```

Python 3.10 a 3.12. El benchmark y el reentrenamiento usan GPU (NVIDIA, CatBoost `task_type=GPU`); funcionan en CPU con mayor tiempo de cómputo.

Entorno con el que se verificaron `verificar_metricas.py` y los scripts 01, 10, 11, 13 y 14: Python 3.10.12, pandas 2.3.3, NumPy 2.2.6, CatBoost 1.2.10, scikit-learn 1.7.2, pyarrow 25.0.1. `requirements.txt` acota pandas a la serie 2.x: con pandas 3 cambia el manejo de faltantes en columnas de texto y algunos scripts (07 y 09) dejan valores nulos en las variables categóricas, que CatBoost rechaza.

---

## Estructura y correspondencia con la tesis

Los archivos están numerados `01`–`14` según el orden del pipeline (el benchmark tiene un orden interno propio, indicado más abajo); el nombre base coincide con el citado en la tesis, para trazar cada resultado a su código. Las secciones corresponden a la versión final del documento (`Tesis_Completa_v42_ITBA_rec`).

| # | Script (`src/`) | Capítulo / Sección | Qué hace |
|---|---|---|---|
| 01 | `01_build_dataset.py` | Cap. 3 (§3.2–3.4) | Filtros de alcance, definición del target (original y refinado *litigio evitable*) y protocolo anti-leakage. |
| 02 | `02_eda_script.py` | Cap. 4 (§4.1–4.5) | Análisis exploratorio: distribuciones, tasas por segmento, correlaciones. |
| 02b | `02b_benchmark5.py` | Cap. 4 (§4.7–4.8) | Primeros cinco modelos del benchmark (regresión logística, Naive Bayes, árbol, Random Forest, GBDT) en NumPy. |
| 03 | `03_benchmark_gpu_windows.py` | Cap. 4 (§4.6–4.10) | Orquesta el benchmark de los 9 modelos bajo validación temporal (2017-20 / 2021 / 2022). |
| 04 | `04_benchmark_lgbm_xgb.py` | Cap. 4 (§4.7–4.8) | Modelos 7 y 8: LightGBM y XGBoost. |
| 05 | `05_benchmark_mlp.py` | Cap. 4 (§4.7–4.8) | Red neuronal (MLP) en NumPy puro. |
| 06 | `06_benchmark_catboost.py` | Cap. 4 (§4.7–4.8) | CatBoost dentro del benchmark. |
| 07 | `07_cap6_tuning_catboost.py` | Cap. 4 (§4.11–4.16) | Optimización bayesiana con Optuna (TPE, 60 trials) + análisis fANOVA. |
| 08 | `08_retrain_v3_gpu.py` | Cap. 4 (§4.10.2) | Modelo final CatBoost v3 sobre target refinado; umbral por máximo F1 (`best_f1_threshold`). |
| 09 | `09_cap7_shap.py` | Cap. 4 (§4.17–4.22) | Interpretabilidad SHAP (TreeSHAP): importancia global, beeswarm, dependencia, casos. |
| 10 | `10_montecarlo_cap8.py` | Cap. 5 (§5.4) | Simulación Monte Carlo del impacto económico (incertidumbre en costos y efectividad). |
| 11 | `11_milliman_comparison.py` | Cap. 5 (§5.6) | Análisis de concentración por decil y comparación con Milliman Nodal. |
| 12 | `12_fnr_economic_analysis.py` | Complementario (ver §6.3.1) | Impacto económico diferencial del FNR por quintil de salario (AWW). No se reporta como sección en la versión final. |
| 13 | `13_fairness_audit.py` | Cap. 6 (§6.3.1) | Auditoría de equidad (Equal Opportunity / Predictive Parity, regla EEOC 4/5). |
| 14 | `14_fairness_calibration.py` | Cap. 6 (§6.3.1) | Calibración diferenciada del umbral por quintil de AWW. |

### Notebooks (`notebooks/`)

- `Tesis_NYSWCB_Colab.ipynb`: notebook reproducible de la tesis (reentrena el modelo final y, en su última sección, verifica las cifras con el modelo exacto).
- `301`–`304`: análisis exploratorio sobre el dataset crudo.
- `305`–`308`: **antecedentes**. Corresponden a una etapa anterior del trabajo, con otro objetivo de predicción (`Has_ANCR`, si el reclamo llega a ser establecido), y no generan resultados de la tesis final. Se conservan como registro del proceso.

---

## Reproducción end-to-end

```bash
export TESIS_BASE_DIR=$(pwd)
python reconstruir_crudo.py               # 0. reconstruye el crudo (raw_data/nyswcb_claims.csv)
python src/01_build_dataset.py            # 1. dataset analítico
python preparar_datos_local.py            #    (alternativa a 0 y 1: CSV desde el Parquet + modelo de la tesis)
# Benchmark de 9 modelos: cada paso amplía el JSON de resultados del anterior, en este orden
python src/02b_benchmark5.py              #    5 modelos base        -> results/benchmark5_results.json
python src/06_benchmark_catboost.py       #    + CatBoost            -> results/benchmark6_results.json
python src/04_benchmark_lgbm_xgb.py       #    + LightGBM y XGBoost  -> results/benchmark8_results.json
python src/03_benchmark_gpu_windows.py    #    versiones GPU         -> results/benchmark_gpu_results.json
python src/05_benchmark_mlp.py            #    + MLP (9 modelos)
python src/07_cap6_tuning_catboost.py     # 7. Optuna
python src/08_retrain_v3_gpu.py           # 8. modelo final v3
python src/09_cap7_shap.py                # 9. SHAP
python src/10_montecarlo_cap8.py          # 10. impacto económico (Monte Carlo, Tabla 35)
python src/11_milliman_comparison.py      # 11. concentración por decil (Tabla 38)
python src/13_fairness_audit.py           # 13. equidad
python src/14_fairness_calibration.py     # 14. calibración por quintil
python verificar_metricas.py              #     verificación contra las cifras de la tesis
```

> Los pasos 09 a 14 usan el modelo de `codigo/model_v3/catboost_v3_full.cbm`. Para obtener las cifras exactas de la tesis debe ser el modelo publicado en `models/` (lo copia `preparar_datos_local.py`); si se reentrena con el paso 08, los resultados coinciden de forma aproximada.

> **Nota (Windows/CatBoost GPU):** si no hay GPU con drivers CUDA compatibles, `08_retrain_v3_gpu.py` cae automáticamente a `task_type=CPU` (con un aviso de "CUDA driver version is insufficient", que puede ignorarse). El entrenamiento en CPU toma ~90 minutos sobre el dataset completo.

> **Troubleshooting:** en versiones antiguas del script, `08_retrain_v3_gpu.py` podía fallar con `FileNotFoundError` al crear `codigo/model_v3` (la carpeta no se versiona en Git por estar vacía). Si esto ocurre, correr `mkdir codigo` (o `mkdir -p codigo` en Linux/Mac) antes de reintentar.

---

## Nota

Las proyecciones económicas son estimaciones bajo supuestos explícitos (no validadas en producción). El dataset del NYSWCB se actualiza periódicamente; los conteos pueden variar según la fecha de descarga.
