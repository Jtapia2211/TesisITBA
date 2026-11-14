"""
preparar_datos_local.py — Reproducción local SIN descargar el crudo de 2,1 GB.

Convierte el dataset analítico comprimido (data/dataset_tesis_clean.parquet, ~25 MB,
ya incluido en el repo) a raw_data/dataset_tesis_clean.csv, que es lo que esperan
los scripts de src/. Con esto podés correr directamente:
    08_retrain_v3_gpu.py, 09_cap7_shap.py, 10_montecarlo_cap8.py,
    13_fairness_audit.py, 14_fairness_calibration.py
sin necesidad de 01_build_dataset.py ni del crudo de 5,4M registros.

También copia el modelo final exacto de la tesis (models/catboost_v3_full.cbm) a
codigo/model_v3/, que es donde lo buscan los scripts 09 a 14, salvo que ya haya
uno ahí (por ejemplo, uno reentrenado con 08_retrain_v3_gpu.py).

Uso (parado en la raíz del repo):
    pip install pandas pyarrow
    python preparar_datos_local.py
"""
import os
import pandas as pd

BASE   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(BASE, "data", "dataset_tesis_clean.parquet")
OUTDIR = os.path.join(BASE, "raw_data")
OUT    = os.path.join(OUTDIR, "dataset_tesis_clean.csv")

assert os.path.exists(SRC), f"No encuentro {SRC}. ¿Clonaste el repo completo?"
os.makedirs(OUTDIR, exist_ok=True)

print("Convirtiendo parquet -> CSV (1.592.919 filas)...")
pd.read_parquet(SRC).to_csv(OUT, index=False)
print(f"Listo: {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")

# Modelo final exacto de la tesis -> donde lo esperan los scripts 09 a 14.
# No pisa un modelo que ya hayas reentrenado con 08_retrain_v3_gpu.py.
import shutil
M_SRC = os.path.join(BASE, "models", "catboost_v3_full.cbm")
M_DIR = os.path.join(BASE, "codigo", "model_v3")
M_DST = os.path.join(M_DIR, "catboost_v3_full.cbm")
if os.path.exists(M_SRC) and not os.path.exists(M_DST):
    os.makedirs(M_DIR, exist_ok=True)
    shutil.copyfile(M_SRC, M_DST)
    print(f"Modelo de la tesis copiado a {M_DST}")
elif os.path.exists(M_DST):
    print(f"Ya existe {M_DST}: no se reemplaza.")

print("Ahora podés correr los scripts 09 a 14 con el modelo de la tesis,")
print("o reentrenar desde cero con: python src/08_retrain_v3_gpu.py")
