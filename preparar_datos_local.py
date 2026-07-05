"""
preparar_datos_local.py — Reproducción local SIN descargar el crudo de 2,1 GB.

Convierte el dataset analítico comprimido (data/dataset_tesis_clean.parquet, ~25 MB,
ya incluido en el repo) a raw_data/dataset_tesis_clean.csv, que es lo que esperan
los scripts de src/. Con esto podés correr directamente:
    08_retrain_v3_gpu.py, 10_cap7_shap.py, 11_montecarlo_cap8.py,
    14_fairness_audit.py, 15_fairness_calibration.py
sin necesidad de 01_build_dataset.py ni del crudo de 5,4M registros.

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
print("Ahora podés correr: python src/08_retrain_v3_gpu.py")
