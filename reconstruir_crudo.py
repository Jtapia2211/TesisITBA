"""
reconstruir_crudo.py  —  Para el DIRECTOR. Correr desde la raíz del repo.

Descarga automáticamente el crudo comprimido (nyswcb_claims_full.parquet, ~128 MB)
desde Google Drive y lo reconstruye como CSV en raw_data/nyswcb_claims.csv (~2,1 GB),
que es donde 01_build_dataset.py lo busca. Después, el pipeline corre sin cambios.

Uso:
    pip install duckdb gdown
    python reconstruir_crudo.py

Luego (reproducir desde el crudo):
    python src/01_build_dataset.py          # regenera el dataset analítico (1,59M filas)
    python src/08_retrain_v3_gpu.py         # modelo final v3
    python src/10_cap7_shap.py              # SHAP
    python src/11_montecarlo_cap8.py        # impacto económico
    python src/14_fairness_audit.py         # equidad
    python src/15_fairness_calibration.py   # calibración por quintil
    # (03_benchmark y 07_optuna son opcionales, pesados y requieren GPU)
"""
import os

# Base = carpeta de este script (raíz del repo). Coincide con el BASE_DIR de los scripts de src/.
BASE     = os.environ.get("TESIS_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
FILE_ID  = "11ADu-u6xZz3JprnpTJr2sOdWe8om596B"   # nyswcb_claims_full.parquet en Google Drive
PARQUET  = os.path.join(BASE, "nyswcb_claims_full.parquet")
OUT_CSV  = os.path.join(BASE, "raw_data", "nyswcb_claims.csv")

try:
    import duckdb, gdown
except ImportError:
    raise SystemExit("Falta una dependencia. Instalá con:  pip install duckdb gdown")

if not os.path.exists(PARQUET):
    print("Descargando el crudo comprimido desde Google Drive (~128 MB)...")
    gdown.download(id=FILE_ID, output=PARQUET, quiet=False)

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
print(f"Reconstruyendo {OUT_CSV} ...")
duckdb.sql(f"COPY (SELECT * FROM '{PARQUET}') TO '{OUT_CSV}' (HEADER, DELIMITER ',')")

mb = os.path.getsize(OUT_CSV) / 1e6
print(f"Listo: {OUT_CSV}  ({mb:.0f} MB). Ya podés correr:  python src/01_build_dataset.py")
