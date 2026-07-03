"""src/generate_data.py — Carga y valida dataset de renovación."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/Dataset Renovacion_prestamo.csv")

REQUIRED_COLUMNS = [
    "LINEA_RENOVADO",
    "PLAZO_RENOVADO",
    "FLAG_VENTA",
    "USO_LINEA_TOTAL_TC_T2",
    "USO_TRIM_LINEA_BBVA",
    "NR_ENTIDADES_TOTAL_T2",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12",
    "SDO_CONSUMO_T2",
    "RESENCIA_OFERTA_PLD_RENOVADO",
    "Ahorro_Sldo_Bco_T1",
    "PConsumo_Sldo_Bco_T1",
    "SDO_BCO_tot_sm_pasivo_Bco_6M",
    "EDAD",
    "SEXO",
    "EST_CIVIL",
    "ANTIGUEDAD_MES",
    "REGION",
    "FLAG_LIMA_PROVINCIA",
    "SUELDO_ESTIMADO",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Lee CSV probando separadores comunes."""
    separadores = [",", ";", "\t", "|"]

    mejor_df = None
    mejor_ncols = 0
    mejor_sep = None

    for sep in separadores:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            df = normalize_columns(df)

            if len(df.columns) > mejor_ncols:
                mejor_df = df
                mejor_ncols = len(df.columns)
                mejor_sep = sep

        except Exception:
            continue

    if mejor_df is None:
        raise ValueError("No se pudo leer el archivo CSV.")

    print(f"Separador detectado: {repr(mejor_sep)} | columnas: {mejor_ncols}")
    return mejor_df


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")

    df = read_csv_auto(path)

    faltantes = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if faltantes:
        print("Columnas encontradas:")
        print(df.columns.tolist())
        raise ValueError(f"Columnas faltantes en el dataset: {faltantes}")

    return df


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset cargado: {df.shape}")
    print(f"Tasa FLAG_VENTA=1: {df['FLAG_VENTA'].mean():.2%}")