"""api/predictor.py — Carga modelo y ejecuta predicciones."""
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = Path("artifacts/modelo.pkl")
METRICS_PATH = Path("artifacts/metrics.json")
FEATURES_PATH = Path("artifacts/feature_names.json")

UMBRAL = float(os.getenv("UMBRAL_PREDICCION", "0.50"))

FEATURES = [
    "LINEA_RENOVADO",
    "PLAZO_RENOVADO",
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


class Predictor:
    def __init__(self):
        self.modelo = None
        self.metricas = {}
        self.features = FEATURES

    def cargar(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            self.modelo = pickle.load(f)

        if METRICS_PATH.exists():
            with open(METRICS_PATH, encoding="utf-8") as f:
                self.metricas = json.load(f)

        if FEATURES_PATH.exists():
            with open(FEATURES_PATH, encoding="utf-8") as f:
                self.features = json.load(f)

        if not hasattr(self.modelo, "predict_proba"):
            raise TypeError("El modelo cargado no tiene predict_proba().")

        log.info("Modelo cargado: %s", type(self.modelo).__name__)

    def _construir_dataframe(self, datos: dict) -> pd.DataFrame:
        faltantes = [f for f in self.features if f not in datos]

        if faltantes:
            raise ValueError(f"Faltan variables requeridas: {faltantes}")

        return pd.DataFrame([{f: datos[f] for f in self.features}])

    def predecir(self, datos: dict) -> dict:
        if self.modelo is None:
            raise RuntimeError("Modelo no cargado.")

        X = self._construir_dataframe(datos)
        proba = float(self.modelo.predict_proba(X)[0][1])

        return {
            "probabilidad_venta": round(proba, 4),
            "decision": "RENOVAR" if proba >= UMBRAL else "NO RENOVAR",
            "umbral_usado": UMBRAL,
            "modelo": type(self.modelo).__name__,
        }


predictor = Predictor()