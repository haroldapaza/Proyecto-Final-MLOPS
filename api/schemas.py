"""api/schemas.py — Modelos Pydantic para Renovación de Préstamo."""
from typing import Literal

from pydantic import BaseModel, Field


class RenovacionInput(BaseModel):
    LINEA_RENOVADO: float = Field(..., ge=0)
    PLAZO_RENOVADO: int = Field(..., ge=1)
    USO_LINEA_TOTAL_TC_T2: float = Field(..., ge=0)
    USO_TRIM_LINEA_BBVA: float = Field(..., ge=0)
    NR_ENTIDADES_TOTAL_T2: int = Field(..., ge=0)
    DIFF_NRO_ENTIDA_TOTALES_T2_T12: float
    SDO_CONSUMO_T2: float = Field(..., ge=0)
    RESENCIA_OFERTA_PLD_RENOVADO: int = Field(..., ge=0)
    Ahorro_Sldo_Bco_T1: float = Field(..., ge=0)
    PConsumo_Sldo_Bco_T1: float = Field(..., ge=0)
    SDO_BCO_tot_sm_pasivo_Bco_6M: float = Field(..., ge=0)
    EDAD: int = Field(..., ge=18, le=100)
    SEXO: str
    EST_CIVIL: str
    ANTIGUEDAD_MES: int = Field(..., ge=0)
    REGION: str
    FLAG_LIMA_PROVINCIA: int = Field(..., ge=0, le=1)
    SUELDO_ESTIMADO: float = Field(..., ge=0)
    CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "LINEA_RENOVADO": 3770,
                "PLAZO_RENOVADO": 12,
                "USO_LINEA_TOTAL_TC_T2": 0,
                "USO_TRIM_LINEA_BBVA": 0,
                "NR_ENTIDADES_TOTAL_T2": 1,
                "DIFF_NRO_ENTIDA_TOTALES_T2_T12": -1,
                "SDO_CONSUMO_T2": 271.12,
                "RESENCIA_OFERTA_PLD_RENOVADO": 21,
                "Ahorro_Sldo_Bco_T1": 1850,
                "PConsumo_Sldo_Bco_T1": 457,
                "SDO_BCO_tot_sm_pasivo_Bco_6M": 3754.333333,
                "EDAD": 25,
                "SEXO": "M",
                "EST_CIVIL": "S",
                "ANTIGUEDAD_MES": 23,
                "REGION": "LIMA NORTE",
                "FLAG_LIMA_PROVINCIA": 1,
                "SUELDO_ESTIMADO": 3531,
                "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 0.071915,
            }
        }
    }


class PrediccionOutput(BaseModel):
    probabilidad_venta: float = Field(..., ge=0, le=1)
    decision: Literal["RENOVAR", "NO RENOVAR"]
    umbral_usado: float = Field(..., ge=0, le=1)
    modelo: str


class HealthResponse(BaseModel):
    status: str
    modelo: str
    version: str
    recall: float = Field(..., ge=0, le=1)
    env: str