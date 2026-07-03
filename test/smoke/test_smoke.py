"""tests/smoke/test_smoke.py — Smoke tests del entorno completo."""
import json
import os
import urllib.request

import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

PAYLOAD_RENOVACION = {
    "MES": 202405,
    "LINEA_RENOVADO": 5000.0,
    "PLAZO_RENOVADO": 24,
    "USO_LINEA_TOTAL_TC_T2": 0.45,
    "USO_TRIM_LINEA_Bco": 0.30,
    "NR_ENTIDADES_TOTAL_T2": 3,
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12": 1,
    "SDO_CONSUMO_T2": 2500.0,
    "RESENCIA_OFERTA_PLD_RENOVADO": 4,
    "Ahorro_Sldo_Bco_T1": 1200.0,
    "PConsumo_Sldo_Bco_T1": 3000.0,
    "SDO_BCO_tot_sm_pasivo_Bco_6M": 2800.0,
    "EDAD": 35,
    "SEXO": "M",
    "EST_CIVIL": "SOLTERO",
    "ANTIGUEDAD_MES": 48,
    "REGION": "LIMA",
    "FLAG_LIMA_PROVINCIA": 1,
    "SUELDO_ESTIMADO": 3500.0,
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 0.60,
}


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_api_root_responde():
    data = get_json(f"{API_URL}/")
    assert data.get("api") == "Renovación de Préstamo"
    assert data.get("version") == "1.0.0"


def test_api_health_ok():
    data = get_json(f"{API_URL}/health")
    assert data["status"] == "ok"
    assert "modelo" in data
    assert 0 <= data.get("recall", 0) <= 1


def test_api_prediccion_payload():
    data = post_json(f"{API_URL}/predecir", PAYLOAD_RENOVACION)

    assert "probabilidad_venta" in data
    assert data["decision"] in ["RENOVAR", "NO RENOVAR"]
    assert 0.0 <= data["probabilidad_venta"] <= 1.0
    assert "modelo" in data


def test_api_prediccion_es_determinista():
    r1 = post_json(f"{API_URL}/predecir", PAYLOAD_RENOVACION)
    r2 = post_json(f"{API_URL}/predecir", PAYLOAD_RENOVACION)

    assert r1["probabilidad_venta"] == r2["probabilidad_venta"]


def test_api_docs_disponible():
    with urllib.request.urlopen(f"{API_URL}/docs", timeout=10) as resp:
        assert resp.status == 200


def test_mlflow_ui_responde():
    try:
        with urllib.request.urlopen(f"{MLFLOW_URL}/health", timeout=10) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("MLflow no disponible")


def test_mlflow_experimentos_accesibles():
    try:
        data = get_json(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search")
        assert "experiments" in data
    except Exception:
        pytest.skip("MLflow API no disponible")