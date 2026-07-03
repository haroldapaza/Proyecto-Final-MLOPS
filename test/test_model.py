"""tests/test_model.py — Tests del modelo serializado."""
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from generate_data import load_dataset  # noqa: E402
from train_pipeline import FEATURES, train  # noqa: E402


@pytest.fixture(scope="module")
def modelo_entrenado(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("workdir")
    cwd = os.getcwd()

    try:
        os.chdir(tmp)

        df = load_dataset()
        train(df)

        with open(tmp / "artifacts" / "modelo.pkl", "rb") as f:
            modelo = pickle.load(f)

        return modelo

    finally:
        os.chdir(cwd)


def test_model_has_predict(modelo_entrenado):
    assert hasattr(modelo_entrenado, "predict")


def test_model_has_predict_proba(modelo_entrenado):
    assert hasattr(modelo_entrenado, "predict_proba")


def test_model_predicts_binary(modelo_entrenado):
    df = load_dataset().head(20)
    X = df[FEATURES]

    y_pred = modelo_entrenado.predict(X)

    assert set(y_pred).issubset({0, 1})


def test_model_predict_proba_shape(modelo_entrenado):
    df = load_dataset().head(10)
    X = df[FEATURES]

    proba = modelo_entrenado.predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_model_predict_proba_sums_to_one(modelo_entrenado):
    df = load_dataset().head(20)
    X = df[FEATURES]

    proba = modelo_entrenado.predict_proba(X)

    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)