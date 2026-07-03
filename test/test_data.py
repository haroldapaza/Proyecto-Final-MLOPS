"""tests/test_data.py — Tests del dataset de renovación."""
import sys

import pandas as pd
import pytest

sys.path.insert(0, "src")

from generate_data import REQUIRED_COLUMNS, load_dataset  # noqa: E402


def test_dataset_loads():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)


def test_dataset_has_required_columns():
    df = load_dataset()
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Columna faltante: {col}"


def test_target_is_binary():
    df = load_dataset()
    assert set(df["FLAG_VENTA"].dropna().unique()).issubset({0, 1})


def test_dataset_has_rows():
    df = load_dataset()
    assert len(df) > 0


def test_no_empty_target():
    df = load_dataset()
    assert df["FLAG_VENTA"].isnull().sum() == 0


def test_cliente_is_not_empty():
    df = load_dataset()
    assert df["CLIENTE"].isnull().sum() == 0