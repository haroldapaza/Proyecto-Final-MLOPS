"""tests/test_pipeline.py — Tests del pipeline de entrenamiento."""
import json
import sys

sys.path.insert(0, "src")

from generate_data import load_dataset  # noqa: E402
from train_pipeline import train  # noqa: E402


def test_train_returns_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = load_dataset()
    metricas = train(df)

    assert "f1" in metricas
    assert "recall" in metricas
    assert "precision" in metricas
    assert "roc_auc" in metricas


def test_train_metrics_in_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = load_dataset()
    metricas = train(df)

    for key in ("f1", "recall", "precision", "accuracy", "roc_auc"):
        assert 0.0 <= metricas[key] <= 1.0


def test_train_saves_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = load_dataset()
    train(df)

    assert (tmp_path / "artifacts" / "modelo.pkl").exists()


def test_train_saves_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = load_dataset()
    train(df)

    metrics_file = tmp_path / "artifacts" / "metrics.json"

    assert metrics_file.exists()

    with open(metrics_file, encoding="utf-8") as f:
        m = json.load(f)

    assert "recall" in m
    assert "f1" in m
    assert "precision" in m
    assert "roc_auc" in m


def test_train_saves_feature_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = load_dataset()
    train(df)

    assert (tmp_path / "artifacts" / "feature_names.json").exists()