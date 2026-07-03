"""src/validate_model.py — Valida que las métricas superen el umbral mínimo."""
import json
import sys
from pathlib import Path

METRICS_PATH = Path("artifacts/metrics.json")


def validate(metrics_path: Path = METRICS_PATH) -> None:
    """Valida métricas. Lanza SystemExit(1) si falla."""
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} no existe. Ejecuta train_pipeline.py primero.")
        sys.exit(1)

    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    umbral = float(m.get("recall_minimo", 0.70))
    recall = float(m.get("recall", 0.0))

    print("=" * 50)
    print("QUALITY GATE — RENOVACIÓN DE PRÉSTAMO")
    print("=" * 50)
    print(f"Recall   : {recall:.4f}  umbral >= {umbral}")
    print(f"F1       : {m.get('f1', 0):.4f}")
    print(f"Precision: {m.get('precision', 0):.4f}")
    print(f"ROC AUC  : {m.get('roc_auc', 0):.4f}")

    if recall < umbral:
        print(f"\nFALLO: Recall {recall:.4f} < {umbral}")
        sys.exit(1)

    print(f"\nAPROBADO: Recall {recall:.4f} >= {umbral}")


if __name__ == "__main__":
    validate()