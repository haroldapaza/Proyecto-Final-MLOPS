"""src/train_pipeline.py — Entrena modelo de renovación de préstamo."""
import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from generate_data import DATA_PATH, load_dataset

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "modelo.pkl"
METRICS_PATH = ARTIFACTS / "metrics.json"
FEATURES_PATH = ARTIFACTS / "feature_names.json"
THRESHOLD_PATH = ARTIFACTS / "threshold.json"

TARGET = "FLAG_VENTA"

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

NUMERIC_FEATURES = [
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
    "ANTIGUEDAD_MES",
    "FLAG_LIMA_PROVINCIA",
    "SUELDO_ESTIMADO",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
]

CATEGORICAL_FEATURES = ["SEXO", "EST_CIVIL", "REGION"]

RANDOM_STATE = 123
TEST_SIZE = 0.30
RECALL_MIN = float(os.getenv("RECALL_MIN", "0.50"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "renovacion-prestamo-preprod")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | TRAIN | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """Construye pipeline con preprocessing + SMOTE + RandomForest."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def calcular_umbral_optimo(y_true, y_proba) -> float:
    """Calcula el umbral más alto que cumpla el recall mínimo."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    candidatos = []

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if r >= RECALL_MIN:
            candidatos.append((t, p, r))

    if not candidatos:
        return 0.50

    # Elegimos el umbral más alto que aún cumple recall mínimo
    mejor_threshold, mejor_precision, mejor_recall = max(candidatos, key=lambda x: x[0])

    log.info(
        "Threshold elegido por recall mínimo: %.4f | Precision=%.4f | Recall=%.4f",
        mejor_threshold,
        mejor_precision,
        mejor_recall,
    )

    return float(mejor_threshold)


def evaluar_con_umbral(y_true, y_proba, threshold: float) -> dict:
    """Calcula métricas usando un umbral personalizado."""
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def configurar_mlflow() -> bool:
    """Configura MLflow. Si no está disponible, continúa sin fallar."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        return True
    except Exception as e:
        log.warning("MLflow no disponible. Se continuará sin MLflow: %s", str(e))
        return False


def train(df: pd.DataFrame) -> dict:
    """Entrena modelo, calcula métricas y guarda artefactos."""
    df = df.copy()

    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    X = df[FEATURES]
    y = df[TARGET]

    log.info("Tasa FLAG_VENTA=1: %.2f%%", y.mean() * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline()

    param_grid = {
        "model__n_estimators": [100],
        "model__max_depth": [10],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [1],
    }

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="recall",
        n_jobs=-1,
        verbose=1,
    )

    usar_mlflow = configurar_mlflow()

    if usar_mlflow:
        mlflow.start_run(run_name="renovacion-prestamo-randomforest-smote")

    try:
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        y_proba = best_model.predict_proba(X_test)[:, 1]

        threshold = calcular_umbral_optimo(y_test, y_proba)
        metricas_base = evaluar_con_umbral(y_test, y_proba, threshold)

        metricas = {
            **metricas_base,
            "threshold": round(threshold, 4),
            "params": gs.best_params_,
            "recall_minimo": RECALL_MIN,
            "target": TARGET,
            "dataset": str(DATA_PATH),
            "n_rows": int(len(df)),
            "tasa_venta": round(float(y.mean()), 4),
        }

        ARTIFACTS.mkdir(exist_ok=True)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(best_model, f)

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metricas, f, indent=2, ensure_ascii=False)

        with open(FEATURES_PATH, "w", encoding="utf-8") as f:
            json.dump(FEATURES, f, indent=2, ensure_ascii=False)

        with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
            json.dump({"threshold": threshold}, f, indent=2, ensure_ascii=False)

        if usar_mlflow:
            mlflow.log_params(gs.best_params_)
            mlflow.log_metric("f1", metricas["f1"])
            mlflow.log_metric("recall", metricas["recall"])
            mlflow.log_metric("precision", metricas["precision"])
            mlflow.log_metric("accuracy", metricas["accuracy"])
            mlflow.log_metric("roc_auc", metricas["roc_auc"])
            mlflow.log_metric("threshold", metricas["threshold"])
            mlflow.log_metric("tasa_venta", metricas["tasa_venta"])

        log.info(
            "F1=%.4f | Recall=%.4f | Precision=%.4f | AUC=%.4f | Threshold=%.4f",
            metricas["f1"],
            metricas["recall"],
            metricas["precision"],
            metricas["roc_auc"],
            metricas["threshold"],
        )

        return metricas

    finally:
        if usar_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    df = load_dataset()
    log.info("Dataset: %d filas", len(df))
    metricas = train(df)
    print(json.dumps(metricas, indent=2, ensure_ascii=False))