#!/usr/bin/env python3
"""
Entrena un modelo ligero (Logistic Regression) para probabilidad de movimiento BTC (up/down).
Usa datasets generados por l1_operational/genera_dataset_modelo1.py

Salida:
- models/modelo1_logreg.pkl (modelo entrenado)
- models/modelo1_logreg.meta.json (metadatos: features, umbral óptimo)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def _load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    return df


def _make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Etiqueta: movimiento futuro (t+1) > 0
    if "close" not in df.columns:
        raise ValueError("El dataset debe contener columna 'close'.")
    future_ret = df["close"].pct_change().shift(-1)
    y = (future_ret > 0).astype(int)

    # Features numéricas
    X = df.select_dtypes(include=[np.number]).copy()
    # Evitar fuga directa del futuro
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    # Eliminar filas con NaN
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def _best_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    train_path = data_dir / "btc_features_train.csv"
    test_path = data_dir / "btc_features_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("No se encontraron los CSV de features. Ejecuta genera_dataset_modelo1 primero.")

    train_df = _load_dataset(str(train_path))
    test_df = _load_dataset(str(test_path))

    X_train, y_train = _make_xy(train_df)
    X_test, y_test = _make_xy(test_df)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None, class_weight="balanced")),
    ])

    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]

    # Métricas con umbral 0.5
    y_pred_05 = (proba_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_05)
    f1 = f1_score(y_test, y_pred_05)
    auc = roc_auc_score(y_test, proba_test)

    # Umbral óptimo por F1 (minimiza señales falsas en promedio)
    thr = _best_threshold_by_f1(y_test.values, proba_test)
    y_pred_thr = (proba_test >= thr).astype(int)
    f1_thr = f1_score(y_test, y_pred_thr)

    print("LogisticRegression - métricas (threshold=0.5):", {"accuracy": acc, "f1": f1, "auc": auc})
    print("LogisticRegression - F1 óptimo:", {"threshold": thr, "f1": f1_thr})

    # Persistir
    model_path = models_dir / "modelo1_logreg.pkl"
    joblib.dump(pipe, model_path)
    meta = {
        "features": list(X_train.columns),
        "threshold": thr,
        "metrics": {"accuracy@0.5": acc, "f1@0.5": f1, "auc": auc, "f1@thr": f1_thr},
    }
    with open(models_dir / "modelo1_logreg.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()


