#!/usr/bin/env python3
"""
Entrena un modelo ligero (Random Forest) para probabilidad de movimiento BTC (up/down).
Salida:
- models/modelo1_rf.pkl
- models/modelo1_rf.meta.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib


def _load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=[0], index_col=0)


def _make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "close" not in df.columns:
        raise ValueError("El dataset debe contener columna 'close'.")
    future_ret = df["close"].pct_change().shift(-1)
    y = (future_ret > 0).astype(int)
    X = df.select_dtypes(include=[np.number]).copy().iloc[:-1]
    y = y.iloc[:-1]
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def _best_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    from l2_tactic.utils import safe_float
    return safe_float(best_t)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)

    train_df = _load_dataset(str(data_dir / "btc_features_train.csv"))
    test_df = _load_dataset(str(data_dir / "btc_features_test.csv"))

    X_train, y_train = _make_xy(train_df)
    X_test, y_test = _make_xy(test_df)

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    y_pred_05 = (proba_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_05)
    f1 = f1_score(y_test, y_pred_05)
    auc = roc_auc_score(y_test, proba_test)

    thr = _best_threshold_by_f1(y_test.values, proba_test)
    f1_thr = f1_score(y_test, (proba_test >= thr).astype(int))

    joblib.dump(clf, models_dir / "modelo1_rf.pkl")
    meta = {
        "features": list(X_train.columns),
        "threshold": thr,
        "metrics": {"accuracy@0.5": acc, "f1@0.5": f1, "auc": auc, "f1@thr": f1_thr},
    }
    with open(models_dir / "modelo1_rf.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("RandomForest - métricas:", {"accuracy": acc, "f1": f1, "auc": auc, "f1@thr": f1_thr, "thr": thr})


if __name__ == "__main__":
    main()
