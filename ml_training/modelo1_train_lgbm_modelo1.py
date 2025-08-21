#!/usr/bin/env python3
"""
Entrena un modelo ligero (LightGBM) para probabilidad de movimiento BTC (up/down).
Salida:
- models/modelo1_lgbm.txt (modelo)
- models/modelo1_lgbm.meta.json (metadatos)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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
    return float(best_t)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Buscando datos en: {data_dir}")
    print(f"📁 Guardando modelos en: {models_dir}")
    
    # Verificar que los archivos existen
    train_path = data_dir / "btc_features_train.csv"
    test_path = data_dir / "btc_features_test.csv"
    
    if not train_path.exists():
        raise FileNotFoundError(f"❌ No se encuentra: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"❌ No se encuentra: {test_path}")
    
    print(f"✅ Archivo entrenamiento: {train_path}")
    print(f"✅ Archivo test: {test_path}")

    train_df = _load_dataset(str(train_path))
    test_df = _load_dataset(str(test_path))
    
    print(f"📊 Datos entrenamiento: {train_df.shape}")
    print(f"📊 Datos test: {test_df.shape}")

    X_train, y_train = _make_xy(train_df)
    X_test, y_test = _make_xy(test_df)
    
    print(f"🎯 X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"🎯 X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"📈 Distribución y_train: {y_train.value_counts().to_dict()}")

    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    params = {
        "objective": "binary",
        "metric": ["auc"],
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbose": -1,
        "scale_pos_weight": float((y_train == 0).sum() / max(1, (y_train == 1).sum())),
    }
    
    print(f"⚖️ Scale pos weight: {params['scale_pos_weight']:.3f}")

    # ✅ VERSIÓN CORREGIDA - Compatible con LightGBM 4.0+
    try:
        # Método moderno (LightGBM 4.0+)
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        print("✅ Usando callbacks (LightGBM 4.0+)")
        
    except Exception as e:
        print(f"⚠️ Callbacks falló, usando método legacy: {e}")
        # Método legacy (LightGBM 3.x)
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

    print(f"🏆 Mejor iteración: {booster.best_iteration}")

    proba_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    y_pred_05 = (proba_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred_05)
    f1 = f1_score(y_test, y_pred_05)
    auc = roc_auc_score(y_test, proba_test)

    thr = _best_threshold_by_f1(y_test.values, proba_test)
    f1_thr = f1_score(y_test, (proba_test >= thr).astype(int))

    model_path = models_dir / "modelo1_lgbm.txt"
    booster.save_model(str(model_path))
    print(f"💾 Modelo guardado: {model_path}")

    meta = {
        "features": list(X_train.columns),
        "threshold": thr,
        "best_iteration": int(booster.best_iteration or 0),
        "metrics": {"accuracy@0.5": acc, "f1@0.5": f1, "auc": auc, "f1@thr": f1_thr},
        "lightgbm_version": lgb.__version__,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(X_train.columns),
    }
    
    meta_path = models_dir / "modelo1_lgbm.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📋 Metadatos guardados: {meta_path}")

    print("\n🎉 RESULTADOS FINALES:")
    print(f"   📊 Accuracy@0.5: {acc:.4f}")
    print(f"   🎯 F1@0.5: {f1:.4f}")
    print(f"   📈 AUC: {auc:.4f}")
    print(f"   ⚡ F1@{thr:.3f}: {f1_thr:.4f}")
    print(f"   🔧 LightGBM: {lgb.__version__}")
    
    # ✅ Mostrar importancia de features (top 10)
    importance = booster.feature_importance(importance_type='gain')
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 TOP 10 FEATURES MÁS IMPORTANTES:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {i+1:2d}. {row['feature']:<20} {row['importance']:>8.0f}")


if __name__ == "__main__":
    main()