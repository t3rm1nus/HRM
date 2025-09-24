#!/usr/bin/env python3
"""
Entrena un modelo Logistic Regression para múltiples activos (BTC, ETH).
Usa datasets generados y los combina en un modelo único multiasset.

Salida:
- models/modelo1_logreg_multiasset.pkl (modelo entrenado)
- models/modelo1_logreg_multiasset.meta.json (metadatos con métricas por símbolo)
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def _load_dataset(path: str) -> pd.DataFrame:
    """Carga dataset con manejo de múltiples formatos de fecha."""
    try:
        return pd.read_csv(path, parse_dates=[0], index_col=0)
    except:
        # Fallback para diferentes formatos
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.set_index('timestamp')
        return df


def _prepare_multiasset_features(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combina features de BTC y ETH en un dataset unificado.
    Añade columnas de símbolo y features cruzadas.
    """
    # Añadir identificador de símbolo
    btc_df = btc_df.copy()
    eth_df = eth_df.copy()
    
    btc_df['symbol'] = 'BTC'
    eth_df['symbol'] = 'ETH'
    
    # Combinar datasets
    combined = pd.concat([btc_df, eth_df], axis=0).sort_index()
    
    # Features cruzadas (correlación BTC-ETH)
    if 'close' in btc_df.columns and 'close' in eth_df.columns:
        btc_close = btc_df['close'].reindex(combined.index, method='ffill')
        eth_close = eth_df['close'].reindex(combined.index, method='ffill')
        
        # Ratio ETH/BTC
        combined['eth_btc_ratio'] = eth_close / btc_close
        combined['eth_btc_ratio_sma'] = combined['eth_btc_ratio'].rolling(20).mean()
        
        # Correlación rolling
        combined['btc_eth_corr'] = btc_close.rolling(50).corr(eth_close)
    
    # Encoding de símbolo (one-hot)
    combined['is_btc'] = (combined['symbol'] == 'BTC').astype(int)
    combined['is_eth'] = (combined['symbol'] == 'ETH').astype(int)
    
    return combined


def _make_xy_multiasset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Genera X,y para múltiples activos con target específico por símbolo.
    """
    if "close" not in df.columns:
        raise ValueError("El dataset debe contener columna 'close'.")
    
    # Target por símbolo (movimiento futuro)
    df_sorted = df.sort_index()
    
    # Calcular retorno futuro por símbolo
    future_ret = pd.Series(index=df_sorted.index, dtype=float)
    
    for symbol in df_sorted['symbol'].unique():
        mask = df_sorted['symbol'] == symbol
        symbol_data = df_sorted.loc[mask, 'close']
        symbol_ret = symbol_data.pct_change().shift(-1)
        future_ret.loc[mask] = symbol_ret
    
    y = (future_ret > 0).astype(int)
    
    # Features (excluir target y metadatos)
    exclude_cols = ['close', 'symbol'] if 'symbol' in df.columns else ['close']
    X = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')
    
    # Filtrar datos válidos
    valid = X.notna().all(axis=1) & y.notna() & (y != -1)
    
    return X.loc[valid], y.loc[valid]


def _best_threshold_by_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Encuentra umbral óptimo por F1-score."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    from l2_tactic.utils import safe_float
    return safe_float(best_t)


def _evaluate_by_symbol(X_test: pd.DataFrame, y_test: pd.Series, 
                       y_proba: np.ndarray, pipeline: Pipeline, 
                       symbol_col: str = 'is_btc') -> Dict:
    """Evalúa rendimiento por símbolo."""
    results = {}
    
    # BTC
    btc_mask = X_test[symbol_col] == 1
    if btc_mask.sum() > 0:
        btc_proba = y_proba[btc_mask]
        btc_pred = (btc_proba >= 0.5).astype(int)
        btc_acc = accuracy_score(y_test[btc_mask], btc_pred)
        btc_f1 = f1_score(y_test[btc_mask], btc_pred)
        btc_auc = roc_auc_score(y_test[btc_mask], btc_proba) if len(y_test[btc_mask].unique()) > 1 else 0.5
        
        # Umbral óptimo para BTC
        btc_thr = _best_threshold_by_f1(y_test[btc_mask].values, btc_proba)
        btc_f1_thr = f1_score(y_test[btc_mask], (btc_proba >= btc_thr).astype(int))
        
        results['BTC'] = {
            'accuracy': float(btc_acc), 
            'f1': float(btc_f1), 
            'auc': float(btc_auc),
            'threshold_optimal': float(btc_thr),
            'f1_optimal': float(btc_f1_thr),
            'samples': int(btc_mask.sum())
        }
    
    # ETH
    eth_mask = X_test[symbol_col] == 0
    if eth_mask.sum() > 0:
        eth_proba = y_proba[eth_mask]
        eth_pred = (eth_proba >= 0.5).astype(int)
        eth_acc = accuracy_score(y_test[eth_mask], eth_pred)
        eth_f1 = f1_score(y_test[eth_mask], eth_pred)
        eth_auc = roc_auc_score(y_test[eth_mask], eth_proba) if len(y_test[eth_mask].unique()) > 1 else 0.5
        
        # Umbral óptimo para ETH
        eth_thr = _best_threshold_by_f1(y_test[eth_mask].values, eth_proba)
        eth_f1_thr = f1_score(y_test[eth_mask], (eth_proba >= eth_thr).astype(int))
        
        results['ETH'] = {
            'accuracy': float(eth_acc), 
            'f1': float(eth_f1), 
            'auc': float(eth_auc),
            'threshold_optimal': float(eth_thr),
            'f1_optimal': float(eth_f1_thr),
            'samples': int(eth_mask.sum())
        }
    
    return results


# Helper para convertir tipos numpy/pandas a JSON serializable
def make_json_serializable(obj):
    """Convierte tipos numpy/pandas a tipos nativos de Python."""
    if hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"🔍 Buscando datos en: {data_dir}")
    print(f"📁 Guardando modelos en: {models_dir}")
    
    # Buscar archivos de datos
    btc_train_path = data_dir / "btc_features_train.csv"
    btc_test_path = data_dir / "btc_features_test.csv"
    eth_train_path = data_dir / "eth_features_train.csv"
    eth_test_path = data_dir / "eth_features_test.csv"
    
    # Verificar archivos BTC
    if not btc_train_path.exists() or not btc_test_path.exists():
        raise FileNotFoundError(f"❌ Archivos BTC no encontrados en {data_dir}")
    
    print(f"✅ BTC train: {btc_train_path}")
    print(f"✅ BTC test: {btc_test_path}")
    
    # Cargar BTC
    btc_train = _load_dataset(str(btc_train_path))
    btc_test = _load_dataset(str(btc_test_path))
    
    # Verificar ETH (opcional)
    has_eth = eth_train_path.exists() and eth_test_path.exists()
    
    if has_eth:
        print(f"✅ ETH train: {eth_train_path}")
        print(f"✅ ETH test: {eth_test_path}")
        
        eth_train = _load_dataset(str(eth_train_path))
        eth_test = _load_dataset(str(eth_test_path))
        
        # Combinar datasets
        train_df = _prepare_multiasset_features(btc_train, eth_train)
        test_df = _prepare_multiasset_features(btc_test, eth_test)
        
        print(f"📊 Dataset combinado - Train: {train_df.shape}, Test: {test_df.shape}")
        print(f"📈 Distribución símbolos train: {train_df['symbol'].value_counts().to_dict()}")
        
    else:
        print("⚠️ Archivos ETH no encontrados, usando solo BTC")
        train_df = btc_train.copy()
        test_df = btc_test.copy()
        train_df['symbol'] = 'BTC'
        test_df['symbol'] = 'BTC'
        train_df['is_btc'] = 1
        test_df['is_btc'] = 1
    
    # Preparar datos para entrenamiento
    X_train, y_train = _make_xy_multiasset(train_df)
    X_test, y_test = _make_xy_multiasset(test_df)
    
    print(f"🎯 X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"🎯 X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"📈 Distribución y_train: {y_train.value_counts().to_dict()}")
    
    # Pipeline con StandardScaler + LogisticRegression
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            max_iter=2000, 
            n_jobs=-1,  # Usar todos los cores disponibles
            class_weight="balanced",
            random_state=42,
            solver='liblinear'  # Bueno para datasets medianos
        )),
    ])
    
    print("🎯 Entrenando Logistic Regression multiasset...")
    pipe.fit(X_train, y_train)
    
    # Predicciones
    proba_test = pipe.predict_proba(X_test)[:, 1]
    y_pred_05 = (proba_test >= 0.5).astype(int)
    
    # Métricas globales
    acc = accuracy_score(y_test, y_pred_05)
    f1 = f1_score(y_test, y_pred_05)
    auc = roc_auc_score(y_test, proba_test)
    
    # Umbral óptimo global
    thr = _best_threshold_by_f1(y_test.values, proba_test)
    y_pred_thr = (proba_test >= thr).astype(int)
    f1_thr = f1_score(y_test, y_pred_thr)
    
    print(f"📊 Métricas globales (threshold=0.5): Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    print(f"⚡ F1 óptimo: threshold={thr:.3f}, F1={f1_thr:.4f}")
    
    # Evaluación por símbolo
    symbol_results = {}
    if 'is_btc' in X_test.columns:
        symbol_results = _evaluate_by_symbol(X_test, y_test, proba_test, pipe)
        
        if symbol_results:
            print(f"\n📈 RESULTADOS POR SÍMBOLO:")
            for symbol, metrics in symbol_results.items():
                print(f"   {symbol}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
                      f"AUC={metrics['auc']:.3f}, Thr_opt={metrics['threshold_optimal']:.3f} "
                      f"({metrics['samples']} samples)")
    
    # Guardar modelo en formato PKL
    model_path_pkl = models_dir / "modelo1_logreg_multiasset.pkl"
    joblib.dump(pipe, model_path_pkl)
    print(f"💾 Modelo PKL guardado: {model_path_pkl}")
    
    # Metadatos extendidos
    meta = {
        "model_type": "multiasset_trend_filter_logistic",
        "assets": list(train_df['symbol'].unique()) if 'symbol' in train_df.columns else ['BTC'],
        "features": list(X_train.columns),
        "threshold": float(thr),
        "metrics": {
            "global": {
                "accuracy@0.5": float(acc),
                "f1@0.5": float(f1),
                "auc": float(auc),
                "f1@thr": float(f1_thr)
            },
            "by_symbol": make_json_serializable(symbol_results)
        },
        "model_params": {
            "algorithm": "LogisticRegression",
            "solver": pipe.named_steps['clf'].solver,
            "class_weight": pipe.named_steps['clf'].class_weight,
            "max_iter": pipe.named_steps['clf'].max_iter,
            "standardized": True
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_features": int(len(X_train.columns)),
        "multiasset": bool(has_eth),
        "usage_notes": {
            "l1_integration": "Usar con threshold óptimo para filtrar señales",
            "input_format": "Requiere is_btc/is_eth y features normalizados por StandardScaler",
            "output": "Probabilidad binaria (0-1) de movimiento alcista",
            "loading": "joblib.load('modelo1_logreg_multiasset.pkl')"
        }
    }
    
    meta_path = models_dir / "modelo1_logreg_multiasset.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📋 Metadatos guardados: {meta_path}")
    
    # Mostrar features más importantes (coeficientes)
    coefficients = pipe.named_steps['clf'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(f"\n🏆 TOP 10 FEATURES MÁS IMPORTANTES (Logistic Regression):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        direction = "📈" if row['coefficient'] > 0 else "📉"
        print(f"   {i+1:2d}. {row['feature']:<25} {direction} {row['coefficient']:>8.4f}")
    
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO:")
    print(f"   📦 Modelo: {model_path_pkl}")
    print(f"   📋 Metadatos: {meta_path}")
    print(f"   🎯 Uso en L1: Cargar con joblib.load() y aplicar threshold={thr:.3f}")


if __name__ == "__main__":
    main()
