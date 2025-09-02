#!/usr/bin/env python3
"""
Entrena un modelo LightGBM para múltiples activos (BTC, ETH) con targets alineados al trading.

Salida:
- models/L1/modelo3_lgbm.pkl (modelo entrenado con target de trading)
- models/L1/modelo3_lgbm.meta.json (metadatos con métricas)
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib


def _load_dataset(path: str) -> pd.DataFrame:
    """Carga dataset con manejo de múltiples formatos de fecha."""
    try:
        return pd.read_csv(path, parse_dates=[0], index_col=0)
    except:
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.set_index('timestamp')
        return df


def _prepare_multiasset_features(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """Combina features de BTC y ETH en un dataset unificado."""
    btc_df = btc_df.copy()
    eth_df = eth_df.copy()
    
    btc_df['symbol'] = 'BTC'
    eth_df['symbol'] = 'ETH'
    
    combined = pd.concat([btc_df, eth_df], axis=0).sort_index()
    
    # Features cruzadas (correlación BTC-ETH)
    if 'close' in btc_df.columns and 'close' in eth_df.columns:
        btc_close = btc_df['close'].reindex(combined.index, method='ffill')
        eth_close = eth_df['close'].reindex(combined.index, method='ffill')
        
        combined['eth_btc_ratio'] = eth_close / btc_close
        combined['eth_btc_ratio_sma'] = combined['eth_btc_ratio'].rolling(20).mean()
        combined['btc_eth_corr'] = btc_close.rolling(50).corr(eth_close)
    
    # Encoding de símbolo
    combined['is_btc'] = (combined['symbol'] == 'BTC').astype(int)
    combined['is_eth'] = (combined['symbol'] == 'ETH').astype(int)
    
    return combined


def create_trading_aligned_target(df: pd.DataFrame, 
                                stop_loss_pct: float = 0.02,
                                take_profit_pct: float = 0.04,
                                max_hold_periods: int = 60,
                                transaction_cost: float = 0.001
                               ) -> pd.Series:
    """Crea target alineado con objetivos reales de trading."""
    closes = df['close'].values
    target = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - max_hold_periods):
        entry_price = closes[i]
        stop_price = entry_price * (1 - stop_loss_pct)
        profit_price = entry_price * (1 + take_profit_pct)
        
        exit_price = None
        
        for j in range(i + 1, min(i + max_hold_periods + 1, len(df))):
            current_price = closes[j]
            
            if current_price <= stop_price:
                exit_price = stop_price
                break
            if current_price >= profit_price:
                exit_price = profit_price
                break
        
        if exit_price is None:
            exit_price = closes[min(i + max_hold_periods, len(df) - 1)]
        
        raw_return = (exit_price - entry_price) / entry_price
        net_return = raw_return - (2 * transaction_cost)
        
        target[i] = 1 if net_return > 0 else 0
    
    return pd.Series(target, index=df.index)


def create_l1_filter_target(df: pd.DataFrame,
                           success_threshold: float = 0.008  # 0.8% ganancia mínima (más realista)
                          ) -> pd.Series:
    """Target específico para L1: ¿Vale la pena ejecutar una señal BUY aquí?"""
    closes = df['close'].values
    target = np.zeros(len(df), dtype=int)
    
    # Mejorar generación de señales L2 simuladas
    if 'rsi' in df.columns:
        rsi = df['rsi'].values
        # RSI oversold más flexible (no solo <30)
        l2_signals = ((rsi < 35) | (rsi > 65)).astype(int)  # Oversold OR overbought
    else:
        # Fallback más balanceado
        returns = pd.Series(closes).pct_change()
        volatility = returns.rolling(20).std()
        # Señales en caídas significativas O subidas fuertes
        strong_moves = (returns < -volatility * 1.5) | (returns > volatility * 1.5)
        l2_signals = strong_moves.fillna(0).astype(int)
    
    # Contar señales generadas para debug
    signal_count = 0
    successful_signals = 0
    
    for i in range(len(df) - 20):
        if l2_signals[i] == 1:  # Solo cuando L2 genera señal
            signal_count += 1
            entry_price = closes[i]
            max_return = -float('inf')
            
            # Evaluar ventana de salida más realista
            for j in range(i + 1, min(i + 15, len(df))):  # Reducido a 15 períodos
                current_return = (closes[j] - entry_price) / entry_price
                max_return = max(max_return, current_return)
            
            if max_return >= success_threshold:
                target[i] = 1
                successful_signals += 1
            else:
                target[i] = 0
    
    # Debug info
    success_rate = successful_signals / signal_count if signal_count > 0 else 0
    print(f"📊 L2 signals simuladas: {signal_count}, Exitosas: {successful_signals} ({success_rate:.1%})")
    
    return pd.Series(target, index=df.index)


def _make_xy_trading_aligned(df: pd.DataFrame, target_type: str = 'trading_pnl') -> Tuple[pd.DataFrame, pd.Series]:
    """Genera X,y con targets alineados al trading."""
    
    if target_type == 'trading_pnl':
        y = create_trading_aligned_target(df)
    elif target_type == 'l1_filter':
        y = create_l1_filter_target(df)
    else:
        raise ValueError(f"Tipo de target desconocido: {target_type}")
    
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
    return float(best_t)


def _evaluate_by_symbol(X_test: pd.DataFrame, y_test: pd.Series, 
                       y_proba: np.ndarray, symbol_col: str = 'is_btc') -> Dict:
    """Evalúa rendimiento por símbolo."""
    results = {}
    
    if symbol_col not in X_test.columns:
        return results
    
    # BTC
    btc_mask = X_test[symbol_col] == 1
    if btc_mask.sum() > 0:
        btc_proba = y_proba[btc_mask]
        btc_pred = (btc_proba >= 0.5).astype(int)
        btc_acc = accuracy_score(y_test[btc_mask], btc_pred)
        btc_f1 = f1_score(y_test[btc_mask], btc_pred, zero_division=0) if len(y_test[btc_mask].unique()) > 1 else 0
        btc_auc = roc_auc_score(y_test[btc_mask], btc_proba) if len(y_test[btc_mask].unique()) > 1 else 0.5
        
        btc_thr = _best_threshold_by_f1(y_test[btc_mask].values, btc_proba)
        btc_f1_thr = f1_score(y_test[btc_mask], (btc_proba >= btc_thr).astype(int), zero_division=0) if len(y_test[btc_mask].unique()) > 1 else 0
        
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
        eth_f1 = f1_score(y_test[eth_mask], eth_pred, zero_division=0) if len(y_test[eth_mask].unique()) > 1 else 0
        eth_auc = roc_auc_score(y_test[eth_mask], eth_proba) if len(y_test[eth_mask].unique()) > 1 else 0.5
        
        eth_thr = _best_threshold_by_f1(y_test[eth_mask].values, eth_proba)
        eth_f1_thr = f1_score(y_test[eth_mask], (eth_proba >= eth_thr).astype(int), zero_division=0) if len(y_test[eth_mask].unique()) > 1 else 0
        
        results['ETH'] = {
            'accuracy': float(eth_acc),
            'f1': float(eth_f1),
            'auc': float(eth_auc),
            'threshold_optimal': float(eth_thr),
            'f1_optimal': float(eth_f1_thr),
            'samples': int(eth_mask.sum())
        }
    
    return results


def make_json_serializable(obj):
    """Convierte tipos numpy/pandas a tipos nativos de Python."""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
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
    models_dir = repo_root / "models" / "L1"
    models_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Verificar ETH
    has_eth = eth_train_path.exists() and eth_test_path.exists()
    
    if has_eth:
        print(f"✅ ETH train: {eth_train_path}")
        print(f"✅ ETH test: {eth_test_path}")
        
        eth_train = _load_dataset(str(eth_train_path))
        eth_test = _load_dataset(str(eth_test_path))
        
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
    
    # ⭐ USAR TARGET ALINEADO AL TRADING ⭐
    print("\n🎯 Generando targets alineados al trading...")
    X_train, y_train = _make_xy_trading_aligned(train_df, target_type='l1_filter')
    X_test, y_test = _make_xy_trading_aligned(test_df, target_type='l1_filter')
    
    print(f"🎯 X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"🎯 X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"📈 Target trading - Positivos: {y_train.mean():.3f} ({y_train.sum()}/{len(y_train)})")
    print(f"📊 Distribución y_train: {y_train.value_counts().to_dict()}")
    
    # Crear datasets LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Parámetros optimizados para tamaño pequeño
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 16,
        'max_depth': 6,
        'learning_rate': 0.1,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'min_child_weight': 0.01,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': 4,
        'is_unbalance': True
    }
    
    print("🎯 Entrenando LightGBM con target de trading...")
    
    # Callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=True),
        lgb.log_evaluation(period=20, show_stdv=False)
    ]
    
    # Entrenar modelo
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )
    
    print(f"🌟 Mejor iteración: {gbm.best_iteration}")
    print(f"📊 Mejor score válido: {gbm.best_score['valid']['binary_logloss']:.4f}")
    
    # Predicciones
    proba_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_05 = (proba_test >= 0.5).astype(int)
    
    # Métricas globales
    acc = accuracy_score(y_test, y_pred_05)
    f1 = f1_score(y_test, y_pred_05, zero_division=0) if len(y_test.unique()) > 1 else 0
    auc = roc_auc_score(y_test, proba_test) if len(y_test.unique()) > 1 else 0.5
    
    # Umbral óptimo
    thr = _best_threshold_by_f1(y_test.values, proba_test)
    y_pred_thr = (proba_test >= thr).astype(int)
    f1_thr = f1_score(y_test, y_pred_thr, zero_division=0) if len(y_test.unique()) > 1 else 0
    
    print(f"📊 Métricas globales (threshold=0.5): Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    print(f"⚡ F1 óptimo: threshold={thr:.3f}, F1={f1_thr:.4f}")
    
    # Evaluación por símbolo
    symbol_results = _evaluate_by_symbol(X_test, y_test, proba_test)
    
    if symbol_results:
        print(f"\n📈 RESULTADOS POR SÍMBOLO:")
        for symbol, metrics in symbol_results.items():
            print(f"   {symbol}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
                  f"AUC={metrics['auc']:.3f}, Thr_opt={metrics['threshold_optimal']:.3f} "
                  f"({metrics['samples']} samples)")
    
    # Verificar tamaño antes de guardar (corregido para Windows)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    # Cerrar el archivo antes de usarlo
    joblib.dump(gbm, tmp_path)
    file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    
    # Eliminar archivo temporal (ahora funciona en Windows)
    try:
        os.unlink(tmp_path)
    except:
        pass  # Ignorar si no se puede eliminar
    
    print(f"📏 Tamaño estimado del modelo: {file_size_mb:.1f} MB")
    
    if file_size_mb > 500:
        print("⚠️  MODELO AÚN MUY GRANDE - Considera reducir más num_leaves o features")
    else:
        print("✅ Tamaño del modelo optimizado exitosamente")
    
    # Guardar modelo
    model_path_pkl = models_dir / "modelo3_lgbm.pkl"
    joblib.dump(gbm, model_path_pkl)
    print(f"💾 Modelo PKL guardado: {model_path_pkl}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': gbm.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    max_importance = feature_importance['importance'].max()
    feature_importance['importance_norm'] = feature_importance['importance'] / max_importance if max_importance > 0 else 0
    
    # Metadatos
    meta = {
        "model_type": "l1_trading_filter_lightgbm",
        "target_type": "l1_filter",
        "assets": list(train_df['symbol'].unique()) if 'symbol' in train_df.columns else ['BTC'],
        "features": list(X_train.columns),
        "threshold": float(thr),
        "metrics": {
            "global": {
                "accuracy@0.5": float(acc),
                "f1@0.5": float(f1),
                "auc": float(auc),
                "f1@thr": float(f1_thr),
                "best_iteration": int(gbm.best_iteration),
                "best_valid_score": float(gbm.best_score['valid']['binary_logloss'])
            },
            "by_symbol": make_json_serializable(symbol_results)
        },
        "model_params": make_json_serializable(params),
        "feature_importance": {
            "top_10": make_json_serializable(
                feature_importance.head(10)[['feature', 'importance']].to_dict('records')
            )
        },
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "num_features": int(len(X_train.columns)),
        "multiasset": bool(has_eth),
        "file_size_mb": float(file_size_mb),
        "usage_notes": {
            "purpose": "Filtro L1 para validar señales de L2",
            "target_meaning": "1 = señal L2 será rentable, 0 = rechazar señal",
            "threshold_usage": f"Usar threshold={thr:.3f} para decisiones de filtrado",
            "loading": "joblib.load('modelo3_lgbm.pkl')",
            "prediction": "model.predict(X, num_iteration=model.best_iteration)"
        }
    }
    
    meta_path = models_dir / "modelo3_lgbm.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"📋 Metadatos guardados: {meta_path}")
    
    # Top features
    print(f"\n🏆 TOP 10 FEATURES MÁS IMPORTANTES:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        bar = "█" * int(row['importance_norm'] * 40)
        print(f"   {i+1:2d}. {row['feature']:<25} {row['importance']:>8.0f} {bar}")
    
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO:")
    print(f"   📦 Modelo: {model_path_pkl}")
    print(f"   📋 Metadatos: {meta_path}")
    print(f"   🎯 Target: Filtro L1 para señales rentables")
    print(f"   ⚡ Threshold óptimo: {thr:.3f}")
    print(f"   💾 Tamaño: {file_size_mb:.1f} MB")
    print(f"   🏆 Mejor iteración: {gbm.best_iteration}/200")


if __name__ == "__main__":
    main()