#!/usr/bin/env python3
"""
Script para revisión completa de modelos L3
"""
import os
import sys
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime

def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")

def check_file_exists(path, description):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "[OK]" if exists else "[FALTA]"
    print(f"{status} {description}: {size} bytes")
    return exists, size

def test_regime_model():
    print_header("2.1 PRUEBA DE CARGA - MODELO DE REGIME DETECTION")
    try:
        from l3_strategy.l3_processor import load_regime_model
        model = load_regime_model()
        if model is None:
            print("[ERROR] No se pudo cargar el modelo de regime")
            return False

        # Verificar estructura del modelo
        if isinstance(model, dict):
            required_keys = ['rf', 'et', 'hgb', 'label_encoder']
            missing_keys = [k for k in required_keys if k not in model]
            if missing_keys:
                print(f"[ERROR] Faltan claves en el ensemble: {missing_keys}")
                return False
            print("[OK] Modelo ensemble cargado correctamente")
            print(f"  - Random Forest: {type(model['rf']).__name__}")
            print(f"  - Extra Trees: {type(model['et']).__name__}")
            print(f"  - Hist Gradient Boosting: {type(model['hgb']).__name__}")
            print(f"  - Label Encoder: {type(model['label_encoder']).__name__}")
        else:
            print(f"[OK] Modelo simple cargado: {type(model).__name__}")

        # Verificar features esperadas
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
            print(f"[OK] Features esperadas: {len(features)}")
            print(f"  Muestra: {features[:5]}...")
        elif isinstance(model, dict) and 'features' in model:
            features = model['features']
            print(f"[OK] Features esperadas: {len(features)}")
            print(f"  Muestra: {features[:5]}...")

        return True
    except Exception as e:
        print(f"[ERROR] cargando modelo de regime: {e}")
        return False

def test_volatility_models():
    print_header("2.2 PRUEBA DE CARGA - MODELOS DE VOLATILIDAD")
    try:
        from l3_strategy.l3_processor import load_vol_models
        garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()

        # Verificar GARCH
        print("GARCH Models:")
        btc_ok = garch_btc is not None
        eth_ok = garch_eth is not None
        print(f"  [{'OK]' if btc_ok else 'ERROR'} BTC-GARCH: {type(garch_btc).__name__ if btc_ok else 'None'}")
        print(f"  [{'OK]' if eth_ok else 'ERROR'} ETH-GARCH: {type(garch_eth).__name__ if eth_ok else 'None'}")

        # Verificar LSTM
        print("LSTM Models:")
        btc_lstm_ok = lstm_btc is not None
        eth_lstm_ok = lstm_eth is not None
        print(f"  [{'OK]' if btc_lstm_ok else 'ERROR'} BTC-LSTM: {type(lstm_btc).__name__ if btc_lstm_ok else 'None'}")
        print(f"  [{'OK]' if eth_lstm_ok else 'ERROR'} ETH-LSTM: {type(lstm_eth).__name__ if eth_lstm_ok else 'None'}")

        return btc_ok or eth_ok or btc_lstm_ok or eth_lstm_ok
    except Exception as e:
        print(f"[ERROR] cargando modelos de volatilidad: {e}")
        return False

def test_sentiment_model():
    print_header("2.3 PRUEBA DE CARGA - MODELO BERT DE SENTIMIENTO")
    try:
        from l3_strategy.l3_processor import load_sentiment_model
        tokenizer, model = load_sentiment_model()

        if tokenizer is None or model is None:
            print("[ERROR] No se pudo cargar el modelo BERT de sentimiento")
            return False

        print("[OK] Modelo BERT cargado correctamente")
        print(f"  - Tokenizer: {type(tokenizer).__name__}")
        print(f"  - Model: {type(model).__name__}")

        # Verificar configuración del modelo
        if hasattr(model, 'config'):
            config = model.config
            print(f"  - Num labels: {getattr(config, 'num_labels', 'N/A')}")
            print(f"  - Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
            print(f"  - Num layers: {getattr(config, 'num_hidden_layers', 'N/A')}")

        return True
    except Exception as e:
        print(f"[ERROR] cargando modelo BERT: {e}")
        return False

def test_portfolio_model():
    print_header("2.4 PRUEBA DE CARGA - MODELO DE PORTFOLIO")
    try:
        from l3_strategy.l3_processor import load_portfolio
        cov, weights = load_portfolio()

        if cov is None or weights is None:
            print("[ERROR] No se pudieron cargar los datos de portfolio")
            return False

        print("[OK] Datos de portfolio cargados correctamente")
        print(f"  - Matriz de covarianza: {cov.shape}")
        print(f"  - Pesos óptimos: {weights.shape}")

        # Verificar que los datos sean válidos
        if cov.empty or weights.empty:
            print("[ERROR] Datos de portfolio están vacíos")
            return False

        # Verificar que los índices coincidan (assets como índice en weights)
        cov_assets = set(cov.index)
        weights_assets = set(weights.index)
        common_assets = cov_assets.intersection(weights_assets)

        if not common_assets:
            print("[ERROR] No hay activos comunes entre covarianza y pesos")
            return False

        print(f"  - Activos comunes: {len(common_assets)}")
        print(f"    {sorted(list(common_assets))}")

        # Verificar que los pesos sumen aproximadamente 1
        total_weight = weights['weight'].sum()
        if abs(total_weight - 1.0) > 0.01:
            print(f"[WARNING] Los pesos no suman 1 (suma={total_weight:.4f})")
        else:
            print(f"[OK] Pesos correctamente normalizados (suma={total_weight:.4f})")

        return True
    except Exception as e:
        print(f"[ERROR] cargando modelo de portfolio: {e}")
        return False

def test_predictions():
    print_header("3. PRUEBA DE PREDICCIONES")
    try:
        # Datos de ejemplo para testing
        sample_market_data = {
            "BTCUSDT": [
                {"timestamp": 1640995200000, "open": 50000, "high": 50500, "low": 49900, "close": 50250, "volume": 1.2},
                {"timestamp": 1641081600000, "open": 50200, "high": 50400, "low": 50000, "close": 50100, "volume": 1.0},
                {"timestamp": 1641168000000, "open": 50150, "high": 50300, "low": 50050, "close": 50200, "volume": 1.5},
            ] * 50,  # Repetir para tener suficientes datos
            "ETHUSDT": [
                {"timestamp": 1640995200000, "open": 3500, "high": 3550, "low": 3480, "close": 3520, "volume": 10},
                {"timestamp": 1641081600000, "open": 3520, "high": 3540, "low": 3490, "close": 3510, "volume": 12},
                {"timestamp": 1641168000000, "open": 3510, "high": 3530, "low": 3500, "close": 3525, "volume": 9},
            ] * 50
        }

        sample_texts = [
            "BTC will rally after the Fed announcement",
            "ETH shows bullish signals in technical analysis",
            "Market sentiment is positive for cryptocurrencies"
        ]

        from l3_strategy.l3_processor import generate_l3_output

        print("Generando output L3 con datos de ejemplo...")
        result = generate_l3_output(sample_market_data, sample_texts)

        if result is None:
            print("✗ ERROR: generate_l3_output retornó None")
            return False

        # Verificar estructura del resultado
        required_keys = ['regime', 'asset_allocation', 'risk_appetite', 'sentiment_score', 'volatility_forecast', 'timestamp']
        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            print(f"✗ ERROR: Faltan claves en el resultado: {missing_keys}")
            return False

        print("✓ Output L3 generado correctamente")
        print(f"  - Regime: {result['regime']}")
        print(f"  - Risk Appetite: {result['risk_appetite']}")
        print(f"  - Sentiment Score: {result['sentiment_score']:.4f}")
        print(f"  - Volatility BTC: {result['volatility_forecast']['BTCUSDT']:.4f}")
        print(f"  - Volatility ETH: {result['volatility_forecast']['ETHUSDT']:.4f}")
        print(f"  - Asset Allocation: {result['asset_allocation']}")

        return True
    except Exception as e:
        print(f"✗ ERROR en predicciones: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print_header("REVISIÓN COMPLETA DE MODELOS L3 - HRM")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Verificar existencia de archivos
    print_header("1. VERIFICACIÓN DE EXISTENCIA DE ARCHIVOS")

    models_status = {}

    # Modelo de regime detection
    exists, size = check_file_exists('models/L3/regime_detection_model_ensemble_optuna.pkl', 'Regime Detection Model')
    models_status['regime'] = {'exists': exists, 'size': size}

    # Modelos GARCH
    exists, size = check_file_exists('models/L3/volatility/BTC-USD_volatility_garch.pkl', 'GARCH BTC')
    models_status['garch_btc'] = {'exists': exists, 'size': size}

    exists, size = check_file_exists('models/L3/volatility/ETH-USD_volatility_garch.pkl', 'GARCH ETH')
    models_status['garch_eth'] = {'exists': exists, 'size': size}

    # Modelos LSTM
    exists, size = check_file_exists('models/L3/volatility/BTC-USD_volatility_lstm.h5', 'LSTM BTC')
    models_status['lstm_btc'] = {'exists': exists, 'size': size}

    exists, size = check_file_exists('models/L3/volatility/ETH-USD_volatility_lstm.h5', 'LSTM ETH')
    models_status['lstm_eth'] = {'exists': exists, 'size': size}

    # Portfolio
    exists, size = check_file_exists('models/L3/portfolio/bl_cov.csv', 'Portfolio Covariance')
    models_status['portfolio_cov'] = {'exists': exists, 'size': size}

    exists, size = check_file_exists('models/L3/portfolio/bl_weights.csv', 'Portfolio Weights')
    models_status['portfolio_weights'] = {'exists': exists, 'size': size}

    # Sentiment BERT
    sentiment_files = ['config.json', 'model.safetensors', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt']
    sentiment_dir = 'models/L3/sentiment/'
    all_exist = all(os.path.exists(os.path.join(sentiment_dir, f)) for f in sentiment_files)
    total_size = sum(os.path.getsize(os.path.join(sentiment_dir, f)) for f in sentiment_files if os.path.exists(os.path.join(sentiment_dir, f)))
    models_status['sentiment'] = {'exists': all_exist, 'size': total_size}

    status = "✓ TODOS EXISTEN" if all_exist else "✗ FALTAN ARCHIVOS"
    print(f"{status} Sentiment BERT Model: {total_size} bytes total")
    for f in sentiment_files:
        f_exists = os.path.exists(os.path.join(sentiment_dir, f))
        print(f"  {'✓' if f_exists else '✗'} {f}")

    # 2. Probar carga de modelos
    load_results = {}
    load_results['regime'] = test_regime_model()
    load_results['volatility'] = test_volatility_models()
    load_results['sentiment'] = test_sentiment_model()
    load_results['portfolio'] = test_portfolio_model()

    # 3. Probar predicciones
    prediction_result = test_predictions()

    # 4. Resumen final
    print_header("4. RESUMEN FINAL")

    print("EXISTENCIA DE ARCHIVOS:")
    for model, status in models_status.items():
        icon = "✓" if status['exists'] else "✗"
        print(f"  {icon} {model}: {'OK' if status['exists'] else 'FALTA'}")

    print("\nCARGA DE MODELOS:")
    for model, success in load_results.items():
        icon = "✓" if success else "✗"
        print(f"  {icon} {model}: {'OK' if success else 'ERROR'}")

    print(f"\nPREDICCIONES: {'✓ OK' if prediction_result else '✗ ERROR'}")

    # Estado general
    all_files_exist = all(s['exists'] for s in models_status.values())
    all_models_load = all(load_results.values())
    predictions_ok = prediction_result

    print(f"\nESTADO GENERAL:")
    print(f"  Archivos: {'✓ TODOS PRESENTES' if all_files_exist else '✗ FALTAN ARCHIVOS'}")
    print(f"  Carga: {'✓ TODOS CARGAN' if all_models_load else '✗ ERRORES DE CARGA'}")
    print(f"  Predicciones: {'✓ FUNCIONAN' if predictions_ok else '✗ FALLAN'}")

    if all_files_exist and all_models_load and predictions_ok:
        print("\n🎉 TODOS LOS MODELOS L3 ESTÁN OPERATIVOS")
    else:
        print("\n⚠️  HAY PROBLEMAS QUE REQUIEREN ATENCIÓN")

if __name__ == "__main__":
    main()
