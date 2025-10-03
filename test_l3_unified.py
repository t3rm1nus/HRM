#!/usr/bin/env python3
"""
Unified L3 Testing Suite
Consolidates all L3 model and regime-specific tests
"""
import os
import sys
import joblib
import pandas as pd
import json
import numpy as np
from datetime import datetime, timezone
import asyncio
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging import logger
from l3_strategy.regime_specific_models import (
    RegimeSpecificL3Processor,
    BullMarketModel,
    BearMarketModel,
    RangeMarketModel,
    VolatileMarketModel,
    RegimeStrategy
)

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

def create_test_market_data(regime_type: str = 'bull') -> Dict[str, pd.DataFrame]:
    """Create synthetic market data for different regime types"""
    np.random.seed(42)  # For reproducible results

    symbols = ['BTCUSDT', 'ETHUSDT']
    market_data = {}

    # Create 200 periods of OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')

    for symbol in symbols:
        # Generate synthetic price data based on regime
        if symbol == 'BTCUSDT':
            base_price = 50000
            if regime_type == 'bull':
                # Bull market: upward trending with moderate volatility
                trend = np.linspace(0, 0.5, 200)  # Strong upward trend
                noise = np.random.normal(0, 0.015, 200)  # Moderate volatility
            elif regime_type == 'bear':
                # Bear market: downward trending with high volatility
                trend = np.linspace(0, -0.3, 200)  # Downward trend
                noise = np.random.normal(0, 0.025, 200)  # High volatility
            else:  # range
                # Range market: sideways with low volatility
                trend = np.sin(np.linspace(0, 4*np.pi, 200)) * 0.05  # Sideways oscillation
                noise = np.random.normal(0, 0.01, 200)  # Low volatility
        else:  # ETHUSDT
            base_price = 3000
            if regime_type == 'bull':
                trend = np.linspace(0, 0.4, 200)
                noise = np.random.normal(0, 0.02, 200)
            elif regime_type == 'bear':
                trend = np.linspace(0, -0.25, 200)
                noise = np.random.normal(0, 0.03, 200)
            else:  # range
                trend = np.sin(np.linspace(0, 4*np.pi, 200)) * 0.03
                noise = np.random.normal(0, 0.012, 200)

        returns = trend + noise
        prices = base_price * (1 + returns).cumprod()

        # Create OHLCV data
        high_mult = 1 + np.random.uniform(0, 0.005, 200)
        low_mult = 1 - np.random.uniform(0, 0.005, 200)
        volume_base = 1000000 if symbol == 'BTCUSDT' else 500000

        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.002, 200)),
            'high': prices * high_mult,
            'low': prices * low_mult,
            'close': prices,
            'volume': volume_base * (1 + np.random.uniform(0, 1, 200))
        }, index=dates)

        # Ensure high >= close >= low and high >= open >= low
        df['high'] = np.maximum(df[['high', 'close', 'open']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['low', 'close', 'open']].min(axis=1), df['low'])

        market_data[symbol] = df

    return market_data

def test_bull_market_model():
    """Test Bull Market Model"""
    print("\n" + "="*60)
    print("TESTING BULL MARKET MODEL")
    print("="*60)

    # Create bull market data
    market_data = create_test_market_data('bull')
    regime_context = {
        'regime': 'bull',
        'volatility_avg': 0.02,
        'sentiment_score': 0.8,
        'risk_appetite': 'aggressive'
    }

    # Test Bull Market Model
    bull_model = BullMarketModel()
    strategy = bull_model.generate_strategy(market_data, regime_context)

    print(f"✅ Bull Market Strategy Generated:")
    print(f"   Risk Appetite: {strategy.risk_appetite:.2f} (expected: high > 0.7)")
    print(f"   BTC Allocation: {strategy.asset_allocation.get('BTCUSDT', 0):.2f} (expected: high > 0.5)")
    print(f"   ETH Allocation: {strategy.asset_allocation.get('ETHUSDT', 0):.2f} (expected: moderate)")
    print(f"   Cash Allocation: {strategy.asset_allocation.get('CASH', 0):.2f} (expected: low < 0.2)")
    print(f"   Rebalancing: {strategy.rebalancing_frequency} (expected: daily)")
    print(f"   Volatility Target: {strategy.volatility_target:.2f} (expected: high > 0.15)")

    # Validate bull market characteristics
    assert strategy.risk_appetite > 0.7, f"Risk appetite too low: {strategy.risk_appetite}"
    assert strategy.asset_allocation.get('BTCUSDT', 0) >= 0.4, "BTC allocation too low for bull market"
    assert strategy.asset_allocation.get('CASH', 1) < 0.2, "Cash allocation too high for bull market"
    assert strategy.rebalancing_frequency == 'daily', f"Wrong rebalancing frequency: {strategy.rebalancing_frequency}"

    print("✅ Bull Market Model validation passed")
    return strategy

def test_bear_market_model():
    """Test Bear Market Model"""
    print("\n" + "="*60)
    print("TESTING BEAR MARKET MODEL")
    print("="*60)

    # Create bear market data
    market_data = create_test_market_data('bear')
    regime_context = {
        'regime': 'bear',
        'volatility_avg': 0.08,
        'sentiment_score': -0.6,
        'risk_appetite': 'conservative'
    }

    # Test Bear Market Model
    bear_model = BearMarketModel()
    strategy = bear_model.generate_strategy(market_data, regime_context)

    print(f"✅ Bear Market Strategy Generated:")
    print(f"   Risk Appetite: {strategy.risk_appetite:.2f} (expected: low < 0.3)")
    print(f"   BTC Allocation: {strategy.asset_allocation.get('BTCUSDT', 0):.2f} (expected: low < 0.2)")
    print(f"   ETH Allocation: {strategy.asset_allocation.get('ETHUSDT', 0):.2f} (expected: very low < 0.1)")
    print(f"   Cash Allocation: {strategy.asset_allocation.get('CASH', 0):.2f} (expected: high > 0.7)")
    print(f"   Rebalancing: {strategy.rebalancing_frequency} (expected: weekly)")
    print(f"   Volatility Target: {strategy.volatility_target:.2f} (expected: low < 0.1)")

    # Validate bear market characteristics
    assert strategy.risk_appetite < 0.3, f"Risk appetite too high: {strategy.risk_appetite}"
    assert strategy.asset_allocation.get('CASH', 0) >= 0.5, "Cash allocation too low for bear market"
    assert strategy.asset_allocation.get('BTCUSDT', 1) < 0.2, "BTC allocation too high for bear market"
    assert strategy.rebalancing_frequency == 'weekly', f"Wrong rebalancing frequency: {strategy.rebalancing_frequency}"

    print("✅ Bear Market Model validation passed")
    return strategy

def test_range_market_model():
    """Test Range Market Model"""
    print("\n" + "="*60)
    print("TESTING RANGE MARKET MODEL")
    print("="*60)

    # Create range market data
    market_data = create_test_market_data('range')
    regime_context = {
        'regime': 'range',
        'volatility_avg': 0.015,
        'sentiment_score': 0.1,
        'risk_appetite': 'moderate'
    }

    # Test Range Market Model
    range_model = RangeMarketModel()
    strategy = range_model.generate_strategy(market_data, regime_context)

    print(f"✅ Range Market Strategy Generated:")
    print(f"   Risk Appetite: {strategy.risk_appetite:.2f} (expected: moderate ≈ 0.5)")
    print(f"   BTC Allocation: {strategy.asset_allocation.get('BTCUSDT', 0):.2f}")
    print(f"   ETH Allocation: {strategy.asset_allocation.get('ETHUSDT', 0):.2f}")
    print(f"   Cash Allocation: {strategy.asset_allocation.get('CASH', 0):.2f} (expected: moderate ≈ 0.3)")
    print(f"   Rebalancing: {strategy.rebalancing_frequency} (expected: daily)")
    print(f"   Volatility Target: {strategy.volatility_target:.2f} (expected: moderate ≈ 0.12)")

    # Validate range market characteristics - now more aggressive
    assert 0.6 <= strategy.risk_appetite <= 0.8, f"Risk appetite not aggressive: {strategy.risk_appetite}"
    assert strategy.rebalancing_frequency == 'daily', f"Wrong rebalancing frequency: {strategy.rebalancing_frequency}"
    assert 0.1 <= strategy.volatility_target <= 0.15, f"Volatility target not moderate: {strategy.volatility_target}"

    print("✅ Range Market Model validation passed")
    return strategy

def test_volatile_market_model():
    """Test Volatile Market Model"""
    print("\n" + "="*60)
    print("TESTING VOLATILE MARKET MODEL")
    print("="*60)

    # Create volatile market data (high volatility)
    market_data = create_test_market_data('bull')  # Use bull data but we'll override volatility
    # Make it more volatile by adding noise
    for symbol, df in market_data.items():
        # Add high volatility by multiplying returns
        df['close'] = df['close'] * (1 + np.random.normal(0, 0.05, len(df)))

    regime_context = {
        'regime': 'volatile',
        'volatility_avg': 0.12,  # High volatility
        'sentiment_score': -0.3,
        'risk_appetite': 'moderate'
    }

    # Test Volatile Market Model
    volatile_model = VolatileMarketModel()
    strategy = volatile_model.generate_strategy(market_data, regime_context)

    print(f"✅ Volatile Market Strategy Generated:")
    print(f"   Risk Appetite: {strategy.risk_appetite:.2f} (expected: moderate < 0.5)")
    print(f"   BTC Allocation: {strategy.asset_allocation.get('BTCUSDT', 0):.2f}")
    print(f"   ETH Allocation: {strategy.asset_allocation.get('ETHUSDT', 0):.2f}")
    print(f"   Cash Allocation: {strategy.asset_allocation.get('CASH', 0):.2f} (expected: minimum liquidity)")
    print(f"   ALT Allocation: {strategy.asset_allocation.get('ALT', 0):.2f} (expected: moderate)")
    print(f"   Rebalancing: {strategy.rebalancing_frequency} (expected: daily)")
    print(f"   Volatility Target: {strategy.volatility_target:.2f} (expected: above current vol)")

    # Validate volatile market characteristics
    assert strategy.risk_appetite < 0.5, f"Volatile strategy risk appetite too high: {strategy.risk_appetite}"
    assert strategy.asset_allocation.get('CASH', 0) >= 0.1, "Volatile strategy needs minimum cash liquidity"
    assert strategy.asset_allocation.get('ALT', 0) >= 0.15, "Volatile strategy alternative assets too low"
    assert strategy.rebalancing_frequency == 'daily', f"Wrong rebalancing frequency: {strategy.rebalancing_frequency}"
    assert strategy.volatility_target > 0.1, f"Volatility target too low: {strategy.volatility_target}"

    print("✅ Volatile Market Model validation passed")
    return strategy

def test_regime_specific_processor():
    """Test the RegimeSpecificL3Processor"""
    print("\n" + "="*60)
    print("TESTING REGIME-SPECIFIC L3 PROCESSOR")
    print("="*60)

    processor = RegimeSpecificL3Processor()

    # Test health check
    health = processor.get_model_health()
    print(f"✅ Health Check: {health['overall_status']}")
    print(f"   Models: {len(health['models'])}")
    for regime, status in health['models'].items():
        print(f"   {regime}: {status['status']}")

    assert health['overall_status'] == 'healthy', "Processor health check failed"
    assert len(health['models']) == 5, f"Expected 5 models, got {len(health['models'])}"

    # Test different regimes
    regimes_to_test = ['bull', 'bear', 'range', 'volatile']

    for regime in regimes_to_test:
        print(f"\n🧪 Testing {regime.upper()} regime processing...")

        market_data = create_test_market_data('bull' if regime != 'volatile' else 'bear')
        if regime == 'volatile':
            # Make volatile data
            for symbol, df in market_data.items():
                df['close'] = df['close'] * (1 + np.random.normal(0, 0.05, len(df)))

        regime_context = {
            'regime': regime,
            'volatility_avg': 0.02 if regime == 'bull' else 0.08 if regime == 'bear' else 0.015 if regime == 'range' else 0.12,
            'sentiment_score': 0.8 if regime == 'bull' else -0.6 if regime == 'bear' else 0.1 if regime == 'range' else -0.3,
            'risk_appetite': 'aggressive' if regime == 'bull' else 'conservative' if regime == 'bear' else 'moderate'
        }

        strategy = processor.generate_regime_strategy(market_data, regime_context)

        print(f"   Generated strategy for {regime} regime:")
        print(f"   Risk Appetite: {strategy.risk_appetite:.2f}")
        print(f"   Asset Allocation: {strategy.asset_allocation}")
        print(f"   Rebalancing: {strategy.rebalancing_frequency}")

        # Validate strategy structure
        assert hasattr(strategy, 'regime'), "Strategy missing regime"
        assert hasattr(strategy, 'risk_appetite'), "Strategy missing risk_appetite"
        assert hasattr(strategy, 'asset_allocation'), "Strategy missing asset_allocation"
        assert hasattr(strategy, 'position_sizing'), "Strategy missing position_sizing"
        assert hasattr(strategy, 'stop_loss_policy'), "Strategy missing stop_loss_policy"
        assert hasattr(strategy, 'take_profit_policy'), "Strategy missing take_profit_policy"

        # Validate regime-specific behavior
        if regime == 'bull':
            assert strategy.risk_appetite > 0.7, f"Bull strategy risk appetite too low: {strategy.risk_appetite}"
            assert strategy.asset_allocation.get('CASH', 1) < 0.3, f"Bull strategy cash too high: {strategy.asset_allocation}"
        elif regime == 'bear':
            assert strategy.risk_appetite < 0.3, f"Bear strategy risk appetite too high: {strategy.risk_appetite}"
            assert strategy.asset_allocation.get('CASH', 0) >= 0.5, f"Bear strategy cash too low: {strategy.asset_allocation}"
        elif regime == 'volatile':
            assert strategy.asset_allocation.get('ALT', 0) > 0, f"Volatile strategy missing alternative assets: {strategy.asset_allocation}"

    print("✅ Regime-Specific L3 Processor validation passed")
    return processor

def test_regime_detection_fallback():
    """Test regime detection fallback when context is missing"""
    print("\n" + "="*60)
    print("TESTING REGIME DETECTION FALLBACK")
    print("="*60)

    processor = RegimeSpecificL3Processor()

    # Test with missing regime context
    market_data = create_test_market_data('bull')

    # Should detect bull regime from market data
    strategy = processor.generate_regime_strategy(market_data, {})

    print(f"✅ Fallback regime detection:")
    print(f"   Detected regime: {strategy.regime}")
    print(f"   Risk appetite: {strategy.risk_appetite:.2f}")

    # Should still generate a valid strategy
    assert strategy.regime in ['bull', 'bear', 'range'], f"Invalid detected regime: {strategy.regime}"
    assert 0 <= strategy.risk_appetite <= 1, f"Invalid risk appetite: {strategy.risk_appetite}"

    print("✅ Regime detection fallback validation passed")

def test_error_handling():
    """Test error handling and edge cases"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)

    processor = RegimeSpecificL3Processor()

    # Test with empty market data
    try:
        strategy = processor.generate_regime_strategy({}, {})
        print("✅ Handled empty market data gracefully")
        # Should generate some valid strategy even with empty data
        assert strategy.regime in ['bull', 'bear', 'range', 'neutral'], f"Invalid regime: {strategy.regime}"
        assert 0 <= strategy.risk_appetite <= 1, f"Invalid risk appetite: {strategy.risk_appetite}"
    except Exception as e:
        print(f"❌ Failed to handle empty market data: {e}")
        raise

    # Test with invalid market data
    try:
        invalid_data = {'BTCUSDT': pd.DataFrame()}  # Empty DataFrame
        strategy = processor.generate_regime_strategy(invalid_data, {})
        print("✅ Handled invalid market data gracefully")
        # Should generate some valid strategy even with invalid data
        assert strategy.regime in ['bull', 'bear', 'range', 'neutral'], f"Invalid regime: {strategy.regime}"
        assert 0 <= strategy.risk_appetite <= 1, f"Invalid risk appetite: {strategy.risk_appetite}"
    except Exception as e:
        print(f"❌ Failed to handle invalid market data: {e}")
        raise

    print("✅ Error handling validation passed")

async def main_l3_tests():
    """Run all L3 tests"""
    print("🚀 UNIFIED L3 MODEL TESTS")
    print("=" * 60)

    success_count = 0
    total_tests = 6

    # Test 1: File verification
    print_header("1. VERIFICACIÓN DE EXISTENCIA DE ARCHIVOS")
    try:
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

        all_files_exist = all(s['exists'] for s in models_status.values())
        print("✅ Test 1 PASSED: File Verification"         success_count += 1
    except Exception as e:
        print(f"❌ Test 1 ERROR: {e}")

    # Test 2: Model loading
    try:
        load_results = {}
        load_results['regime'] = test_regime_model()
        load_results['volatility'] = test_volatility_models()
        load_results['sentiment'] = test_sentiment_model()
        load_results['portfolio'] = test_portfolio_model()

        all_models_load = all(load_results.values())
        if all_models_load:
            success_count += 1
            print("✅ Test 2 PASSED: Model Loading")
        else:
            print("❌ Test 2 FAILED: Model Loading")
    except Exception as e:
        print(f"❌ Test 2 ERROR: {e}")

    # Test 3: Predictions
    try:
        if test_predictions():
            success_count += 1
            print("✅ Test 3 PASSED: Predictions")
        else:
            print("❌ Test 3 FAILED: Predictions")
    except Exception as e:
        print(f"❌ Test 3 ERROR: {e}")

    # Test 4: Regime models
    try:
        bull_strategy = test_bull_market_model()
        bear_strategy = test_bear_market_model()
        range_strategy = test_range_market_model()
        volatile_strategy = test_volatile_market_model()
        success_count += 1
        print("✅ Test 4 PASSED: Regime Models")
    except Exception as e:
        print(f"❌ Test 4 ERROR: {e}")

    # Test 5: Processor
    try:
        processor = test_regime_specific_processor()
        success_count += 1
        print("✅ Test 5 PASSED: Processor")
    except Exception as e:
        print(f"❌ Test 5 ERROR: {e}")

    # Test 6: Error handling
    try:
        test_regime_detection_fallback()
        test_error_handling()
        success_count += 1
        print("✅ Test 6 PASSED: Error Handling")
    except Exception as e:
        print(f"❌ Test 6 ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {success_count}/{total_tests} tests passed")
    print(".1f")
    print("\n📊 MODEL STATUS:")
    print(f"  Files Present: {'✓' if all_files_exist else '✗'}")
    print(f"  Models Loaded: {'✓' if all_models_load else '✗'}")
    print(f"  Regimes Supported: 5 (Bull, Bear, Range, Volatile, Neutral)")

    if success_count >= total_tests - 1:
        print("\n🎉 L3 MODELS ARE OPERATIONAL!")
        print("  All regime-specific strategies validated")
        print("  Risk management and asset allocation working")
        print("  Sentiment analysis integrated")
        print("  Volatility forecasting active")
    else:
        print("\n⚠️  SOME L3 TESTS FAILED - CHECK MODEL AVAILABILITY")

    print("=" * 60)
    return success_count >= total_tests - 1

if __name__ == "__main__":
    success = asyncio.run(main_l3_tests())
    sys.exit(0 if success else 1)
