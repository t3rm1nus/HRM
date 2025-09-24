#!/usr/bin/env python3
"""
Test script for L1 Operational Models
Tests the implementation of the 3 L1 models and their integration with L2
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging import logger
from l1_operational.models import (
    L1Model, MomentumModel, TechnicalIndicatorsModel, VolumeSignalsModel,
    L1Signal, L1SignalType
)
from l1_operational.l1_operational import L1OperationalProcessor
from l2_tactic.models import TacticalSignal

def create_test_market_data() -> Dict[str, pd.DataFrame]:
    """Create synthetic market data for testing"""
    np.random.seed(42)  # For reproducible results

    symbols = ['BTCUSDT', 'ETHUSDT']
    market_data = {}

    # Create 100 periods of OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

    for symbol in symbols:
        # Generate synthetic price data with trend and volatility
        base_price = 50000 if symbol == 'BTCUSDT' else 3000

        # Create a trending price series
        trend = np.linspace(0, 0.1, 100)  # Slight upward trend
        noise = np.random.normal(0, 0.02, 100)  # Random noise
        returns = trend + noise

        # Convert to price series
        prices = base_price * (1 + returns).cumprod()

        # Create OHLCV data
        high_mult = 1 + np.random.uniform(0, 0.01, 100)
        low_mult = 1 - np.random.uniform(0, 0.01, 100)
        volume_base = 1000000 if symbol == 'BTCUSDT' else 500000

        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.005, 100)),
            'high': prices * high_mult,
            'low': prices * low_mult,
            'close': prices,
            'volume': volume_base * (1 + np.random.uniform(0, 2, 100))
        }, index=dates)

        # Ensure high >= close >= low and high >= open >= low
        df['high'] = np.maximum(df[['high', 'close', 'open']].max(axis=1), df['high'])
        df['low'] = np.minimum(df[['low', 'close', 'open']].min(axis=1), df['low'])

        market_data[symbol] = df

    return market_data

def test_individual_models():
    """Test each L1 model individually"""
    print("\n" + "="*60)
    print("TESTING INDIVIDUAL L1 MODELS")
    print("="*60)

    market_data = create_test_market_data()

    # Test Momentum Model
    print("\n1. Testing MomentumModel...")
    momentum_model = MomentumModel()
    momentum_signals = momentum_model.generate_signals(market_data)

    print(f"   Generated {len(momentum_signals)} momentum signals")
    for signal in momentum_signals[:3]:  # Show first 3
        print(f"   {signal.symbol}: {signal.direction} ({signal.signal_type.value}) "
              f"conf={signal.confidence:.3f} "
              f"str={signal.strength:.3f}")

    # Test Technical Indicators Model
    print("\n2. Testing TechnicalIndicatorsModel...")
    technical_model = TechnicalIndicatorsModel()
    technical_signals = technical_model.generate_signals(market_data)

    print(f"   Generated {len(technical_signals)} technical signals")
    for signal in technical_signals[:3]:  # Show first 3
        print(f"   {signal.symbol}: {signal.direction} ({signal.signal_type.value}) "
              f"conf={signal.confidence:.3f} "
              f"str={signal.strength:.3f}")

    # Test Volume Signals Model
    print("\n3. Testing VolumeSignalsModel...")
    volume_model = VolumeSignalsModel()
    volume_signals = volume_model.generate_signals(market_data)

    print(f"   Generated {len(volume_signals)} volume signals")
    for signal in volume_signals[:3]:  # Show first 3
        print(f"   {signal.symbol}: {signal.direction} ({signal.signal_type.value}) "
              f"conf={signal.confidence:.3f} "
              f"str={signal.strength:.3f}")

    return momentum_signals, technical_signals, volume_signals

def test_combined_l1_model():
    """Test the combined L1 model"""
    print("\n" + "="*60)
    print("TESTING COMBINED L1 MODEL")
    print("="*60)

    market_data = create_test_market_data()
    l1_model = L1Model()

    result = l1_model.predict(market_data)

    signals = result['signals']
    metrics = result['metrics']

    print(f"\nTotal signals generated: {len(signals)}")
    print("Metrics:")
    print(f"  Buy signals: {metrics['buy_signals']}")
    print(f"  Sell signals: {metrics['sell_signals']}")
    print(f"  Hold signals: {metrics['hold_signals']}")
    print(f"  Avg confidence: {metrics['avg_confidence']:.3f}")
    print(f"  Avg strength: {metrics['avg_strength']:.3f}")

    print("\nSignal types breakdown:")
    for signal_type, count in metrics['signal_types'].items():
        print(f"  {signal_type}: {count}")

    print("\nSample signals:")
    for i, signal in enumerate(signals[:5]):
        print(f"  {i+1}. {signal.symbol} {signal.direction} "
              f"conf={signal.confidence:.3f} "
              f"str={signal.strength:.3f}")

    return result

async def test_l1_l2_integration():
    """Test L1-L2 integration"""
    print("\n" + "="*60)
    print("TESTING L1-L2 INTEGRATION")
    print("="*60)

    market_data = create_test_market_data()

    # Create L1 processor
    l1_processor = L1OperationalProcessor()

    # Process market data
    tactical_signals = await l1_processor.process_market_data(market_data)

    print(f"\nGenerated {len(tactical_signals)} tactical signals for L2")

    # Validate signal format
    print("\nValidating signal format...")
    valid_signals = 0
    for i, signal in enumerate(tactical_signals):
        try:
            # Check required attributes
            assert hasattr(signal, 'symbol'), f"Signal {i} missing symbol"
            assert hasattr(signal, 'side'), f"Signal {i} missing side"
            assert hasattr(signal, 'strength'), f"Signal {i} missing strength"
            assert hasattr(signal, 'confidence'), f"Signal {i} missing confidence"
            assert hasattr(signal, 'features'), f"Signal {i} missing features"
            assert hasattr(signal, 'timestamp'), f"Signal {i} missing timestamp"

            # Check types
            assert isinstance(signal.symbol, str), f"Signal {i} symbol not string"
            assert signal.side in ['buy', 'sell', 'hold'], f"Signal {i} invalid side: {signal.side}"
            assert 0 <= signal.strength <= 1, f"Signal {i} invalid strength: {signal.strength}"
            assert 0 <= signal.confidence <= 1, f"Signal {i} invalid confidence: {signal.confidence}"

            valid_signals += 1

        except Exception as e:
            print(f"   ❌ Signal {i} validation failed: {e}")

    print(f"   ✅ {valid_signals}/{len(tactical_signals)} signals passed validation")

    # Show sample tactical signals
    print("\nSample tactical signals:")
    for i, signal in enumerate(tactical_signals[:3]):
        print(f"  {i+1}. {signal.symbol} {signal.side} "
              f"conf={signal.confidence:.3f} "
              f"str={signal.strength:.3f}")
        print(f"      Source: {getattr(signal, 'source', 'unknown')}")
        print(f"      L1 Type: {signal.metadata.get('l1_signal_type', 'unknown')}")

    # Test health check
    health = await l1_processor.health_check()
    print(f"\nHealth check: {health['status']}")
    print(f"Active models: {health['active_models']}/{health['total_models']}")

    return tactical_signals

def test_signal_consistency():
    """Test signal consistency and edge cases"""
    print("\n" + "="*60)
    print("TESTING SIGNAL CONSISTENCY")
    print("="*60)

    # Test with insufficient data
    print("\n1. Testing with insufficient data...")
    short_data = create_test_market_data()
    # Truncate to only 5 periods
    for symbol in short_data:
        short_data[symbol] = short_data[symbol].head(5)

    l1_model = L1Model()
    result = l1_model.predict(short_data)
    print(f"   Signals with insufficient data: {len(result['signals'])} (expected: 0)")

    # Test with empty data
    print("\n2. Testing with empty data...")
    empty_result = l1_model.predict({})
    print(f"   Signals with empty data: {len(empty_result['signals'])} (expected: 0)")

    # Test with NaN values
    print("\n3. Testing with NaN values...")
    nan_data = create_test_market_data()
    nan_data['BTCUSDT'].loc[:, 'close'] = np.nan
    nan_result = l1_model.predict(nan_data)
    print(f"   Signals with NaN data: {len(nan_result['signals'])} (expected: 0)")

    print("\n✅ Consistency tests completed")

async def main():
    """Run all tests"""
    print("🧪 STARTING L1 MODELS TESTS")
    print("="*60)

    try:
        # Test individual models
        momentum_sigs, technical_sigs, volume_sigs = test_individual_models()

        # Test combined model
        combined_result = test_combined_l1_model()

        # Test L1-L2 integration
        tactical_signals = await test_l1_l2_integration()

        # Test consistency
        test_signal_consistency()

        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Summary
        total_signals = len(momentum_sigs) + len(technical_sigs) + len(volume_sigs)
        print("\n📊 SUMMARY:")
        print(f"  Momentum signals: {len(momentum_sigs)}")
        print(f"  Technical signals: {len(technical_sigs)}")
        print(f"  Volume signals: {len(volume_sigs)}")
        print(f"  Total L1 signals: {total_signals}")
        print(f"  L2 tactical signals: {len(tactical_signals)}")
        print(f"  Combined model signals: {len(combined_result['signals'])}")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
