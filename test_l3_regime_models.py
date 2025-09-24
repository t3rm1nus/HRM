#!/usr/bin/env python3
"""
Test script for L3 Regime-Specific Models
Tests the implementation of the 3 regime-specific L3 models:
- Bull Market Model
- Bear Market Model
- Range Market Model
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
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

    # Validate range market characteristics
    assert 0.4 <= strategy.risk_appetite <= 0.6, f"Risk appetite not moderate: {strategy.risk_appetite}"
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
    assert len(health['models']) == 4, f"Expected 4 models, got {len(health['models'])}"

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

async def main():
    """Run all tests"""
    print("🧪 STARTING L3 REGIME-SPECIFIC MODELS TESTS")
    print("="*60)

    try:
        # Test individual models
        bull_strategy = test_bull_market_model()
        bear_strategy = test_bear_market_model()
        range_strategy = test_range_market_model()
        volatile_strategy = test_volatile_market_model()

        # Test processor
        processor = test_regime_specific_processor()

        # Test fallback and error handling
        test_regime_detection_fallback()
        test_error_handling()

        print("\n" + "="*60)
        print("🎉 ALL L3 REGIME-SPECIFIC MODEL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Summary
        print("\n📊 SUMMARY:")
        print(f"  Bull Market Strategy: Risk={bull_strategy.risk_appetite:.2f}, BTC={bull_strategy.asset_allocation.get('BTCUSDT', 0):.2f}")
        print(f"  Bear Market Strategy: Risk={bear_strategy.risk_appetite:.2f}, Cash={bear_strategy.asset_allocation.get('CASH', 0):.2f}")
        print(f"  Range Market Strategy: Risk={range_strategy.risk_appetite:.2f}, Rebalance={range_strategy.rebalancing_frequency}")
        print(f"  Volatile Market Strategy: Risk={volatile_strategy.risk_appetite:.2f}, ALT={volatile_strategy.asset_allocation.get('ALT', 0):.2f}")
        print(f"  Processor Health: {processor.get_model_health()['overall_status']}")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
