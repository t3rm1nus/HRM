#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for L2 error fixes

This script tests the fixes applied to resolve:
1. TacticalSignal object missing 'action' attribute
2. NaN values after validation in core.logging
3. 'str' object has no attribute 'keys' error in main cycle

Author: Cline AI Assistant
Date: 2025-01-15
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging import logger
from l2_tactic.models import TacticalSignal

def test_tactical_signal_action_attribute():
    """Test that TacticalSignal has action attribute"""
    logger.info("🧪 Testing TacticalSignal action attribute...")
    
    try:
        # Create a test signal
        signal = TacticalSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=0.8,
            confidence=0.7,
            signal_type="test",
            source="test"
        )
        
        # Test that action attribute exists and works
        assert hasattr(signal, 'action'), "TacticalSignal missing 'action' attribute"
        assert signal.action == "buy", f"Expected action='buy', got '{signal.action}'"
        
        # Test setting action
        signal.action = "sell"
        assert signal.side == "sell", f"Setting action didn't update side: {signal.side}"
        assert signal.action == "sell", f"Action not updated: {signal.action}"
        
        logger.info("✅ TacticalSignal action attribute test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ TacticalSignal action attribute test FAILED: {e}")
        return False

def test_nan_validation():
    """Test NaN validation utilities"""
    logger.info("🧪 Testing NaN validation...")
    
    try:
        from core.data_validation import validate_and_clean_data
        
        # Test DataFrame with NaN values
        df_with_nans = pd.DataFrame({
            'price': [100.0, np.nan, 102.0],
            'volume': [1000, 2000, np.nan],
            'symbol': ['BTC', None, 'ETH']
        })
        
        cleaned_df = validate_and_clean_data(df_with_nans, "test_dataframe")
        
        # Check that NaN values are cleaned
        assert not cleaned_df.isna().any().any(), "DataFrame still contains NaN values"
        assert cleaned_df['price'].iloc[1] == 0.0, "NaN price not replaced with 0.0"
        assert cleaned_df['volume'].iloc[2] == 0.0, "NaN volume not replaced with 0.0"
        assert cleaned_df['symbol'].iloc[1] == '', "None symbol not replaced with empty string"
        
        # Test dictionary with NaN values
        dict_with_nans = {
            'price': 100.0,
            'volume': np.nan,
            'rsi': float('nan')
        }
        
        cleaned_dict = validate_and_clean_data(dict_with_nans, "test_dict")
        assert cleaned_dict['volume'] == 0.0, "NaN volume in dict not cleaned"
        assert cleaned_dict['rsi'] == 0.0, "NaN rsi in dict not cleaned"
        
        logger.info("✅ NaN validation test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ NaN validation test FAILED: {e}")
        return False

def test_safe_dict_access():
    """Test safe dictionary access utilities"""
    logger.info("🧪 Testing safe dictionary access...")
    
    try:
        from core.data_validation import safe_dict_access, ensure_dict, safe_market_data_access
        
        # Test safe_dict_access with various inputs
        test_dict = {'key1': 'value1', 'key2': 'value2'}
        assert safe_dict_access(test_dict, 'key1') == 'value1'
        assert safe_dict_access(test_dict, 'missing_key', 'default') == 'default'
        assert safe_dict_access("not_a_dict", 'key', 'default') == 'default'
        assert safe_dict_access(None, 'key', 'default') == 'default'
        
        # Test ensure_dict with various inputs
        assert ensure_dict({'a': 1}) == {'a': 1}
        assert ensure_dict('{"a": 1}') == {'a': 1}  # JSON string
        assert ensure_dict("not json") == {}
        assert ensure_dict(None) == {}
        
        # Test safe_market_data_access
        state = {'market_data': {'BTCUSDT': 'some_data'}}
        market_data = safe_market_data_access(state)
        assert isinstance(market_data, dict)
        
        # Test with invalid market_data
        state_invalid = {'market_data': "not_a_dict"}
        market_data_fixed = safe_market_data_access(state_invalid)
        assert isinstance(market_data_fixed, dict)
        assert isinstance(state_invalid['market_data'], dict)  # Should be fixed in place
        
        logger.info("✅ Safe dictionary access test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Safe dictionary access test FAILED: {e}")
        return False

def test_signal_validator():
    """Test signal validation utilities"""
    logger.info("🧪 Testing signal validator...")
    
    try:
        from l2_tactic.signal_validator import validate_tactical_signal, validate_signal_list, create_fallback_signal
        
        # Test valid signal
        valid_signal = TacticalSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=0.8,
            confidence=0.7
        )
        
        validated = validate_tactical_signal(valid_signal)
        assert validated is not None, "Valid signal was rejected"
        assert hasattr(validated, 'action'), "Validated signal missing action attribute"
        
        # Test invalid signal (missing attributes)
        class InvalidSignal:
            def __init__(self):
                self.symbol = "BTCUSDT"
                # Missing required attributes
        
        invalid_signal = InvalidSignal()
        validated_invalid = validate_tactical_signal(invalid_signal)
        assert validated_invalid is None, "Invalid signal was not rejected"
        
        # Test signal list validation
        signals = [valid_signal, invalid_signal, None]
        validated_list = validate_signal_list(signals)
        assert len(validated_list) == 1, f"Expected 1 valid signal, got {len(validated_list)}"
        
        # Test fallback signal creation
        fallback = create_fallback_signal("ETHUSDT", "test")
        assert fallback.symbol == "ETHUSDT"
        assert fallback.side == "hold"
        assert hasattr(fallback, 'action')
        
        logger.info("✅ Signal validator test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ Signal validator test FAILED: {e}")
        return False

def test_finrl_signal_structure():
    """Test FinRL signal structure fixes"""
    logger.info("🧪 Testing FinRL signal structure...")
    
    try:
        # Test that we can create a proper TacticalSignal with all required attributes
        signal = TacticalSignal(
            symbol="BTCUSDT",
            side="buy",
            strength=0.8,
            confidence=0.7,
            signal_type="finrl",
            source="finrl",
            timestamp=pd.Timestamp.utcnow(),
            features={'rsi': 45.0, 'macd': 0.1}
        )
        
        # Verify all attributes exist
        required_attrs = ['symbol', 'side', 'strength', 'confidence', 'timestamp', 'features']
        for attr in required_attrs:
            assert hasattr(signal, attr), f"Signal missing required attribute: {attr}"
        
        # Test action attribute
        assert hasattr(signal, 'action'), "Signal missing action attribute"
        assert signal.action == signal.side, "Action doesn't match side"
        
        # Test to_order_signal method
        order_signal = signal.to_order_signal()
        assert isinstance(order_signal, dict), "to_order_signal didn't return dict"
        assert 'symbol' in order_signal, "Order signal missing symbol"
        assert 'side' in order_signal, "Order signal missing side"
        
        logger.info("✅ FinRL signal structure test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ FinRL signal structure test FAILED: {e}")
        return False

async def test_l2_signal_processing():
    """Test L2 signal processing with fixes"""
    logger.info("🧪 Testing L2 signal processing...")
    
    try:
        from l2_tactic.signal_generator import L2TacticProcessor
        from l2_tactic.config import L2Config
        
        # Create test configuration
        config = L2Config()
        processor = L2TacticProcessor(config)
        
        # Create test state with market data
        test_state = {
            'market_data': {
                'BTCUSDT': pd.DataFrame({
                    'open': [50000, 50100, 50200],
                    'high': [50200, 50300, 50400],
                    'low': [49800, 49900, 50000],
                    'close': [50100, 50200, 50300],
                    'volume': [1000, 1100, 1200]
                })
            },
            'portfolio': {'total': 10000.0}
        }
        
        # Test technical signals generation
        tech_signals = await processor.technical_signals(test_state)
        assert isinstance(tech_signals, list), "Technical signals not returned as list"
        
        # Validate each signal if any were generated
        for signal in tech_signals:
            assert isinstance(signal, TacticalSignal), f"Invalid signal type: {type(signal)}"
            assert hasattr(signal, 'action'), "Signal missing action attribute"
            assert signal.action in ['buy', 'sell', 'hold'], f"Invalid action: {signal.action}"
        
        logger.info("✅ L2 signal processing test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"❌ L2 signal processing test FAILED: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting comprehensive L2 fix tests...")
    
    tests = [
        ("TacticalSignal action attribute", test_tactical_signal_action_attribute),
        ("NaN validation", test_nan_validation),
        ("Safe dictionary access", test_safe_dict_access),
        ("Signal validator", test_signal_validator),
        ("FinRL signal structure", test_finrl_signal_structure),
        ("L2 signal processing", test_l2_signal_processing),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            logger.info(f"🧪 Running test: {test_name}")
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            if result:
                passed_tests += 1
                logger.info(f"✅ {test_name} - PASSED")
            else:
                logger.error(f"❌ {test_name} - FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
    
    logger.info(f"🎯 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests PASSED! L2 fixes are working correctly.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests FAILED. Please check the issues.")
        return False

async def main():
    """Main test function"""
    success = await run_all_tests()
    
    if success:
        logger.info("✅ All L2 error fixes have been verified and are working correctly!")
        logger.info("🔄 The HRM system should now run without the previous L2 errors.")
    else:
        logger.error("❌ Some tests failed. Please review the fixes and try again.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
