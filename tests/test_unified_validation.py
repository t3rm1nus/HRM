# -*- coding: utf-8 -*-
"""
Test for Unified Validation System
Tests the centralized validation utilities that eliminate code duplication.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core.unified_validation import UnifiedValidator, validate_market_data_structure, validate_ohlcv_data, validate_and_fix_market_data


class TestUnifiedValidation:
    """Test cases for the unified validation system."""

    def test_validate_market_data_structure_valid_dict(self):
        """Test validation of valid market data structure."""
        valid_data = {
            'BTCUSDT': pd.DataFrame({'close': [50000, 50100]}),
            'ETHUSDT': pd.DataFrame({'close': [3500, 3550]})
        }
        is_valid, message = UnifiedValidator.validate_market_data_structure(valid_data)
        assert is_valid == True
        assert 'Valid symbols' in message

    def test_validate_market_data_structure_invalid(self):
        """Test validation of invalid market data structure."""
        invalid_data = None
        is_valid, message = UnifiedValidator.validate_market_data_structure(invalid_data)
        assert is_valid == False
        assert 'Data is None' in message

    def test_validate_ohlcv_data_valid_dataframe(self):
        """Test validation of valid OHLCV DataFrame."""
        df = pd.DataFrame({
            'open': [50000, 50100],
            'high': [50500, 50400],
            'low': [49500, 49900],
            'close': [50200, 50100],
            'volume': [1.2, 1.5]
        })
        validated_df, message = UnifiedValidator.validate_ohlcv_data(df)
        assert validated_df is not None
        assert 'Successfully validated' in message

    def test_validate_ohlcv_data_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'open': [50000, 50100],
            'close': [50200, 50100]
        })
        validated_df, message = UnifiedValidator.validate_ohlcv_data(df)
        assert validated_df is None
        assert 'Missing required columns' in message

    def test_validate_symbol_data_required_valid(self):
        """Test validation of required symbols with valid data."""
        symbols = ['BTCUSDT', 'ETHUSDT']
        market_data = {
            'BTCUSDT': pd.DataFrame({'open': [50000], 'high': [50500], 'low': [49500], 'close': [50200], 'volume': [1.2]}),
            'ETHUSDT': pd.DataFrame({'open': [3500], 'high': [3550], 'low': [3480], 'close': [3520], 'volume': [10]})
        }
        valid_data, message = UnifiedValidator.validate_symbol_data_required(symbols, market_data)
        assert len(valid_data) == 2
        assert 'Validated 2/2 symbols' in message

    def test_validate_symbol_data_required_missing_symbol(self):
        """Test validation with missing required symbol."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        market_data = {
            'BTCUSDT': pd.DataFrame({'open': [50000], 'high': [50500], 'low': [49500], 'close': [50200], 'volume': [1.2]}),
            'ETHUSDT': pd.DataFrame({'open': [3500], 'high': [3550], 'low': [3480], 'close': [3520], 'volume': [10]})
        }
        valid_data, message = UnifiedValidator.validate_symbol_data_required(symbols, market_data)
        assert len(valid_data) == 2
        assert 'Missing symbols: [\'ADAUSDT\']' in message

    def test_validate_and_fix_market_data_valid(self):
        """Test comprehensive market data validation and fix."""
        state = {
            'market_data': {
                'BTCUSDT': pd.DataFrame({'open': [50000], 'high': [50500], 'low': [49500], 'close': [50200], 'volume': [1.2]}),
                'ETHUSDT': pd.DataFrame({'open': [3500], 'high': [3550], 'low': [3480], 'close': [3520], 'volume': [10]})
            }
        }
        config = {'SYMBOLS': ['BTCUSDT', 'ETHUSDT']}

        fixed_data, message = UnifiedValidator.validate_and_fix_market_data(state, config)
        assert len(fixed_data) == 2
        assert 'Validated 2/2 symbols' in message

    def test_validate_trading_parameters_valid(self):
        """Test validation of valid trading parameters."""
        is_valid, message = UnifiedValidator.validate_trading_parameters(
            symbol='BTCUSDT',
            quantity=0.01,
            price=50000.0,
            side='buy'
        )
        assert is_valid == True
        assert 'Valid trading parameters' in message

    def test_validate_trading_parameters_invalid_quantity(self):
        """Test validation with invalid quantity."""
        is_valid, message = UnifiedValidator.validate_trading_parameters(
            symbol='BTCUSDT',
            quantity=-0.01,
            price=50000.0,
            side='buy'
        )
        assert is_valid == False
        assert 'Invalid quantity' in message

    def test_sanitize_numeric_value_with_default(self):
        """Test sanitizing numeric values with default handling."""
        assert UnifiedValidator.sanitize_numeric_value(None, default=5.0) == 5.0
        assert UnifiedValidator.sanitize_numeric_value(np.nan, default=5.0) == 5.0
        assert UnifiedValidator.sanitize_numeric_value('invalid', default=5.0) == 5.0
        assert UnifiedValidator.sanitize_numeric_value(10.5) == 10.5

    def test_clean_portfolio_data(self):
        """Test cleaning of portfolio data."""
        portfolio_data = {
            'BTCUSDT': 0.5,
            'ETHUSDT': 0.3,
            'USDT': 100.0,
            'invalid_key': None,
            'also_invalid': 'not_a_number',
            123: 45.0  # Non-string key should be filtered out
        }
        cleaned = UnifiedValidator.clean_portfolio_data(portfolio_data)
        assert 'BTCUSDT' in cleaned
        assert 'ETHUSDT' in cleaned
        assert 'USDT' in cleaned
        assert 'invalid_key' in cleaned  # String keys with None values get converted to 0.0
        assert 123 not in cleaned  # Non-string keys are filtered out
        assert cleaned['invalid_key'] == 0.0  # None converted to 0.0
        assert all(isinstance(v, float) for v in cleaned.values())

    def test_backward_compatibility_functions(self):
        """Test that backward compatibility functions work."""
        # Test validate_market_data_structure wrapper
        valid_data = {'BTCUSDT': pd.DataFrame({'close': [50000]})}
        is_valid, message = validate_market_data_structure(valid_data)
        assert is_valid == True

        # Test validate_ohlcv_data wrapper
        df = pd.DataFrame({'open': [50000], 'high': [50500], 'low': [49500], 'close': [50200], 'volume': [1.2]})
        validated_df, message = validate_ohlcv_data(df)
        assert validated_df is not None

        # Test validate_and_fix_market_data wrapper
        state = {'market_data': valid_data}
        config = {'SYMBOLS': ['BTCUSDT']}
        fixed_data, message = validate_and_fix_market_data(state, config)
        assert len(fixed_data) >= 0

    def test_integration_main_loop_validation(self):
        """Test integration with main loop validation patterns."""
        # Simulate the pattern used in main.py's cycle validation
        state = {
            "market_data": {
                'BTCUSDT': pd.DataFrame({'open': [50000], 'high': [50500], 'low': [49500], 'close': [50200], 'volume': [1.2]}),
                'ETHUSDT': pd.DataFrame({'open': [3500], 'high': [3550], 'low': [3480], 'close': [3520], 'volume': [10]})
            }
        }
        config = {"SYMBOLS": ["BTCUSDT", "ETHUSDT"]}

        # This simulates the main loop validation pattern
        try:
            # Get market data (simulating loader.get_realtime_data())
            market_data = state["market_data"]

            # Validate structure (centralized validation)
            is_valid, validation_msg = UnifiedValidator.validate_market_data_structure(market_data)

            if is_valid:
                # Further validate required symbols
                symbols = config["SYMBOLS"]
                missing_symbols = [sym for sym in symbols if sym not in market_data]
                if not missing_symbols:
                    # Validation successful - this would continue normal processing
                    assert True  # Test passes if we reach here without exceptions
                else:
                    assert False, f"Missing symbols: {missing_symbols}"
            else:
                assert False, f"Validation failed: {validation_msg}"

        except Exception as e:
            assert False, f"Unexpected exception during integration test: {e}"

if __name__ == "__main__":
    pytest.main([__file__])
