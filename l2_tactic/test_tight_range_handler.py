"""
Test for tight range handler
"""

import pytest
import numpy as np
import pandas as pd
from l2_tactic.tight_range_handler import PATH2TightRangeFix


def test_tight_range_handler_buy_signal():
    """Test that tight range handler generates buy signal when RSI < 40"""
    handler = PATH2TightRangeFix()
    
    # Create mock market data with RSI < 40
    np.random.seed(42)
    close_prices = np.random.uniform(50000, 51000, 60)
    # Create a downtrend to get RSI < 40
    close_prices[-14:] = np.linspace(51000, 49500, 14)
    
    market_data = pd.DataFrame({
        'close': close_prices,
        'high': close_prices + 100,
        'low': close_prices - 100,
        'volume': np.random.uniform(1000, 5000, 60)
    })
    
    signal = handler.process_tight_range_signal('BTCUSDT', market_data, 0.8, 'HOLD')
    
    assert signal['action'] == 'BUY'
    assert signal['confidence'] > 0.0
    assert signal['allow_partial_rebalance'] == True
    assert signal['market_making_enabled'] == True
    assert 'RSI < 40' in signal['reason']


def test_tight_range_handler_sell_signal():
    """Test that tight range handler generates sell signal when RSI > 60"""
    handler = PATH2TightRangeFix()
    
    # Create mock market data with RSI > 60
    np.random.seed(43)
    close_prices = np.random.uniform(50000, 51000, 60)
    # Create an uptrend to get RSI > 60
    close_prices[-14:] = np.linspace(50000, 51500, 14)
    
    market_data = pd.DataFrame({
        'close': close_prices,
        'high': close_prices + 100,
        'low': close_prices - 100,
        'volume': np.random.uniform(1000, 5000, 60)
    })
    
    signal = handler.process_tight_range_signal('BTCUSDT', market_data, 0.8, 'HOLD')
    
    assert signal['action'] == 'SELL'
    assert signal['confidence'] > 0.0
    assert signal['allow_partial_rebalance'] == True
    assert signal['market_making_enabled'] == True
    assert 'RSI > 60' in signal['reason']


def test_tight_range_handler_hold_signal():
    """Test that tight range handler generates hold signal when RSI is between 40 and 60"""
    handler = PATH2TightRangeFix()
    
    # Create mock market data with RSI between 40 and 60
    np.random.seed(44)
    close_prices = np.random.uniform(50000, 51000, 60)
    
    market_data = pd.DataFrame({
        'close': close_prices,
        'high': close_prices + 100,
        'low': close_prices - 100,
        'volume': np.random.uniform(1000, 5000, 60)
    })
    
    signal = handler.process_tight_range_signal('BTCUSDT', market_data, 0.8, 'HOLD')
    
    assert signal['action'] == 'HOLD'
    assert signal['confidence'] > 0.0
    assert signal['allow_partial_rebalance'] == True
    assert signal['market_making_enabled'] == True
    assert 'Balanced RSI' in signal['reason']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
