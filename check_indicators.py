import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from datetime import datetime
from l2_tactic.technical.multi_timeframe import MultiTimeframeTechnical
from test_pullback_strategy import create_mock_market_data

# Create pullback market data
market_data = create_mock_market_data(price_trend='pullback')
df = market_data['BTCUSDT']

# Create mock config
class MockConfig:
    pass

mock_config = MockConfig()

# Calculate technical indicators
processor = MultiTimeframeTechnical(mock_config)
indicators = processor.calculate_technical_indicators(df)

print('BTCUSDT Indicators:')
rsi = indicators.get('rsi', pd.Series([50]))
if isinstance(rsi, pd.Series) and not rsi.empty:
    print(f'RSI: {rsi.iloc[-1]:.1f}')
else:
    print(f'RSI: N/A')

# Check if ma50 is available (it might be called close_sma or sma_50)
ma50 = indicators.get('ma50')
if ma50 is None:
    ma50 = indicators.get('close_sma')  # Check if it's called close_sma
if ma50 is None:
    ma50 = indicators.get('sma_50')    # Check if it's called sma_50

if isinstance(ma50, pd.Series) and not ma50.empty:
    print(f'MA50: {ma50.iloc[-1]:.2f}')
else:
    print(f'MA50: N/A')

current_price = df['close'].iloc[-1]
print(f'Current Price: {current_price:.2f}')
print()

# Check if pullback conditions met
rsi_value = rsi.iloc[-1] if isinstance(rsi, pd.Series) and not rsi.empty else 50
ma50_value = ma50.iloc[-1] if isinstance(ma50, pd.Series) and not ma50.empty else 0

print(f'Pullback Conditions:')
print(f'RSI < 50: {rsi_value < 50}')
print(f'Price < MA50: {current_price < ma50_value}')
print(f'Both conditions: {rsi_value < 50 and current_price < ma50_value}')
