import pandas as pd
import numpy as np


def safe_divide(a, b, default=0):
    """Safe division that handles division by zero"""
    return np.where(b != 0, a / b, default)

def calculate_rsi_safe(prices, period=14):
    """Calculate RSI with safety checks"""
    if len(prices) < period + 1:
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    
    # Use safe division
    rs = safe_divide(gain, loss, 0)
    rsi = 100 - safe_divide(100, (1 + rs), 50)
    
    return rsi

def calculate_indicators_safe(df):
    """Calculate technical indicators with error handling"""
    try:
        if len(df) == 0:
            return df
        
        # RSI
        df['rsi'] = calculate_rsi_safe(df['close'])
        
        # Moving averages
        df['ema_10'] = df['close'].ewm(span=10, min_periods=1).mean()
        df['ema_20'] = df['close'].ewm(span=20, min_periods=1).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12, min_periods=1).mean()
        ema_26 = df['close'].ewm(span=26, min_periods=1).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume indicators
        vol_mean = df['volume'].rolling(window=20, min_periods=1).mean()
        df['vol_rel'] = safe_divide(df['volume'], vol_mean, 1.0)
        
        return df
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return df
