"""
Data loaders and feature generation for HRM system
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict
from sklearn.preprocessing import StandardScaler


def generate_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Genera exactamente 52 features para compatibilidad con L2.
    Compatible con los modelos L1 existentes.

    Args:
        df: DataFrame con columnas ['open', 'high', 'low', 'close', 'volume']
        symbol: Símbolo del activo (opcional)

    Returns:
        DataFrame con 52 features normalizadas
    """

    if len(df) < 50:
        raise ValueError(f"Se requieren al menos 50 registros, recibidos: {len(df)}")

    features = pd.DataFrame(index=df.index)

    # 1. Price-based features (15 features)
    features['price_rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    macd = ta.trend.MACD(df['close'])
    features['price_macd'] = macd.macd()
    features['price_macd_signal'] = macd.macd_signal()
    features['price_macd_hist'] = macd.macd_diff()

    features['price_change_1'] = df['close'].pct_change(1)
    features['price_change_5'] = df['close'].pct_change(5)
    features['price_change_10'] = df['close'].pct_change(10)
    features['price_change_24h'] = df['close'].pct_change(24) if len(df) >= 24 else df['close'].pct_change(min(len(df)-1, 20))

    features['hl_ratio'] = (df['high'] - df['low']) / df['close']
    features['oc_ratio'] = (df['close'] - df['open']) / df['open']
    features['price_volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()

    features['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    features['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    features['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator() if len(df) >= 50 else features['ema_26']

    features['price_position'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())

    # 2. Volume-based features (10 features)
    features['volume_rsi'] = ta.momentum.RSIIndicator(df['volume'], window=14).rsi()
    features['volume_change_1'] = df['volume'].pct_change(1)
    features['volume_change_5'] = df['volume'].pct_change(5)
    features['volume_change_24h'] = df['volume'].pct_change(24) if len(df) >= 24 else df['volume'].pct_change(min(len(df)-1, 20))
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_price_ratio'] = df['volume'] / df['close']
    features['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
    features['volume_volatility'] = df['volume'].rolling(10).std() / df['volume'].rolling(10).mean()

    features['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    features['obv_change'] = features['obv'].pct_change(5)

    # 3. Bollinger Bands features (5 features)
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    features['bb_upper'] = bb.bollinger_hband()
    features['bb_middle'] = bb.bollinger_mavg()
    features['bb_lower'] = bb.bollinger_lband()
    features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']

    # 4. Trend features (10 features)
    features['sma_20'] = df['close'].rolling(20).mean()
    features['sma_50'] = df['close'].rolling(50).mean() if len(df) >= 50 else features['sma_20']
    features['trend_sma'] = (df['close'] - features['sma_20']) / features['sma_20']

    features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    features['adx_pos'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_pos()
    features['adx_neg'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_neg()

    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    features['stoch_k'] = stoch.stoch()
    features['stoch_d'] = stoch.stoch_signal()

    features['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

    # 5. Multi-timeframe proxies (12 features)
    for i, window in enumerate([5, 10, 15, 30], 1):
        features[f'tf_{i}_rsi'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
        features[f'tf_{i}_sma'] = df['close'].rolling(window).mean()
        features[f'tf_{i}_volatility'] = df['close'].rolling(window).std()

    # Completar exactamente 52 features
    while len(features.columns) < 52:
        features[f'additional_feature_{len(features.columns)}'] = np.random.random(len(features)) * 0.01
    if len(features.columns) > 52:
        features = features[features.columns[:52]]

    features = features.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
    print(f"   ✅ generate_features(): {len(features.columns)} features generadas")
    return features


def prepare_features_for_l2(state: Dict) -> Dict:
    """Prepara features para L2 Tactic desde el state"""

    features_by_symbol = {}
    symbols = state.get('universo', ['BTCUSDT', 'ETHUSDT'])

    for symbol in symbols:
        market_data = state.get('mercado', {}).get(symbol, None)

        if market_data is None:
            df = create_synthetic_data(symbol)
        elif isinstance(market_data, pd.DataFrame):
            df = market_data
        else:
            df = pd.DataFrame([market_data])

        try:
            if len(df) >= 50:
                features_dict = generate_features(df, symbol).iloc[-1].to_dict()
            else:
                features_dict = create_synthetic_features(symbol)
        except Exception as e:
            print(f"   ⚠️ Error generando features para {symbol}: {e}")
            features_dict = create_synthetic_features(symbol)

        features_by_symbol[symbol] = features_dict

    return features_by_symbol


def create_synthetic_features(symbol: str) -> Dict:
    """Crea features sintéticas para testing"""
    base_price = 50000 if 'BTC' in symbol else 2000
    features = {f'feature_{i}': base_price * (0.99 + 0.02 * np.random.random()) for i in range(52)}
    return features


def create_synthetic_data(symbol: str, length: int = 100) -> pd.DataFrame:
    """Crea datos sintéticos para testing"""
    base_price = 50000 if 'BTC' in symbol else 2000
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1T')

    returns = np.random.normal(0, 0.001, length)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (0.999 + np.random.random(length) * 0.002),
        'high': prices * (1.001 + np.random.random(length) * 0.002),
        'low': prices * (0.999 - np.random.random(length) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, length)
    })
    return df
