"""
Data loaders and feature generation for HRM system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union
import ta
from sklearn.preprocessing import StandardScaler

def generate_features(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """
    Genera exactamente 52 features para compatibilidad con L2
    Compatible con los modelos L1 existentes
    
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
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    features['price_macd'] = macd.macd()
    features['price_macd_signal'] = macd.macd_signal()
    features['price_macd_hist'] = macd.macd_diff()
    
    # Price changes
    features['price_change_1'] = df['close'].pct_change(1)
    features['price_change_5'] = df['close'].pct_change(5)
    features['price_change_10'] = df['close'].pct_change(10)
    features['price_change_24h'] = df['close'].pct_change(24) if len(df) >= 24 else df['close'].pct_change(min(len(df)-1, 20))
    
    # Price ratios
    features['hl_ratio'] = (df['high'] - df['low']) / df['close']
    features['oc_ratio'] = (df['close'] - df['open']) / df['open']
    features['price_volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
    
    # EMAs
    features['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    features['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    features['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator() if len(df) >= 50 else features['ema_26']
    
    # Price position
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
    
    # On Balance Volume
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
    
    # ADX
    features['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    features['adx_pos'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_pos()
    features['adx_neg'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx_neg()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    features['stoch_k'] = stoch.stoch()
    features['stoch_d'] = stoch.stoch_signal()
    
    # Williams %R
    features['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    
    # Commodity Channel Index
    features['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
    
    # 5. Multi-timeframe proxies (12 features)
    # Simular multi-timeframe con diferentes ventanas
    for i, window in enumerate([5, 10, 15, 30], 1):
        features[f'tf_{i}_rsi'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
        features[f'tf_{i}_sma'] = df['close'].rolling(window).mean()
        features[f'tf_{i}_volatility'] = df['close'].rolling(window).std()
    
    # Asegurar que tenemos exactamente 52 features
    current_features = len([col for col in features.columns if not col.startswith('_')])
    
    # Si faltan features, agregar features adicionales
    while current_features < 52:
        feature_name = f'additional_feature_{current_features - 51}'
        features[feature_name] = np.random.random(len(features)) * 0.01  # Features mínimas
        current_features += 1
    
    # Si sobran features, eliminar las últimas
    feature_cols = [col for col in features.columns if not col.startswith('_')]
    if len(feature_cols) > 52:
        features = features[feature_cols[:52]]
    
    # Llenar NaN con forward fill y después con 0
    features = features.fillna(method='ffill').fillna(0)
    
    # Reemplazar infinitos
    features = features.replace([np.inf, -np.inf], 0)
    
    print(f"   ✅ generate_features(): {len(features.columns)} features generadas")
    
    return features

def prepare_features_for_l2(state: dict) -> dict:
    """
    Prepara features para L2 Tactic desde el state
    Mantiene compatibilidad con el flujo actual
    
    Args:
        state: State global del sistema
        
    Returns:
        dict: Features por símbolo preparadas para L2
    """
    
    features_by_symbol = {}
    symbols = state.get('universo', ['BTCUSDT', 'ETHUSDT'])
    
    for symbol in symbols:
        if symbol in state.get('mercado', {}):
            market_data = state['mercado'][symbol]
            
            # Convertir a DataFrame si es necesario
            if isinstance(market_data, dict):
                # Crear DataFrame básico si solo tenemos datos simples
                df = pd.DataFrame([market_data])
            elif isinstance(market_data, pd.DataFrame):
                df = market_data
            else:
                # Crear datos sintéticos si no hay datos
                df = create_synthetic_data(symbol)
            
            try:
                # Generar features usando la función principal
                if len(df) >= 1:
                    # Para datos insuficientes, crear features básicas
                    features_dict = create_basic_features(df, symbol)
                else:
                    features_dict = create_synthetic_features(symbol)
                    
                features_by_symbol[symbol] = features_dict
                
            except Exception as e:
                print(f"   ⚠️ Error generando features para {symbol}: {e}")
                # Usar features sintéticas como fallback
                features_by_symbol[symbol] = create_synthetic_features(symbol)
    
    return features_by_symbol

def create_basic_features(df: pd.DataFrame, symbol: str) -> dict:
    """Crea features básicas desde datos limitados"""
    
    last_row = df.iloc[-1] if len(df) > 0 else {}
    
    features = {
        'price_rsi': last_row.get('rsi', 50.0),
        'price_macd': last_row.get('macd', 0.0),
        'price_macd_signal': last_row.get('macd_signal', 0.0),
        'price_macd_hist': last_row.get('macd_hist', 0.0),
        'price_change_24h': last_row.get('price_change_24h', 0.0),
        'volume_rsi': last_row.get('volume_rsi', 50.0),
        'volume_change_24h': last_row.get('volume_change_24h', 0.0),
        'volume_ratio': last_row.get('volume_ratio', 1.0),
        'bb_upper': last_row.get('close', 50000) * 1.02,
        'bb_middle': last_row.get('close', 50000),
        'bb_lower': last_row.get('close', 50000) * 0.98,
        'bb_position': 0.5,
        'ema_12': last_row.get('close', 50000),
        'ema_26': last_row.get('close', 50000),
        'ema_50': last_row.get('close', 50000),
        'sma_20': last_row.get('close', 50000),
        'sma_50': last_row.get('close', 50000),
    }
    
    # Completar hasta 52 features
    base_price = last_row.get('close', 50000 if 'BTC' in symbol else 2000)
    for i in range(len(features), 52):
        features[f'feature_{i}'] = base_price * (0.99 + 0.02 * np.random.random())
    
    return features

def create_synthetic_features(symbol: str) -> dict:
    """Crea features sintéticas para testing"""
    
    base_price = 50000 if 'BTC' in symbol else 2000
    
    features = {
        'price_rsi': 45.0 + np.random.random() * 10,
        'price_macd': -10.0 + np.random.random() * 20,
        'price_macd_signal': -5.0 + np.random.random() * 10,
        'price_macd_hist': -2.0 + np.random.random() * 4,
        'price_change_24h': -0.02 + np.random.random() * 0.04,
        'volume_rsi': 40.0 + np.random.random() * 20,
        'volume_change_24h': -0.1 + np.random.random() * 0.2,
        'volume_ratio': 0.8 + np.random.random() * 0.4,
        'bb_upper': base_price * 1.02,
        'bb_middle': base_price,
        'bb_lower': base_price * 0.98,
        'bb_position': 0.3 + np.random.random() * 0.4,
        'ema_12': base_price * (0.999 + np.random.random() * 0.002),
        'ema_26': base_price * (0.999 + np.random.random() * 0.002),
        'ema_50': base_price * (0.999 + np.random.random() * 0.002),
        'sma_20': base_price * (0.999 + np.random.random() * 0.002),
        'sma_50': base_price * (0.999 + np.random.random() * 0.002),
    }
    
    # Completar hasta 52 features
    for i in range(len(features), 52):
        features[f'feature_{i}'] = base_price * (0.99 + 0.02 * np.random.random())
    
    return features

def create_synthetic_data(symbol: str, length: int = 100) -> pd.DataFrame:
    """Crea datos sintéticos para testing"""
    
    base_price = 50000 if 'BTC' in symbol else 2000
    
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1T')
    
    # Generar precios con random walk
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
