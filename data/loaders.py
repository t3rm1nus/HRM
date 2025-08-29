import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from .connectors.binance_connector import BinanceConnector
from comms.config import SYMBOLS

logger = logging.getLogger(__name__)

def normalize_column_names(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Normaliza nombres de columnas independientemente del origen."""
    column_mapping = {
        'close': 'close', 'Close': 'close', 'CLOSE': 'close',
        'open': 'open', 'Open': 'open', 'OPEN': 'open',
        'high': 'high', 'High': 'high', 'HIGH': 'high', 
        'low': 'low', 'Low': 'low', 'LOW': 'low',
        'volume': 'volume', 'Volume': 'volume', 'VOLUME': 'volume',
        'timestamp': 'timestamp', 'Timestamp': 'timestamp', 'TIMESTAMP': 'timestamp',
        'datetime': 'timestamp', 'Datetime': 'timestamp', 'DATETIME': 'timestamp'
    }
    
    df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    return df

def calculate_technical_indicators(df: pd.DataFrame, window_sizes: list = [10, 20, 50]) -> pd.DataFrame:
    """Calcula indicadores técnicos para el DataFrame."""
    df = df.copy()
    
    # Precios
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Medias móviles
    for window in window_sizes:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    
    # Volatilidad
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    df['atr_14'] = calculate_atr(df, window=14)
    
    # Momentum
    df['rsi_14'] = calculate_rsi(df['close'], window=14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    # Bollinger Bands
    df['bb_upper_20'], df['bb_lower_20'] = calculate_bollinger_bands(df['close'], window=20)
    df['bb_width_20'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['sma_20']
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    return df.dropna()

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calcula RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calcula MACD."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Calcula Bollinger Bands."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calcula Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum.reduce([high_low, high_close, low_close])
    return true_range.rolling(window=window).mean()

def build_multitimeframe_features(
    df_1m: pd.DataFrame, 
    df_5m: Optional[pd.DataFrame] = None, 
    symbol: str = ""
) -> pd.DataFrame:
    """
    Construye features multi-timeframe a partir de datos de mercado.
    """
    logger.info(f"🔧 Construyendo features para {symbol} - 1m shape: {df_1m.shape}")
    
    # Normalizar datos
    df_1m = normalize_column_names(df_1m, symbol)
    df_1m_features = calculate_technical_indicators(df_1m)
    
    # Features multi-timeframe si hay datos 5m
    if df_5m is not None and not df_5m.empty:
        df_5m = normalize_column_names(df_5m, symbol)
        df_5m_features = calculate_technical_indicators(df_5m)
        
        # Reindexar features 5m a 1m y añadir sufijo
        df_5m_reindexed = df_5m_features.reindex(df_1m_features.index, method='ffill')
        df_5m_reindexed = df_5m_reindexed.add_suffix('_5m')
        
        # Combinar features
        df_1m_features = pd.concat([df_1m_features, df_5m_reindexed], axis=1)
    
    # Añadir identificador de símbolo
    df_1m_features['symbol'] = symbol
    df_1m_features['is_btc'] = 1 if 'BTC' in symbol else 0
    df_1m_features['is_eth'] = 1 if 'ETH' in symbol else 0
    
    logger.info(f"✅ Features construidas para {symbol} - shape final: {df_1m_features.shape}")
    return df_1m_features.dropna()

def temporal_train_test_split(
    features: pd.DataFrame, 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide los datos en train/test temporal."""
    split_idx = int(len(features) * (1 - test_size))
    train = features.iloc[:split_idx]
    test = features.iloc[split_idx:]
    
    logger.info(f"📊 Split temporal: Train={len(train)}, Test={len(test)}")
    return train, test

def prepare_features(
    df_1m: pd.DataFrame,
    df_5m: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    symbol: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completa: normaliza columnas, genera features 1m + 5m y divide train/test.
    """
    features = build_multitimeframe_features(df_1m=df_1m, df_5m=df_5m, symbol=symbol)
    return temporal_train_test_split(features, test_size=test_size)

class RealTimeDataLoader:
    """Cargador de datos en tiempo real para todas las capas."""
    
    def __init__(self, real_time=True):
        self.real_time = real_time
        self.connector = BinanceConnector(testnet=True) if real_time else None
        self.cache = {}
    
    async def get_market_data(self, symbol: str, timeframe: str = "1m", limit: int = 100) -> pd.DataFrame:
        """Obtiene datos de mercado en tiempo real."""
        try:
            if self.real_time and self.connector:
                klines = self.connector.get_klines(symbol, timeframe, limit)
                if klines:
                    df = pd.DataFrame(klines, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignored"
                    ])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    numeric_cols = ["open", "high", "low", "close", "volume"]
                    df[numeric_cols] = df[numeric_cols].astype(float)
                    return df
            
            # Fallback: datos simulados o cached
            return self._get_cached_data(symbol, timeframe, limit)
            
        except Exception as e:
            logger.error(f"Error obteniendo datos para {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_cached_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Datos de respaldo para testing."""
        # Implementar lógica de cache o datos simulados
        return pd.DataFrame()
    
    async def get_features_for_symbol(self, symbol: str, timeframes: list = ["1m", "5m"]) -> pd.DataFrame:
        """Obtiene features completas para un símbolo."""
        try:
            df_1m = await self.get_market_data(symbol, "1m", 100)
            df_5m = await self.get_market_data(symbol, "5m", 100)
            
            if df_1m.empty:
                logger.warning(f"No hay datos para {symbol}")
                return pd.DataFrame()
            
            features = build_multitimeframe_features(df_1m, df_5m, symbol)
            return features
            
        except Exception as e:
            logger.error(f"Error generando features para {symbol}: {e}")
            return pd.DataFrame()