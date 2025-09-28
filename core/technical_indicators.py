# core/technical_indicators.py
import pandas as pd
import numpy as np
import logging

from core.logging import logger

def calculate_technical_indicators(market_data: dict) -> dict:
    """
    Calcula indicadores técnicos para múltiples símbolos.
    
    Args:
        market_data: Dict de DataFrames OHLCV por símbolo, p.ej. {"BTCUSDT": df, "ETHUSDT": df}
                     Cada DataFrame debe tener columnas: ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
        Dict con DataFrames de indicadores por símbolo, con columnas:
        ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
         'macd', 'macd_signal', 'rsi', 'bollinger_middle', 'bollinger_std', 'bollinger_upper',
         'bollinger_lower', 'vol_mean_20', 'vol_std_20', 'vol_zscore']
    """
    indicators = {}
    for symbol, df in market_data.items():
        if not validate_dataframe_for_indicators(df):
            logger.warning(f"{symbol}: No hay datos válidos para calcular indicadores")
            indicators[symbol] = pd.DataFrame()
            continue

        df_ind = df.copy()

        # Convertir columnas a float64 para evitar errores de tipo
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_ind.columns:
                df_ind[col] = pd.to_numeric(df_ind[col], errors='coerce')

        # SMA y EMA
        df_ind['sma_20'] = df_ind['close'].rolling(window=20, min_periods=20).mean()
        if len(df_ind) >= 50:
            df_ind['sma_50'] = df_ind['close'].rolling(window=50, min_periods=50).mean()
        else:
            logger.warning(f"{symbol}: Menos de 50 filas ({len(df_ind)}), sma_50 no calculado")
            df_ind['sma_50'] = np.nan
        df_ind['ema_12'] = df_ind['close'].ewm(span=12, adjust=False).mean()
        df_ind['ema_26'] = df_ind['close'].ewm(span=26, adjust=False).mean()

        # MACD
        df_ind['macd'] = df_ind['ema_12'] - df_ind['ema_26']
        df_ind['macd_signal'] = df_ind['macd'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df_ind['close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-9)  # Evitar división por cero
        df_ind['rsi'] = 100 - (100 / (1 + rs))
        # Log especial si RSI es 0 por falta de variación
        if (df_ind['rsi'] == 0).all():
            logger.warning(f"{symbol}: RSI=0 en todos los puntos. Posible datos corruptos o sin variación de precios.")
            indicators[symbol] = pd.DataFrame()
            continue

        # Bollinger Bands
        df_ind['bollinger_middle'] = df_ind['close'].rolling(window=20, min_periods=20).mean()
        df_ind['bollinger_std'] = df_ind['close'].rolling(window=20, min_periods=20).std()
        df_ind['bollinger_upper'] = df_ind['bollinger_middle'] + 2 * df_ind['bollinger_std']
        df_ind['bollinger_lower'] = df_ind['bollinger_middle'] - 2 * df_ind['bollinger_std']

        # Volumen normalizado
        df_ind['vol_mean_20'] = df_ind['volume'].rolling(window=20, min_periods=20).mean()
        df_ind['vol_std_20'] = df_ind['volume'].rolling(window=20, min_periods=20).std()
        df_ind['vol_zscore'] = (df_ind['volume'] - df_ind['vol_mean_20']) / (df_ind['vol_std_20'] + 1e-9)

        # MOMENTUM INDICATORS - NEW ADDITION
        # Rate of Change (ROC) - momentum over different periods
        df_ind['roc_5'] = df_ind['close'].pct_change(periods=5) * 100
        df_ind['roc_10'] = df_ind['close'].pct_change(periods=10) * 100
        df_ind['roc_20'] = df_ind['close'].pct_change(periods=20) * 100

        # Momentum (close - close_n_periods_ago)
        df_ind['momentum_5'] = df_ind['close'] - df_ind['close'].shift(5)
        df_ind['momentum_10'] = df_ind['close'] - df_ind['close'].shift(10)
        df_ind['momentum_20'] = df_ind['close'] - df_ind['close'].shift(20)

        # Williams %R - momentum oscillator
        highest_high_14 = df_ind['high'].rolling(window=14, min_periods=14).max()
        lowest_low_14 = df_ind['low'].rolling(window=14, min_periods=14).min()
        df_ind['williams_r'] = -100 * (highest_high_14 - df_ind['close']) / (highest_high_14 - lowest_low_14 + 1e-9)

        # Commodity Channel Index (CCI) - momentum and volatility
        typical_price = (df_ind['high'] + df_ind['low'] + df_ind['close']) / 3
        sma_tp_20 = typical_price.rolling(window=20, min_periods=20).mean()
        mad_tp_20 = (typical_price - sma_tp_20).abs().rolling(window=20, min_periods=20).mean()
        df_ind['cci'] = (typical_price - sma_tp_20) / (0.015 * mad_tp_20 + 1e-9)

        # Stochastic Oscillator - momentum
        lowest_low_14 = df_ind['low'].rolling(window=14, min_periods=14).min()
        highest_high_14 = df_ind['high'].rolling(window=14, min_periods=14).max()
        df_ind['stoch_k'] = 100 * (df_ind['close'] - lowest_low_14) / (highest_high_14 - lowest_low_14 + 1e-9)
        df_ind['stoch_d'] = df_ind['stoch_k'].rolling(window=3, min_periods=3).mean()

        # Average Directional Index (ADX) - trend strength
        # Calculate True Range
        df_ind['tr'] = np.maximum(
            df_ind['high'] - df_ind['low'],
            np.maximum(
                (df_ind['high'] - df_ind['close'].shift(1)).abs(),
                (df_ind['low'] - df_ind['close'].shift(1)).abs()
            )
        )

        # Calculate Directional Movement
        df_ind['dm_plus'] = np.where(
            (df_ind['high'] - df_ind['high'].shift(1)) > (df_ind['low'].shift(1) - df_ind['low']),
            np.maximum(df_ind['high'] - df_ind['high'].shift(1), 0),
            0
        )
        df_ind['dm_minus'] = np.where(
            (df_ind['low'].shift(1) - df_ind['low']) > (df_ind['high'] - df_ind['high'].shift(1)),
            np.maximum(df_ind['low'].shift(1) - df_ind['low'], 0),
            0
        )

        # Calculate Directional Indicators
        atr_14 = df_ind['tr'].rolling(window=14, min_periods=14).mean()
        di_plus = 100 * (df_ind['dm_plus'].rolling(window=14, min_periods=14).mean() / (atr_14 + 1e-9))
        di_minus = 100 * (df_ind['dm_minus'].rolling(window=14, min_periods=14).mean() / (atr_14 + 1e-9))

        # Calculate ADX
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9)
        df_ind['adx'] = dx.rolling(window=14, min_periods=14).mean()
        df_ind['di_plus'] = di_plus
        df_ind['di_minus'] = di_minus

        # Eliminar filas con NaN en indicadores clave (excepto sma_50)
        required_indicators = ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']
        df_ind = df_ind.dropna(subset=required_indicators)

        # Verificar si el DataFrame resultante está vacío tras eliminar NaN
        if df_ind.empty:
            logger.warning(f"{symbol}: DataFrame vacío tras eliminar NaN en indicadores clave")
            indicators[symbol] = pd.DataFrame()
            continue

        indicators[symbol] = df_ind

        debug_dataframe_types(df_ind, name=f"{symbol} indicadores")
    return indicators

def validate_dataframe_for_indicators(df: pd.DataFrame) -> bool:
    """
    Valida que un DataFrame sea apto para calcular indicadores técnicos.
    
    Args:
        df: DataFrame con datos OHLCV
    
    Returns:
        bool: True si el DataFrame es válido, False si no
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if df is None or df.empty:
        logger.warning("DataFrame vacío o None")
        return False
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning(f"DataFrame falta columnas: {missing}")
        return False
    
    if len(df) < 20:  # Mínimo para SMA_20, Bollinger Bands y volumen (más estricto que 14 para RSI)
        logger.warning(f"DataFrame tiene menos de 20 filas: {len(df)}")
        return False
    
    # Verificar tipos numéricos
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Columna {col} tiene tipo no numérico: {df[col].dtype}")
            return False
    
    # Verificar que no haya NaN en las columnas requeridas
    if df[required_cols].isna().any().any():
        logger.warning(f"DataFrame tiene valores NaN en columnas requeridas: {df[required_cols].isna().sum()}")
        return False
    
    return True

def debug_dataframe_types(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Imprime información de depuración sobre tipos de datos y uso de memoria.

    Args:
        df: DataFrame a inspeccionar
        name: Nombre para identificar el DataFrame en los logs
    """
    if df is None or df.empty:
        logger.info(f"{name} está vacío")
        return

    logger.info(f"{name} info:")
    logger.info(df.info())
    logger.info(df.dtypes.value_counts())
    logger.info(f"{name} memoria: {df.memory_usage(deep=True).sum()/1024:.2f} KB")

def calculate_range_indicators(df):
    """
    Calcula indicadores específicos para detectar y operar en mercados en rango.

    Args:
        df: DataFrame con columnas OHLCV

    Returns:
        DataFrame con indicadores de rango añadidos
    """
    try:
        df_ind = df.copy()

        # Rango de precios reciente (20 periodos)
        df_ind['range_high_20'] = df_ind['high'].rolling(20).max()
        df_ind['range_low_20'] = df_ind['low'].rolling(20).min()
        df_ind['range_middle'] = (df_ind['range_high_20'] + df_ind['range_low_20']) / 2

        # Fuerza del rango (ancho del rango relativo al precio medio)
        df_ind['range_strength'] = (df_ind['range_high_20'] - df_ind['range_low_20']) / df_ind['range_middle']

        # Indicador de consolidación (menor volatilidad = mayor consolidación)
        df_ind['consolidation_ratio'] = df_ind['range_strength'].rolling(10).mean()

        # Mean reversion signal (distancia del precio al rango medio)
        df_ind['range_deviation'] = (df_ind['close'] - df_ind['range_middle']) / df_ind['range_middle']

        logger.info("📊 Indicadores de rango calculados exitosamente")
        return df_ind

    except Exception as e:
        logger.error(f"❌ Error calculando indicadores de rango: {e}")
        return df
