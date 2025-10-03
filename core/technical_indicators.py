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

    logger.debug(f"{name} info: {len(df)} rows, {len(df.columns)} columns")
    logger.debug(f"{name} dtypes: {df.dtypes.value_counts().to_dict()}")
    logger.info(f"{name} memoria: {df.memory_usage(deep=True).sum()/1024:.2f} KB")

def calculate_technical_strength_score(indicators_df: pd.DataFrame, symbol: str = "") -> float:
    """
    Calculate technical strength score combining RSI, MACD, volume, and trend indicators.

    Args:
        indicators_df: DataFrame with technical indicators
        symbol: Symbol name for logging

    Returns:
        Technical strength score (0.0 to 1.0)
    """
    try:
        if indicators_df is None or indicators_df.empty:
            logger.warning(f"⚠️ No indicators data for technical strength calculation{symbol}")
            return 0.5  # Neutral score

        # Get latest values
        latest = indicators_df.iloc[-1]

        # RSI Component (0-1 scale, higher when RSI is in favorable ranges)
        rsi = latest.get('rsi', 50.0)
        if rsi <= 30:
            rsi_score = 0.8  # Oversold - bullish signal
        elif rsi <= 45:
            rsi_score = 0.6  # Moderately oversold
        elif rsi <= 55:
            rsi_score = 0.5  # Neutral
        elif rsi <= 70:
            rsi_score = 0.6  # Moderately overbought
        else:
            rsi_score = 0.2  # Overbought - bearish signal

        # MACD Component (0-1 scale based on signal strength)
        macd = latest.get('macd', 0.0)
        macd_signal = latest.get('macd_signal', 0.0)
        macd_diff = macd - macd_signal

        # Normalize MACD difference (assuming typical range of -50 to +50)
        macd_normalized = max(-1.0, min(1.0, macd_diff / 50.0))
        macd_score = 0.5 + (macd_normalized * 0.5)  # Convert to 0-1 scale

        # Volume Component (0-1 scale based on volume strength)
        vol_zscore = latest.get('vol_zscore', 0.0)
        # Higher volume (positive z-score) indicates stronger conviction
        volume_score = max(0.0, min(1.0, 0.5 + (vol_zscore * 0.3)))

        # Trend Strength Component using ADX
        adx = latest.get('adx', 25.0)
        # ADX > 25 indicates trending market, higher ADX = stronger trend
        trend_score = max(0.0, min(1.0, adx / 50.0))

        # Momentum Component using ROC and Williams %R
        roc_5 = latest.get('roc_5', 0.0)
        williams_r = latest.get('williams_r', -50.0)

        # Combine momentum indicators
        momentum_score = 0.5
        if roc_5 > 2.0 and williams_r < -20:
            momentum_score = 0.8  # Strong bullish momentum
        elif roc_5 > 1.0 and williams_r < -30:
            momentum_score = 0.7  # Moderate bullish momentum
        elif roc_5 < -2.0 and williams_r > -80:
            momentum_score = 0.2  # Strong bearish momentum
        elif roc_5 < -1.0 and williams_r > -70:
            momentum_score = 0.3  # Moderate bearish momentum

        # Weighted combination of all components
        weights = {
            'rsi': 0.25,
            'macd': 0.25,
            'volume': 0.20,
            'trend': 0.15,
            'momentum': 0.15
        }

        strength_score = (
            rsi_score * weights['rsi'] +
            macd_score * weights['macd'] +
            volume_score * weights['volume'] +
            trend_score * weights['trend'] +
            momentum_score * weights['momentum']
        )

        # Ensure score is within bounds
        strength_score = max(0.0, min(1.0, strength_score))

        logger.debug(f"🎯 Technical Strength Score for {symbol}: {strength_score:.3f}")
        logger.debug(f"   Components - RSI: {rsi_score:.3f}, MACD: {macd_score:.3f}, Volume: {volume_score:.3f}, Trend: {trend_score:.3f}, Momentum: {momentum_score:.3f}")

        return strength_score

    except Exception as e:
        logger.error(f"❌ Error calculating technical strength score{symbol}: {e}")
        return 0.5  # Return neutral score on error


def calculate_convergence_multiplier(l1_l2_agreement: float, l1_confidence: float = 0.5, l2_confidence: float = 0.5) -> float:
    """
    Calculate convergence multiplier based on L1+L2 agreement levels.

    Args:
        l1_l2_agreement: Agreement level between L1 and L2 signals (0.0 to 1.0)
        l1_confidence: L1 signal confidence (0.0 to 1.0)
        l2_confidence: L2 signal confidence (0.0 to 1.0)

    Returns:
        Convergence multiplier for position sizing (0.5 to 2.0)
    """
    try:
        # Base multiplier from agreement level
        if l1_l2_agreement >= 0.9:
            base_multiplier = 2.0  # Perfect agreement - maximum sizing
        elif l1_l2_agreement >= 0.8:
            base_multiplier = 1.8  # Strong agreement
        elif l1_l2_agreement >= 0.7:
            base_multiplier = 1.5  # Good agreement
        elif l1_l2_agreement >= 0.6:
            base_multiplier = 1.2  # Moderate agreement
        elif l1_l2_agreement >= 0.5:
            base_multiplier = 1.0  # Neutral agreement
        elif l1_l2_agreement >= 0.4:
            base_multiplier = 0.8  # Weak agreement
        elif l1_l2_agreement >= 0.3:
            base_multiplier = 0.7  # Poor agreement
        else:
            base_multiplier = 0.5  # Very poor agreement - minimum sizing

        # Adjust based on confidence levels
        avg_confidence = (l1_confidence + l2_confidence) / 2.0

        # Confidence bonus/penalty
        if avg_confidence >= 0.8:
            confidence_adjustment = 1.2  # High confidence bonus
        elif avg_confidence >= 0.7:
            confidence_adjustment = 1.1  # Moderate confidence bonus
        elif avg_confidence >= 0.6:
            confidence_adjustment = 1.0  # Neutral
        elif avg_confidence >= 0.5:
            confidence_adjustment = 0.9  # Low confidence penalty
        else:
            confidence_adjustment = 0.8  # Very low confidence penalty

        final_multiplier = base_multiplier * confidence_adjustment

        # Ensure reasonable bounds
        final_multiplier = max(0.3, min(2.5, final_multiplier))

        logger.debug(f"🔄 Convergence Multiplier: agreement={l1_l2_agreement:.3f}, avg_conf={avg_confidence:.3f} → multiplier={final_multiplier:.3f}")

        return final_multiplier

    except Exception as e:
        logger.error(f"❌ Error calculating convergence multiplier: {e}")
        return 1.0  # Return neutral multiplier on error


def validate_technical_strength_for_position_size(strength_score: float, position_size_usd: float, symbol: str = "") -> bool:
    """
    Validate if technical strength meets minimum requirements for large positions.
    Includes circuit breaker logic for extreme conditions.

    Args:
        strength_score: Technical strength score (0.0 to 1.0)
        position_size_usd: Position size in USD
        symbol: Symbol name for logging

    Returns:
        True if position size is allowed, False if rejected
    """
    try:
        # Input validation with circuit breakers
        if not isinstance(strength_score, (int, float)) or not (0.0 <= strength_score <= 1.0):
            logger.error(f"🚨 CIRCUIT BREAKER: Invalid strength score {strength_score} for {symbol}")
            return False

        if not isinstance(position_size_usd, (int, float)) or position_size_usd < 0:
            logger.error(f"🚨 CIRCUIT BREAKER: Invalid position size {position_size_usd} for {symbol}")
            return False

        # Emergency circuit breaker: Reject all positions if strength is extremely low
        if strength_score < 0.1:
            logger.error(f"🚨 EMERGENCY CIRCUIT BREAKER: Extremely weak technical strength ({strength_score:.3f}) for {symbol} - rejecting all positions")
            return False

        # Emergency circuit breaker: Reject extremely large positions regardless of strength
        if position_size_usd > 100000:  # $100K+ positions are too risky
            logger.error(f"🚨 EMERGENCY CIRCUIT BREAKER: Position size ${position_size_usd:.0f} too large for {symbol} - rejecting")
            return False

        # Define minimum strength requirements based on position size
        if position_size_usd >= 10000:  # Large positions ($10K+)
            min_strength = 0.7
            size_category = "LARGE"
        elif position_size_usd >= 5000:  # Medium positions ($5K+)
            min_strength = 0.6
            size_category = "MEDIUM"
        elif position_size_usd >= 1000:  # Small positions ($1K+)
            min_strength = 0.5
            size_category = "SMALL"
        else:  # Micro positions
            min_strength = 0.3  # Lower requirement for very small positions
            size_category = "MICRO"

        # Check if strength meets requirements
        if strength_score >= min_strength:
            logger.debug(f"✅ Technical strength validation PASSED for {symbol} {size_category} position (${position_size_usd:.0f}): strength={strength_score:.3f} >= {min_strength:.3f}")
            return True
        else:
            logger.warning(f"❌ Technical strength validation FAILED for {symbol} {size_category} position (${position_size_usd:.0f}): strength={strength_score:.3f} < {min_strength:.3f}")
            return False

    except Exception as e:
        logger.error(f"❌ Error validating technical strength{symbol}: {e}")
        # Circuit breaker: On error, only allow micro positions to prevent system failure
        if position_size_usd < 1000:
            logger.warning(f"⚠️ Allowing micro position due to validation error: ${position_size_usd:.0f}")
            return True
        else:
            logger.error(f"🚨 CIRCUIT BREAKER: Rejecting position due to validation error: ${position_size_usd:.0f}")
            return False


def get_convergence_safety_mode() -> str:
    """
    Determine the safety mode for convergence calculations based on system state.

    Returns:
        Safety mode: 'conservative', 'moderate', 'aggressive', or 'emergency'
    """
    try:
        # Check for emergency conditions (could be expanded with more system health checks)
        # For now, default to moderate safety
        return 'moderate'
    except Exception as e:
        logger.error(f"Error determining convergence safety mode: {e}")
        return 'conservative'  # Safe default


def apply_convergence_safety_limits(multiplier: float, safety_mode: str = None) -> float:
    """
    Apply safety limits to convergence multipliers based on system safety mode.

    Args:
        multiplier: Raw convergence multiplier
        safety_mode: Safety mode ('conservative', 'moderate', 'aggressive', 'emergency')

    Returns:
        Limited multiplier within safe bounds
    """
    if safety_mode is None:
        safety_mode = get_convergence_safety_mode()

    # Define safety limits for each mode
    safety_limits = {
        'emergency': {'min': 0.5, 'max': 1.0},    # Very conservative during emergencies
        'conservative': {'min': 0.6, 'max': 1.5},  # Conservative limits
        'moderate': {'min': 0.5, 'max': 2.0},     # Standard operational limits
        'aggressive': {'min': 0.3, 'max': 2.5}    # Aggressive limits for testing
    }

    limits = safety_limits.get(safety_mode, safety_limits['moderate'])

    # Apply limits
    limited_multiplier = max(limits['min'], min(limits['max'], multiplier))

    if limited_multiplier != multiplier:
        logger.info(f"🛡️ CONVERGENCE SAFETY LIMIT applied: {multiplier:.2f} → {limited_multiplier:.2f} (mode: {safety_mode})")

    return limited_multiplier


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
