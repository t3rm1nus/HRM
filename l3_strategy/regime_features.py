# regime_features.py
"""
Functions to prepare features for regime detection with complete technical indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula TODAS las features necesarias para regime detection, incluyendo
    indicadores técnicos completos (Bollinger Bands, ADX, Momentum).
    
    Args:
        df: DataFrame con columnas OHLCV
        
    Returns:
        DataFrame con todas las features para regime detection
    """
    from core import logging as log
    
    # Verificar y convertir datos a numérico
    numeric_df = pd.DataFrame()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        try:
            numeric_df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            log.error(f"Error convirtiendo {col} a numérico: {e}")
            numeric_df[col] = 0.0
    
    # Inicializar DataFrame de features
    features = pd.DataFrame(index=df.index)
    
    # OHLCV base (ya convertido a numérico)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        features[col] = numeric_df[col].fillna(method='ffill').fillna(0)
    
    try:
        # Returns y log returns
        returns = features['close'].pct_change()
        features['return'] = returns.fillna(0)
        features['log_return'] = np.log1p(returns).fillna(0)
        
        # Volatility features (múltiples ventanas)
        for w in [5, 15, 30, 60, 120]:
            features[f'volatility_{w}'] = returns.rolling(w, min_periods=1).std().fillna(0)
            features[f'return_{w}'] = returns.rolling(w, min_periods=1).mean().fillna(0)
        
        # ============================================
        # RSI (14-period standard)
        # ============================================
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        
        # Evitar división por cero
        rs = gain / loss.replace(0, np.inf)
        features['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # ============================================
        # BOLLINGER BANDS (20-period, 2 std dev)
        # ============================================
        ma20 = features['close'].rolling(window=20, min_periods=1).mean()
        std20 = features['close'].rolling(window=20, min_periods=1).std()
        
        # Middle band
        features['bollinger_middle'] = ma20.fillna(method='ffill').fillna(features['close'])
        
        # Upper band (middle + 2*std)
        upper_calc = ma20 + (std20 * 2)
        features['bollinger_upper'] = upper_calc.fillna(method='ffill').fillna(features['close'] * 1.02)
        
        # Lower band (middle - 2*std)
        lower_calc = ma20 - (std20 * 2)
        features['bollinger_lower'] = lower_calc.fillna(method='ffill').fillna(features['close'] * 0.98)
        
        # BB width (útil para análisis)
        bb_width = (features['bollinger_upper'] - features['bollinger_lower']) / features['bollinger_middle']
        features['bb_width'] = bb_width.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Aliases para compatibilidad con regime_classifier
        features['boll_middle'] = features['bollinger_middle']
        features['boll_upper'] = features['bollinger_upper']
        features['boll_lower'] = features['bollinger_lower']
        
        # ============================================
        # MACD (12, 26, 9)
        # ============================================
        ema12 = features['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = features['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        features['macd'] = (ema12 - ema26).fillna(0)
        features['macdsig'] = features['macd'].ewm(span=9, adjust=False, min_periods=1).mean().fillna(0)
        features['macdhist'] = (features['macd'] - features['macdsig']).fillna(0)
        
        # ============================================
        # ADX (Average Directional Index) - 14 period
        # ============================================
        features = _calculate_adx(features, period=14)
        
        # ============================================
        # MOMENTUM (5 and 10 period)
        # ============================================
        features['momentum_5'] = features['close'].diff(5).fillna(0)
        features['momentum_10'] = features['close'].diff(10).fillna(0)
        
        # ============================================
        # ATR (Average True Range) - útil para volatilidad
        # ============================================
        features = _calculate_atr(features, period=14)
        
        # ============================================
        # MOVING AVERAGES adicionales
        # ============================================
        features['sma_20'] = features['close'].rolling(window=20, min_periods=1).mean().fillna(features['close'])
        features['sma_50'] = features['close'].rolling(window=50, min_periods=1).mean().fillna(features['close'])
        features['ema_20'] = features['close'].ewm(span=20, adjust=False, min_periods=1).mean().fillna(features['close'])
        
        # ============================================
        # VALIDACIÓN FINAL - Eliminar cualquier NaN residual
        # ============================================
        na_columns = features.columns[features.isna().any()].tolist()
        if na_columns:
            log.warning(f"Limpiando NaN en columnas: {na_columns}")
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Verificación final
        final_na_count = features.isna().sum().sum()
        if final_na_count > 0:
            log.error(f"❌ {final_na_count} NaN residuales después de limpieza")
            features = features.fillna(0)
        
        log.info(f"✅ Features calculadas: {len(features.columns)} columnas, {len(features)} filas")
        
    except Exception as e:
        log.error(f"❌ Error calculando features técnicas: {e}")
        import traceback
        log.error(traceback.format_exc())
        
        # Asegurar columnas mínimas en caso de error
        for col in features.columns:
            if features[col].isna().any():
                features[col] = features[col].fillna(0)
    
    return features


def _calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calcula ADX (Average Directional Index) y direccionales DI+/DI-.
    
    ADX mide la fuerza de la tendencia (0-100):
    - 0-20: Tendencia débil/ausente
    - 20-40: Tendencia moderada
    - 40+: Tendencia fuerte
    
    Args:
        df: DataFrame con OHLC
        period: Período para el cálculo (default 14)
        
    Returns:
        DataFrame con columnas adx, di_plus, di_minus añadidas
    """
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 1. Calcular +DM y -DM (Directional Movement)
        high_diff = high.diff()
        low_diff = -low.diff()
        
        # +DM cuando high actual > high anterior Y high_diff > low_diff
        dm_plus = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        
        # -DM cuando low actual < low anterior Y low_diff > high_diff
        dm_minus = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # 2. Calcular True Range (TR)
        tr_candidates = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        })
        tr = tr_candidates.max(axis=1)
        
        # 3. Suavizar con Wilder's smoothing (similar a EMA)
        # ATR = Average True Range
        atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        
        # Smoothed +DM y -DM
        dm_plus_smooth = dm_plus.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        dm_minus_smooth = dm_minus.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        
        # 4. Calcular +DI y -DI (Directional Indicators)
        # DI = (smoothed DM / ATR) * 100
        df['di_plus'] = (100 * dm_plus_smooth / atr).replace([np.inf, -np.inf], 0).fillna(0)
        df['di_minus'] = (100 * dm_minus_smooth / atr).replace([np.inf, -np.inf], 0).fillna(0)
        
        # 5. Calcular DX (Directional Index)
        # DX = 100 * |DI+ - DI-| / (DI+ + DI-)
        di_sum = df['di_plus'] + df['di_minus']
        di_diff = abs(df['di_plus'] - df['di_minus'])
        dx = (100 * di_diff / di_sum).replace([np.inf, -np.inf], 0).fillna(0)
        
        # 6. ADX = Smoothed DX
        df['adx'] = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean().fillna(0)
        
        # Limitar valores a rango razonable [0, 100]
        df['adx'] = df['adx'].clip(0, 100)
        df['di_plus'] = df['di_plus'].clip(0, 100)
        df['di_minus'] = df['di_minus'].clip(0, 100)
        
    except Exception as e:
        from core import logging as log
        log.error(f"Error calculando ADX: {e}")
        # Valores por defecto seguros
        df['adx'] = 20.0  # Neutral
        df['di_plus'] = 25.0
        df['di_minus'] = 25.0
    
    return df


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calcula ATR (Average True Range) - medida de volatilidad.
    
    Args:
        df: DataFrame con OHLC
        period: Período para el cálculo (default 14)
        
    Returns:
        DataFrame con columna atr añadida
    """
    try:
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range = max(high-low, |high-close_prev|, |low-close_prev|)
        tr_candidates = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        })
        tr = tr_candidates.max(axis=1)
        
        # ATR = Smoothed TR usando Wilder's smoothing
        df['atr'] = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().fillna(0)
        
    except Exception as e:
        from core import logging as log
        log.error(f"Error calculando ATR: {e}")
        df['atr'] = 0.0
    
    return df


def validate_regime_features(features: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """
    Valida que todas las features requeridas estén presentes con valores reales.
    
    Args:
        features: DataFrame con las features calculadas
        required_features: Lista de features requeridas por el modelo
        
    Returns:
        DataFrame con todas las features requeridas validadas
    """
    from core import logging as log
    
    # Identificar features faltantes, en cero, con NaN
    missing_features = []
    zero_variance_features = []
    nan_features = []
    
    for feature in required_features:
        if feature not in features.columns:
            missing_features.append(feature)
            features[feature] = 0.0
        else:
            # Verificar NaN
            if features[feature].isna().any():
                nan_features.append(feature)
                features[feature] = features[feature].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Verificar varianza cero (indica valores constantes/fallback)
            if len(features) > 1 and features[feature].std() == 0:
                zero_variance_features.append(feature)
    
    # Log detallado
    if missing_features:
        log.warning(f"⚠️ Features faltantes ({len(missing_features)}): {', '.join(missing_features[:5])}")
    
    if zero_variance_features:
        log.warning(f"⚠️ Features sin variación ({len(zero_variance_features)}): {', '.join(zero_variance_features[:5])}")
        log.warning("   Estas features pueden estar usando valores por defecto")
    
    if nan_features:
        log.warning(f"⚠️ Features con NaN corregidas ({len(nan_features)}): {', '.join(nan_features[:5])}")
    
    # Estadísticas finales
    log.info(f"📊 Validación de features:")
    log.info(f"   - Requeridas: {len(required_features)}")
    log.info(f"   - Proporcionadas: {len(features.columns)}")
    log.info(f"   - Faltantes: {len(missing_features)}")
    log.info(f"   - Sin variación: {len(zero_variance_features)}")
    log.info(f"   - Con NaN: {len(nan_features)}")
    
    # Seleccionar solo las features requeridas
    features = features[required_features].copy()
    
    # Limpieza final exhaustiva
    na_count = features.isna().sum().sum()
    if na_count > 0:
        log.warning(f"⚠️ Limpiando {na_count} NaN residuales")
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Verificación post-limpieza
        final_na = features.isna().sum().sum()
        if final_na > 0:
            log.error(f"❌ Quedan {final_na} NaN después de limpieza - forzando a 0")
            features = features.fillna(0)
        else:
            log.info(f"✅ Todos los NaN eliminados")
    
    return features


def calculate_technical_summary(df: pd.DataFrame) -> Dict:
    """
    Calcula un resumen de indicadores técnicos actuales.
    Útil para logging y debugging.
    
    Returns:
        Dict con valores actuales de indicadores clave
    """
    try:
        latest = df.iloc[-1]
        
        summary = {
            'price': latest.get('close', 0),
            'rsi': latest.get('rsi', 50),
            'adx': latest.get('adx', 20),
            'di_plus': latest.get('di_plus', 25),
            'di_minus': latest.get('di_minus', 25),
            'bb_upper': latest.get('bollinger_upper', 0),
            'bb_middle': latest.get('bollinger_middle', 0),
            'bb_lower': latest.get('bollinger_lower', 0),
            'bb_width': latest.get('bb_width', 0),
            'macd': latest.get('macd', 0),
            'macd_signal': latest.get('macdsig', 0),
            'momentum_5': latest.get('momentum_5', 0),
            'momentum_10': latest.get('momentum_10', 0),
            'atr': latest.get('atr', 0),
            'volatility_30': latest.get('volatility_30', 0)
        }
        
        # Calcular posición respecto a Bollinger Bands
        if summary['bb_upper'] != summary['bb_lower']:
            bb_position = (summary['price'] - summary['bb_lower']) / (summary['bb_upper'] - summary['bb_lower'])
            summary['bb_position'] = bb_position
        else:
            summary['bb_position'] = 0.5
        
        return summary
        
    except Exception as e:
        from core import logging as log
        log.error(f"Error calculando technical summary: {e}")
        return {}