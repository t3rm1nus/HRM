# comms/data_validation.py - Utilidades para validar y limpiar datos de trading

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

from core.logging import logger

def clean_price_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Limpia y valida datos de precio asegurándose de que sean numéricos.
    
    Args:
        data: Diccionario con datos de mercado (precios, volúmenes, etc.)
        
    Returns:
        Diccionario con datos limpios y validados
    """
    cleaned_data = {}
    
    for symbol, symbol_data in data.items():
        if not isinstance(symbol_data, dict):
            logger.warning(f"Datos inválidos para {symbol}: {type(symbol_data)}")
            continue
            
        cleaned_symbol_data = {}
        
        # Campos numéricos esperados
        numeric_fields = [
            'price', 'close', 'open', 'high', 'low', 'volume',
            'priceChange', 'priceChangePercent', 'weightedAvgPrice',
            'prevClosePrice', 'lastPrice', 'bidPrice', 'askPrice'
        ]
        
        for field in numeric_fields:
            if field in symbol_data:
                cleaned_value = safe_float_conversion(symbol_data[field])
                if cleaned_value is not None:
                    cleaned_symbol_data[field] = cleaned_value
                else:
                    logger.warning(f"No se pudo convertir {field} para {symbol}: {symbol_data[field]}")
        
        # Mantener campos no numéricos como están
        for field, value in symbol_data.items():
            if field not in numeric_fields:
                cleaned_symbol_data[field] = value
                
        cleaned_data[symbol] = cleaned_symbol_data
    
    return cleaned_data

def safe_float_conversion(value: Any) -> Optional[float]:
    """
    Convierte un valor a float de forma segura.
    
    Args:
        value: Valor a convertir
        
    Returns:
        Float convertido o None si no es posible
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    
    if isinstance(value, str):
        # Remover espacios y caracteres especiales
        value = value.strip()
        if not value or value.lower() in ['nan', 'none', 'null', '']:
            return None
        
        try:
            return float(value)
        except ValueError:
            return None
    
    return None

def validate_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y limpia un DataFrame con datos OHLCV.
    
    Args:
        df: DataFrame con columnas ['open', 'high', 'low', 'close', 'volume']
        
    Returns:
        DataFrame limpio y validado
    """
    if df.empty:
        logger.warning("DataFrame vacío recibido")
        return df
    
    # Asegurar que las columnas numéricas sean float
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    existing_cols = [col for col in numeric_cols if col in df.columns]
    
    for col in existing_cols:
        # Convertir a string primero para manejar mixed types
        df[col] = df[col].astype(str)
        
        # Convertir a float usando safe_float_conversion
        df[col] = df[col].apply(safe_float_conversion)
        
        # Rellenar valores None con el valor anterior válido
        df[col] = df[col].fillna(method='ffill')
        
        # Si aún hay NaN al inicio, usar interpolación
        if df[col].isna().any():
            df[col] = df[col].interpolate()
        
        # Último recurso: rellenar con 0 (para volumen) o precio medio
        if df[col].isna().any():
            if col == 'volume':
                df[col] = df[col].fillna(0)
            else:
                # Para precios, usar la media de la serie
                mean_price = df[col].mean()
                if not np.isnan(mean_price):
                    df[col] = df[col].fillna(mean_price)
                else:
                    # Si no hay datos válidos, usar un precio por defecto
                    default_price = 50000 if 'BTC' in str(df.index.name) else 3000
                    df[col] = df[col].fillna(default_price)
                    logger.warning(f"Usando precio por defecto {default_price} para {col}")
    
    # Validar que high >= low, close entre high y low, etc.
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # Corregir inconsistencias básicas
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
    
    logger.info(f"DataFrame validado: {len(df)} filas, columnas: {list(df.columns)}")
    return df

def calculate_safe_indicators(df: pd.DataFrame, symbol: str = "") -> Dict[str, float]:
    """
    Calcula indicadores técnicos de forma segura evitando errores de tipo.
    
    Args:
        df: DataFrame con datos OHLCV
        symbol: Símbolo del activo (para logging)
        
    Returns:
        Diccionario con indicadores calculados
    """
    try:
        if df.empty or len(df) < 20:
            logger.warning(f"Datos insuficientes para {symbol}: {len(df)} filas")
            return {}
        
        # Validar datos antes del cálculo
        df = validate_ohlcv_data(df.copy())
        
        indicators = {}
        
        # RSI
        if 'close' in df.columns:
            rsi = calculate_rsi(df['close'], period=14)
            if not np.isnan(rsi):
                indicators['rsi'] = rsi
        
        # MACD
        if 'close' in df.columns:
            macd_data = calculate_macd(df['close'])
            if macd_data:
                indicators.update(macd_data)
        
        # EMAs
        if 'close' in df.columns:
            ema_10 = calculate_ema(df['close'], period=10)
            ema_20 = calculate_ema(df['close'], period=20)
            
            if not np.isnan(ema_10):
                indicators['ema_10'] = ema_10
            if not np.isnan(ema_20):
                indicators['ema_20'] = ema_20
        
        # Volatilidad
        if 'close' in df.columns and len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(24 * 365)  # Anualizada para crypto
                if not np.isnan(volatility):
                    indicators['volatility'] = volatility
        
        logger.info(f"Indicadores calculados para {symbol}: {list(indicators.keys())}")
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculando indicadores para {symbol}: {e}")
        return {}

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calcula RSI de forma segura."""
    try:
        if len(prices) < period + 1:
            return np.nan
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else np.nan
        
    except Exception as e:
        logger.error(f"Error calculando RSI: {e}")
        return np.nan

def calculate_ema(prices: pd.Series, period: int) -> float:
    """Calcula EMA de forma segura."""
    try:
        if len(prices) < period:
            return np.nan
            
        ema = prices.ewm(span=period).mean()
        return float(ema.iloc[-1]) if not np.isnan(ema.iloc[-1]) else np.nan
        
    except Exception as e:
        logger.error(f"Error calculando EMA: {e}")
        return np.nan

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calcula MACD de forma segura."""
    try:
        if len(prices) < max(fast, slow) + signal:
            return {}
            
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        result = {}
        if not np.isnan(macd_line.iloc[-1]):
            result['macd'] = float(macd_line.iloc[-1])
        if not np.isnan(signal_line.iloc[-1]):
            result['macd_signal'] = float(signal_line.iloc[-1])
        if not np.isnan(histogram.iloc[-1]):
            result['macd_histogram'] = float(histogram.iloc[-1])
            
        return result
        
    except Exception as e:
        logger.error(f"Error calculando MACD: {e}")
        return {}

# Función de utilidad para integrar en tu código existente
def fix_market_data(market_data: Any) -> Dict[str, Any]:
    """
    Función principal para limpiar datos de mercado antes de calcular indicadores.
    Úsala en main.py antes de llamar a los cálculos de indicadores técnicos.

    Args:
        market_data: Datos crudos del mercado (puede ser dict, str, o cualquier tipo)

    Returns:
        Datos limpios listos para cálculos
    """
    # Handle string input (from JSON etc)
    if isinstance(market_data, str):
        try:
            import json
            parsed_data = json.loads(market_data)
            if isinstance(parsed_data, dict):
                market_data = parsed_data
            else:
                logger.error(f"Parsed data is not a dict: {type(parsed_data)}")
                return {}
        except Exception as e:
            logger.error(f"Failed to parse market data string: {e}")
            return {}

    # Ensure dictionary type
    if not isinstance(market_data, dict):
        logger.error(f"Invalid market data type after parsing: {type(market_data)}")
        return {}

    cleaned_data = {}
    for symbol, data in market_data.items():
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                logger.error(f"Failed to parse data for {symbol}: {e}")
                continue

        if isinstance(data, dict):
            cleaned_data[symbol] = clean_price_data(data)
        elif isinstance(data, pd.DataFrame):
            cleaned_data[symbol] = data
        else:
            logger.error(f"Invalid data type for {symbol}: {type(data)}")
            continue

    return cleaned_data
