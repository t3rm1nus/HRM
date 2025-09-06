# main.py - Versión corregida con features técnicas
import os
import time
import logging
import asyncio
import pandas as pd
import numpy as np
import csv
from dotenv import load_dotenv
from datetime import datetime

from l1_operational.data_feed import DataFeed
from l2_tactic.main_processor import L2MainProcessor
from l1_operational.order_manager import OrderManager
from comms.config import SYMBOLS, USE_TESTNET
from core.logging import setup_logger
from core.persistent_logger import PersistentLogger
from l2_tactic.config import L2Config
from l2_tactic.models import TacticalSignal
from l1_operational.order_manager import Signal
from l2_tactic.signal_generator import L2TacticProcessor
# Importar el logger persistente
from core.persistent_logger import persistent_logger

# Configurar logging
from core.logging import logger

config_l2 = L2Config()

# Cargar variables de entorno
load_dotenv()

# Inicializar componentes
data_feed = DataFeed()
l2_processor = L2TacticProcessor(config=config_l2)
l1_order_manager = OrderManager()

# Estado global
state = {
    "mercado": {symbol: {} for symbol in SYMBOLS}, 
    "estrategia": "neutral",
    "portfolio": {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3000.0},  # Inicializar con 3000 USDT
    "universo": SYMBOLS,
    "exposicion": {symbol: 0.0 for symbol in SYMBOLS},
    "senales": {},
    "ordenes": [],
    "riesgo": {},
    "deriva": False,
    "ciclo_id": 0,
}

import pandas as pd
import numpy as np
import logging

from core.logging import logger

def calculate_technical_indicators(df: pd.DataFrame) -> dict:
    """
    Calcula indicadores técnicos desde OHLCV data con validación robusta de tipos
    """
    try:
        if df.empty or len(df) < 20:
            logger.warning("DataFrame vacío o insuficientes datos para indicadores técnicos")
            return {}
        
        # Asegurar que tenemos las columnas necesarias
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning("DataFrame no tiene todas las columnas OHLCV necesarias")
            return {}
        
        # ===== VALIDACIÓN Y CONVERSIÓN DE TIPOS =====
        df_clean = df.copy()
        
        # Convertir todas las columnas numéricas asegurándose que sean float
        for col in required_cols:
            try:
                # Si la columna es object (strings), convertir a numeric
                if df_clean[col].dtype == 'object':
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Asegurar que sea float
                df_clean[col] = df_clean[col].astype(float)
                
                # Rellenar NaN con forward fill, luego backward fill si es necesario
                df_clean[col] = df_clean[col].ffill().bfill()
                
                # Si aún hay NaN, usar interpolación
                if df_clean[col].isna().any():
                    df_clean[col] = df_clean[col].interpolate(method='linear')
                
                # Último recurso: rellenar con la mediana de la serie
                if df_clean[col].isna().any():
                    median_val = df_clean[col].median()
                    if not pd.isna(median_val):
                        df_clean[col] = df_clean[col].fillna(median_val)
                    else:
                        # Si no hay datos válidos, usar valores por defecto
                        default_val = 50000 if col in ['open', 'high', 'low', 'close'] else 1000
                        df_clean[col] = df_clean[col].fillna(default_val)
                        logger.warning(f"Usando valor por defecto {default_val} para columna {col}")
                
            except Exception as col_error:
                logger.error(f"Error procesando columna {col}: {col_error}")
                return {}
        
        # Validar consistencia OHLC
        df_clean['high'] = np.maximum(df_clean['high'], 
                                    np.maximum(df_clean['open'], df_clean['close']))
        df_clean['low'] = np.minimum(df_clean['low'], 
                                   np.minimum(df_clean['open'], df_clean['close']))
        
        # Extraer series limpias
        close = df_clean['close']
        high = df_clean['high']
        low = df_clean['low']
        volume = df_clean['volume']
        
        logger.info(f"Datos limpiados: {len(df_clean)} filas, close range: {close.min():.2f} - {close.max():.2f}")
        
        # ===== CÁLCULO DE INDICADORES =====
        
        # RSI (14 períodos) - con validación de división
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            
            # Evitar división por cero
            rs = gain / (loss + 1e-10)  # Añadir epsilon pequeño
            rsi = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculando RSI: {e}")
            rsi = pd.Series([50] * len(close), index=close.index)
        
        # MACD (12, 26, 9)
        try:
            ema_12 = close.ewm(span=12, min_periods=1).mean()
            ema_26 = close.ewm(span=26, min_periods=1).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, min_periods=1).mean()
            macd_hist = macd - macd_signal
        except Exception as e:
            logger.error(f"Error calculando MACD: {e}")
            macd = pd.Series([0] * len(close), index=close.index)
            macd_signal = pd.Series([0] * len(close), index=close.index)
            macd_hist = pd.Series([0] * len(close), index=close.index)
        
        # Bollinger Bands (20, 2)
        try:
            sma_20 = close.rolling(window=20, min_periods=1).mean()
            std_20 = close.rolling(window=20, min_periods=1).std()
            bb_upper = sma_20 + (std_20 * 2)
            bb_lower = sma_20 - (std_20 * 2)
        except Exception as e:
            logger.error(f"Error calculando Bollinger Bands: {e}")
            sma_20 = close.rolling(window=20, min_periods=1).mean()
            bb_upper = sma_20
            bb_lower = sma_20
        
        # Medias móviles adicionales
        try:
            sma_10 = close.rolling(window=10, min_periods=1).mean()
            ema_10 = close.ewm(span=10, min_periods=1).mean()
        except Exception as e:
            logger.error(f"Error calculando medias móviles: {e}")
            sma_10 = close
            ema_10 = close
        
        # Volatilidad
        try:
            returns = close.pct_change()
            volatility = returns.rolling(window=20, min_periods=1).std() * np.sqrt(252)
        except Exception as e:
            logger.error(f"Error calculando volatilidad: {e}")
            volatility = pd.Series([0] * len(close), index=close.index)
        
        # Volume indicators
        try:
            vol_sma = volume.rolling(window=20, min_periods=1).mean()
            vol_ratio = volume / (vol_sma + 1e-10)  # Evitar división por cero
        except Exception as e:
            logger.error(f"Error calculando indicadores de volumen: {e}")
            vol_ratio = pd.Series([1] * len(volume), index=volume.index)
        
        # ===== EXTRACCIÓN DE VALORES FINALES =====
        
        # Obtener valores más recientes (evitar NaN)
        latest_idx = len(df_clean) - 1
        
        def safe_extract_value(series, default_value):
            """Extrae valor de forma segura con fallback"""
            try:
                value = series.iloc[latest_idx]
                if pd.isna(value) or np.isinf(value):
                    return default_value
                return float(value)
            except (IndexError, TypeError, ValueError):
                return default_value
        
        # Valores por defecto basados en precio actual
        current_price = safe_extract_value(close, 50000)
        
        indicators = {
            'rsi': safe_extract_value(rsi, 50.0),
            'macd': safe_extract_value(macd, 0.0),
            'macd_signal': safe_extract_value(macd_signal, 0.0),
            'macd_hist': safe_extract_value(macd_hist, 0.0),
            'bb_upper': safe_extract_value(bb_upper, current_price * 1.05),
            'bb_lower': safe_extract_value(bb_lower, current_price * 0.95),
            'sma_20': safe_extract_value(sma_20, current_price),
            'sma_10': safe_extract_value(sma_10, current_price),
            'ema_12': safe_extract_value(ema_12, current_price),
            'ema_10': safe_extract_value(ema_10, current_price),
            'volatility': safe_extract_value(volatility, 0.5),
            'vol_ratio': safe_extract_value(vol_ratio, 1.0)
        }
        
        # Calcular cambio 24h si tenemos suficientes datos
        try:
            if len(df_clean) >= 24:
                price_24h_ago = close.iloc[-24]
                if not pd.isna(price_24h_ago) and price_24h_ago > 0:
                    change_24h = (close.iloc[-1] - price_24h_ago) / price_24h_ago
                    indicators['change_24h'] = safe_extract_value(pd.Series([change_24h]), 0.0)
                else:
                    indicators['change_24h'] = 0.0
            else:
                indicators['change_24h'] = 0.0
        except Exception as e:
            logger.error(f"Error calculando change_24h: {e}")
            indicators['change_24h'] = 0.0
        
        # ===== VALIDACIÓN FINAL =====
        
        # Validar que no hay valores infinitos o NaN
        for key, value in indicators.items():
            if pd.isna(value) or np.isinf(value):
                if key in ['rsi']:
                    indicators[key] = 50.0
                elif key in ['vol_ratio']:
                    indicators[key] = 1.0
                elif key in ['bb_upper', 'bb_lower', 'sma_20', 'sma_10', 'ema_12', 'ema_10']:
                    indicators[key] = current_price
                else:
                    indicators[key] = 0.0
                logger.warning(f"Valor inválido en {key}, usando fallback: {indicators[key]}")
        
        # Validar rangos lógicos
        indicators['rsi'] = max(0, min(100, indicators['rsi']))  # RSI entre 0-100
        indicators['vol_ratio'] = max(0, indicators['vol_ratio'])  # Vol ratio positivo
        indicators['volatility'] = max(0, indicators['volatility'])  # Volatilidad positiva
        
        logger.info(f"✅ Indicadores calculados exitosamente: RSI={indicators['rsi']:.2f}, "
                   f"MACD={indicators['macd']:.4f}, Vol_Ratio={indicators['vol_ratio']:.2f}")
        
        return indicators
        
    except Exception as e:
        logger.error(f"❌ Error crítico calculando indicadores técnicos: {e}")
        logger.error(f"DataFrame shape: {df.shape if not df.empty else 'Empty'}")
        logger.error(f"DataFrame columns: {list(df.columns) if not df.empty else 'None'}")
        
        # Retornar indicadores por defecto en caso de error crítico
        return {
            'rsi': 50.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_hist': 0.0,
            'bb_upper': 50000 * 1.05,
            'bb_lower': 50000 * 0.95,
            'sma_20': 50000,
            'sma_10': 50000,
            'ema_12': 50000,
            'ema_10': 50000,
            'volatility': 0.5,
            'vol_ratio': 1.0,
            'change_24h': 0.0
        }
def prepare_features_for_l2(state):
    """
    Prepara las features de los indicadores técnicos para que L2 las pueda usar.
    Expandido a 52 features para compatibilidad con LightGBM.
    """
    try:
        features_by_symbol = {}
        
        # Verificar que tenemos indicadores técnicos calculados
        if 'indicadores_tecnicos' not in state or not state['indicadores_tecnicos']:
            logger.warning("⚠️ No hay indicadores técnicos en state")
            return features_by_symbol
        
        # Convertir indicadores a formato de features para L2
        for symbol, indicators in state['indicadores_tecnicos'].items():
            symbol_features = {}
            
            # Features de precio y tendencia (8 features)
            symbol_features.update({
                'price_rsi': indicators.get('rsi', 50.0),
                'price_macd': indicators.get('macd', 0.0),
                'price_macd_signal': indicators.get('macd_signal', 0.0),
                'price_macd_hist': indicators.get('macd_hist', 0.0),
                'price_change_24h': indicators.get('change_24h', 0.0),
                'price_volatility': indicators.get('volatility', 0.5),
                'volume_ratio': indicators.get('vol_ratio', 1.0),
                'price_momentum': indicators.get('macd_hist', 0.0) * 100  # Momentum derivado
            })
            
            # Features de medias móviles (12 features)
            current_price = state.get('mercado', {}).get(symbol, {}).get('price', 50000)
            sma_20 = indicators.get('sma_20', current_price)
            sma_10 = indicators.get('sma_10', current_price)
            ema_10 = indicators.get('ema_10', current_price)
            ema_12 = indicators.get('ema_12', current_price)
            
            symbol_features.update({
                'sma_10': sma_10,
                'sma_20': sma_20,
                'ema_10': ema_10,
                'ema_12': ema_12,
                'price_vs_sma20': (current_price - sma_20) / sma_20 if sma_20 > 0 else 0,
                'price_vs_sma10': (current_price - sma_10) / sma_10 if sma_10 > 0 else 0,
                'price_vs_ema10': (current_price - ema_10) / ema_10 if ema_10 > 0 else 0,
                'price_vs_ema12': (current_price - ema_12) / ema_12 if ema_12 > 0 else 0,
                'sma10_vs_sma20': (sma_10 - sma_20) / sma_20 if sma_20 > 0 else 0,
                'ema10_vs_ema12': (ema_10 - ema_12) / ema_12 if ema_12 > 0 else 0,
                'ma_convergence': abs((sma_10 - ema_10) / current_price) if current_price > 0 else 0,
                'ma_strength': (sma_10 + ema_10) / (2 * current_price) if current_price > 0 else 1
            })
            
            # Features de Bollinger Bands (8 features)
            bb_upper = indicators.get('bb_upper', current_price * 1.02)
            bb_lower = indicators.get('bb_lower', current_price * 0.98)
            bb_middle = (bb_upper + bb_lower) / 2
            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            
            symbol_features.update({
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'bb_middle': bb_middle,
                'bb_width': bb_width,
                'bb_position': (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5,
                'bb_squeeze': 1.0 if bb_width < 0.1 else 0.0,
                'bb_breakout_upper': 1.0 if current_price > bb_upper else 0.0,
                'bb_breakout_lower': 1.0 if current_price < bb_lower else 0.0
            })
            
            # Señales básicas derivadas (10 features)
            rsi_val = indicators.get('rsi', 50.0)
            symbol_features.update({
                'signal_rsi_oversold': 1.0 if rsi_val < 30 else 0.0,
                'signal_rsi_overbought': 1.0 if rsi_val > 70 else 0.0,
                'signal_rsi_neutral': 1.0 if 30 <= rsi_val <= 70 else 0.0,
                'signal_macd_bullish': 1.0 if indicators.get('macd_hist', 0) > 0 else 0.0,
                'signal_macd_bearish': 1.0 if indicators.get('macd_hist', 0) < 0 else 0.0,
                'signal_volume_high': 1.0 if indicators.get('vol_ratio', 1) > 1.5 else 0.0,
                'signal_volume_low': 1.0 if indicators.get('vol_ratio', 1) < 0.8 else 0.0,
                'signal_trend_up': 1.0 if sma_10 > sma_20 and current_price > sma_10 else 0.0,
                'signal_trend_down': 1.0 if sma_10 < sma_20 and current_price < sma_10 else 0.0,
                'signal_consolidation': 1.0 if abs((sma_10 - sma_20) / sma_20) < 0.01 else 0.0
            })
            
            # Features adicionales de volatilidad y momentum (8 features)
            volatility_val = indicators.get('volatility', 0.5)
            symbol_features.update({
                'volatility_norm': min(volatility_val / 2.0, 1.0),  # Normalizada
                'volatility_high': 1.0 if volatility_val > 1.0 else 0.0,
                'volatility_low': 1.0 if volatility_val < 0.3 else 0.0,
                'momentum_rsi': (rsi_val - 50) / 50,  # RSI normalizado
                'momentum_macd': indicators.get('macd', 0.0) / max(abs(indicators.get('macd', 0.001)), 0.001),
                'price_acceleration': indicators.get('macd_hist', 0.0) - indicators.get('macd', 0.0),
                'trend_strength': abs(indicators.get('macd_hist', 0.0)),
                'market_pressure': (indicators.get('vol_ratio', 1.0) - 1.0) * (rsi_val - 50) / 50
            })
            
            # Features de contexto de mercado (6 features)
            symbol_features.update({
                'current_price': current_price,
                'price_normalized': current_price / max(current_price, 1000),  # Normalizado
                'session_progress': 0.5,  # Progreso de la sesión (placeholder)
                'market_cap_factor': 1.0 if 'BTC' in symbol else 0.5,  # Factor de capitalización
                'liquidity_score': min(indicators.get('vol_ratio', 1.0), 3.0) / 3.0,
                'risk_score': min(volatility_val, 2.0) / 2.0
            })
            
            # Verificar que tenemos exactamente 52 features
            current_count = len(symbol_features)
            if current_count < 52:
                # Añadir features de relleno si es necesario
                for i in range(52 - current_count):
                    symbol_features[f'feature_padding_{i}'] = 0.0
                    
            elif current_count > 52:
                # Limitar a 52 features si tenemos más
                symbol_features = dict(list(symbol_features.items())[:52])
            
            features_by_symbol[symbol] = symbol_features
            logger.info(f"✅ Features preparadas para {symbol}: {len(symbol_features)} features (objetivo: 52)")
        
        return features_by_symbol
        
    except Exception as e:
        logger.error(f"❌ Error preparando features para L2: {e}")
        return {}

def update_state_with_features(state):
    """
    Actualiza el state con las features preparadas para L2.
    """
    try:
        features = prepare_features_for_l2(state)
        
        if features:
            # Añadir las features al state para que L2 las encuentre
            state['features'] = features
            
            # También añadirlas en el formato específico que L2 espera
            state['l2_features'] = {}
            for symbol, symbol_features in features.items():
                # Convertir símbolos al formato que L2 espera
                l2_symbol = symbol.replace('USDT', '/USDT')  # BTC/USDT en lugar de BTCUSDT
                state['l2_features'][l2_symbol] = symbol_features
            
            logger.info(f"✅ Features añadidas al state para {len(features)} símbolos")
            
            # Log de debug para verificar
            for symbol, features_dict in features.items():
                sample_features = list(features_dict.keys())[:5]  # Primeras 5 features
                logger.info(f"   {symbol}: {sample_features}... (total: {len(features_dict)})")
        else:
            logger.warning("⚠️ No se generaron features para L2")
            
        return state
        
    except Exception as e:
        logger.error(f"❌ Error actualizando state con features: {e}")
        return state

# Función para integrar en tu main.py después del cálculo de indicadores técnicos
def integrate_features_with_l2(state):
    """
    Función principal para integrar indicadores técnicos con L2.
    Llamar después de calculate_technical_indicators.
    """
    try:
        # 1. Preparar features desde indicadores técnicos
        state = update_state_with_features(state)
        
        # 2. Validar que las features lleguen a L2
        if 'features' in state and state['features']:
            logger.info("✅ Features disponibles para L2 Tactic")
        else:
            logger.warning("⚠️ No hay features disponibles para L2 Tactic")
        
        # 3. Añadir metadatos para L2
        if 'l2_metadata' not in state:
            state['l2_metadata'] = {}
        
        state['l2_metadata'].update({
            'features_timestamp': time.time(),
            'features_count': sum(len(f) for f in state.get('features', {}).values()),
            'symbols_with_features': list(state.get('features', {}).keys())
        })
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error integrando features con L2: {e}")
        return state

# Función de debug para verificar el flujo
def debug_l2_features(state):
    """
    Función de debug para verificar que las features llegan correctamente a L2.
    """
    logger.info("=== DEBUG L2 FEATURES ===")
    
    # Verificar structure del state
    keys_in_state = list(state.keys())
    logger.info(f"Keys en state: {keys_in_state}")
    
    # Verificar features
    if 'features' in state:
        logger.info(f"Features encontradas: {list(state['features'].keys())}")
        for symbol, features in state['features'].items():
            logger.info(f"  {symbol}: {len(features)} features")
    else:
        logger.warning("❌ No hay 'features' en state")
    
    # Verificar l2_features
    if 'l2_features' in state:
        logger.info(f"L2 Features encontradas: {list(state['l2_features'].keys())}")
    else:
        logger.warning("❌ No hay 'l2_features' en state")
    
    # Verificar indicadores técnicos
    if 'indicadores_tecnicos' in state:
        logger.info(f"Indicadores técnicos: {list(state['indicadores_tecnicos'].keys())}")
    else:
        logger.warning("❌ No hay 'indicadores_tecnicos' en state")
    
    logger.info("=== END DEBUG L2 FEATURES ===")

def validate_dataframe_for_indicators(df: pd.DataFrame) -> bool:
    """
    Valida que un DataFrame sea apto para calcular indicadores técnicos.
    
    Args:
        df: DataFrame a validar
        
    Returns:
        True si es válido, False si no
    """
    if df.empty:
        logger.warning("DataFrame está vacío")
        return False
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Faltan columnas requeridas: {missing_cols}")
        return False
    
    # Verificar que hay datos numéricos
    for col in required_cols:
        if df[col].dtype == 'object':
            # Intentar convertir una muestra
            try:
                pd.to_numeric(df[col].iloc[:5], errors='raise')
            except (ValueError, TypeError):
                logger.warning(f"Columna {col} contiene datos no convertibles a numérico")
                return False
    
    logger.info(f"✅ DataFrame validado: {len(df)} filas, columnas OK")
    return True


# Función auxiliar para debugging
def debug_dataframe_types(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Imprime información de debug sobre tipos de datos en el DataFrame.
    """
    logger.info(f"=== DEBUG {name} ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    for col in df.columns:
        logger.info(f"{col}: dtype={df[col].dtype}, "
                   f"sample={df[col].iloc[0] if len(df) > 0 else 'N/A'} "
                   f"(type: {type(df[col].iloc[0]) if len(df) > 0 else 'N/A'})")
    
    logger.info(f"=== END DEBUG {name} ===")


def prepare_market_features(ohlcv_data, indicators):
    """
    Combina datos OHLCV e indicadores técnicos en una estructura completa.
    [OPCIONAL - solo si necesitas una función separada]
    """
    return {
        'price': ohlcv_data['close'],
        'ohlcv': ohlcv_data,
        'indicators': indicators,
        'change_24h': indicators.get('change_24h', 0.0),
        'volatility': indicators.get('volatility', 0.0),
        'volume_ratio': indicators.get('vol_ratio', 1.0),
        'rsi': indicators.get('rsi', 50.0),
        'macd': indicators.get('macd', 0.0),
        'timestamp': datetime.now().timestamp(),
        'data_quality': 'good'
    }
def validate_state_structure(state):
    """
    Valida que el state tenga la estructura mínima requerida.
    """
    required_keys = ['ciclo_id', 'mercado', 'portfolio', 'universo', 'exposicion', 
                     'senales', 'ordenes', 'riesgo', 'deriva']
    
    for key in required_keys:
        if key not in state:
            logger.warning(f"⚠️ Clave faltante en state: {key}")
            
            # Añadir valores por defecto
            if key == 'ciclo_id':
                state[key] = 0
            elif key in ['mercado', 'portfolio', 'exposicion', 'senales', 'ordenes', 'riesgo']:
                state[key] = {}
            elif key == 'universo':
                state[key] = SYMBOLS
            elif key == 'deriva':
                state[key] = False
    
    return state

async def save_portfolio_to_csv(state, total_value, btc_balance, btc_value, eth_balance, eth_value, usdt_balance, cycle_id):
    """
    Guarda la línea del portfolio en un CSV externo para seguimiento histórico.
    
    Args:
        state: Estado del sistema
        total_value: Valor total del portfolio en USDT
        btc_balance: Balance de BTC
        btc_value: Valor de BTC en USDT
        eth_balance: Balance de ETH  
        eth_value: Valor de ETH en USDT
        usdt_balance: Balance de USDT
        cycle_id: ID del ciclo actual
    """
    try:
        # Crear directorio si no existe
        csv_dir = "data/portfolio"
        os.makedirs(csv_dir, exist_ok=True)
        
        # Nombre del archivo CSV con fecha
        today_date = datetime.now().strftime("%Y%m%d")
        csv_file = os.path.join(csv_dir, f"portfolio_history_.csv")
        
        # Datos a guardar
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Obtener precios actuales para referencia
        btc_price = state['mercado'].get('BTCUSDT', {}).get('price', 0)
        eth_price = state['mercado'].get('ETHUSDT', {}).get('price', 0)
        
        portfolio_data = {
            'timestamp': timestamp,
            'cycle_id': cycle_id,
            'total_value_usdt': round(total_value, 2),
            'btc_balance': round(btc_balance, 8),
            'btc_value_usdt': round(btc_value, 2),
            'btc_price': round(btc_price, 2),
            'eth_balance': round(eth_balance, 6),
            'eth_value_usdt': round(eth_value, 2),
            'eth_price': round(eth_price, 2),
            'usdt_balance': round(usdt_balance, 2),
            'portfolio_line': f"PORTFOLIO TOTAL: {total_value:.2f} USDT | BTC: {btc_balance:.5f} ({btc_value:.2f}$) | ETH: {eth_balance:.3f} ({eth_value:.2f}$) | USDT: {usdt_balance:.2f}$"
        }
        
        # Verificar si el archivo existe para añadir headers
        file_exists = os.path.exists(csv_file)
        
        # Escribir al CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=portfolio_data.keys())
            
            # Escribir header solo si el archivo es nuevo
            if not file_exists:
                writer.writeheader()
                logger.info(f"Archivo CSV creado: {csv_file}")
            
            # Escribir datos
            writer.writerow(portfolio_data)
        
        # Log cada 10 ciclos para no saturar
        if cycle_id % 10 == 0:
            logger.info(f"Portfolio guardado en CSV: {csv_file}")
            
    except Exception as e:
        logger.error(f"Error guardando portfolio en CSV: {e}")


async def update_portfolio_from_orders(state, orders):
    """
    Actualiza el portfolio basado en las órdenes ejecutadas.
    
    Args:
        state: Estado del sistema
        orders: Lista de órdenes procesadas
    """
    try:
        if not orders:
            logger.debug("No hay órdenes para procesar")
            return
        
        logger.info(f"Procesando {len(orders)} órdenes para actualizar portfolio...")
        
        portfolio_changes = {}
        orders_processed = 0
        
        for order in orders:
            # Debug: mostrar estructura de la orden
            logger.debug(f"Procesando orden: {type(order)} - {order}")
            
            # Diferentes formas de acceder a los atributos según el tipo de objeto
            try:
                # Intentar obtener atributos de diferentes maneras
                if hasattr(order, 'symbol'):
                    symbol = order.symbol
                elif hasattr(order, 'get') and callable(order.get):
                    symbol = order.get('symbol')
                else:
                    symbol = str(order).split()[1] if len(str(order).split()) > 1 else None
                
                if hasattr(order, 'status'):
                    status = order.status
                elif hasattr(order, 'get') and callable(order.get):
                    status = order.get('status')
                else:
                    # Buscar "filled" en la representación string
                    status = 'filled' if 'filled' in str(order) else 'unknown'
                
                if hasattr(order, 'side'):
                    side = order.side
                elif hasattr(order, 'get') and callable(order.get):
                    side = order.get('side')
                else:
                    # Buscar "buy" o "sell" en la representación string
                    order_str = str(order).lower()
                    if 'buy' in order_str:
                        side = 'buy'
                    elif 'sell' in order_str:
                        side = 'sell'
                    else:
                        side = 'unknown'
                
                # Obtener cantidad de diferentes formas
                quantity = 0
                for attr_name in ['qty', 'quantity', 'amount', 'size']:
                    if hasattr(order, attr_name):
                        quantity = getattr(order, attr_name, 0)
                        break
                
                # Si no encontramos quantity, buscar en string
                if quantity == 0:
                    import re
                    match = re.search(r'(\d+\.?\d*)', str(order))
                    if match:
                        quantity = float(match.group(1))
                
                # Obtener precio del mercado actual
                price = 0
                if symbol and symbol in state.get('mercado', {}):
                    price = state['mercado'][symbol].get('price', 0)
                    if price == 0:
                        price = state['mercado'][symbol].get('ohlcv', {}).get('close', 0)
                
                logger.info(f"Orden extraída: symbol={symbol}, status={status}, side={side}, "
                           f"quantity={quantity}, price={price}")
                
            except Exception as extract_error:
                logger.error(f"Error extrayendo datos de orden: {extract_error}")
                continue
            
            # Validar datos extraídos
            if not all([symbol, status, side, quantity > 0, price > 0]):
                logger.warning(f"Orden incompleta ignorada: symbol={symbol}, status={status}, "
                              f"side={side}, qty={quantity}, price={price}")
                continue
            
            # Solo procesar órdenes completadas exitosamente
            if status.lower() != 'filled':
                logger.debug(f"Orden no completada, status: {status}")
                continue
            
            # Calcular cambios en el portfolio
            if side.lower() == 'buy':
                # Comprar: reducir USDT, aumentar activo base
                usdt_cost = quantity * price
                
                # Verificar que tenemos suficiente USDT
                current_usdt = state['portfolio'].get('USDT', 0)
                if current_usdt >= usdt_cost:
                    portfolio_changes['USDT'] = portfolio_changes.get('USDT', 0) - usdt_cost
                    portfolio_changes[symbol] = portfolio_changes.get(symbol, 0) + quantity
                    
                    logger.info(f"✅ COMPRA ejecutada: {quantity:.5f} {symbol} a {price:.2f}$ "
                               f"(costo: {usdt_cost:.2f} USDT)")
                    orders_processed += 1
                else:
                    logger.warning(f"❌ Compra cancelada: USDT insuficiente "
                                 f"({current_usdt:.2f} < {usdt_cost:.2f})")
            
            elif side.lower() == 'sell':
                # Vender: aumentar USDT, reducir activo base
                current_asset = state['portfolio'].get(symbol, 0)
                if current_asset >= quantity:
                    usdt_received = quantity * price
                    portfolio_changes['USDT'] = portfolio_changes.get('USDT', 0) + usdt_received
                    portfolio_changes[symbol] = portfolio_changes.get(symbol, 0) - quantity
                    
                    logger.info(f"✅ VENTA ejecutada: {quantity:.5f} {symbol} a {price:.2f}$ "
                               f"(recibido: {usdt_received:.2f} USDT)")
                    orders_processed += 1
                else:
                    logger.warning(f"❌ Venta cancelada: {symbol} insuficiente "
                                 f"({current_asset:.5f} < {quantity:.5f})")
        
        # Aplicar cambios al portfolio
        if portfolio_changes:
            logger.info(f"Aplicando cambios al portfolio: {portfolio_changes}")
            for asset, change in portfolio_changes.items():
                current_balance = state['portfolio'].get(asset, 0)
                new_balance = max(0, current_balance + change)  # No permitir saldos negativos
                state['portfolio'][asset] = new_balance
                
                if change != 0:
                    change_str = f"{'+'if change > 0 else ''}{change:.6f}"
                    logger.info(f"📊 Portfolio actualizado: {asset} {current_balance:.6f} -> {new_balance:.6f} ({change_str})")
        else:
            logger.info("No se realizaron cambios en el portfolio")
        
        logger.info(f"Portfolio update completado: {orders_processed} órdenes procesadas")
    
    except Exception as e:
        logger.error(f"❌ Error actualizando portfolio: {e}", exc_info=True)

async def log_cycle_data(state: dict, cycle_id: int, start_time: float):
    """Loggear todos los datos del ciclo."""
    try:
        cycle_duration = (time.time() - start_time) * 1000
        
        # Datos de mercado
        btc_price = state['mercado'].get('BTCUSDT', {}).get('ohlcv', {}).get('close', 0)
        eth_price = state['mercado'].get('ETHUSDT', {}).get('ohlcv', {}).get('close', 0)
        
        # Log ciclo
        await persistent_logger.log_cycle({
            'timestamp': datetime.now().isoformat(),
            'cycle_id': cycle_id,
            'duration_ms': cycle_duration,
            'signals_generated': len(state.get('senales', {}).get('signals', [])),
            'orders_executed': len(state.get('ordenes', [])),
            'market_condition': state.get('estrategia', 'neutral'),
            'btc_price': btc_price,
            'eth_price': eth_price,
            'total_operations': len(state.get('ordenes', [])),
            'successful_operations': len([o for o in state.get('ordenes', []) if o.get('status') == 'filled']),
            'failed_operations': len([o for o in state.get('ordenes', []) if o.get('status') == 'rejected'])
        })
        
        # Log señales
        signals = state.get('senales', {}).get('signals', [])
        for signal in signals:
            if hasattr(signal, 'symbol'):
                await persistent_logger.log_signal({
                    'timestamp': datetime.now().isoformat(),
                    'cycle_id': cycle_id,
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'confidence': signal.confidence,
                    'quantity': getattr(signal, 'qty', 0),
                    'stop_loss': getattr(signal, 'stop_loss', 0),
                    'take_profit': getattr(signal, 'take_profit', 0),
                    'signal_id': f'cycle_{cycle_id}_{signal.symbol}',
                    'strategy': getattr(signal, 'signal_type', ''),
                    'ai_score': 0,
                    'tech_score': 0,
                    'risk_score': 0,
                    'ensemble_decision': '',
                    'market_regime': state.get('estrategia', 'neutral')
                })
        
        # Log datos de mercado
        for symbol in SYMBOLS:
            market_data = state['mercado'].get(symbol, {})
            ohlcv = market_data.get('ohlcv', {})
            indicators = market_data.get('indicators', {})
            
            if ohlcv:
                await persistent_logger.log_market_data({
                    'symbol': symbol,
                    'price': ohlcv.get('close', 0),
                    'volume': ohlcv.get('volume', 0),
                    'high': ohlcv.get('high', 0),
                    'low': ohlcv.get('low', 0),
                    'open': ohlcv.get('open', 0),
                    'close': ohlcv.get('close', 0),
                    'spread': 0,
                    'liquidity': 0,
                    'volatility': indicators.get('volatility', 0),
                    'rsi': indicators.get('rsi', 0),
                    'macd': indicators.get('macd', 0),
                    'bollinger_upper': indicators.get('bb_upper', 0),
                    'bollinger_lower': indicators.get('bb_lower', 0)
                })
        
        # Log performance cada 10 ciclos
        if cycle_id % 10 == 0:
            portfolio_value = sum(state["portfolio"].values())
            await persistent_logger.log_performance({
                'timestamp': datetime.now().isoformat(),
                'cycle_id': cycle_id,
                'portfolio_value': portfolio_value,
                'total_exposure': state.get('exposicion', {}).get('total', 0),
                'btc_exposure': state.get('exposicion', {}).get('BTCUSDT', 0),
                'eth_exposure': state.get('exposicion', {}).get('ETHUSDT', 0),
                'cash_balance': state.get('portfolio', {}).get('USDT', 0),
                'total_pnl': 0,
                'daily_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'correlation_btc_eth': 0,
                'signals_count': len(signals),
                'trades_count': len(state.get('ordenes', []))
            })
        
        # Log estado completo cada 30 ciclos
        if cycle_id % 30 == 0:
            await persistent_logger.log_state(state, cycle_id)
            
    except Exception as e:
        logger.error(f"Error en logging del ciclo: {e}")

# ------------------------------------------------------------------ #
# Bucle principal
# ------------------------------------------------------------------ #
async def _run_loop(state):
    """
    Loop principal del sistema HRM con cálculo de indicadores técnicos y features para L2.
    
    Args:
        state: Diccionario de estado del sistema
    """
    while True:
        try:
            cycle_start = time.time()
            state['ciclo_id'] += 1
            current_cycle = state['ciclo_id']
            
            logger.info(f"[TICK] Iniciando ciclo {current_cycle}")
            
            # ===== PASO 0: RECOLECTAR Y PROCESAR DATOS DE MERCADO =====
            processed_symbols = []
            
            for symbol in SYMBOLS:
                try:
                    # Obtener datos históricos (necesarios para indicadores)
                    df = data_feed.fetch_data(symbol, limit=100)
                    
                    if not df.empty and len(df) > 20:  # Necesitamos al menos 20 períodos
                        # Datos OHLCV básicos del último período
                        last_row = df.iloc[-1]
                        ohlcv_data = {
                            'close': float(last_row['close']),
                            'open': float(last_row['open']),
                            'high': float(last_row['high']),
                            'low': float(last_row['low']),
                            'volume': float(last_row['volume']),
                            'timestamp': datetime.now().timestamp()
                        }
                        
                        # Calcular indicadores técnicos usando la función corregida
                        indicators = calculate_technical_indicators(df)
                        
                        # Preparar estructura completa para state
                        market_data = {
                            # Datos OHLCV actuales
                            'price': ohlcv_data['close'],  # Para compatibilidad
                            'ohlcv': ohlcv_data,
                            
                            # Indicadores técnicos calculados
                            'indicators': indicators,
                            
                            # Datos adicionales derivados
                            'change_24h': indicators.get('change_24h', 0.0),
                            'volatility': indicators.get('volatility', 0.0),
                            'volume_ratio': indicators.get('vol_ratio', 1.0),
                            'rsi': indicators.get('rsi', 50.0),
                            'macd': indicators.get('macd', 0.0),
                            
                            # Metadatos
                            'timestamp': datetime.now().timestamp(),
                            'data_quality': 'good'
                        }
                        
                        # Guardar en state
                        state['mercado'][symbol] = market_data
                        processed_symbols.append(symbol)
                        
                        logger.debug(f"✅ Datos procesados para {symbol}: "
                                   f"Price={ohlcv_data['close']:.2f}, "
                                   f"RSI={indicators.get('rsi', 0):.1f}, "
                                   f"MACD={indicators.get('macd', 0):.3f}, "
                                   f"VolRatio={indicators.get('vol_ratio', 0):.2f}")
                        
                    else:
                        logger.warning(f"⚠️ Datos insuficientes para {symbol} (len={len(df)})")
                        
                        # Datos mínimos para evitar crashes en L2
                        state['mercado'][symbol] = {
                            'price': 0,
                            'ohlcv': {
                                'close': 0, 'open': 0, 'high': 0, 'low': 0, 'volume': 0,
                                'timestamp': datetime.now().timestamp()
                            },
                            'indicators': {
                                'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
                                'sma_20': 0, 'sma_10': 0, 'ema_12': 0, 'ema_10': 0,
                                'bb_upper': 0, 'bb_lower': 0, 'volatility': 0.0, 'vol_ratio': 1.0,
                                'change_24h': 0.0
                            },
                            'change_24h': 0.0,
                            'volatility': 0.0,
                            'volume_ratio': 1.0,
                            'rsi': 50.0,
                            'macd': 0.0,
                            'timestamp': datetime.now().timestamp(),
                            'data_quality': 'insufficient'
                        }
                        
                except Exception as e:
                    logger.error(f"❌ Error procesando datos para {symbol}: {e}")
                    
                    # En caso de error, proporcionar datos seguros mínimos
                    state['mercado'][symbol] = {
                        'price': 0,
                        'ohlcv': {
                            'close': 0, 'open': 0, 'high': 0, 'low': 0, 'volume': 0,
                            'timestamp': datetime.now().timestamp()
                        },
                        'indicators': {
                            'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
                            'sma_20': 0, 'sma_10': 0, 'ema_12': 0, 'ema_10': 0,
                            'bb_upper': 0, 'bb_lower': 0, 'volatility': 0.0, 'vol_ratio': 1.0,
                            'change_24h': 0.0
                        },
                        'change_24h': 0.0,
                        'volatility': 0.0,
                        'volume_ratio': 1.0,
                        'rsi': 50.0,
                        'macd': 0.0,
                        'timestamp': datetime.now().timestamp(),
                        'data_quality': 'error'
                    }
                    continue

            
            
            # ===== PASO 0.5: PREPARAR INDICADORES TÉCNICOS STATE =====
            # Crear estructura de indicadores técnicos separada para compatibilidad
            state['indicadores_tecnicos'] = {}
            for symbol in state['mercado'].keys():
                if 'indicators' in state['mercado'][symbol]:
                    state['indicadores_tecnicos'][symbol] = state['mercado'][symbol]['indicators']
                else:
                    state['indicadores_tecnicos'][symbol] = {}
            
            logger.info(f"✅ Datos de mercado actualizados para {len(processed_symbols)} símbolos")
            logger.info(f"✅ Indicadores técnicos preparados para {len(state['indicadores_tecnicos'])} símbolos")
            logger.debug(f"📊 Contenido de indicadores: {state['indicadores_tecnicos']}")
            
            # ===== PASO 1: PREPARAR FEATURES PARA L2 =====
            logger.info("[FEATURES] Preparando features para L2...")
            state = integrate_features_with_l2(state)
            
            # Debug solo en los primeros ciclos o cada 100 ciclos
            if current_cycle <= 3 or current_cycle % 100 == 0:
                debug_l2_features(state)
            
            # Verificar que las features se prepararon correctamente
            features_ready = ('features' in state and state['features'] and 
                            len(state['features']) > 0)
            
            if features_ready:
                feature_count = sum(len(f) for f in state['features'].values())
                logger.info(f"✅ Features preparadas: {feature_count} total para "
                           f"{len(state['features'])} símbolos")
            else:
                logger.warning("⚠️ No se prepararon features para L2 - continuando con datos básicos")
            
            # ===== PASO 2: EJECUTAR CAPAS JERÁRQUICAS =====
            
            logger.info("[L4] Ejecutando capa Meta...")
            # Aquí iría tu lógica L4 si la tienes implementada
            
            logger.info("[L3] Ejecutando capa Strategy...")
            # Aquí iría tu lógica L3 si la tienes implementada
            
            logger.info("[L2] Ejecutando capa Tactic...")
            
            # PASO 2.1: Generar señales tácticas con L2 (ahora con features completas)
            try:
                l2_result = await l2_processor.process(state=state, market_data=state["mercado"], features_by_symbol=state.get("features", {}))
                signals = l2_result.get("signals", [])  # Extraer señales del diccionario
                
                # Guardar señales y órdenes en el estado
                state["senales"] = {"signals": signals, "orders": l2_result.get("orders_for_l1", [])}
                
                if signals:
                    signal_types = {}
                    for signal in signals:
                        signal_type = getattr(signal, 'side', 'unknown')  # Usar 'side' en lugar de 'action'
                        symbol = getattr(signal, 'symbol', 'unknown')
                        signal_types[f"{symbol}_{signal_type}"] = signal_types.get(f"{symbol}_{signal_type}", 0) + 1
                    
                    logger.info(f"[L2] Detalle señales: {dict(signal_types)}")
                else:
                    logger.info("[L2] No se generaron señales tácticas")
                    
            except Exception as l2_error:
                logger.error(f"❌ Error en L2 Processor: {l2_error}", exc_info=True)
                signals = []
                state["senales"] = {"signals": [], "orders": []}

            # ===== PASO 3: EJECUTAR CAPA OPERACIONAL =====
            
            logger.info("[VALIDATION] Validando state['portfolio'] antes de L1...")
            
            # Validar portfolio existe
            if 'portfolio' not in state:
                state['portfolio'] = {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3000.0}
                logger.info("✅ Portfolio inicializado con valores por defecto")
            
            logger.info("[L1] Ejecutando capa Operational...")
            
            # PASO 3.1: Procesar señales con Order Manager
            orders = []
            processed_signals = 0
            
            for signal in signals:
                try:
                    if isinstance(signal, (TacticalSignal, Signal)): 
                        order_report = await l1_order_manager.handle_signal(signal)
                        orders.append(order_report)
                        processed_signals += 1
                        
                        logger.debug(f"✅ Señal procesada: {signal.symbol} {signal.side}")
                        
                    else:
                        logger.warning(f"⚠️ Se ignoró objeto no válido: {type(signal)}")
                        
                except Exception as signal_error:
                    logger.error(f"❌ Error procesando señal: {signal_error}")
                    continue

            state["ordenes"] = orders
            logger.info(f"[L1] Órdenes procesadas: {len(orders)} de {len(signals)} señales")
            
            # DEBUG: Mostrar detalles de las órdenes antes de actualizar portfolio
            if orders:
                logger.info("=== DEBUG ÓRDENES ===")
                for i, order in enumerate(orders):
                    logger.info(f"Orden {i}: {type(order)} - {order}")
                    # Intentar mostrar atributos disponibles
                    if hasattr(order, '__dict__'):
                        logger.info(f"  Atributos: {order.__dict__}")
                logger.info("=== END DEBUG ÓRDENES ===")
            
            # Actualizar portfolio basado en órdenes ejecutadas
            await update_portfolio_from_orders(state, orders)
            
            # Log del Portfolio con color azul oscuro
            btc_balance = state['portfolio'].get('BTCUSDT', 0)
            eth_balance = state['portfolio'].get('ETHUSDT', 0)
            usdt_balance = state['portfolio'].get('USDT', 0)

            # Obtener precios actuales
            btc_price = state['mercado'].get('BTCUSDT', {}).get('price', 0)
            eth_price = state['mercado'].get('ETHUSDT', {}).get('price', 0)

            # Calcular valor total
            btc_value_usdt = btc_balance * btc_price
            eth_value_usdt = eth_balance * eth_price
            total_portfolio_value = btc_value_usdt + eth_value_usdt + usdt_balance

            # Log con color azul oscuro
            logger.info(f"\033[34m\033[1mPORTFOLIO TOTAL: {total_portfolio_value:.2f} USDT | "
                    f"BTC: {btc_balance:.5f} ({btc_value_usdt:.2f}$) | "
                    f"ETH: {eth_balance:.3f} ({eth_value_usdt:.2f}$) | "
                    f"USDT: {usdt_balance:.2f}$\033[0m")
            
            # Guardar portfolio en CSV
            try:
                await save_portfolio_to_csv(
                    state, total_portfolio_value, btc_balance, btc_value_usdt, 
                    eth_balance, eth_value_usdt, usdt_balance, current_cycle
                )
            except Exception as csv_error:
                logger.error(f"Error guardando portfolio en CSV: {csv_error}")
            if orders:
                # Log resumen de órdenes
                order_summary = {}
                for order in orders:
                    if hasattr(order, 'symbol') and hasattr(order, 'status'):
                        key = f"{order.symbol}_{order.status}"
                        order_summary[key] = order_summary.get(key, 0) + 1
                
                logger.info(f"[L1] Resumen órdenes: {dict(order_summary)}")

            # ===== PASO 4: LOGGING Y MÉTRICAS =====
            
            # Logging persistente con manejo de errores
            try:
                await log_cycle_data(state, current_cycle, cycle_start)
            except Exception as log_error:
                logger.error(f"❌ Error en logging persistente: {log_error}")

            # ===== PASO 5: CONTROL DE TIMING Y CICLO =====
            
            # Calcular tiempo de ciclo
            elapsed_time = time.time() - cycle_start
            target_cycle_time = 10  # segundos
            sleep_time = max(0, target_cycle_time - elapsed_time)
            
            # Log de rendimiento
            performance_msg = f"[TICK] Ciclo {current_cycle} completado en {elapsed_time:.2f}s"
            if elapsed_time > target_cycle_time:
                performance_msg += f" ⚠️ LENTO (objetivo: {target_cycle_time}s)"
            
            logger.info(f"{performance_msg}. Esperando {sleep_time:.2f}s.")
            
            # Estadísticas cada 50 ciclos
            if current_cycle % 50 == 0:
                try:
                    stats = persistent_logger.get_log_stats()
                    logger.info(f"📊 ESTADÍSTICAS (Ciclo {current_cycle}): {stats}")
                    
                    # Estadísticas de rendimiento adicionales
                    avg_cycle_time = elapsed_time  # Simplificado, podrías mantener un promedio
                    logger.info(f"📈 RENDIMIENTO: Tiempo promedio de ciclo: {avg_cycle_time:.2f}s, "
                               f"Símbolos procesados: {len(processed_symbols)}, "
                               f"Señales generadas: {len(signals)}, "
                               f"Órdenes ejecutadas: {len(orders)}")
                               
                except Exception as stats_error:
                    logger.error(f"❌ Error obteniendo estadísticas: {stats_error}")
            
            # Esperar antes del próximo ciclo
            await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("🛑 Interrupción por usuario - cerrando sistema limpiamente")
            break
            
        except Exception as e:
            logger.error(f"❌ Error fatal en el loop principal: {e}", exc_info=True)
            
            # Intentar recuperación básica
            try:
                # Reinicializar state básico si está corrupto
                if 'ciclo_id' not in state:
                    state['ciclo_id'] = 0
                if 'mercado' not in state:
                    state['mercado'] = {}
                if 'portfolio' not in state:
                    state['portfolio'] = {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3000.0}
                    
                logger.info("✅ State básico reinicializado después del error")
                
            except Exception as recovery_error:
                logger.error(f"❌ Error en recuperación: {recovery_error}")
            
            # Esperar más tiempo después de un error fatal
            await asyncio.sleep(30)


# ------------------------------------------------------------------ #
async def main():
    """
    Función principal que inicializa el sistema y ejecuta el loop.
    """
    global data_feed, l2_processor, l1_order_manager, persistent_logger
    
    try:
        # ===== INICIALIZACIÓN DEL SISTEMA =====
        logger.info("🚀 INICIANDO SISTEMA HRM CON LOGGING PERSISTENTE")
        logger.info(f"📁 Logs guardados en: data/logs/")
        
        # Inicializar state principal
        state = {
            'mercado': {symbol: {} for symbol in SYMBOLS},
            'estrategia': 'neutral',
            'portfolio': {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3000.0},  # Inicializar con 3000 USDT
            'universo': SYMBOLS,
            'exposicion': {symbol: 0.0 for symbol in SYMBOLS},
            'senales': {},
            'ordenes': [],
            'riesgo': {},
            'deriva': False,
            'ciclo_id': 0
        }
        
        # Validar estructura del state
        state = validate_state_structure(state)
        
        # Inicializar componentes del sistema
        logger.info("🔧 Inicializando componentes del sistema...")
        
        # DataFeed
        data_feed = DataFeed()
        await data_feed.start()
        logger.info("✅ DataFeed iniciado")
        
        # L2 Processor
        l2_processor = L2TacticProcessor(config=L2Config())
        logger.info("✅ L2 Processor iniciado")
        
        # L1 Order Manager
        l1_order_manager = OrderManager()
        logger.info("✅ L1 Order Manager iniciado")
        
        # Persistent Logger
        persistent_logger = PersistentLogger()
        logger.info("✅ Persistent Logger iniciado")
        
        logger.info(f"✅ Símbolos: {SYMBOLS}")
        logger.info(f"✅ Modo Testnet: {os.getenv('USE_TESTNET', 'True')}")
        logger.info("🌙 Sistema listo para ejecución prolongada")
        
        # ===== EJECUTAR LOOP PRINCIPAL =====
        await _run_loop(state)  # ✅ PASAR STATE COMO PARÁMETRO
        
    except KeyboardInterrupt:
        logger.info("🛑 Cierre del sistema solicitado por usuario")
        
    except Exception as e:
        logger.error(f"❌ Error crítico en función main: {e}", exc_info=True)
        
    finally:
        # Limpieza de recursos
        logger.info("🧹 Cerrando recursos del sistema...")
        
        try:
            if 'data_feed' in globals():
                await data_feed.close()
                logger.info("✅ DataFeed cerrado")
        except:
            logger.warning("⚠️ Error cerrando DataFeed")
            
        try:
            if 'persistent_logger' in globals():
                await persistent_logger.close()
                logger.info("✅ Persistent Logger cerrado")
        except:
            logger.warning("⚠️ Error cerrando Persistent Logger")
        
        logger.info("👋 Sistema HRM terminado")


if __name__ == "__main__":
    # Configurar constantes
    SYMBOLS = ['BTCUSDT', 'ETHUSDT']
    
    # Ejecutar sistema
    asyncio.run(main())