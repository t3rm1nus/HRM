# main.py - Versión corregida con features técnicas
import os
import time
import logging
import asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

from l1_operational.data_feed import DataFeed
from l2_tactic.main_processor import L2MainProcessor
from l1_operational.order_manager import OrderManager
from comms.config import SYMBOLS, USE_TESTNET
from core.logging import setup_logger
from l2_tactic.config import L2Config
from l2_tactic.models import TacticalSignal
from l1_operational.order_manager import Signal

# Importar el logger persistente
from core.persistent_logger import persistent_logger

# Configurar logging
setup_logger()
logger = logging.getLogger(__name__)

config_l2 = L2Config()

# Cargar variables de entorno
load_dotenv()

# Inicializar componentes
data_feed = DataFeed()
l2_processor = L2MainProcessor(config=config_l2)
l1_order_manager = OrderManager()

# Estado global
state = {
    "mercado": {symbol: {} for symbol in SYMBOLS}, 
    "estrategia": "neutral",
    "portfolio": {symbol: 0.0 for symbol in SYMBOLS + ["USDT"]},
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

logger = logging.getLogger(__name__)

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
            
            # Features de precio y tendencia
            symbol_features.update({
                'price_rsi': indicators.get('rsi', 50.0),
                'price_macd': indicators.get('macd', 0.0),
                'price_macd_signal': indicators.get('macd_signal', 0.0),
                'price_macd_hist': indicators.get('macd_hist', 0.0),
                'price_change_24h': indicators.get('change_24h', 0.0),
            })
            
            # Features de medias móviles
            current_price = state.get('mercado', {}).get(symbol, {}).get('price', 0)
            if current_price and current_price > 0:
                sma_20 = indicators.get('sma_20', current_price)
                sma_10 = indicators.get('sma_10', current_price)
                ema_10 = indicators.get('ema_10', current_price)
                
                symbol_features.update({
                    'price_vs_sma20': (current_price - sma_20) / sma_20 if sma_20 > 0 else 0,
                    'price_vs_sma10': (current_price - sma_10) / sma_10 if sma_10 > 0 else 0,
                    'price_vs_ema10': (current_price - ema_10) / ema_10 if ema_10 > 0 else 0,
                    'sma10_vs_sma20': (sma_10 - sma_20) / sma_20 if sma_20 > 0 else 0,
                })
            
            # Features de volatilidad y volumen
            symbol_features.update({
                'volatility': indicators.get('volatility', 0.5),
                'volume_ratio': indicators.get('vol_ratio', 1.0),
            })
            
            # Features de Bollinger Bands
            bb_upper = indicators.get('bb_upper', current_price * 1.05)
            bb_lower = indicators.get('bb_lower', current_price * 0.95)
            if current_price and bb_upper > bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                symbol_features['bb_position'] = max(0, min(1, bb_position))
            else:
                symbol_features['bb_position'] = 0.5
            
            # Señales básicas derivadas
            symbol_features.update({
                'signal_rsi_oversold': 1.0 if indicators.get('rsi', 50) < 30 else 0.0,
                'signal_rsi_overbought': 1.0 if indicators.get('rsi', 50) > 70 else 0.0,
                'signal_macd_bullish': 1.0 if indicators.get('macd_hist', 0) > 0 else 0.0,
                'signal_volume_high': 1.0 if indicators.get('vol_ratio', 1) > 1.5 else 0.0,
            })
            
            features_by_symbol[symbol] = symbol_features
            logger.info(f"✅ Features preparadas para {symbol}: {len(symbol_features)} features")
        
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

            logger.info(f"✅ Datos de mercado actualizados para {len(processed_symbols)} símbolos")
            
            # ===== PASO 0.5: PREPARAR INDICADORES TÉCNICOS PARA STATE =====
            # Crear estructura de indicadores técnicos separada para compatibilidad
            state['indicadores_tecnicos'] = {}
            for symbol in state['mercado'].keys():
                if 'indicators' in state['mercado'][symbol]:
                    state['indicadores_tecnicos'][symbol] = state['mercado'][symbol]['indicators']
                else:
                    state['indicadores_tecnicos'][symbol] = {}
            
            logger.info(f"✅ Indicadores técnicos preparados para {len(state['indicadores_tecnicos'])} símbolos")
            
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
                signals = await l2_processor.process(state)
                state["senales"] = {"signals": signals, "orders": []}
                
                logger.info(f"[L2] Señales generadas: {len(signals)}")
                
                # Log adicional sobre las señales generadas
                if signals:
                    signal_types = {}
                    for signal in signals:
                        signal_type = getattr(signal, 'action', 'unknown')
                        symbol = getattr(signal, 'symbol', 'unknown')
                        signal_types[f"{symbol}_{signal_type}"] = signal_types.get(f"{symbol}_{signal_type}", 0) + 1
                    
                    logger.info(f"[L2] Detalle señales: {dict(signal_types)}")
                else:
                    logger.info("[L2] No se generaron señales tácticas")
                    
            except Exception as l2_error:
                logger.error(f"❌ Error en L2 Processor: {l2_error}")
                signals = []
                state["senales"] = {"signals": [], "orders": []}

            # ===== PASO 3: EJECUTAR CAPA OPERACIONAL =====
            
            logger.info("[VALIDATION] Validando state['portfolio'] antes de L1...")
            
            # Validar portfolio existe
            if 'portfolio' not in state:
                state['portfolio'] = {'BTC': 0.0, 'ETH': 0.0, 'USDT': 10000.0}
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
                        
                        logger.debug(f"✅ Señal procesada: {signal.symbol} {signal.action}")
                        
                    else:
                        logger.warning(f"⚠️ Se ignoró objeto no válido: {type(signal)}")
                        
                except Exception as signal_error:
                    logger.error(f"❌ Error procesando señal: {signal_error}")
                    continue

            state["ordenes"] = orders
            logger.info(f"[L1] Órdenes procesadas: {len(orders)} de {len(signals)} señales")
            
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
                    state['portfolio'] = {'BTC': 0.0, 'ETH': 0.0, 'USDT': 10000.0}
                    
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
            'portfolio': {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 0.0},
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
        await data_feed.initialize()
        logger.info("✅ DataFeed iniciado")
        
        # L2 Processor
        l2_processor = L2TacticProcessor()
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