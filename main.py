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

def calculate_technical_indicators(df: pd.DataFrame) -> dict:
    """
    Calcula indicadores técnicos desde OHLCV data
    """
    try:
        if df.empty or len(df) < 20:
            return {}
        
        # Asegurar que tenemos las columnas necesarias
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            logger.warning("DataFrame no tiene todas las columnas OHLCV necesarias")
            return {}
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # RSI (14 períodos)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        macd_hist = macd - macd_signal
        
        # Bollinger Bands (20, 2)
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        bb_upper = sma_20 + (std_20 * 2)
        bb_lower = sma_20 - (std_20 * 2)
        
        # Medias móviles adicionales
        sma_10 = close.rolling(window=10).mean()
        ema_10 = close.ewm(span=10).mean()
        
        # Volatilidad
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Anualizada
        
        # Volume indicators
        vol_sma = volume.rolling(window=20).mean()
        vol_ratio = volume / vol_sma
        
        # Obtener valores más recientes (evitar NaN)
        latest_idx = len(df) - 1
        
        indicators = {
            'rsi': float(rsi.iloc[latest_idx]) if not pd.isna(rsi.iloc[latest_idx]) else 50.0,
            'macd': float(macd.iloc[latest_idx]) if not pd.isna(macd.iloc[latest_idx]) else 0.0,
            'macd_signal': float(macd_signal.iloc[latest_idx]) if not pd.isna(macd_signal.iloc[latest_idx]) else 0.0,
            'macd_hist': float(macd_hist.iloc[latest_idx]) if not pd.isna(macd_hist.iloc[latest_idx]) else 0.0,
            'bb_upper': float(bb_upper.iloc[latest_idx]) if not pd.isna(bb_upper.iloc[latest_idx]) else float(close.iloc[latest_idx]),
            'bb_lower': float(bb_lower.iloc[latest_idx]) if not pd.isna(bb_lower.iloc[latest_idx]) else float(close.iloc[latest_idx]),
            'sma_20': float(sma_20.iloc[latest_idx]) if not pd.isna(sma_20.iloc[latest_idx]) else float(close.iloc[latest_idx]),
            'sma_10': float(sma_10.iloc[latest_idx]) if not pd.isna(sma_10.iloc[latest_idx]) else float(close.iloc[latest_idx]),
            'ema_12': float(ema_12.iloc[latest_idx]) if not pd.isna(ema_12.iloc[latest_idx]) else float(close.iloc[latest_idx]),
            'ema_10': float(ema_10.iloc[latest_idx]) if not pd.isna(ema_10.iloc[latest_idx]) else float(close.iloc[latest_idx]),
            'volatility': float(volatility.iloc[latest_idx]) if not pd.isna(volatility.iloc[latest_idx]) else 0.0,
            'vol_ratio': float(vol_ratio.iloc[latest_idx]) if not pd.isna(vol_ratio.iloc[latest_idx]) else 1.0
        }
        
        # Calcular cambio 24h si tenemos suficientes datos
        if len(df) >= 24:
            price_24h_ago = close.iloc[-24]
            change_24h = (close.iloc[-1] - price_24h_ago) / price_24h_ago
            indicators['change_24h'] = float(change_24h)
        else:
            indicators['change_24h'] = 0.0
        
        # Validar que no hay valores infinitos
        for key, value in indicators.items():
            if np.isinf(value) or np.isnan(value):
                indicators[key] = 0.0
                
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculando indicadores técnicos: {e}")
        return {}

def prepare_market_features(ohlcv_data: dict, indicators: dict) -> dict:
    """
    Prepara estructura de datos completa para L2
    """
    try:
        return {
            'ohlcv': ohlcv_data,
            'indicators': indicators,
            'change_24h': indicators.get('change_24h', 0.0),
            'volatility': indicators.get('volatility', 0.0),
            'volume_ratio': indicators.get('vol_ratio', 1.0),
            'timestamp': datetime.now().timestamp()
        }
    except Exception as e:
        logger.error(f"Error preparando features de mercado: {e}")
        return {}

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
async def _run_loop():
    while True:
        try:
            cycle_start = time.time()
            state['ciclo_id'] += 1
            current_cycle = state['ciclo_id']
            
            logger.info(f"[TICK] Iniciando ciclo {current_cycle}")
            
            # PASO 0: Recolectar y procesar datos de mercado con indicadores técnicos
            for symbol in SYMBOLS:
                try:
                    # Obtener datos históricos (necesarios para indicadores)
                    df = data_feed.fetch_data(symbol, limit=100)  # Obtener más datos para indicadores
                    
                    if not df.empty and len(df) > 20:  # Necesitamos al menos 20 períodos
                        # Datos OHLCV básicos
                        last_row = df.iloc[-1]
                        ohlcv_data = {
                            'close': float(last_row['close']),
                            'open': float(last_row['open']),
                            'high': float(last_row['high']),
                            'low': float(last_row['low']),
                            'volume': float(last_row['volume'])
                        }
                        
                        # Calcular indicadores técnicos
                        indicators = calculate_technical_indicators(df)
                        
                        # Preparar estructura completa de features
                        market_features = prepare_market_features(ohlcv_data, indicators)
                        
                        # Guardar en state
                        state['mercado'][symbol] = market_features
                        
                        logger.debug(f"✅ Features generadas para {symbol}: RSI={indicators.get('rsi', 0):.1f}, "
                                   f"MACD={indicators.get('macd', 0):.3f}, VOL={indicators.get('vol_ratio', 0):.2f}")
                        
                    else:
                        logger.warning(f"⚠️ Datos insuficientes para {symbol} (len={len(df)})")
                        # Datos mínimos para evitar crashes
                        state['mercado'][symbol] = {
                            'ohlcv': {'close': 0, 'open': 0, 'high': 0, 'low': 0, 'volume': 0},
                            'indicators': {},
                            'change_24h': 0.0,
                            'volatility': 0.0,
                            'volume_ratio': 1.0,
                            'timestamp': datetime.now().timestamp()
                        }
                        
                except Exception as e:
                    logger.error(f"❌ Error procesando datos para {symbol}: {e}")
                    continue

            logger.info("[L4] Ejecutando capa Meta...")
            logger.info("[L3] Ejecutando capa Strategy...")
            logger.info("[L2] Ejecutando capa Tactic...")
            
            # PASO 1: Generar señales tácticas con L2 (ahora con features completas)
            signals = await l2_processor.process(state)
            state["senales"] = {"signals": signals, "orders": []}
            
            logger.info(f"[L2] Señales generadas: {len(signals)}")

            logger.info("[VALIDATION] Validando state['portfolio'] antes de L1...")
            logger.info("[L1] Ejecutando capa Operational...")
            
            orders = []
            for signal in signals:
                if isinstance(signal, (TacticalSignal, Signal)): 
                    try:
                        order_report = await l1_order_manager.handle_signal(signal)
                        orders.append(order_report)
                    except Exception as e:
                        logger.error(f"Error procesando señal: {e}")
                else:
                    logger.warning(f"Se ignoró objeto no válido: {type(signal)}")

            state["ordenes"] = orders
            logger.info(f"[L1] Órdenes procesadas: {len(orders)}")

            # 📊 LOGGING PERSISTENTE
            await log_cycle_data(state, current_cycle, cycle_start)

            # Tiempo de ciclo y sleep
            elapsed_time = time.time() - cycle_start
            sleep_time = max(0, 10 - elapsed_time)
            
            logger.info(f"[TICK] Ciclo {current_cycle} completado en {elapsed_time:.2f}s. Esperando {sleep_time:.2f}s.")
            
            # Mostrar estadísticas cada 50 ciclos
            if current_cycle % 50 == 0:
                stats = persistent_logger.get_log_stats()
                logger.info(f"📊 ESTADÍSTICAS: {stats}")
            
            await asyncio.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("🛑 Interrupción por usuario")
            break
        except Exception as e:
            logger.error(f"Error fatal en el loop principal: {e}", exc_info=True)
            await asyncio.sleep(10)  # Esperar antes de reintentar

# ------------------------------------------------------------------ #
async def main():
    try:
        logger.info("🚀 INICIANDO SISTEMA HRM CON LOGGING PERSISTENTE")
        logger.info(f"📁 Logs guardados en: data/logs/")
        
        # Iniciar componentes
        await data_feed.start()
        
        # Mostrar información inicial
        logger.info(f"✅ DataFeed iniciado")
        logger.info(f"✅ Símbolos: {SYMBOLS}")
        logger.info(f"✅ Modo Testnet: {USE_TESTNET}")
        logger.info("🌙 Sistema listo para ejecución prolongada")
        
        await _run_loop()
        
    except KeyboardInterrupt:
        logger.info("🛑 Ejecución interrumpida por usuario")
    except Exception as e:
        logger.error(f"❌ Error crítico en main: {e}", exc_info=True)
    finally:
        try:
            await data_feed.stop()
            logger.info("✅ DataFeed detenido")
        except:
            pass
        
        # Estadísticas finales
        stats = persistent_logger.get_log_stats()
        logger.info(f"📈 ESTADÍSTICAS FINALES: {stats}")
        logger.info("🎯 EJECUCIÓN FINALIZADA")

if __name__ == "__main__":
    asyncio.run(main())