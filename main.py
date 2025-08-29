# main.py - Versión corregida con logging persistente
import os
import time
import logging
import asyncio
import pandas as pd
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

async def log_cycle_data(state: dict, cycle_id: int, start_time: float):
    """Loggear todos los datos del ciclo."""
    try:
        cycle_duration = (time.time() - start_time) * 1000
        
        # Datos de mercado
        btc_price = state['mercado'].get('BTCUSDT', {}).get('close', 0)
        eth_price = state['mercado'].get('ETHUSDT', {}).get('close', 0)
        
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
            await persistent_logger.log_signal({
                'timestamp': datetime.now().isoformat(),
                'cycle_id': cycle_id,
                'symbol': signal.get('symbol', ''),
                'side': signal.get('side', ''),
                'confidence': signal.get('confidence', 0),
                'quantity': signal.get('qty', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0),
                'signal_id': signal.get('signal_id', f'cycle_{cycle_id}'),
                'strategy': signal.get('strategy', ''),
                'ai_score': signal.get('ai_score', 0),
                'tech_score': signal.get('tech_score', 0),
                'risk_score': signal.get('risk_score', 0),
                'ensemble_decision': signal.get('ensemble_decision', ''),
                'market_regime': state.get('estrategia', 'neutral')
            })
        
        # Log datos de mercado
        for symbol in SYMBOLS:
            market_data = state['mercado'].get(symbol, {})
            if market_data:
                await persistent_logger.log_market_data({
                    'symbol': symbol,
                    'price': market_data.get('close', 0),
                    'volume': market_data.get('volume', 0),
                    'high': market_data.get('high', 0),
                    'low': market_data.get('low', 0),
                    'open': market_data.get('open', 0),
                    'close': market_data.get('close', 0),
                    'spread': 0,  # Puedes calcularlo si tienes bid/ask
                    'liquidity': 0,
                    'volatility': 0,
                    'rsi': 0,
                    'macd': 0,
                    'bollinger_upper': 0,
                    'bollinger_lower': 0
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
            
            # PASO 0: Recolectar datos de mercado para cada símbolo
            for symbol in SYMBOLS:
                df = data_feed.fetch_data(symbol)
                if not df.empty:
                    # Guardar el último precio y datos
                    last_row = df.iloc[-1]
                    state['mercado'][symbol] = {
                        'close': last_row['close'],
                        'open': last_row['open'],
                        'high': last_row['high'],
                        'low': last_row['low'],
                        'volume': last_row['volume']
                    }
                    logger.debug(f"Datos mercado {symbol}: {last_row['close']}")
                else:
                    logger.warning(f"No se pudieron obtener datos para {symbol}")

            logger.info("[L4] Ejecutando capa Meta...")
            logger.info("[L3] Ejecutando capa Strategy...")
            logger.info("[L2] Ejecutando capa Tactic...")
            
            # PASO 1: Generar señales tácticas con L2
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