# main.py
import asyncio
import sys
import os
import json
import pandas as pd
from datetime import datetime
from core.state_manager import initialize_state, validate_state_structure, log_cycle_data
from core.portfolio_manager import update_portfolio_from_orders, save_portfolio_to_csv
from core.technical_indicators import calculate_technical_indicators
from core.feature_engineering import integrate_features_with_l2, debug_l2_features
from l1_operational.data_feed import DataFeed
from l2_tactic.signal_generator import L2TacticProcessor
from l2_tactic.models import L2State, TacticalSignal
from l1_operational.order_manager import OrderManager
from comms.config import config
from l2_tactic.config import L2Config
from l1_operational.realtime_loader import RealTimeDataLoader
from l3_strategy.l3_processor import generate_l3_output
from l1_operational.bus_adapter import BusAdapterAsync
from comms.message_bus import MessageBus
from core.logging import logger
from dotenv import load_dotenv

# Añadir la raíz del proyecto al path para imports relativos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Paths y configuración L3
DATA_DIR = "data/datos_inferencia"
L3_OUTPUT = os.path.join(DATA_DIR, "l3_output.json")
L3_UPDATE_INTERVAL = 600  # 10 minutos
L3_TIMEOUT = 30           # timeout en segundos


# Inicializar bus global
bus = MessageBus()

# -----------------------------
# Funciones auxiliares L3
# -----------------------------
async def execute_l3_pipeline(state=None):
    logger.debug("[DEBUG] Entrando en execute_l3_pipeline, logger id: %s" % id(logger))
    """
    Ejecuta L3 directamente, sin subprocess, compatible con IDE 'play'.
    Llama a la función main() de procesar_l3.py.
    """
    from l3_strategy.procesar_l3 import main as procesar_l3_main
    
    if state is None:
        # Si no se pasa state, intentar obtenerlo del caller
        import inspect
        caller_locals = inspect.currentframe().f_back.f_locals
        state = caller_locals.get('state', {})
        
    # Validar el state antes de usarlo
    from core.state_manager import validate_state_structure
    state = validate_state_structure(state)
    market_data = state.get("mercado", {})
    market_data_serializable = {
        k: v.reset_index().to_dict(orient="records") if isinstance(v, (pd.Series, pd.DataFrame)) else v
        for k, v in market_data.items()
    }
    try:
        if asyncio.iscoroutinefunction(procesar_l3_main):
            await procesar_l3_main({"mercado": market_data_serializable})
        else:
            procesar_l3_main({"mercado": market_data_serializable})
        logger.info("✅ L3 ejecutado correctamente")
    except Exception as e:
        logger.error(f"❌ Error ejecutando L3: {e}", exc_info=True)

def load_l3_output():
    logger.debug("[DEBUG] Entrando en load_l3_output, logger id: %s" % id(logger))
    """Carga el output consolidado de L3"""
    if os.path.exists(L3_OUTPUT):
        with open(L3_OUTPUT, "r") as f:
            return json.load(f)
    logger.warning("⚠️ No se encontró l3_output.json, usando fallback")
    market_data_example = {}
    texts_example = []
    return generate_l3_output(market_data_example, texts_example)

async def l3_periodic_task(state):
    logger.debug("[DEBUG] Entrando en l3_periodic_task, logger id: %s" % id(logger))
    """Ejecuta L3 periódicamente y actualiza el contexto estratégico"""
    while True:
        try:
            try:
                await asyncio.wait_for(execute_l3_pipeline(state), timeout=L3_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Timeout ejecutando L3 (>{L3_TIMEOUT}s), usando última estrategia conocida")
            l3_context = load_l3_output()
            state["estrategia"] = l3_context.get("strategic_context", {})
            logger.info("🔄 l3_output.json actualizado en state['estrategia']")
            await asyncio.sleep(L3_UPDATE_INTERVAL)
        except asyncio.CancelledError:
            logger.info("🛑 Tarea L3 cancelada")
            break
        except Exception as e:
            logger.error(f"❌ Error en tarea L3: {e}", exc_info=True)
            await asyncio.sleep(L3_UPDATE_INTERVAL)

def log_error(e: Exception, msg: str = ""):
    logger.error(f"❌ {msg}: {str(e)}", exc_info=True)

async def main():
    logger.debug("[DEBUG] Entrando en main, logger id: %s" % id(logger))
    try:
        logger.info("🚀 INICIANDO SISTEMA HRM CON LOGGING CENTRALIZADO Y L3 FORZADO")

        # Inicializar y validar state
        from l2_tactic.models import L2State  # Importar L2State
        state = initialize_state(config["SYMBOLS"])
        
        # Validación agresiva del state
        logger.debug(f"[DEBUG] Antes de validate_state_structure, tipo de state['l2']: {type(state.get('l2', 'No existe'))}")
        state = validate_state_structure(state)
        logger.debug(f"[DEBUG] Después de validate_state_structure, tipo de state['l2']: {type(state.get('l2', 'No existe'))}")
        
        # Doble validación para asegurar L2State
        if not isinstance(state.get("l2"), L2State):
            logger.warning("⚠️ Forzando L2State en state['l2']")
            state["l2"] = L2State()
            
        logger.debug(f"[DEBUG] State final: {state.keys()}")
        logger.debug(f"[DEBUG] State['l2'] tipo final: {type(state['l2'])}")
        logger.debug(f"[DEBUG] State['l2'] signals: {state['l2'].signals if hasattr(state.get('l2', {}), 'signals') else 'NO SIGNALS'}")

        # Inicializar componentes
        loader = RealTimeDataLoader(config)
        data_feed = DataFeed(config)
        # Asegurar que el state está validado antes de pasarlo al bus
        state = validate_state_structure(state)
        bus_adapter = BusAdapterAsync(config, state)
        l2_processor = L2TacticProcessor(config)
        l1_order_manager = OrderManager(market_data=state["mercado"])
        logger.info(f"✅ Componentes iniciados, símbolos: {config['SYMBOLS']}")

        # Forzar L3 inicial
        await execute_l3_pipeline(state)
        initial_l3 = load_l3_output()
        state["estrategia"] = initial_l3.get("strategic_context", {})
        logger.info("🔄 l3_output.json inicial cargado en state['estrategia']")

        # Ejecutar L3 periódico
        asyncio.create_task(l3_periodic_task(state))

        # Ejecutar loop principal L2/L1
        await main_loop(state, data_feed, loader, l2_processor, l1_order_manager, bus_adapter)

    except KeyboardInterrupt:
        logger.info("🛑 Cierre solicitado por usuario")
    except Exception as e:
        import traceback
        logger.error("❌ Error en main con traza completa:")
        logger.error(traceback.format_exc())
        
        # Intentar obtener más información sobre el estado
        try:
            logger.error(f"[DEBUG] Estado del state al fallar:")
            logger.error(f"- state type: {type(state)}")
            logger.error(f"- state keys: {state.keys() if isinstance(state, dict) else 'NO DICT'}")
            logger.error(f"- state['l2'] type: {type(state.get('l2', 'No existe'))}")
            
            if hasattr(state, 'l2') and hasattr(state.l2, '__dict__'):
                logger.error(f"- state.l2.__dict__: {state.l2.__dict__}")
        except Exception as debug_e:
            logger.error(f"Error al intentar debuggear: {debug_e}")
    finally:
        if 'data_feed' in locals():
            await data_feed.close()
            logger.info("✅ Conexiones de DataFeed cerradas")
        if 'loader' in locals():
            await loader.close()
            logger.info("✅ RealTimeDataLoader cerrado")
        if 'bus_adapter' in locals():
            await bus_adapter.close()
            logger.info("✅ BusAdapterAsync cerrado")

async def main_loop(state, data_feed: DataFeed, realtime_loader: RealTimeDataLoader, l2_processor: L2TacticProcessor, l1_order_manager: OrderManager, bus_adapter: BusAdapterAsync):
    from l2_tactic.models import L2State  # Importar L2State
    logger.debug("[DEBUG] Entrando en main_loop, logger id: %s" % id(logger))
    """Loop principal L2/L1"""
    consecutive_zero_signals = 0  # Contador para detectar 0 señales persistentes
    max_zero_signals = 5  # Umbral para fallback

    while True:
        try:
            ciclo_start = pd.Timestamp.utcnow()
            state["cycle_id"] = state.get("cycle_id", 0) + 1

            # Validar que state["l2"] sea L2State
            if not isinstance(state.get("l2"), L2State):
                logger.warning("⚠️ state['l2'] no es L2State, inicializando...")
                # Si es dict, intenta rescatar señales
                signals = []
                l2_val = state.get("l2", {})
                if isinstance(l2_val, dict):
                    signals = l2_val.get("signals", [])
                state["l2"] = L2State()
                state["l2"].signals = signals
                logger.debug(f"[DEBUG] Tipo de state['l2'] en main_loop: {type(state['l2'])}")

            # 1️⃣ Obtener datos de mercado
            logger.debug("Getting market data")
            market_data = await realtime_loader.get_realtime_data()
            if not market_data or all(df.empty for df in market_data.values()):
                logger.warning("⚠️ No se obtuvieron datos en tiempo real, usando DataFeed...")
                market_data = await data_feed.get_market_data()
                if not market_data or all(df.empty for df in market_data.values()):
                    logger.warning("⚠️ No se obtuvieron datos de mercado válidos, usando valores por defecto")
                    market_data = {
                        "BTCUSDT": pd.DataFrame({'close': [110758.76]}, index=[pd.Timestamp.utcnow()]),
                        "ETHUSDT": pd.DataFrame({'close': [4301.11]}, index=[pd.Timestamp.utcnow()])
                    }

            logger.debug(f"Market data keys: {list(market_data.keys())}")
            for symbol, df in market_data.items():
                logger.debug(f"Market data {symbol} shape: {df.shape if not df.empty else 'empty'}")
                logger.debug(f"Market data {symbol} columns: {df.columns.tolist() if not df.empty else 'empty'}")
                logger.debug(f"Market data {symbol} dtypes: {df.dtypes if not df.empty else 'empty'}")

            # 2️⃣ Calcular indicadores técnicos
            logger.debug("Calculating technical indicators")
            technical_indicators = calculate_technical_indicators(market_data)
            logger.debug(f"Technical indicators keys: {list(technical_indicators.keys())}")
            for symbol, df in technical_indicators.items():
                logger.debug(f"Technical indicators {symbol} shape: {df.shape if not df.empty else 'empty'}")
                logger.debug(f"Technical indicators {symbol} columns: {df.columns.tolist() if not df.empty else 'empty'}")
                logger.debug(f"Technical indicators {symbol} dtypes: {df.dtypes if not df.empty else 'empty'}")
            state["technical_indicators"] = technical_indicators

            # 3️⃣ Generar señales técnicas
            logger.debug("Generating technical signals")
            technical_signals = await l2_processor.technical_signals(market_data, technical_indicators)
            state["technical_signals"] = technical_signals
            logger.debug(f"Señales técnicas generadas: {len(technical_signals)}")

            # 4️⃣ Actualizar market_data para L1
            state["mercado"] = {
                symbol: {
                    'close': float(technical_indicators[symbol]['close'].iloc[-1] if symbol in technical_indicators and not technical_indicators[symbol].empty else (110758.76 if symbol == "BTCUSDT" else 4301.11))
                } for symbol in config["SYMBOLS"]
            }
            l1_order_manager.market_data = state["mercado"]

            # 5️⃣ Procesar señales en L2
            logger.debug("Processing L2 signals")
            l2_result = await l2_processor.process(market_data=market_data, features_by_symbol=technical_indicators, state=state)
            # Refuerzo defensivo: asegurar que state['l2'] es L2State antes de asignar signals
            if not isinstance(state.get("l2"), L2State):
                logger.warning("⚠️ state['l2'] no es L2State tras process, corrigiendo...")
                state["l2"] = L2State()
            
            # Asegurar que l2_result es un dict y extraer señales de forma segura
            if isinstance(l2_result, dict):
                signals = l2_result.get("signals", [])
            else:
                signals = []
                logger.warning("⚠️ l2_result no es dict, usando lista vacía")
            
            # Asignar señales validadas
            state["l2"].signals = signals if isinstance(signals, list) else []
            
            signal_count = len(state["l2"].signals)
            logger.debug(f"Señales L2 generadas: {signal_count}")

            # Fallback si 0 señales persisten
            if not state["l2"].signals:
                consecutive_zero_signals += 1
                if consecutive_zero_signals >= max_zero_signals:
                    logger.warning(f"⚠️ {consecutive_zero_signals} ciclos con 0 señales, usando señales técnicas como fallback")
                    state["l2"].signals = technical_signals
                else:
                    consecutive_zero_signals = 0

            # 6️⃣ Ejecutar L1 → validación y envío de órdenes
            logger.debug("Processing L1 signals")
            l1_orders = l2_result.get("orders_for_l1", [])
            orders_result = await l1_order_manager.process_signals(l1_orders, state=state)

            # 7️⃣ Actualizar portfolio
            logger.debug("Updating portfolio")
            # Ensure orders_result is a list or convert dict to list of orders
            if isinstance(orders_result, dict):
                processed_orders = orders_result.get("orders", [])
            elif isinstance(orders_result, list):
                processed_orders = orders_result
            else:
                processed_orders = []
                logger.warning("⚠️ Tipo de orders_result inesperado: %s", type(orders_result))

            await update_portfolio_from_orders(state, processed_orders)
            await save_portfolio_to_csv(state)

            # 8️⃣ Logging persistente
            logger.debug("Logging cycle data")
            cycle_id = state.get("cycle_id", 0)
            
            # Get accurate counts
            signals = state['l2'].signals if isinstance(state.get('l2'), L2State) else []
            valid_signals = [s for s in signals if isinstance(s, TacticalSignal)]
            valid_orders = [o for o in processed_orders if isinstance(o, dict) and o.get('status') != 'rejected']
            
            # Update state with accurate counts
            state['cycle_stats'] = {
                'signals_count': len(valid_signals),
                'orders_count': len(valid_orders),
                'rejected_orders': len([o for o in processed_orders if isinstance(o, dict) and o.get('status') == 'rejected']),
                'cycle_time': (pd.Timestamp.utcnow() - ciclo_start).total_seconds()
            }
            
            # Log cycle data with updated counts
            await log_cycle_data(state, cycle_id, ciclo_start)

            logger.info(f"📊 Ciclo {cycle_id} completado en {state['cycle_stats']['cycle_time']:.2f}s con " + 
                       f"{state['cycle_stats']['signals_count']} señales y {state['cycle_stats']['orders_count']} órdenes " +
                       f"({state['cycle_stats']['rejected_orders']} rechazadas)")

            await asyncio.sleep(10)  # Ciclo principal cada 10s

        except asyncio.CancelledError:
            logger.info("🛑 Loop principal cancelado")
            break
        except Exception as e:
            logger.error(f"❌ Error en ciclo {state.get('cycle_id', 0)}: {e}", exc_info=True)
            await asyncio.sleep(10)  # Prevent tight loop on failure

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())