# main.py
import asyncio
import sys
import os
from dotenv import load_dotenv
load_dotenv()
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
from l2_tactic.risk_controls.manager import RiskControlManager
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
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
    Ejecuta L3 directamente. Intenta el pipeline completo (4 IAs) y si falla
    cae al pipeline ligero de procesar_l3.py
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
    
    # Intentar obtener datos de diferentes fuentes en orden de prioridad
    market_data = state.get("market_data", {})  # Datos OHLCV recién obtenidos
    if not market_data or all(not isinstance(v, pd.DataFrame) or v.empty for v in market_data.values()):
        market_data = state.get("market_data_full", {})  # Backup: datos históricos
        
    # Debug de datos de mercado
    logger.debug(f"[L3] Market data keys: {list(market_data.keys())}")
    logger.debug("[L3] Estado de los DataFrames:")
    for k, v in market_data.items():
        if isinstance(v, pd.DataFrame):
            logger.debug(f"[L3] {k} shape: {v.shape}, columns: {v.columns.tolist()}")
            
    # Serializar manteniendo la estructura OHLCV
    market_data_serializable = {}
    for symbol, df in market_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Asegurar que tenemos las columnas necesarias
            needed_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in needed_cols):
                logger.warning(f"[L3] {symbol} falta alguna columna OHLCV")
                continue
                
            # Convertir a lista de diccionarios preservando el orden temporal
            try:
                df_clean = df[needed_cols].copy()
                # Convertir a valores numéricos
                for col in needed_cols:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Eliminar NaN y convertir a lista de diccionarios
                df_clean = df_clean.dropna()
                market_data_serializable[symbol] = df_clean.to_dict(orient="records")
                logger.debug(f"[L3] {symbol} serializado: {len(market_data_serializable[symbol])} registros")
            except Exception as e:
                logger.error(f"[L3] Error serializando {symbol}: {e}")
                continue
        else:
            logger.warning(f"[L3] {symbol} DataFrame vacío o inválido")
    try:
        # Descargar y analizar sentimiento
        texts_for_sentiment = []
        try:
            import l3_strategy.sentiment_inference as si
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Si ya hay un loop, usar create_task y await
                    import nest_asyncio
                    nest_asyncio.apply()
                    df_reddit = loop.run_until_complete(si.download_reddit())
                else:
                    df_reddit = loop.run_until_complete(si.download_reddit())
            except Exception as e:
                logger.error(f"⚠️ Error en event loop Reddit: {e}")
                df_reddit = pd.DataFrame()
            df_news = si.download_news()
            
            # Combinar textos
            for df in [df_reddit, df_news]:
                if not df.empty:
                    texts_for_sentiment.extend(df['text'].tolist())
            logger.info(f"✅ Descargados {len(texts_for_sentiment)} textos para análisis de sentimiento")
        except Exception as e:
            logger.error(f"⚠️ Error descargando datos de sentimiento: {e}")
            texts_for_sentiment = []  # Fallback
            
        if not texts_for_sentiment:  # Usar fallback si no hay textos
            texts_for_sentiment = [
                "macro neutral", "crypto sentiment mixed", "btc outlook uncertain",
                "eth consolidation phase", "crypto market sentiment cautious"
            ]
            
        l3_output = generate_l3_output(market_data_serializable, texts_for_sentiment)
        logger.info("\x1b[32m✅ L3 (pipeline completo) output regenerado\x1b[0m")
        return l3_output
    except Exception as e:
        logger.error(f"❌ L3 completo falló, usando pipeline ligero: {e}", exc_info=True)
        try:
            if asyncio.iscoroutinefunction(procesar_l3_main):
                await procesar_l3_main({"mercado": market_data_serializable})
            else:
                procesar_l3_main({"mercado": market_data_serializable})
            logger.info("\x1b[32m✅ L3 (pipeline ligero) output regenerado\x1b[0m")
            return {"strategic_context": {}}
        except Exception as e2:
            logger.error(f"❌ Error ejecutando L3 ligero: {e2}", exc_info=True)
            return {"strategic_context": {}}

def load_l3_output():
    logger.debug("[DEBUG] Entrando en load_l3_output, logger id: %s" % id(logger))
    """Carga el output consolidado de L3"""
    if os.path.exists(L3_OUTPUT):
        with open(L3_OUTPUT, "r") as f:
            data = json.load(f)
            # Normalizar: si viene del pipeline completo (dict plano), envolver
            if isinstance(data, dict) and 'strategic_context' not in data:
                return {"strategic_context": data}
            return data
    logger.warning("⚠️ No se encontró l3_output.json, usando fallback vacío")
    # No ejecutar L3 sin datos de mercado - retornar contexto vacío
    return {"strategic_context": {}}

async def l3_periodic_task(state):
    logger.debug("[DEBUG] Entrando en l3_periodic_task, logger id: %s" % id(logger))
    """Ejecuta L3 periódicamente y actualiza el contexto estratégico"""
    
    while True:
        try:
            # Verificar que tenemos datos válidos
            market_data = state.get("market_data", {})
            if not market_data:
                logger.warning("⚠️ L3: No hay datos de mercado disponibles")
                await asyncio.sleep(10)
                continue
                
            # Verificar que tenemos suficientes datos
            min_rows = 120  # Necesitamos al menos 120 períodos
            data_valid = True
            data_info = []
            
            for symbol, df in market_data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.warning(f"⚠️ L3: {symbol} sin datos válidos")
                    data_valid = False
                    continue
                    
                if len(df) < min_rows:
                    logger.warning(f"⚠️ L3: {symbol} insuficientes datos ({len(df)} < {min_rows})")
                    data_valid = False
                    continue
                    
                if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    logger.warning(f"⚠️ L3: {symbol} faltan columnas OHLCV")
                    data_valid = False
                    continue
                    
                data_info.append(f"{symbol}: {len(df)}")
            
            if not data_valid:
                logger.warning("⚠️ L3: Esperando datos válidos...")
                await asyncio.sleep(10)
                continue
                
            # Ejecutar L3 con datos válidos
            logger.info(f"🔄 L3: Ejecutando con datos ({', '.join(data_info)})")
            try:
                l3_output = await asyncio.wait_for(execute_l3_pipeline(state), timeout=L3_TIMEOUT)
                if l3_output:
                    state["estrategia"] = l3_output.get("strategic_context", {})
                    logger.info("✅ L3: Output actualizado en state['estrategia']")
                else:
                    # Fallback: cargar desde archivo si execute_l3_pipeline falló
                    l3_context = load_l3_output()
                    state["estrategia"] = l3_context.get("strategic_context", {})
                    logger.info("⚠️ L3: Usando output desde archivo")
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ L3: Timeout ({L3_TIMEOUT}s), usando última estrategia")
                l3_context = load_l3_output()
                state["estrategia"] = l3_context.get("strategic_context", {})
                
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
        initial_capital = 3000.0  # Capital inicial configurable
        state = initialize_state(config["SYMBOLS"], initial_capital)
        logger.info(f"💰 Capital inicial configurado: {initial_capital} USDT")
        
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
        binance_client = BinanceClient()
        l1_order_manager = OrderManager(binance_client=binance_client, market_data=state["mercado"])
        
        # Inicializar RiskControlManager
        l2_config = L2Config()
        risk_manager = RiskControlManager(l2_config)
        logger.info(f"✅ Componentes iniciados, símbolos: {config['SYMBOLS']}")
        logger.info(f"🛡️ RiskControlManager inicializado para stop-loss y take-profit")

        # L3 se ejecutará después de obtener datos de mercado por primera vez
        logger.info("🔄 L3 se ejecutará después de obtener datos de mercado")

        # Obtener datos iniciales antes de iniciar cualquier tarea
        logger.info("🔄 Obteniendo datos iniciales de mercado...")
        initial_data = await loader.get_realtime_data()
        if not initial_data or all(df.empty for df in initial_data.values()):
            initial_data = await data_feed.get_market_data()
            
        if initial_data and not all(df.empty for df in initial_data.values()):
            formatted_data = {}
            for symbol, df in initial_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                        formatted_data[symbol] = df
                        logger.info(f"✅ Datos iniciales {symbol}: {len(df)} períodos")
                    else:
                        logger.warning(f"⚠️ {symbol} falta alguna columna OHLCV: {df.columns.tolist()}")
                        
            if formatted_data:
                state["market_data"] = formatted_data
                state["market_data_full"] = formatted_data.copy()
                logger.info("✅ Datos iniciales cargados en state")
            else:
                logger.error("❌ No se pudieron obtener datos iniciales válidos")
                return
        else:
            logger.error("❌ No se pudieron obtener datos iniciales")
            return

        # Ejecutar L3 periódico solo después de tener datos
        asyncio.create_task(l3_periodic_task(state))

        # Ejecutar loop principal L2/L1
        await main_loop(state, data_feed, loader, l2_processor, l1_order_manager, bus_adapter, risk_manager)

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

async def main_loop(state, data_feed: DataFeed, realtime_loader: RealTimeDataLoader, l2_processor: L2TacticProcessor, l1_order_manager: OrderManager, bus_adapter: BusAdapterAsync, risk_manager: RiskControlManager):
    from l2_tactic.models import L2State  # Importar L2State
    logger.debug("[DEBUG] Entrando en main_loop, logger id: %s" % id(logger))
    """Loop principal L2/L1"""
    consecutive_zero_signals = 0  # Contador para detectar 0 señales persistentes
    max_zero_signals = 5  # Umbral para fallback
    l3_initialized = False  # Flag para ejecutar L3 inicial solo una vez

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
                    logger.error("🚨 ALERTA: Usando precios fallback por falta de conectividad real. Revisa la conexión a Binance/DataFeed.")
                    market_data = {
                        "BTCUSDT": pd.DataFrame({'close': [110758.76]}, index=[pd.Timestamp.utcnow()]),
                        "ETHUSDT": pd.DataFrame({'close': [4301.11]}, index=[pd.Timestamp.utcnow()])
                    }

            # Verificar y formatear datos de mercado
            formatted_market_data = {}
            for symbol, df in market_data.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Asegurar que tenemos las columnas correctas
                    if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                        formatted_market_data[symbol] = df
                        logger.debug(f"Market data {symbol} shape: {df.shape}")
                        logger.debug(f"Market data {symbol} columns: {df.columns.tolist()}")
                    else:
                        logger.warning(f"⚠️ {symbol} falta alguna columna OHLCV: {df.columns.tolist()}")

            if not formatted_market_data:
                logger.error("❌ No hay datos de mercado válidos después del formateo")
                return
                
            # Guardar datos completos para L3
            state["market_data"] = formatted_market_data
            state["market_data_full"] = formatted_market_data.copy()

            # 2️⃣ Calcular indicadores técnicos y enriquecer con L3
            logger.debug("Calculating technical indicators")
            technical_indicators = calculate_technical_indicators(formatted_market_data)
            
            # Ejecutar L3 después de tener los datos formateados
            if not l3_initialized and formatted_market_data:
                min_rows = 120  # Necesitamos al menos 120 períodos para los retornos
                if all(len(df) >= min_rows for df in formatted_market_data.values()):
                    logger.info(f"🚀 Ejecutando L3 inicial (datos: {[f'{k}: {len(v)}' for k,v in formatted_market_data.items()]})")
                    l3_output = await execute_l3_pipeline(state)
                    if l3_output:
                        state["estrategia"] = l3_output.get("strategic_context", {})
                        logger.info("🔄 L3 output inicial cargado")
                        l3_initialized = True
                    
            # Integrar features con datos de L3
            l3_context = state.get("estrategia", {})
            technical_indicators = integrate_features_with_l2(technical_indicators, l3_context)
            debug_l2_features(technical_indicators)  # Debug para ver los features integrados
            
            logger.debug(f"Technical indicators keys: {list(technical_indicators.keys())}")
            for symbol, df in technical_indicators.items():
                logger.debug(f"Technical indicators {symbol} shape: {df.shape if not df.empty else 'empty'}")
                logger.debug(f"Technical indicators {symbol} columns: {df.columns.tolist() if not df.empty else 'empty'}")
                logger.debug(f"Technical indicators {symbol} dtypes: {df.dtypes if not df.empty else 'empty'}")
            state["technical_indicators"] = technical_indicators

            # 🚀 Ejecutar L3 inicial si es la primera vez que tenemos datos de mercado
            if not l3_initialized and formatted_market_data:
                # Verificar que tenemos suficientes datos (120 períodos mínimo para features L3)
                min_rows_l3 = 120  # Necesitamos 120 períodos para calcular features como return_120
                data_valid = all(
                    isinstance(df, pd.DataFrame) and 
                    len(df) >= min_rows_l3 and
                    all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
                    for df in formatted_market_data.values()
                )
                if not data_valid:
                    logger.warning(f"⚠️ Insuficientes datos para L3. Se necesitan {min_rows_l3} períodos con OHLCV completo")
                min_rows = 120  # Necesitamos al menos 120 períodos para calcular todos los retornos
                
                for symbol, df in market_data.items():
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        logger.warning(f"[L3] {symbol}: Sin datos válidos")
                        data_valid = False
                        continue
                    
                    if len(df) < min_rows:
                        logger.warning(f"[L3] {symbol}: Insuficientes datos ({len(df)} < {min_rows})")
                        data_valid = False
                        continue
                        
                    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                        logger.warning(f"[L3] {symbol}: Faltan columnas OHLCV. Columnas: {df.columns.tolist()}")
                        data_valid = False
                        continue
                        
                    logger.info(f"[L3] {symbol}: {len(df)} períodos disponibles, columnas OK")
                
                if data_valid:
                    logger.info("🚀 Ejecutando L3 inicial con datos de mercado disponibles")
                    l3_output = await execute_l3_pipeline(state)
                    if l3_output:
                        state["estrategia"] = l3_output.get("strategic_context", {})
                        logger.info("🔄 l3_output inicial cargado en state['estrategia']")
                    else:
                        # Fallback: cargar desde archivo si execute_l3_pipeline falló
                        initial_l3 = load_l3_output()
                        state["estrategia"] = initial_l3.get("strategic_context", {})
                        logger.info("🔄 l3_output.json inicial cargado en state['estrategia']")
                    l3_initialized = True
                else:
                    logger.warning("⚠️ No hay suficientes datos para L3, esperando más datos...")

            # 3️⃣ Generar señales técnicas
            logger.debug("Generating technical signals")
            technical_signals = await l2_processor.technical_signals(market_data, technical_indicators)
            state["technical_signals"] = technical_signals
            logger.debug(f"Señales técnicas generadas: {len(technical_signals)}")

            # 4️⃣ Actualizar market_data para L1 y L3
            state["mercado"] = {
                symbol: {
                    'close': float(technical_indicators[symbol]['close'].iloc[-1] if symbol in technical_indicators and not technical_indicators[symbol].empty else (110758.76 if symbol == "BTCUSDT" else 4301.11))
                } for symbol in config["SYMBOLS"]
            }
            
            # Guardar datos completos para L3
            state["market_data"] = market_data  # Datos OHLCV actuales
            state["market_data_full"] = market_data.copy()  # Copia para histórico
            
            # Debug de datos guardados
            logger.debug(f"[DEBUG] Market data guardado - shapes:")
            for symbol, df in market_data.items():
                if isinstance(df, pd.DataFrame):
                    logger.debug(f"  {symbol}: {df.shape}, columnas: {df.columns.tolist()}")
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
            logger.info(f"🔍 L1 orders recibidas: {len(l1_orders)}")
            for i, order in enumerate(l1_orders):
                logger.info(f"   Orden {i}: {order.get('symbol')} {order.get('side')} qty={order.get('quantity', 'None')}")
            orders_result = await l1_order_manager.process_signals(l1_orders, state=state)

            # 7️⃣ Monitorear posiciones existentes para stop-loss/take-profit
            logger.debug("Monitoring existing positions")
            current_prices = {}
            for symbol in config["SYMBOLS"]:
                if symbol in market_data and not market_data[symbol].empty:
                    current_prices[symbol] = float(market_data[symbol]['close'].iloc[-1])
            
            if current_prices:
                portfolio_value = state.get("total_value", state.get("initial_capital", 1000.0))
                risk_alerts = risk_manager.monitor_existing_positions(current_prices, portfolio_value)
                
                # Procesar alertas de riesgo (stop-loss/take-profit activados)
                if risk_alerts:
                    logger.info(f"🚨 {len(risk_alerts)} alertas de riesgo detectadas")
                    for alert in risk_alerts:
                        logger.warning(f"⚠️ {alert.alert_type.value}: {alert.message}")
                        # Aquí se podrían generar órdenes de cierre automático
                        # Por ahora solo logueamos las alertas

            # 8️⃣ Actualizar portfolio
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

            # 9️⃣ Logging persistente
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


            # Log extra: timeframe y duración del ciclo
            timeframe = config.get('INTERVAL', '10s')
            logger.info(f"⏱️ Timeframe: {timeframe} | Ciclo {cycle_id} | Duración: {state['cycle_stats']['cycle_time']:.2f}s")

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

