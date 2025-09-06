# l2_tactic/procesar_l2.py
import logging
import pandas as pd
from typing import Dict, Any
from .config import L2Config
from .signal_generator import L2TacticProcessor
from l1_operational.realtime_loader import RealTimeDataLoader
from core.logging import logger
logger.info("l2_tactic")

async def procesar_l2(state: dict, config: L2Config, bus=None) -> dict:
    """
    Procesa la capa táctica (L2), generando señales y órdenes listas para L1.
    Ahora con datos REALES de Binance.
    """
    logger.info("🎯 INICIANDO procesamiento L2 - Nivel Táctico (Datos REALES)")

    # --- 1. Obtener datos de mercado REALES ---
    loader = RealTimeDataLoader(real_time=True)
    market_data = {}
    features_by_symbol = {}
    
    # Símbolos a procesar (del state o por defecto)
    symbols = state.get("universo", ["BTCUSDT", "ETHUSDT"])
    logger.info(f"📊 Obteniendo datos REALES para símbolos: {symbols}")

    for symbol in symbols:
        try:
            # Obtener datos de mercado en tiempo real
            symbol_data = await loader.get_market_data(symbol, "1m", 100)
            if not symbol_data.empty:
                market_data[symbol] = symbol_data
                logger.info(f"✅ Datos REALES para {symbol}: {len(symbol_data)} registros")
                
                # Generar features técnicas en tiempo real
                features = await loader.get_features_for_symbol(symbol)
                if not features.empty:
                    features_by_symbol[symbol] = features
                    logger.info(f"🔧 Features REALES para {symbol}: {features.shape}")
                else:
                    logger.warning(f"⚠️ No se pudieron generar features para {symbol}")
            else:
                logger.warning(f"⚠️ No hay datos para {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos para {symbol}: {e}")

    # Actualizar state con datos reales
    state["mercado"] = market_data
    state["features_by_symbol"] = features_by_symbol
    
    logger.info(f"📈 Datos de mercado REALES obtenidos: {len(market_data)} símbolos")
    logger.info(f"🔧 Features generadas: {len(features_by_symbol)} símbolos")

    # --- 2. Verificar que tenemos datos suficientes ---
    if not market_data:
        logger.warning("⚠️ No hay datos de mercado disponibles. Saltando procesamiento L2.")
        state["senales"] = {"signals": [], "orders": []}
        return state

    # --- 3. Procesar señales con pipeline completo y datos REALES ---
    processor = L2TacticProcessor(config)
    result = await processor.process(
        state=state,
        market_data=market_data,           # ✅ Datos de mercado REALES
        features_by_symbol=features_by_symbol,  # ✅ Features REALES
        bus=bus
    )

    orders_for_l1 = result.get("orders_for_l1", [])
    signals_generated = result.get("signals", [])
    
    logger.info(f"📦 Señales generadas: {len(signals_generated)}")
    logger.info(f"📦 Órdenes finales preparadas para L1: {len(orders_for_l1)}")

    # Log detallado de las señales generadas
    for signal in signals_generated:
        logger.info(f"   🎯 Señal: {signal.get('symbol', 'N/A')} "
                   f"{signal.get('side', 'N/A')} "
                   f"Conf: {signal.get('confidence', 0):.2f}")

    # --- 4. Publicar al bus (si está disponible) ---
    if bus and orders_for_l1:
        try:
            for order in orders_for_l1:
                bus.publish("l2/orders", order)
            logger.info(f"🚀 {len(orders_for_l1)} órdenes enviadas al bus L1")
        except Exception as e:
            logger.error(f"❌ Error publicando órdenes en el bus: {e}")

    # --- 5. Guardar señales y órdenes en el estado ---
    state["senales"] = {
        "signals": signals_generated, 
        "orders": orders_for_l1,
        "metadata": {
            "data_quality": "realtime" if market_data else "simulated",
            "symbols_processed": list(market_data.keys()),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    }

    # L2 nunca modifica el portafolio directamente
    logger.debug(f"[L2] Portfolio sin cambios: {state.get('portfolio', {})}")

    # --- 6. Métricas de performance ---
    logger.info(f"✅ PROCESAMIENTO L2 COMPLETADO - "
               f"Señales: {len(signals_generated)}, "
               f"Órdenes: {len(orders_for_l1)}")

    return state


async def procesar_l2_standalone():
    """
    Función standalone para testing de L2 sin dependencias del sistema completo.
    """
    logger.info("🧪 EJECUTANDO L2 EN MODO STANDALONE (TEST)")
    
    # Configuración por defecto
    config = L2Config()
    loader = RealTimeDataLoader(real_time=True)
    
    # Estado simulado
    state = {
        "universo": ["BTCUSDT", "ETHUSDT"],
        "portfolio": {"USDT": 10000, "BTC": 0, "ETH": 0},
        "estrategia": "agresiva",
        "exposicion": {"BTC": 0, "ETH": 0},
        "ciclo_id": 1
    }
    
    # Procesar
    result_state = await procesar_l2(state, config)
    
    # Resultados
    signals = result_state["senales"]["signals"]
    orders = result_state["senales"]["orders"]
    
    logger.info(f"🧪 RESULTADOS STANDALONE - Señales: {len(signals)}, Órdenes: {len(orders)}")
    
    return result_state


# Para ejecución directa de testing
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        logger.info("🧪 Iniciando test standalone de L2...")
        try:
            await procesar_l2_standalone()
        except Exception as e:
            logger.error(f"❌ Error en test: {e}")
    
    asyncio.run(test())