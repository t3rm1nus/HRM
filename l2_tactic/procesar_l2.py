# l2_tactic/procesar_l2.py
import logging
from .config import L2Config
from .signal_generator import L2TacticProcessor

logger = logging.getLogger("l2_tactic")


async def procesar_l2(state: dict, config: L2Config, bus=None) -> dict:
    """
    Procesa la capa táctica (L2), generando señales y órdenes listas para L1.
    """
    logger.info("🎯 INICIANDO procesamiento L2 - Nivel Táctico")

    # --- 1. Extraer datos de mercado ---
    market_data = state.get("mercado", {})
    symbols = list(market_data.keys())
    logger.info(f"📈 Datos de mercado extraídos para símbolos: {symbols}")

    # --- 2. Procesar señales con pipeline completo ---
    processor = L2TacticProcessor(config)
    result = await processor.process(
        state,
        state.get("mercado", {}),                 # ✅ market_data
        state.get("features_by_symbol", {}),      # ✅ pasamos features_by_symbol si existe en el state
        bus
    )

    orders_for_l1 = result.get("orders_for_l1", [])
    logger.info(f"📦 Órdenes finales preparadas para L1: {len(orders_for_l1)}")

    # --- 3. Publicar al bus (si está disponible) ---
    if bus:
        for order in orders_for_l1:
            bus.publish("l2/orders", order)
        logger.info("🚀 Órdenes enviadas al bus L1")

    # --- 4. Guardar señales y órdenes en el estado ---
    state["senales"] = {"signals": result.get("signals", []), "orders": orders_for_l1}

    # L2 nunca modifica el portafolio directamente
    logger.debug(f"[L2] Portfolio sin cambios: {state['portfolio']}")

    return state
