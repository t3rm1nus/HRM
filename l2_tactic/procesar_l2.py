import logging
from typing import Dict, List
from .config import L2Config
from .signal_generator import SignalGenerator

logger = logging.getLogger("l2_tactic")

def procesar_l2(state: Dict, config: L2Config) -> Dict:
    """
    Procesa la capa táctica L2, generando señales para los símbolos en el universo.
    Args:
        state: Estado actual del sistema.
        config: Configuración de la capa L2.
    Returns:
        Estado actualizado con señales tácticas.
    """
    logger.info("🎯 INICIANDO procesamiento L2 - Nivel Táctico")
    logger.info("🚀 Inicializando SignalGenerator...")
    
    try:
        signal_generator = SignalGenerator(config)
        logger.info("✅ SignalGenerator inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando SignalGenerator: {e}", exc_info=True)
        return state

    logger.info("📊 Extrayendo datos de mercado del state...")
    market_data = state.get("mercado", {})
    symbols = config.signals.universe
    logger.info(f"📈 Datos de mercado extraídos para símbolos: {symbols}")

    logger.info("🧠 Extrayendo contexto de régimen...")
    regime_context = state.get("regime_context", {})
    logger.info(f"🧠 Contexto de régimen: {regime_context}")

    logger.info("🔍 Generando señales tácticas...")
    signals = []
    for symbol in symbols:
        try:
            symbol_signals = signal_generator.generate(state, symbol)
            signals.extend(symbol_signals)
            logger.info(f"Generated {len(symbol_signals)} signals for {symbol}")
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}", exc_info=True)

    logger.info(f"✅ Generadas {len(signals)} señales tácticas para {len(symbols)} símbolos")
    state["senales"] = {"signals": signals}
    logger.info(f"✅ L2 completado - {len(signals)} señales agregadas al state")
    return state