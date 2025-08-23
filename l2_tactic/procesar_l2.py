# l2_tactic/procesar_l2.py
import logging
from .config import L2Config
from .signal_generator import SignalGenerator

logger = logging.getLogger("l2_tactic")

def procesar_l2(state: dict, config: L2Config) -> dict:
    """
    Procesa la capa táctica (L2), generando señales para todos los símbolos.
    No modifica el portafolio.
    """
    logger.info("🎯 INICIANDO procesamiento L2 - Nivel Táctico")
    
    logger.info("🚀 Inicializando SignalGenerator...")
    signal_generator = SignalGenerator(config)
    logger.info("✅ SignalGenerator inicializado correctamente")

    logger.info("📊 Extrayendo datos de mercado del state...")
    market_data = state.get("mercado", {})
    symbols = list(market_data.keys())
    logger.info(f"📈 Datos de mercado extraídos para símbolos: {symbols}")

    logger.info("🧠 Extrayendo contexto de régimen...")
    regimen_context = state.get("regimen_context", {})
    logger.info(f"🧠 Contexto de régimen: {regimen_context}")

    logger.info("🔍 Generando señales tácticas...")
    signals = []
    for symbol in symbols:
        logger.info(f"Generating signals for {symbol}")
        try:
            symbol_signals = signal_generator.generate(state, symbol)
            signals.extend(symbol_signals)
            logger.info(f"Generated {len(symbol_signals)} signals for {symbol}")
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}", exc_info=True)

    logger.info(f"✅ Generadas {len(signals)} señales tácticas para {len(symbols)} símbolos")
    state["senales"] = {"signals": signals}
    logger.info(f"✅ L2 completado - {len(signals)} señales agregadas al state")
    
    # Asegurarse de que el portafolio no se modifique
    logger.debug(f"[L2] Portfolio sin cambios: {state['portfolio']}")
    return state