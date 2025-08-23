# l3_strategy.py
import logging
import time
from typing import Dict, Any

logger = logging.getLogger("l3_strategy")

def procesar_l3(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesa la capa estratégica (L3), convirtiendo señales tácticas en órdenes.
    Args:
        state: Estado actual del sistema.
    Returns:
        Estado actualizado con órdenes.
    """
    logger.info("🚀 Procesando capa L3 - Estratégica")
    ordenes = []

    # Obtener señales tácticas
    signals = state.get("senales", {}).get("signals", [])
    logger.debug(f"[L3] Procesando {len(signals)} señales tácticas")

    for signal in signals:
        try:
            # Validar señal
            if signal["confidence"] < 0.6:
                logger.debug(f"[L3] Señal para {signal['symbol']} descartada: confianza {signal['confidence']} < 0.6")
                continue

            # Obtener precio actual del mercado
            symbol = signal["symbol"]
            if symbol not in state["mercado"]:
                logger.error(f"[L3] Símbolo {symbol} no encontrado en datos de mercado")
                continue

            price = float(state["mercado"][symbol]["close"].iloc[-1])
            amount = 0.1  # Cantidad fija para pruebas

            # Crear orden
            orden = {
                "id": f"order_{symbol}_{len(ordenes)}",
                "symbol": symbol,
                "side": signal["direction"],
                "amount": amount,
                "price": price,
                "type": "market",
                "strategy_id": "l3_strategy",
                "timestamp": time.time(),
                "metadata": {
                    "confidence": signal["confidence"],
                    "source": signal["source"]
                },
                "risk": {
                    "stop_loss": price * 0.95 if signal["direction"] == "buy" else price * 1.05,
                    "take_profit": price * 1.05 if signal["direction"] == "buy" else price * 0.95
                }
            }
            ordenes.append(orden)
            logger.info(f"[L3] Generada orden para {symbol}: {signal['direction']} {amount} @ {price}")
        except Exception as e:
            logger.error(f"[L3] Error procesando señal para {symbol}: {e}", exc_info=True)

    state["ordenes"] = ordenes
    logger.debug(f"[L3] Generadas {len(ordenes)} órdenes: {ordenes}")
    return state