import time
import os
import json
import random
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional
from core.logging import logger
from l3_strategy.l3_processor import generate_l3_output

L3_OUTPUT = "data/datos_inferencia/l3_output.json"

logger.info("l3_strategy")


def make_json_serializable(obj):
    """Convierte dicts, listas, sets y DataFrames para json.dump"""
    import pandas as pd

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")  # <- convierte DataFrame a lista de dicts
    else:
        return obj


def determine_regime(state: Dict[str, Any]) -> str:
    """Simula la detección del régimen de mercado basado en volatilidad y señales"""
    signals = state.get("senales", [])
    bullish_signals = sum(1 for s in signals if s.get("direction") == "buy")
    bearish_signals = sum(1 for s in signals if s.get("direction") == "sell")
    if bullish_signals > bearish_signals:
        return "bullish"
    elif bearish_signals > bullish_signals:
        return "bearish"
    return "neutral"


def generate_asset_allocation(regime: str) -> Dict[str, float]:
    """Genera distribución de activos según régimen"""
    if regime == "bullish":
        return {"BTC": 0.6, "ETH": 0.3, "CASH": 0.1}
    elif regime == "bearish":
        return {"BTC": 0.3, "ETH": 0.2, "CASH": 0.5}
    return {"BTC": 0.5, "ETH": 0.5, "CASH": 0.0}


def procesar_l3(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("🚀 Procesando capa L3 - Estratégica")
    ordenes = []

    # Procesar señales tácticas
    signals = state.get("senales", [])
    for idx, signal in enumerate(signals):
        try:
            if signal.get("confidence", 0) < 0.6:
                continue
            symbol = signal.get("symbol")
            if not symbol or symbol not in state.get("mercado", {}):
                continue
            price = float(state["mercado"][symbol]["close"].iloc[-1])
            amount = 0.1  # cantidad fija de prueba

            orden = {
                "id": f"order_{symbol}_{idx}",
                "symbol": symbol,
                "side": signal.get("direction", "buy"),
                "amount": amount,
                "price": price,
                "type": "market",
                "strategy_id": "l3_strategy",
                "timestamp": time.time(),
                "metadata": {"confidence": signal.get("confidence"), "source": signal.get("source")},
                "risk": {
                    "stop_loss": price * 0.95 if signal.get("direction") == "buy" else price * 1.05,
                    "take_profit": price * 1.05 if signal.get("direction") == "buy" else price * 0.95
                }
            }
            ordenes.append(orden)
        except Exception as e:
            logger.error(f"[L3] Error procesando señal {symbol}: {e}", exc_info=True)

    # Determinar contexto estratégico dinámico
    regime = determine_regime(state)
    asset_allocation = generate_asset_allocation(regime)
    risk_appetite = "moderate" if regime == "neutral" else "high" if regime == "bullish" else "low"

    # Guardar resultados en el state
    state.update({
        "ordenes": ordenes,
        "regime": regime,
        "asset_allocation": asset_allocation,
        "risk_appetite": risk_appetite,
        "strategic_guidelines": {
            "regime": regime,
            "allocation": asset_allocation,
            "risk_appetite": risk_appetite
        },
        "strategic_context": state.get("strategic_context", {})  # Asegura que la clave exista
    })

    return state


def main(state: Optional[Dict[str, Any]] = None, fallback: bool = True) -> Dict[str, Any]:
    logger.info("⚡ Ejecutando L3 main()")

    if state is None:
        market_data = {}
        texts = []
        initial_output = generate_l3_output(market_data, texts)
        state = {
            "senales": initial_output.get("signals", []),
            "mercado": {},
            "regime": initial_output.get("regime"),
            "asset_allocation": initial_output.get("asset_allocation"),
            "risk_appetite": initial_output.get("risk_appetite"),
            "strategic_guidelines": initial_output.get("strategic_guidelines")
        }

    state = procesar_l3(state)
    # Asegura que la clave 'strategic_context' exista en el output
    if "strategic_context" not in state:
        state["strategic_context"] = {}

    # Guardar output
    os.makedirs(os.path.dirname(L3_OUTPUT), exist_ok=True)
    with open(L3_OUTPUT, "w") as f:
        json.dump(make_json_serializable(state), f, indent=2)

    logger.info(f"✅ L3 output guardado en {L3_OUTPUT}")
    return state


if __name__ == "__main__":
    main(fallback=True)
