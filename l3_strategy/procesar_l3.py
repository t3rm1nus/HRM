#l3_strategy/procesar_l3.py 

import time
import os
import json
from typing import Dict, Any, Optional
from core.logging import logger
from .decision_maker import make_decision, save_decision, load_inputs, ensure_dir
from .regime_classifier import clasificar_regimen
from .exposure_manager import gestionar_exposicion
from .bus_integration import publish_event, subscribe_event, L3MessageType
from datetime import datetime, timezone

L3_OUTPUT = "data/datos_inferencia/l3_output.json"
STRATEGIC_DECISION_FILE = "data/datos_inferencia/strategic_decision.json"

logger.info("l3_strategy - Actualizado con gestión de exposición basada en capital real")


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
        return obj.to_dict(orient="records")
    else:
        return obj


def procesar_l3(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesa la capa L3 usando el decision maker actualizado.
    No depende de fallbacks tácticos obsoletos.
    """
    logger.info("🚀 Procesando capa L3 - Estratégica (Actualizada)")

    # Obtener datos necesarios del state
    portfolio_state = state.get("portfolio", {})
    market_data = state.get("market_data", {})
    inputs = load_inputs()  # Cargar inputs de otros módulos L3

    # Generar decisión estratégica usando el nuevo decision maker
    strategic_decision = make_decision(inputs, portfolio_state, market_data)
    # Extraer decisiones de exposición
    exposure_decisions = strategic_decision.get("exposure_decisions", {})

    # Generar órdenes basadas en decisiones de exposición
    ordenes = []
    for symbol, decision in exposure_decisions.items():
        if symbol == "USDT":
            continue  # USDT es para liquidez, no genera órdenes

        adjustment = decision.get("adjustment", 0.0)
        action = decision.get("action", "hold")

        if action in ["buy", "sell"] and abs(adjustment) > 0.0001:  # Threshold mínimo
            try:
                price = market_data.get(symbol, {}).get("close", 50000.0)
                if isinstance(price, (list, tuple)):
                    price = price[-1] if price else 50000.0

                from l2_tactic.utils import safe_float
                orden = {
                    "id": f"l3_exposure_{symbol}_{int(time.time())}",
                    "symbol": symbol,
                    "side": action,
                    "quantity": abs(adjustment),
                    "price": safe_float(price),
                    "type": "market",
                    "strategy_id": "l3_exposure_management",
                    "timestamp": time.time(),
                    "metadata": {
                        "source": "l3_exposure_manager",
                        "regime": strategic_decision.get("market_regime"),
                        "target_position": decision.get("target_position", 0.0)
                    },
                    "risk": {
                        "stop_loss": price * 0.98 if action == "buy" else price * 1.02,
                        "take_profit": price * 1.02 if action == "buy" else price * 0.98
                    }
                }
                ordenes.append(orden)
                logger.info(f"📋 Orden L3 generada: {action} {abs(adjustment):.6f} {symbol} @ {price:.2f}")

            except Exception as e:
                logger.error(f"❌ Error generando orden para {symbol}: {e}")

   

    # Actualizar state con nueva información estratégica
    state.update({
        "ordenes": ordenes,
        "strategic_decision": {
            **strategic_decision,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        },
        "regime": strategic_decision.get("market_regime"),
        "risk_appetite": strategic_decision.get("risk_appetite"),
        "exposure_decisions": exposure_decisions,
        "strategic_guidelines": strategic_decision.get("strategic_guidelines", {}),
        "strategic_context": {
            "regime": strategic_decision.get("market_regime"),
            "risk_appetite": strategic_decision.get("risk_appetite"),
            "liquidity_maintained": True,
            "capital_based_sizing": True,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        }
    })

    # Guardar decisión estratégica
    ensure_dir(os.path.dirname(STRATEGIC_DECISION_FILE))
    save_decision(strategic_decision, STRATEGIC_DECISION_FILE)

    logger.info(f"✅ L3 procesado exitosamente - Régimen: {strategic_decision.get('market_regime')}, Órdenes: {len(ordenes)}")
    return state


def main(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ejecuta el pipeline L3 actualizado sin depender de fallbacks obsoletos.
    """
    logger.info("⚡ Ejecutando L3 main() - Versión actualizada")

    if state is None:
        # Estado mínimo para testing
        state = {
            "portfolio": {
                "total_value": 3000.0,
                "usdt_balance": 1500.0,
                "btc_balance": 0.05,
                "eth_balance": 1.0
            },
            "market_data": {
                "BTCUSDT": {"close": 50000.0},
                "ETHUSDT": {"close": 3000.0}
            }
        }

    state = procesar_l3(state)

    # Guardar output principal
    os.makedirs(os.path.dirname(L3_OUTPUT), exist_ok=True)
    with open(L3_OUTPUT, "w") as f:
        json.dump(make_json_serializable(state), f, indent=2)

    logger.info(f"✅ L3 output actualizado en {L3_OUTPUT}")
    return state


if __name__ == "__main__":
    main()
