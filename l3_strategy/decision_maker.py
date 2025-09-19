"""
Decision Maker - L3
Toma los outputs de todos los módulos de L3 (regime, sentiment, portfolio, risk, macro)
y genera las directrices estratégicas unificadas para L2.
Incluye gestión de exposición basada en capital real y régimen de mercado.
"""

import os
import json
from datetime import datetime
from .regime_classifier import clasificar_regimen
from .exposure_manager import gestionar_exposicion
from core.logging import logger

# Directorio de inferencias
INFER_DIR = "data/datos_inferencia"
OUTPUT_FILE = os.path.join(INFER_DIR, "strategic_decision.json")


def ensure_dir(directory: str):
    """Crea el directorio si no existe"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_inputs():
    """Carga todos los JSON disponibles en data/datos_inferencia"""
    results = {}
    for file in os.listdir(INFER_DIR):
        if file.endswith(".json") and file not in ["l3_output.json", "strategic_decision.json"]:
            path = os.path.join(INFER_DIR, file)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    results[file.replace(".json", "")] = data
            except Exception as e:
                print(f"⚠️ Error cargando {file}: {e}")
    return results


def make_decision(inputs: dict, portfolio_state: dict = None, market_data: dict = None):
    """
    Combina todos los outputs de L3 en una decisión estratégica.
    Ahora incluye gestión de exposición basada en capital real.
    """
    # Obtener régimen usando el clasificador actualizado
    regime = clasificar_regimen(market_data) if market_data else inputs.get("regime_detection", {}).get("predicted_regime", "neutral")

    sentiment = inputs.get("sentiment", {}).get("sentiment_score", 0.0)
    portfolio = inputs.get("portfolio", {}).get("weights", {})
    risk_appetite = inputs.get("risk", {}).get("risk_appetite", "moderate")
    macro = inputs.get("macro", {})

    # Gestionar exposición si tenemos datos de portfolio y mercado
    exposure_decisions = {}
    if portfolio_state and market_data:
        universo = ["BTCUSDT", "ETHUSDT"]  # Universos disponibles
        exposure_decisions = gestionar_exposicion(universo, portfolio_state, market_data, regime)
        logger.info("📊 Decisiones de exposición calculadas exitosamente")
    else:
        logger.warning("⚠️ Datos insuficientes para gestión de exposición - usando configuración por defecto")

    # Ajustar guidelines basados en régimen y apetito de riesgo
    max_single_exposure = 0.7 if risk_appetite == "high" else 0.5
    if regime == "bear":
        max_single_exposure = 0.3  # Reducir exposición máxima en bear market

    decision = {
        "timestamp": datetime.utcnow().isoformat(),
        "market_regime": regime,
        "sentiment_score": sentiment,
        "asset_allocation": portfolio,
        "risk_appetite": risk_appetite,
        "macro_context": macro,
        "exposure_decisions": exposure_decisions,
        "strategic_guidelines": {
            "rebalance_frequency": "daily" if regime == "volatile" else "weekly",
            "max_single_asset_exposure": max_single_exposure,
            "volatility_target": 0.25 if risk_appetite == "high" else 0.15,
            "liquidity_requirement": "high" if risk_appetite != "high" or regime == "bear" else "medium",
            "btc_max_exposure": 0.2 if regime == "bear" else 0.5,
            "usdt_min_liquidity": 0.10  # 10% mínimo en liquidez
        }
    }
    return decision


def save_decision(data: dict, output_path: str):
    """Guarda la decisión estratégica en JSON"""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Decisión estratégica guardada en {output_path}")


if __name__ == "__main__":
    print("🔄 Ejecutando Decision Maker...")
    ensure_dir(INFER_DIR)

    inputs = load_inputs()
    decision = make_decision(inputs)
    save_decision(decision, OUTPUT_FILE)

    print("📊 Resumen Decision Maker:")
    print(json.dumps(decision, indent=4))
