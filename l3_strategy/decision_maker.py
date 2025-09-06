"""
Decision Maker - L3
Toma los outputs de todos los módulos de L3 (regime, sentiment, portfolio, risk, macro)
y genera las directrices estratégicas unificadas para L2.
"""

import os
import json
from datetime import datetime

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


def make_decision(inputs: dict):
    """
    Combina todos los outputs de L3 en una decisión estratégica.
    """
    regime = inputs.get("regime_detection", {}).get("predicted_regime", "unknown")
    sentiment = inputs.get("sentiment", {}).get("sentiment_score", 0.0)
    portfolio = inputs.get("portfolio", {}).get("weights", {})
    risk_appetite = inputs.get("risk", {}).get("risk_appetite", "moderate")
    macro = inputs.get("macro", {})

    decision = {
        "timestamp": datetime.utcnow().isoformat(),
        "market_regime": regime,
        "sentiment_score": sentiment,
        "asset_allocation": portfolio,
        "risk_appetite": risk_appetite,
        "macro_context": macro,
        "strategic_guidelines": {
            "rebalance_frequency": "weekly",
            "max_single_asset_exposure": 0.7 if risk_appetite == "high" else 0.5,
            "volatility_target": 0.25 if risk_appetite == "high" else 0.15,
            "liquidity_requirement": "high" if risk_appetite != "high" else "medium"
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
