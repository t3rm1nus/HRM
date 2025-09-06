"""
Risk Manager - L3
Calcula el apetito de riesgo estratégico a partir de inputs de mercado
(volatilidad, régimen, sentimiento, macro).
Genera un output JSON para integrarse en el pipeline HRM.
"""

import os
import json
from datetime import datetime

# Directorio de salida
OUTPUT_DIR = "data/datos_inferencia"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "risk.json")


def ensure_dir(directory: str):
    """Crea el directorio si no existe"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_risk_appetite(volatility: float, sentiment: float, regime: str) -> str:
    """
    Define el apetito de riesgo según inputs de mercado.
    volatility: valor entre 0 y 1 (normalizado)
    sentiment: valor entre -1 (muy negativo) y 1 (muy positivo)
    regime: bull, bear, range, volatile
    """
    if regime == "bear" or volatility > 0.7:
        return "low"
    elif sentiment > 0.3 and volatility < 0.5 and regime == "bull":
        return "high"
    else:
        return "moderate"


def risk_analysis():
    """
    Simulación: en la versión final, cargará datos de l3_output.json.
    """
    volatility = 0.45  # ejemplo normalizado
    sentiment = 0.2    # ejemplo
    regime = "bull"

    appetite = calculate_risk_appetite(volatility, sentiment, regime)

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": {
            "volatility": volatility,
            "sentiment": sentiment,
            "regime": regime
        },
        "risk_appetite": appetite
    }
    return results


def save_risk(data: dict, output_path: str):
    """Guarda el resultado de risk manager en JSON"""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Output risk guardado en {output_path}")


if __name__ == "__main__":
    print("🔄 Ejecutando Risk Manager...")
    ensure_dir(OUTPUT_DIR)

    results = risk_analysis()
    save_risk(results, OUTPUT_FILE)

    print("📊 Resumen Risk Manager:")
    print(json.dumps(results, indent=4))
