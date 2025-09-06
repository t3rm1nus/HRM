# l3_aggregator.py
import os
import json
import pandas as pd
from datetime import datetime

# ======== CONFIG ========
DATA_DIR = "data/datos_inferencia"
OUTPUT_DIR = "data/l3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pesos jerárquicos por defecto si algún input falta
DEFAULT_WEIGHTS = {
    "BTC": 0.5,
    "ETH": 0.3,
    "USDT": 0.2
}

# ======== FUNCIONES AUXILIARES ========
def load_csv_or_empty(path):
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    else:
        print(f"⚠️ Archivo no encontrado: {path}, usando DataFrame vacío")
        return pd.DataFrame()

def load_json_or_empty(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        print(f"⚠️ Archivo no encontrado: {path}, usando diccionario vacío")
        return {}

# ======== CARGA DE OUTPUTS L3 ========
def load_l3_outputs():
    outputs = {}

    # Regime detection
    outputs['regime'] = load_json_or_empty(os.path.join(DATA_DIR, "regime_detection.json"))

    # Volatilidad
    outputs['volatility'] = load_csv_or_empty(os.path.join(DATA_DIR, "volatility_forecast.csv"))

    # Portfolio Black-Litterman
    outputs['bl_weights'] = load_csv_or_empty(os.path.join(DATA_DIR, "bl_weights.csv"))

    # Sentiment
    outputs['sentiment'] = load_csv_or_empty(os.path.join(DATA_DIR, "sentiment_scores.csv"))

    return outputs

# ======== COMBINACIÓN JERÁRQUICA ========
def combine_outputs(outputs):
    """
    Combina outputs de L3 para generar pesos jerárquicos finales.
    """
    # Inicializar pesos finales con BL si existe, si no, default
    if not outputs['bl_weights'].empty:
        weights = outputs['bl_weights'].to_dict(orient='records')[0]
    else:
        weights = DEFAULT_WEIGHTS.copy()
    
    # Ajuste por régimen de mercado
    regime = outputs['regime'].get("market_regime", "neutral")
    if regime == "bull_market":
        factor = 1.1
    elif regime == "bear_market":
        factor = 0.9
    else:
        factor = 1.0

    weights = {k: min(v*factor, 1.0) for k, v in weights.items()}

    # Normalizar para que sumen 1
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}

    # Ajuste final por volatilidad (fallback a histórica si no hay modelo)
    if not outputs['volatility'].empty:
        for asset in weights:
            if asset in outputs['volatility'].columns:
                vol = outputs['volatility'][asset].iloc[-1]
                weights[asset] *= 1 / (1 + vol)
        # Renormalizar
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

    return weights

# ======== GENERACIÓN JSON FINAL ========
def generate_l2_json(weights):
    timestamp = datetime.utcnow().isoformat()
    output = {
        "timestamp": timestamp,
        "asset_allocation": weights,
        "market_regime": outputs['regime'].get("market_regime", "neutral"),
        "risk_appetite": outputs['regime'].get("risk_appetite", "moderate"),
        "confidence_level": outputs['regime'].get("confidence_level", 0.8)
    }
    out_path = os.path.join(OUTPUT_DIR, f"l3_to_l2_{datetime.utcnow().strftime('%Y%m%d')}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ JSON final para L2 generado en '{out_path}'")
    return out_path

# ======== MAIN ========
if __name__ == "__main__":
    print("⏳ Cargando outputs L3...")
    outputs = load_l3_outputs()

    print("🔗 Combinando outputs para generar pesos jerárquicos...")
    final_weights = combine_outputs(outputs)

    print("📤 Generando JSON final para L2/L1...")
    json_path = generate_l2_json(final_weights)

    print("🎉 Pipeline de agregación L3 completo.")
