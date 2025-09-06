import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# ================= CONFIG =================
OUTPUT_DIR = "data/datos_inferencia"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ASSETS = ["BTC-USD", "ETH-USD", "USDT"]  # extensible
REGIME_FILE = "data/datos_para_modelos_l3/regime/regime_output.csv"
SENTIMENT_FILE = "data/datos_para_modelos_l3/sentiment/sentiment_output.csv"
VOLATILITY_DIR = "data/datos_para_modelos_l3/volatility"
PORTFOLIO_FILE = "models/L3/portfolio/bl_weights.csv"
HISTORICAL_VOL_FILE = "data/datos_para_modelos_l3/volatility/historical_vol.csv"

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ================= FUNCIONES =================
def load_regime():
    if os.path.exists(REGIME_FILE):
        df = pd.read_csv(REGIME_FILE, index_col=0, parse_dates=True)
        logging.info(f"Regime detection cargado con {len(df)} registros")
        return df.to_dict(orient='records')[-1]
    logging.warning("Regime detection no encontrado, usando default 'neutral'")
    return {"regime": "neutral"}

def load_sentiment():
    if os.path.exists(SENTIMENT_FILE):
        df = pd.read_csv(SENTIMENT_FILE, index_col=0, parse_dates=True)
        logging.info(f"Sentiment cargado con {len(df)} registros")
        return df.to_dict(orient='records')[-1]
    logging.warning("Sentiment no encontrado, usando valores neutros")
    return {asset: 0.5 for asset in ASSETS}

def load_volatility(assets):
    vol_data = {}
    for asset in assets:
        garch_file = os.path.join(VOLATILITY_DIR, f"{asset}_volatility_garch.pkl")
        lstm_file = os.path.join(VOLATILITY_DIR, f"{asset}_volatility_lstm.h5")
        fallback_file = HISTORICAL_VOL_FILE

        vol = None
        # intentar GARCH
        if os.path.exists(garch_file):
            try:
                vol = pd.read_pickle(garch_file).iloc[-1]["volatility"]
                logging.info(f"{asset} - Volatility GARCH cargada")
            except Exception as e:
                logging.warning(f"{asset} - Error cargando GARCH: {e}")

        # intentar LSTM
        if vol is None and os.path.exists(lstm_file):
            try:
                from volatility_model_utils import load_lstm_volatility
                vol = load_lstm_volatility(lstm_file)
                logging.info(f"{asset} - Volatility LSTM cargada")
            except Exception as e:
                logging.warning(f"{asset} - Error cargando LSTM: {e}")

        # fallback a histórica
        if vol is None:
            if os.path.exists(fallback_file):
                df_hist = pd.read_csv(fallback_file, index_col=0)
                vol = df_hist[asset].iloc[-1] if asset in df_hist.columns else 0.1
                logging.info(f"{asset} - Volatility histórica usada")
            else:
                vol = 0.1
                logging.warning(f"{asset} - Volatility fallback default 0.1 usada")

        vol_data[asset] = vol
    return vol_data

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        df = pd.read_csv(PORTFOLIO_FILE, index_col=0)
        logging.info("Portfolio Black-Litterman cargado")
        return df["weight"].to_dict()
    logging.warning("Portfolio BL no encontrado, usando distribución uniforme")
    return {asset: 1/len(ASSETS) for asset in ASSETS}

def combine_hierarchical_weights(portfolio_weights, sentiment, volatility, regime):
    combined = {}
    total_weight = 0.0
    for asset, w in portfolio_weights.items():
        s = sentiment.get(asset, 0.5)
        v = volatility.get(asset, 0.1)
        factor = (1 - v) * s
        combined[asset] = w * factor
        total_weight += combined[asset]

    if total_weight > 0:
        for asset in combined:
            combined[asset] /= total_weight
    else:
        n = len(portfolio_weights)
        combined = {asset: 1/n for asset in portfolio_weights}
        logging.warning("Normalización falló, usando distribución uniforme")

    logging.info(f"Pesos jerárquicos calculados: {combined}")
    return combined

# ================= PIPELINE =================
def run_pipeline(assets=ASSETS):
    logging.info("=== Ejecutando pipeline L3 HRM ===")
    regime = load_regime()
    sentiment = load_sentiment()
    volatility = load_volatility(assets)
    portfolio_weights = load_portfolio()
    final_weights = combine_hierarchical_weights(portfolio_weights, sentiment, volatility, regime)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "regime": regime,
        "sentiment": sentiment,
        "volatility": volatility,
        "portfolio_weights": portfolio_weights,
        "final_weights": final_weights
    }

    out_file = os.path.join(OUTPUT_DIR, f"l3_inference_{datetime.utcnow().strftime('%Y%m%d')}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=4)

    logging.info(f"L3 Inference final guardada en: {out_file}")
    return output

# ================= MAIN =================
if __name__ == "__main__":
    run_pipeline()
