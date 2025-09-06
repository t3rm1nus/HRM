# l3_strategy/l3_processor.py
"""
L3 Processor - HRM
Genera un output jerárquico para L2 combinando:
- Macro conditions
- Regime detection
- Market sentiment
- Volatility forecasts
- Portfolio optimization
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model

from core import logging as log

# ---------------------------
# Paths
# ---------------------------
INFER_DIR = "data/datos_inferencia"
OUTPUT_FILE = os.path.join(INFER_DIR, "l3_output.json")

REGIME_MODEL_PATH = "models/L3/regime_detection_model_ensemble_optuna.pkl"
SENTIMENT_MODEL_DIR = "models/L3/sentiment/"
VOL_GARCH_PATH_BTC = "models/L3/volatility/BTC-USD_volatility_garch.pkl"
VOL_GARCH_PATH_ETH = "models/L3/volatility/ETH-USD_volatility_garch.pkl"
VOL_LSTM_PATH_BTC = "models/L3/volatility/BTC-USD_volatility_lstm.h5"
VOL_LSTM_PATH_ETH = "models/L3/volatility/ETH-USD_volatility_lstm.h5"
PORTFOLIO_COV_PATH = "models/L3/portfolio/bl_cov.csv"
PORTFOLIO_WEIGHTS_PATH = "models/L3/portfolio/bl_weights.csv"

# ---------------------------
# Helpers
# ---------------------------
def save_json(data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    def make_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4, default=make_serializable)
    log.info(f"L3 output guardado en {output_path}")

# ---------------------------
# Load Models
# ---------------------------
def load_regime_model():
    import joblib
    log.info("Cargando modelo de Regime Detection")
    if not os.path.exists(REGIME_MODEL_PATH):
        log.critical(f"Modelo de Regime Detection faltante: {REGIME_MODEL_PATH}")
        raise FileNotFoundError(REGIME_MODEL_PATH)
    model = joblib.load(REGIME_MODEL_PATH)
    log.info("Modelo de Regime Detection cargado")
    return model

def load_sentiment_model():
    log.info("Cargando modelo BERT de Sentimiento")
    tokenizer = BertTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
    log.info("Modelo BERT cargado")
    return tokenizer, model

def load_vol_models():
    import joblib
    log.info("Cargando modelos de volatilidad GARCH y LSTM")
    garch_btc = joblib.load(VOL_GARCH_PATH_BTC)
    garch_eth = joblib.load(VOL_GARCH_PATH_ETH)
    lstm_btc = load_model(VOL_LSTM_PATH_BTC, compile=False)
    lstm_eth = load_model(VOL_LSTM_PATH_ETH, compile=False)
    log.info("Modelos de volatilidad cargados")
    return garch_btc, garch_eth, lstm_btc, lstm_eth

def load_portfolio():
    log.info("Cargando optimización de cartera Black-Litterman")
    cov = pd.read_csv(PORTFOLIO_COV_PATH, index_col=0)
    weights = pd.read_csv(PORTFOLIO_WEIGHTS_PATH, index_col=0)
    log.info("Cartera cargada")
    return cov, weights

# ---------------------------
# Inference Functions
# ---------------------------
def predict_regime(features: pd.DataFrame, model):
    # Si el modelo es un ensemble dict, usa el clasificador 'rf' por defecto
    if isinstance(model, dict) and 'rf' in model:
        regime = model['rf'].predict(features)[0]
    else:
        regime = model.predict(features)[0]
    log.info(f"Regime detectado: {regime}")
    return regime

def predict_sentiment(texts: list, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        sentiment_score = float(torch.mean(probs[:, 2] - probs[:, 0]))  # positivo-negativo
    log.info(f"Score de sentimiento calculado: {sentiment_score}")
    return sentiment_score

def predict_vol_garch(model, returns: np.ndarray):
    vol = float(model.forecast(returns))
    log.info(f"Volatilidad GARCH: {vol}")
    return vol

def predict_vol_lstm(model, returns: np.ndarray):
    returns = returns.reshape(-1,1,1)
    pred = model.predict(returns, verbose=0)
    vol = float(pred[-1,0])
    log.info(f"Volatilidad LSTM: {vol}")
    return vol

def compute_risk_appetite(volatility_avg, sentiment_score):
    if volatility_avg > 0.05:
        base = 0.3
    else:
        base = 0.6
    appetite = base + 0.4 * np.tanh(sentiment_score)
    if appetite < 0.2:
        ra = "conservative"
    elif appetite < 0.5:
        ra = "moderate"
    else:
        ra = "aggressive"
    log.info(f"Risk appetite calculado: {ra}")
    return ra

# ---------------------------
# Main L3 Output
# ---------------------------
def generate_l3_output(market_data: dict, texts_for_sentiment: list):
    log.info("Generando L3 output estratégico")
    regime_model = load_regime_model()
    tokenizer, sentiment_model = load_sentiment_model()
    garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()
    cov_matrix, optimal_weights = load_portfolio()

    df_features = pd.DataFrame(market_data)  # adaptar según features reales
    regime = predict_regime(df_features, regime_model)
    sentiment_score = predict_sentiment(texts_for_sentiment, tokenizer, sentiment_model)

    returns_btc = np.array([c["close"] for c in market_data["BTCUSDT"]])
    returns_eth = np.array([c["close"] for c in market_data["ETHUSDT"]])
    vol_btc = 0.5*(predict_vol_garch(garch_btc, returns_btc)+predict_vol_lstm(lstm_btc, returns_btc))
    vol_eth = 0.5*(predict_vol_garch(garch_eth, returns_eth)+predict_vol_lstm(lstm_eth, returns_eth))
    volatility_avg = np.mean([vol_btc, vol_eth])

    asset_allocation = {col: float(optimal_weights.loc[0, col]) for col in optimal_weights.columns}
    risk_appetite = compute_risk_appetite(volatility_avg, sentiment_score)

    strategic_guidelines = {
        "regime": regime,
        "asset_allocation": asset_allocation,
        "risk_appetite": risk_appetite,
        "sentiment_score": sentiment_score,
        "volatility_forecast": {"BTCUSDT": vol_btc, "ETHUSDT": vol_eth},
        "timestamp": datetime.utcnow().isoformat()
    }

    save_json(strategic_guidelines, OUTPUT_FILE)
    log.info("L3 output generado correctamente")
    return strategic_guidelines

# ---------------------------
# CLI Execution
# ---------------------------
if __name__ == "__main__":
    market_data_example = {
        "BTCUSDT": [{"open":50000,"high":50500,"low":49900,"close":50250,"volume":1.2},
                    {"open":50200,"high":50400,"low":50000,"close":50100,"volume":1.0},
                    {"open":50150,"high":50300,"low":50050,"close":50200,"volume":1.5}],
        "ETHUSDT": [{"open":3500,"high":3550,"low":3480,"close":3520,"volume":10},
                    {"open":3520,"high":3540,"low":3490,"close":3510,"volume":12},
                    {"open":3510,"high":3530,"low":3500,"close":3525,"volume":9}]
    }
    texts_example = [
        "BTC will rally after the Fed announcement",
        "ETH bearish sentiment in crypto news"
    ]
    try:
        output = generate_l3_output(market_data_example, texts_example)
        log.info("Ejecución de L3 finalizada con éxito", extra={"output": output})
        print(json.dumps(output, indent=2))
    except Exception as e:
        log.critical(f"L3 falló: {e}")
        raise
