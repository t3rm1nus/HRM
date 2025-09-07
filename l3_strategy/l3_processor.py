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
def _build_regime_features(market_data: dict, model) -> pd.DataFrame:
    """Construye un DataFrame con los nombres de features que el modelo espera.
    Rellena con 0.0 cualquier columna faltante y mapea OHLCV básicos si existen.
    """
    # Obtener nombres esperados desde el modelo si es posible
    if hasattr(model, 'feature_names_in_'):
        expected = list(getattr(model, 'feature_names_in_'))
    else:
        # Fallback razonable si el modelo no expone nombres
        expected = [
            'open','high','low','close','volume',
            'boll_lower','boll_middle','boll_upper'
        ]

    row = {name: 0.0 for name in expected}

    # Intentar mapear OHLCV básicos desde BTCUSDT (o ETHUSDT)
    def _last_close(sym: str) -> float:
        val = market_data.get(sym)
        try:
            if isinstance(val, list) and val:
                return float(val[-1].get('close', 0.0))
            if isinstance(val, dict):
                return float(val.get('close', 0.0))
        except Exception:
            pass
        return 0.0

    def _last_ohlcv(sym: str, key: str) -> float:
        val = market_data.get(sym)
        try:
            if isinstance(val, list) and val:
                return float(val[-1].get(key, 0.0))
            if isinstance(val, dict):
                return float(val.get(key, 0.0))
        except Exception:
            pass
        return 0.0

    primary = 'BTCUSDT' if 'BTCUSDT' in market_data else ('ETHUSDT' if 'ETHUSDT' in market_data else None)
    if primary:
        for key in ('open','high','low','close','volume'):
            if key in row:
                row[key] = _last_ohlcv(primary, key)

    return pd.DataFrame([row], columns=expected)

def _safe_close_series(val) -> np.ndarray:
    """Convierte market_data[symbol] a un np.array de cierres de longitud >=1."""
    try:
        if isinstance(val, list) and val:
            arr = [float(d.get('close', 0.0)) for d in val]
            return np.array(arr, dtype=float)
        if isinstance(val, dict):
            return np.array([float(val.get('close', 0.0))], dtype=float)
    except Exception:
        pass
    return np.array([0.0], dtype=float)

def generate_l3_output(market_data: dict, texts_for_sentiment: list):
    log.info("Generando L3 output estratégico")

    # Carga robusta de modelos con fallbacks por componente
    try:
        regime_model = load_regime_model()
        df_features = _build_regime_features(market_data, regime_model)
        regime = predict_regime(df_features, regime_model)
    except Exception as e:
        log.critical(f"Regime detection fallback por error: {e}")
        regime = 'neutral'

    try:
        tokenizer, sentiment_model = load_sentiment_model()
        sentiment_score = predict_sentiment(texts_for_sentiment or [], tokenizer, sentiment_model)
    except Exception as e:
        log.critical(f"Sentiment fallback por error: {e}")
        sentiment_score = 0.0

    # Volatilidad
    try:
        garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()
        btc_series = _safe_close_series(market_data.get('BTCUSDT'))
        eth_series = _safe_close_series(market_data.get('ETHUSDT'))
        if len(btc_series) < 10 or len(eth_series) < 10:
            raise ValueError("series demasiado cortas para modelos de volatilidad")
        vol_btc = 0.5*(predict_vol_garch(garch_btc, btc_series)+predict_vol_lstm(lstm_btc, btc_series))
        vol_eth = 0.5*(predict_vol_garch(garch_eth, eth_series)+predict_vol_lstm(lstm_eth, eth_series))
    except Exception as e:
        log.critical(f"Volatility fallback por error: {e}")
        vol_btc = 0.03
        vol_eth = 0.03

    volatility_avg = float(np.mean([vol_btc, vol_eth]))

    # Black-Litterman / pesos
    try:
        cov_matrix, optimal_weights = load_portfolio()
        asset_allocation = {col: float(optimal_weights.loc[0, col]) for col in optimal_weights.columns}
    except Exception as e:
        log.critical(f"Portfolio allocation fallback por error: {e}")
        asset_allocation = {"BTC": 0.5, "ETH": 0.4, "CASH": 0.1}

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
