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
FEATURES_FILE = os.path.join(INFER_DIR, "regime_features_expected.json")

REGIME_MODEL_PATH = "models/L3/regime_detection_model_ensemble_optuna.pkl"
SENTIMENT_MODEL_DIR = "models/L3/sentiment/"
VOL_GARCH_PATH_BTC = "models/L3/volatility/BTC-USD_volatility_garch.pkl"
VOL_GARCH_PATH_ETH = "models/L3/volatility/ETH-USD_volatility_garch.pkl"
VOL_LSTM_PATH_BTC = "models/L3/volatility/BTC-USD_volatility_lstm.h5"
VOL_LSTM_PATH_ETH = "models/L3/volatility/ETH-USD_volatility_lstm.h5"
PORTFOLIO_COV_PATH = "models/L3/portfolio/bl_cov.csv"
PORTFOLIO_WEIGHTS_PATH = "models/L3/portfolio/bl_weights.csv"

# Volatility warmup: mínimo de cierres requeridos y valor por defecto
MIN_VOL_BARS = 10
WARMUP_VOL = 0.03

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
    # Log y persistencia de las features esperadas, si existen
    try:
        if hasattr(model, 'feature_names_in_'):
            names = list(getattr(model, 'feature_names_in_'))
            log.info(f"Regime features esperadas ({len(names)}): {names}")
            os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
            with open(FEATURES_FILE, 'w', encoding='utf-8') as f:
                json.dump({"feature_names_in_": names}, f, indent=2)
    except Exception as e:
        log.warning(f"No se pudieron registrar las features de Regime: {e}")
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
    weights = pd.read_csv(PORTFOLIO_WEIGHTS_PATH)
    if weights.empty:
        raise ValueError("weights vacío")
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
    inputs = tokenizer(texts or ["market"], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        num_classes = probs.shape[-1]
        if num_classes == 2:
            score = probs[:, 1] - probs[:, 0]
        elif num_classes >= 3:
            score = probs[:, 2] - probs[:, 0]
        else:
            score = probs[:, -1]
        sentiment_score = float(torch.mean(score))
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
    """Construye un DataFrame con EXACTAMENTE las columnas que el modelo espera.
    - Si el modelo expone feature_names_in_, se crean todas con defaults 0.0 y se rellenan las disponibles.
    - Calcula rápidamente algunas features comunes (return, log_return, bollinger, macd) si hay serie de cierres.
    """
    # Columnas esperadas
    if hasattr(model, 'feature_names_in_'):
        expected = list(getattr(model, 'feature_names_in_'))
    else:
        expected = [
            'open','high','low','close','volume',
            'return','log_return','macd','macdsig','macdhist',
            'boll_lower','boll_middle','boll_upper'
        ]

    row = {name: 0.0 for name in expected}

    # Helpers de OHLCV
    def _series(sym: str):
        val = market_data.get(sym)
        if isinstance(val, list) and val:
            return pd.DataFrame(val)
        if isinstance(val, dict):
            return pd.DataFrame([val])
        if isinstance(val, pd.DataFrame):
            return val
        if isinstance(val, pd.Series):
            return pd.DataFrame({'close': val})
        return pd.DataFrame()

    def _last_ohlcv(df: pd.DataFrame, key: str) -> float:
        try:
            if key in df.columns and not df.empty:
                return float(df[key].iloc[-1])
        except Exception:
            pass
        return 0.0

    primary = 'BTCUSDT' if 'BTCUSDT' in market_data else ('ETHUSDT' if 'ETHUSDT' in market_data else None)
    dfp = _series(primary) if primary else pd.DataFrame()

    # Mapear OHLCV básicos
    for key in ('open','high','low','close','volume'):
        if key in row:
            row[key] = _last_ohlcv(dfp, key)

    # Si hay serie de cierres suficiente, calcular retornos y MACD/Bollinger mínimos
    try:
        if not dfp.empty and 'close' in dfp.columns:
            close = pd.to_numeric(dfp['close'], errors='coerce').dropna()
            if len(close) >= 2:
                ret = close.pct_change().iloc[-1]
                lret = np.log(close).diff().iloc[-1]
                if 'return' in row:
                    row['return'] = float(ret if np.isfinite(ret) else 0.0)
                if 'log_return' in row:
                    row['log_return'] = float(lret if np.isfinite(lret) else 0.0)

            # Bollinger 20
            if len(close) >= 20:
                ma = close.rolling(20).mean().iloc[-1]
                std = close.rolling(20).std().iloc[-1]
                if 'boll_middle' in row: row['boll_middle'] = float(ma if np.isfinite(ma) else 0.0)
                if 'boll_upper' in row: row['boll_upper'] = float((ma + 2*std) if np.isfinite(ma) and np.isfinite(std) else 0.0)
                if 'boll_lower' in row: row['boll_lower'] = float((ma - 2*std) if np.isfinite(ma) and np.isfinite(std) else 0.0)

            # MACD (12,26,9) simple con EMA
            if len(close) >= 35:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = (ema12 - ema26)
                macdsig = macd.ewm(span=9, adjust=False).mean()
                macdhist = macd - macdsig
                if 'macd' in row: row['macd'] = float(macd.iloc[-1])
                if 'macdsig' in row: row['macdsig'] = float(macdsig.iloc[-1])
                if 'macdhist' in row: row['macdhist'] = float(macdhist.iloc[-1])
    except Exception:
        pass

    # Devolver DataFrame en el orden exacto esperado
    df = pd.DataFrame([row])
    # Garantizar todas las columnas y el orden
    for c in expected:
        if c not in df.columns:
            df[c] = 0.0
    
    # Debug: log de features esperadas vs proporcionadas
    missing_features = [c for c in expected if c not in df.columns or df[c].iloc[0] == 0.0]
    if missing_features:
        log.debug(f"Features faltantes o en 0: {missing_features[:10]}...")  # Solo primeras 10
    
    # Log detallado de features return_*
    return_features = [c for c in expected if isinstance(c, str) and c.startswith('return_')]
    if return_features:
        log.info(f"Features return_* esperadas: {return_features[:5]}...")  # Solo primeras 5
    # Si el modelo espera columnas tipo return_{N}, calcularlas dinámicamente
    try:
        if not dfp.empty and 'close' in dfp.columns:
            close_full = pd.to_numeric(dfp['close'], errors='coerce').dropna()
            log.info(f"Calculando retornos con {len(close_full)} puntos de datos")
            
            # Calcular todos los retornos de una vez
            for name in expected:
                if isinstance(name, str) and name.startswith('return_'):
                    try:
                        n = int(name.split('_', 1)[1])
                        if n > 0 and len(close_full) > n:
                            # Calcular retorno de N períodos hacia atrás
                            current_price = close_full.iloc[-1]
                            past_price = close_full.iloc[-n-1]
                            if past_price > 0:
                                r = (current_price / past_price) - 1.0
                                df.loc[0, name] = float(r) if np.isfinite(r) else 0.0
                                log.info(f"Calculado {name}: {df.loc[0, name]:.6f}")
                            else:
                                df.loc[0, name] = 0.0
                        else:
                            # Si no hay suficientes datos, usar 0.0
                            df.loc[0, name] = 0.0
                            log.info(f"Insuficientes datos para {name}: {len(close_full)} < {n+1}")
                    except (ValueError, IndexError, ZeroDivisionError) as e:
                        log.debug(f"Error calculando {name}: {e}")
                        df.loc[0, name] = 0.0
        else:
            log.info(f"No hay datos suficientes: dfp.empty={dfp.empty}, close in columns={'close' in dfp.columns if not dfp.empty else False}")
    except Exception as e:
        log.debug(f"Error en cálculo de retornos: {e}")
        pass
    df = df[expected]
    return df

def _safe_close_series(val) -> np.ndarray:
    """Convierte market_data[symbol] a un np.array de cierres de longitud >=1."""
    try:
        if isinstance(val, list) and val:
            arr = [float(d.get('close', 0.0)) for d in val if isinstance(d, dict)]
            return np.array(arr, dtype=float) if arr else np.array([0.0], dtype=float)
        if isinstance(val, dict):
            return np.array([float(val.get('close', 0.0))], dtype=float)
        if isinstance(val, pd.DataFrame) and 'close' in val.columns:
            return val['close'].dropna().values.astype(float)
        if isinstance(val, pd.Series):
            return val.dropna().values.astype(float)
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

    # Volatilidad (con warmup forzado si hay pocas velas)
    try:
        garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()
        btc_series = _safe_close_series(market_data.get('BTCUSDT'))
        eth_series = _safe_close_series(market_data.get('ETHUSDT'))
        if len(btc_series) < MIN_VOL_BARS or len(eth_series) < MIN_VOL_BARS:
            log.warning(f"Volatility warmup: series cortas (BTC={len(btc_series)}, ETH={len(eth_series)}), usando valor por defecto {WARMUP_VOL}")
            vol_btc = WARMUP_VOL
            vol_eth = WARMUP_VOL
        else:
            vol_btc = 0.5*(predict_vol_garch(garch_btc, btc_series)+predict_vol_lstm(lstm_btc, btc_series))
            vol_eth = 0.5*(predict_vol_garch(garch_eth, eth_series)+predict_vol_lstm(lstm_eth, eth_series))
    except Exception as e:
        log.critical(f"Volatility fallback por error: {e}")
        vol_btc = WARMUP_VOL
        vol_eth = WARMUP_VOL

    volatility_avg = float(np.mean([vol_btc, vol_eth]))

    # Black-Litterman / pesos
    try:
        cov_matrix, optimal_weights = load_portfolio()
        first_row = optimal_weights.iloc[0]
        # Filtrar solo columnas cripto conocidas o numéricas
        allowed = {"BTC","ETH","CASH","BTCUSDT","ETHUSDT"}
        cols = [c for c in first_row.index if (c in allowed)]
        if not cols:
            # fallback: elegir solo columnas cuyo valor sea convertible a float
            cols = [c for c in first_row.index if pd.api.types.is_number(first_row[c]) or str(first_row[c]).replace('.','',1).isdigit()]
        asset_allocation = {str(col): float(first_row[col]) for col in cols}
        if not asset_allocation:
            raise ValueError("sin columnas de asignación válidas")
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
# Bloque de ejecución CLI comentado para evitar ejecución automática al importar
# if __name__ == "__main__":
#     market_data_example = {
#         "BTCUSDT": [{"open":50000,"high":50500,"low":49900,"close":50250,"volume":1.2},
#                     {"open":50200,"high":50400,"low":50000,"close":50100,"volume":1.0},
#                     {"open":50150,"high":50300,"low":50050,"close":50200,"volume":1.5}],
#         "ETHUSDT": [{"open":3500,"high":3550,"low":3480,"close":3520,"volume":10},
#                     {"open":3520,"high":3540,"low":3490,"close":3510,"volume":12},
#                     {"open":3510,"high":3530,"low":3500,"close":3525,"volume":9}]
#     }
#     texts_example = [
#         "BTC will rally after the Fed announcement",
#         "ETH bearish sentiment in crypto news"
#     ]
#     try:
#         output = generate_l3_output(market_data_example, texts_example)
#         log.info("Ejecución de L3 finalizada con éxito", extra={"output": output})
#         print(json.dumps(output, indent=2))
#     except Exception as e:
#         log.critical(f"L3 falló: {e}")
#         raise
