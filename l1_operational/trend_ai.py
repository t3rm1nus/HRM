import os
import csv
import numpy as np
from typing import Dict, Any
from loguru import logger
import joblib

from .config import TREND_THRESHOLD

# --- Paths a modelos ---
MODEL_RF_PATH = "C:/proyectos/HRM/models/L1/modelo2_rf.pkl"
MODEL_LGBM_PATH = "C:/proyectos/HRM/models/L1/modelo3_lgbm.pkl"
MODEL_LR_PATH = "C:/proyectos/HRM/models/L1/modelo1_lr.pkl"  # si existiera

HIST_PATH = "C:/proyectos/HRM/logs/trend_ai_history.csv"

# --- Cargar modelos ---
models = {}
for name, path in [("RF", MODEL_RF_PATH), ("LGBM", MODEL_LGBM_PATH), ("LR", MODEL_LR_PATH)]:
    if os.path.exists(path):
        models[name] = joblib.load(path)
        logger.info(f"[TrendAI] Modelo {name} cargado exitosamente desde {path}")
    else:
        models[name] = None
        logger.warning(f"[TrendAI] Modelo {name} NO encontrado en {path}. Usando fallback si es necesario.")

# --- Función interna para fallback minimalista ---
def _score_trend(signal: Dict[str, Any]) -> float:
    features = signal.get("features", {}) or {}
    if not features:
        logger.warning("[TrendAI-Fallback] No features disponibles. Retornando score default 1.0")
        return 1.0
    rsi = features.get("rsi_trend", 0.5)
    macd = features.get("macd_trend", 0.5)
    slope = features.get("price_slope", 0.5)
    score = 0.4 * rsi + 0.4 * macd + 0.2 * slope
    score = float(max(0.0, min(1.0, score)))
    logger.info(f"[TrendAI-Fallback] Calculado score: {score:.3f} (rsi={rsi}, macd={macd}, slope={slope})")
    return score

# --- Función para convertir features a vector ML ---
def _extract_features(signal: dict):
    features = signal.get("features", {}) or {}
    symbol = signal.get("symbol", "BTC")

    # Lista completa de features en el orden exacto usado en entrenamiento (52 features)
    ordered = [
        features.get("delta_close", 0.0),
        features.get("ema_10", 0.0),
        features.get("ema_20", 0.0),
        features.get("sma_10", 0.0),
        features.get("sma_20", 0.0),
        features.get("volume", 0.0),
        features.get("vol_rel", 0.0),
        features.get("rsi", 0.0),
        features.get("macd", 0.0),
        features.get("macd_signal", 0.0),
        features.get("macd_hist", 0.0),
        features.get("trend_adx", 0.0),
        features.get("momentum_stoch", 0.0),
        features.get("momentum_stoch_signal", 0.0),
        features.get("volume_obv", 0.0),
        features.get("volatility_bbw", 0.0),
        features.get("volatility_atr", 0.0),
        features.get("trend_sma_fast", 0.0),
        features.get("trend_sma_slow", 0.0),
        features.get("trend_ema_fast", 0.0),
        features.get("trend_ema_slow", 0.0),
        features.get("trend_macd", 0.0),
        features.get("momentum_rsi", 0.0),
        features.get("close_5m", 0.0),
        features.get("delta_close_5m", 0.0),
        features.get("ema_10_5m", 0.0),
        features.get("ema_20_5m", 0.0),
        features.get("sma_10_5m", 0.0),
        features.get("sma_20_5m", 0.0),
        features.get("volume_5m", 0.0),
        features.get("vol_rel_5m", 0.0),
        features.get("rsi_5m", 0.0),
        features.get("macd_5m", 0.0),
        features.get("macd_signal_5m", 0.0),
        features.get("macd_hist_5m", 0.0),
        features.get("trend_adx_5m", 0.0),
        features.get("momentum_stoch_5m", 0.0),
        features.get("momentum_stoch_signal_5m", 0.0),
        features.get("volume_obv_5m", 0.0),
        features.get("volatility_bbw_5m", 0.0),
        features.get("volatility_atr_5m", 0.0),
        features.get("trend_sma_fast_5m", 0.0),
        features.get("trend_sma_slow_5m", 0.0),
        features.get("trend_ema_fast_5m", 0.0),
        features.get("trend_ema_slow_5m", 0.0),
        features.get("trend_macd_5m", 0.0),
        features.get("momentum_rsi_5m", 0.0),
        features.get("eth_btc_ratio", 0.0),
        features.get("eth_btc_ratio_sma", 0.0),
        features.get("btc_eth_corr", 0.0),
        1.0 if symbol == "BTC" else 0.0,  # is_btc
        1.0 if symbol == "ETH" else 0.0   # is_eth
    ]

    logger.debug(f"[TrendAI] Features extraídas para {symbol}: {len(ordered)} features")
    return [ordered]  # sklearn espera 2D: [n_samples, n_features]

# --- Guardar histórico ---
def _save_history(signal, probs, final_decision):
    os.makedirs(os.path.dirname(HIST_PATH), exist_ok=True)
    with open(HIST_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            signal.get("symbol"),
            signal.get("timeframe"),
            signal.get("price"),
            signal.get("volume"),
            probs.get("RF"),
            probs.get("LGBM"),
            probs.get("LR"),
            final_decision
        ])
    logger.info(f"[TrendAI] Historia guardada para signal {signal.get('signal_id')}")

# --- Función pública ---
def filter_signal(signal: Dict[str, Any]) -> bool:
    """
    Retorna True si la señal supera el umbral de tendencia.
    Combina los 3 modelos ML + fallback minimalista.
    """
    try:
        probs = {}
        X = _extract_features(signal)

        # --- Calcular probabilidad de cada modelo ---
        for name in ["RF", "LGBM", "LR"]:
            if models.get(name):
                model = models[name]
                # Handle different model types
                if hasattr(model, 'predict_proba'):
                    # sklearn models (RF, LR) have predict_proba
                    probs[name] = float(model.predict_proba(X)[0][1])
                elif hasattr(model, 'predict'):
                    # LightGBM native Booster returns probabilities directly with predict()
                    pred = model.predict(X)
                    if isinstance(pred, (list, np.ndarray)) and len(pred) > 0:
                        # For binary classification, predict() returns probabilities
                        probs[name] = float(pred[0])
                    else:
                        probs[name] = 0.5  # fallback
                else:
                    probs[name] = 0.5  # fallback
                logger.info(f"[TrendAI] Modelo {name} predijo prob: {probs[name]:.3f}")
            else:
                probs[name] = None
                logger.warning(f"[TrendAI] Modelo {name} no disponible. Saltando.")

        # --- Ensemble ponderado ---
        weights = {"RF": 0.3, "LGBM": 0.5, "LR": 0.2}  # ajustar según desempeño
        valid_probs = {k: v for k, v in probs.items() if v is not None}
        if not valid_probs:
            raise ValueError("Ningún modelo disponible para ensemble.")
        
        weighted_sum = sum(p * weights[name] for name, p in valid_probs.items())
        total_weight = sum(weights[name] for name in valid_probs)
        decision_prob = weighted_sum / total_weight
        final_decision = decision_prob >= TREND_THRESHOLD

        logger.info(
            f"[TrendAI-Ensemble] symbol={signal.get('symbol')} timeframe={signal.get('timeframe')} "
            f"probs={probs} -> ensemble_prob={decision_prob:.3f} -> {'PASS' if final_decision else 'BLOCK'}"
        )

        _save_history(signal, probs, final_decision)
        return final_decision

    except Exception as e:
        logger.error(f"[TrendAI] Error evaluando señal: {e} | signal={signal}")
        # fallback
        score = _score_trend(signal)
        final_decision = score >= TREND_THRESHOLD
        logger.info(f"[TrendAI-Fallback] score={score:.3f} -> {'PASS' if final_decision else 'BLOCK'}")
        return final_decision
