import os
import csv
from typing import Dict, Any
from loguru import logger
import joblib

from .config import TREND_THRESHOLD

# --- Paths a modelos ---
MODEL_RF_PATH = "C:/proyectos/HRM/models/modelo2_rf.pkl"
MODEL_LGBM_PATH = "C:/proyectos/HRM/models/modelo3_lgbm.pkl"
MODEL_LR_PATH = "C:/proyectos/HRM/models/modelo1_lr.pkl"  # si existiera

HIST_PATH = "C:/proyectos/HRM/logs/trend_ai_history.csv"

# --- Cargar modelos ---
models = {}
for name, path in [("RF", MODEL_RF_PATH), ("LGBM", MODEL_LGBM_PATH), ("LR", MODEL_LR_PATH)]:
    if os.path.exists(path):
        models[name] = joblib.load(path)
        logger.info(f"[TrendAI] Modelo {name} cargado desde {path}")
    else:
        models[name] = None
        logger.warning(f"[TrendAI] Modelo {name} NO encontrado en {path}")

# --- Función interna para fallback minimalista ---
def _score_trend(signal: Dict[str, Any]) -> float:
    features = signal.get("features", {}) or {}
    if not features:
        return 1.0
    rsi = features.get("rsi_trend", 0.5)
    macd = features.get("macd_trend", 0.5)
    slope = features.get("price_slope", 0.5)
    score = 0.4 * rsi + 0.4 * macd + 0.2 * slope
    return float(max(0.0, min(1.0, score)))

# --- Función para convertir features a vector ML ---
def _extract_features(signal: dict):
    features = signal.get("features", {}) or {}
    ordered = [
        features.get("delta_close", 0.0),
        features.get("delta_close_5m", 0.0),
        features.get("momentum_stoch", 0.0),
        features.get("momentum_stoch_5m", 0.0),
        features.get("macd", 0.0),
        features.get("macd_hist", 0.0),
        features.get("volatility_atr", 0.0),
        features.get("volatility_bbw", 0.0),
        # Añadir todas las demás features usadas en entrenamiento
    ]
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
        if models.get("RF"):
            probs["RF"] = float(models["RF"].predict_proba(X)[0][1])
        else:
            probs["RF"] = None

        if models.get("LGBM"):
            probs["LGBM"] = float(models["LGBM"].predict_proba(X)[0][1])
        else:
            probs["LGBM"] = None

        if models.get("LR"):
            probs["LR"] = float(models["LR"].predict_proba(X)[0][1])
        else:
            probs["LR"] = None

        # --- Ensemble ponderado ---
        weights = {"RF": 0.3, "LGBM": 0.5, "LR": 0.2}  # ajustar según desempeño
        weighted_probs = [
            (p * weights[name]) for name, p in probs.items() if p is not None
        ]
        decision_prob = sum(weighted_probs) / sum([weights[name] for name, p in probs.items() if p is not None])
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
