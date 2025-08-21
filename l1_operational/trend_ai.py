"""
Filtro de tendencia (Trend AI) minimalista para L1/L2.

Interfaz pública:
    filter_signal(signal: dict) -> bool

Entrada esperada:
    signal: dict con al menos las claves:
        - symbol: str
        - timeframe: str (ej. '1m', '5m')
        - price: float (último precio)
        - volume: float (volumen reciente)
        - features: dict opcional con señales derivadas (ej. rsi, macd, slope)

Salida:
    bool indicando si la operación pasa el filtro de tendencia.
"""

from typing import Dict, Any
from loguru import logger

from .config import TREND_THRESHOLD


def _score_trend(signal: Dict[str, Any]) -> float:
    """
    Calcula un score simple de tendencia a partir de features opcionales.
    Política mínima: media ponderada de señales disponibles.
    Si no hay features, retorna 1.0 para no bloquear por defecto.
    """
    features = (signal or {}).get("features", {}) or {}
    if not features:
        return 1.0

    # Tomamos algunas señales comunes si existen
    rsi = features.get("rsi_trend", 0.5)         # [0,1]
    macd = features.get("macd_trend", 0.5)       # [0,1]
    slope = features.get("price_slope", 0.5)     # [0,1]

    # Ponderación simple
    score = 0.4 * rsi + 0.4 * macd + 0.2 * slope
    return float(max(0.0, min(1.0, score)))


def filter_signal(signal: Dict[str, Any]) -> bool:
    """Retorna True si la señal supera el umbral de tendencia.

    Ejemplo de uso:
        ok = filter_signal({"symbol":"BTC/USDT", "features": {"rsi_trend":0.7}})
    """
    try:
        score = _score_trend(signal)
        decision = score >= TREND_THRESHOLD
        logger.info(
            f"[TrendAI] symbol={signal.get('symbol')} timeframe={signal.get('timeframe')} "
            f"score={score:.3f} threshold={TREND_THRESHOLD} -> {'PASS' if decision else 'BLOCK'}"
        )
        return decision
    except Exception as e:
        logger.error(f"[TrendAI] Error evaluando señal: {e} | signal={signal}")
        # En caso de error, por seguridad no bloquear por defecto en L1; devolver True
        return True


