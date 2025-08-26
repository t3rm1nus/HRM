"""
BlenderEnsemble
Combina señales mediante pesos configurables
(similar a un stacking lineal).
"""

from typing import List, Dict, Any
import numpy as np
from core.logging import logger


class BlenderEnsemble:
    """
    Combina señales usando pesos fijos o dinámicos.
    """

    def __init__(self,
                 weights: Dict[str, float],
                 default: float = 0.0):
        """
        weights: dict {source_name: peso}
                 Ej.: {"model_ppo": 0.5, "rsi": 0.3, "macd": 0.2}
        default: peso para cualquier fuente no listada.
        """
        self.weights = weights
        self.default = default
        logger.info(
            f"[BlenderEnsemble] inicializado: {weights} (default={default})"
        )

    # ------------------------------------------------------------------ #
    def blend(self,
              signals: List[Dict[str, Any]]
              ) -> Dict[str, Any]:
        """
        Entrada:
            signals = [
                {"symbol": "BTC/USDT", "side": "buy", "prob": 0.9,
                 "source": "model_ppo"},
                {"symbol": "BTC/USDT", "side": "buy", "prob": 0.6,
                 "source": "rsi"},
                ...
            ]
        Salida:
            dict con la señal final y su score.
        """
        if not signals:
            logger.warning("[BlenderEnsemble] Lista vacía")
            return {}

        grouped = {}
        for sig in signals:
            key = (sig["symbol"], sig["side"])
            weight = self.weights.get(sig.get("source", ""), self.default)
            grouped.setdefault(key, 0.0)
            grouped[key] += sig.get("prob", 1.0) * weight

        if not grouped:
            return {}

        (symbol, side), score = max(grouped.items(), key=lambda x: x[1])
        logger.debug(
            f"[BlenderEnsemble] blended winner={symbol} {side} (score={score:.2f})"
        )
        return {
            "symbol": symbol,
            "side": side,
            "score": score,
            "origin": "blender",
        }