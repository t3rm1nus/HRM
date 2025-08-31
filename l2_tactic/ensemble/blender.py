# l2_tactic/ensemble/blender.py
"""
BlenderEnsemble
Combina señales mediante pesos configurables
(similar a un stacking lineal).
"""

from typing import List, Dict, Any
import numpy as np
from core.logging import logger
from ..models import TacticalSignal

class BlenderEnsemble:
    """
    Combina señales usando pesos fijos o dinámicos.
    """

    def __init__(self,
                 weights = {
                    'ai': 0.5,
                    'technical': 0.3, 
                    'risk': 0.2
                },
                 default: float = 0.0):
        """
        weights: dict {source_name: peso}
                 Ej.: {"model_ppo": 0.5, "rsi": 0.3, "macd": 0.2}
        default: peso para cualquier fuente no listada.
        """
        self.weights = weights
        self.default = default
        self.strategies = {
            'technical': 0.4,      # Peso para señales técnicas
            'finrl_ppo': 0.4,      # Peso para modelo FinRL PPO
            'mean_reversion': 0.2   # Peso para mean reversion
        }
        logger.info(f"[BlenderEnsemble] inicializado: {self.strategies}")
        logger.info(
            f"[BlenderEnsemble] inicializado: {weights} (default={default})"
        )

    # ------------------------------------------------------------------ #
    def blend(self,
              signals: List[TacticalSignal]
              ) -> TacticalSignal:
        """
        Entrada:
            signals = [
                TacticalSignal(symbol="BTC/USDT", side="buy", ...),
                TacticalSignal(symbol="BTC/USDT", side="buy", ...),
                ...
            ]
        Salida:
            Objeto TacticalSignal con la señal final y su score compuesto.
        """
        if not signals:
            logger.warning("[BlenderEnsemble] Lista vacía")
            return None

        grouped = {}
        winning_signal = None  # Almacenar la señal ganadora para tomar su precio

        for sig in signals:
            key = (sig.symbol, sig.side)
            weight = self.weights.get(sig.source, self.default)
            grouped.setdefault(key, {"score": 0.0, "signal": sig})
            
            composite_score = sig.strength * sig.confidence
            grouped[key]["score"] += composite_score * weight
            
            # Si el score actual es el mejor, guardar esta señal como candidata
            if grouped[key]["score"] > grouped.get("best_score", -1):
                grouped["best_score"] = grouped[key]["score"]
                winning_signal = sig

        if not winning_signal:
            return None
        
        # Usar el símbolo, lado y precio de la señal ganadora
        final_signal = TacticalSignal(
            symbol=winning_signal.symbol,
            side=winning_signal.side,
            source="ensemble_blender",
            confidence=grouped["best_score"],
            strength=grouped["best_score"],
            price=winning_signal.price, # <--- AQUI: Usar el precio de la señal ganadora
        )

        logger.debug(
            f"[BlenderEnsemble] blended winner={final_signal}"
        )
        return final_signal