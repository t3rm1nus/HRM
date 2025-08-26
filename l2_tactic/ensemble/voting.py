"""
VotingEnsemble
Realiza una votación simple (majority / hard-vote o soft-vote)
sobre las señales generadas por cada modelo.
"""

from typing import List, Dict, Any
import pandas as pd
from core.logging import logger


class VotingEnsemble:
    """
    Combina señales mediante votación.
      - hard: 1 si ≥ umbral de modelos la emiten.
      - soft: promedia probabilidades (si las hay).
    """

    def __init__(self,
                 method: str = "hard",
                 threshold: float = 0.5):
        """
        method: "hard" | "soft"
        threshold: fracción de modelos que deben estar de acuerdo
                   (solo aplica a hard-vote).
        """
        if method not in {"hard", "soft"}:
            raise ValueError("method debe ser 'hard' o 'soft'")
        self.method = method
        self.threshold = threshold
        logger.info(
            f"[VotingEnsemble] inicializado: method={method}, "
            f"threshold={threshold}"
        )

    # ------------------------------------------------------------------ #
    def vote(self,
             signals: List[Dict[str, Any]]
             ) -> Dict[str, Any]:
        """
        Entrada:
            signals = [
                {"symbol": "BTC/USDT", "side": "buy",  "prob": 0.7},
                {"symbol": "BTC/USDT", "side": "sell", "prob": 0.3},
                ...
            ]
        Salida:
            dict con la señal consensuada o None si no hay consenso.
        """
        if not signals:
            logger.warning("[VotingEnsemble] Lista vacía de señales")
            return {}

        df = pd.DataFrame(signals)

        # hard-vote
        if self.method == "hard":
            counts = (
                df.groupby(["symbol", "side"])
                  .size()
                  .reset_index(name="votes")
            )
            total_models = df["symbol"].value_counts().max()
            counts = counts[counts["votes"] >= total_models * self.threshold]

            if counts.empty:
                logger.debug("[VotingEnsemble] Sin consenso")
                return {}

            # gana la fila con más votos
            winner = counts.sort_values("votes", ascending=False).iloc[0]
            logger.debug(
                f"[VotingEnsemble] winner={winner['symbol']} "
                f"{winner['side']} ({winner['votes']}/{total_models})"
            )
            return {
                "symbol": winner["symbol"],
                "side": winner["side"],
                "origin": "voting",
            }

        # soft-vote
        else:
            summary = (
                df.groupby(["symbol", "side"])["prob"]
                  .mean()
                  .reset_index(name="avg_prob")
            )
            summary = summary.sort_values("avg_prob", ascending=False)
            winner = summary.iloc[0]

            logger.debug(
                f"[VotingEnsemble] soft-vote winner={winner['symbol']} "
                f"{winner['side']} (avg_prob={winner['avg_prob']:.2f})"
            )
            return {
                "symbol": winner["symbol"],
                "side": winner["side"],
                "prob": winner["avg_prob"],
                "origin": "voting",
            }