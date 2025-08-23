# l2_tactic/signal_composer.py

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from .models import TacticalSignal, SignalSource, SignalDirection

logger = logging.getLogger(__name__)

class SignalComposer:
    """
    Compositor de señales tácticas.
    Combina señales de IA, técnicas y patrones usando un sistema de pesos dinámicos.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_weights = {
            "ai": 0.5,
            "technical": 0.3,
            "pattern": 0.2
        }

    def compose(self, signals: List[TacticalSignal], regime_context: Optional[Dict] = None) -> List[TacticalSignal]:
        """
        Punto de entrada: combina señales aplicando pesos y resolviendo conflictos.
        Siempre devuelve una lista, aunque no haya señales válidas.
        """
        if not signals:
            return []

        try:
            logger.info("Composing signals")
            weights = self._calculate_dynamic_weights(regime_context)

            composite = self._create_composite_signal(signals, weights)

            if composite is None:
                logger.warning("No composite signal could be created")
                return []

            final = self._resolve_conflicts([composite])
            logger.info(f"Generated {len(final)} final signals from {len(signals)} candidates")
            return final
        except Exception as e:
            logger.error(f"Signal composition failed: {e}")
            return []

    def _create_composite_signal(self, signals: List[TacticalSignal], weights: Dict[str, float]) -> Optional[TacticalSignal]:
        """
        Crea una señal compuesta ponderando las señales según su origen.
        """
        if not signals:
            return None

        symbol = signals[0].symbol
        timestamp = pd.Timestamp.now(tz="UTC")

        # Agrupamos por source (no signal_type)
        ai_signals = [s for s in signals if s.source == SignalSource.AI_MODEL]
        tech_signals = [s for s in signals if s.source == SignalSource.TECHNICAL]
        pattern_signals = [s for s in signals if s.source == SignalSource.PATTERN]

        def avg_strength(sig_list: List[TacticalSignal]) -> float:
            return np.mean([s.strength for s in sig_list]) if sig_list else 0.0

        composite_strength = (
            weights.get("ai", 0.5) * avg_strength(ai_signals)
            + weights.get("technical", 0.3) * avg_strength(tech_signals)
            + weights.get("pattern", 0.2) * avg_strength(pattern_signals)
        )

        if composite_strength == 0:
            return None

        # Dirección: LONG si positivo, SHORT si negativo
        direction = SignalDirection.LONG if composite_strength > 0 else SignalDirection.SHORT
        confidence = float(np.clip(abs(composite_strength), 0, 1))

        last_price = None
        for s in reversed(signals):
            if s.price is not None:
                last_price = s.price
                break

        return TacticalSignal(
            symbol=symbol,
            direction=direction,
            strength=abs(composite_strength),
            confidence=confidence,
            price=last_price if last_price is not None else 0.0,  # ✅ evita None
            timestamp=timestamp,
            source=SignalSource.AGGREGATED,
            metadata={"weights": weights, "method": "weighted_average"},
            expires_at=timestamp + pd.Timedelta(minutes=5)  # ✅ mejor con expiración
        )

    def _calculate_dynamic_weights(self, regime_context: Optional[Dict]) -> Dict[str, float]:
        """
        Ajusta pesos según el contexto de régimen.
        """
        if not regime_context:
            return self.default_weights

        strategy = regime_context.get("strategy", "default")
        weights = self.default_weights.copy()

        if strategy == "estrategia_agresiva":
            weights["ai"] = 0.6
            weights["technical"] = 0.3
            weights["pattern"] = 0.1
        elif strategy == "estrategia_defensiva":
            weights["ai"] = 0.3
            weights["technical"] = 0.4
            weights["pattern"] = 0.3

        return weights

    def _resolve_conflicts(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        """
        Resuelve conflictos eliminando señales redundantes.
        """
        if not signals:
            return []

        seen = set()
        resolved = []
        for s in signals:
            key = (s.symbol, s.source, s.direction)
            if key not in seen:
                resolved.append(s)
                seen.add(key)

        return resolved
