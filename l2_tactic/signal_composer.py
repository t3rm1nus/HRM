import logging
from typing import Dict, List, Any
import pandas as pd
from .config import L2Config

logger = logging.getLogger("l2_tactic.signal_composer")

class SignalComposer:
    """
    Combina señales tácticas de múltiples fuentes (IA, técnicas, patrones).
    """
    def __init__(self, config: L2Config):
        self.config = config
        self.min_signal_strength = getattr(config, "min_signal_strength", 0.5)

        # Pesos para señales según la fuente
        self.ai_model_weight = getattr(config, "ai_model_weight", 0.5)
        self.technical_weight = getattr(config, "technical_weight", 0.3)
        self.pattern_weight = getattr(config, "pattern_weight", 0.2)

        logger.info("SignalComposer initialized with config")

    def compose(self, signals: List[Dict], market_data: pd.DataFrame) -> List[Dict]:
        """
        Combina señales de diferentes fuentes en una lista final de señales.
        Args:
            signals: Lista de diccionarios de señales con 'symbol', 'direction', 'confidence'.
            market_data: DataFrame con datos de mercado (OHLCV).
        Returns:
            Lista de señales combinadas.
        """
        logger.info("Composing signals")
        try:
            if not signals:
                logger.warning("No signals provided for composition")
                return []

            # Validar market_data
            if market_data.empty:
                logger.warning("Empty market data provided")
                return signals  # Devolver señales sin combinar si no hay datos

            # Agrupar señales por símbolo y dirección
            grouped_signals = {}
            for signal in signals:
                symbol = signal["symbol"]
                direction = signal["direction"]
                key = (symbol, direction)
                if key not in grouped_signals:
                    grouped_signals[key] = []
                grouped_signals[key].append(signal)

            final_signals = []
            for (symbol, direction), signal_group in grouped_signals.items():
                # Calcular confianza ponderada
                weighted_confidence = 0.0
                total_weight = 0.0
                source_counts = {"ai": 0, "technical": 0, "pattern": 0}

                for signal in signal_group:
                    confidence = signal["confidence"]
                    source = signal.get("source", "unknown")
                    if source == "ai":
                        weight = self.ai_model_weight
                        source_counts["ai"] += 1
                    elif source == "technical":
                        weight = self.technical_weight
                        source_counts["technical"] += 1
                    elif source == "pattern":
                        weight = self.pattern_weight
                        source_counts["pattern"] += 1
                    else:
                        weight = 0.1  # Peso por defecto para fuentes desconocidas
                    weighted_confidence += confidence * weight
                    total_weight += weight

                if total_weight > 0:
                    final_confidence = weighted_confidence / total_weight
                else:
                    final_confidence = max(s["confidence"] for s in signal_group)

                # Filtrar por fuerza mínima
                if final_confidence >= self.min_signal_strength:
                    final_signals.append({
                        "symbol": symbol,
                        "direction": direction,
                        "confidence": final_confidence,
                        "source": "composite",
                        "sources_used": source_counts
                    })

            logger.info(f"Composed {len(final_signals)} signals from {len(signals)} candidates")
            return final_signals

        except Exception as e:
            logger.error(f"Signal composition failed: {e}", exc_info=True)
            return signals  # Devolver señales originales como respaldo