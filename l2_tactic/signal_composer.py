# l2_tactic/signal_composer.py
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple, Iterable

from .config import L2Config
from .models import TacticalSignal
from datetime import datetime

logger = logging.getLogger("l2_tactic.signal_composer")

class SignalComposer:
    """
    Combina señales tácticas de múltiples fuentes (IA, técnicas, patrones).
    - Weighted voting / score average por símbolo+lado
    - Resolución de conflictos BUY vs SELL en el mismo símbolo
    - Ajuste dinámico de pesos por performance histórica (inyectada vía metrics)
    """

    def __init__(self, config: L2Config, metrics: Optional[object] = None):
        self.cfg = config
        self.metrics = metrics

        # Pesos base por fuente
        self.w_ai = getattr(config, "ai_model_weight", 0.50)
        self.w_tech = getattr(config, "technical_weight", 0.30)
        self.w_pattern = getattr(config, "pattern_weight", 0.20)

        # Filtros de calidad / mínimos para aceptar la señal compuesta
        self.min_conf = getattr(config, "min_signal_confidence", 0.50)
        self.min_strength = getattr(config, "min_signal_strength", 0.10)

        # Cómo resolver conflictos buy/sell
        self.conflict_tie_threshold = getattr(config, "conflict_tie_threshold", 0.05)
        self.keep_both_when_far = getattr(config, "keep_both_when_far", False)
    
    # --- MÉTODO CORREGIDO ---
    def compose(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        if not signals:
            logger.warning("⚠️ No hay señales para componer")
            return []
        
        signals_by_symbol = {}
        for signal in signals:
            if not hasattr(signal, 'symbol') or not hasattr(signal, 'side') or not hasattr(signal, 'source'):
                logger.error(f"❌ Señal inválida: {signal.__dict__}")
                continue
            symbol = signal.symbol
            signals_by_symbol.setdefault(symbol, []).append(signal)
        
        composed_signals = []
        for symbol, sym_signals in signals_by_symbol.items():
            total_weight = 0.0
            weighted_strength = 0.0
            weighted_confidence = 0.0
            features = {}
            
            for signal in sym_signals:
                weight = self._get_dynamic_weight(signal)
                total_weight += weight
                weighted_strength += signal.strength * weight
                weighted_confidence += signal.confidence * weight
                features.update(signal.features)
            
            if total_weight > 0:
                avg_strength = weighted_strength / total_weight
                avg_confidence = weighted_confidence / total_weight
                dominant_signal = max(sym_signals, key=lambda s: self._get_dynamic_weight(s))
                
                composed_signal = TacticalSignal(
                    symbol=symbol,
                    strength=avg_strength,
                    confidence=avg_confidence,
                    side=dominant_signal.side,
                    features=features,
                    timestamp=datetime.now().timestamp(),
                    source='composed',
                    metadata={'composed_from': [s.source for s in sym_signals]}
                )
                composed_signals.append(composed_signal)
                logger.debug(f"✅ Señal compuesta para {symbol}: side={dominant_signal.side}, strength={avg_strength:.3f}, confidence={avg_confidence:.3f}")
        
        logger.info(f"✅ Señales compuestas generadas: {len(composed_signals)}")
        return composed_signals

    # --- Métodos auxiliares ---
    def _group_and_weight_signals(self, signals: List[TacticalSignal]) -> Dict[Tuple, List[Tuple[TacticalSignal, float]]]:
        """Agrupa las señales por (símbolo, lado) y calcula el peso dinámico."""
        grouped = {}
        for signal in signals:
            key = (signal.symbol, signal.side)
            weight = self._get_dynamic_weight(signal)
            grouped.setdefault(key, []).append((signal, weight))
        return grouped

    def _create_composed_signal(self, key: Tuple, weighted_signals: List[Tuple]) -> TacticalSignal:
        """Crea una señal compuesta a partir de las señales ponderadas."""
        symbol, side = key
        
        # Calcular promedios ponderados
        total_weight = sum(w for _, w in weighted_signals)
        if total_weight == 0:
            return None

        # Ejemplo de promedio ponderado para confianza y fuerza
        avg_confidence = sum(s.confidence * w for s, w in weighted_signals) / total_weight
        avg_strength = sum(s.strength * w for s, w in weighted_signals) / total_weight
        
        # Tomar la primera señal como base para los otros atributos
        base_signal = weighted_signals[0][0]
        
        return replace(
            base_signal,
            confidence=avg_confidence,
            strength=avg_strength,
            sources=[s.source for s, _ in weighted_signals],
            composite_score=avg_confidence * avg_strength
        )

    def _get_dynamic_weight(self, signal: TacticalSignal) -> float:
        logger.debug(f"Calculando peso para señal: source={signal.source}, confidence={signal.confidence}")
        base_weight = 1.0
        if signal.source == 'ai':
            base_weight *= 1.5
        elif signal.source == 'technical':
            base_weight *= 1.0
        elif signal.source == 'risk':
            base_weight *= 2.0
        weight = base_weight * signal.confidence
        logger.debug(f"Peso calculado: {weight}")
        return max(weight, 0.01)

    def _resolve_conflicts_and_filter(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        """
        Resuelve conflictos BUY vs SELL y filtra por umbrales de calidad.
        """
        signals_by_symbol = {}
        for s in signals:
            if s and s.confidence >= self.min_conf and s.strength >= self.min_strength:
                signals_by_symbol.setdefault(s.symbol, {}).update({s.side: s})
        
        final = []
        for symbol, signals_in_conflict in signals_by_symbol.items():
            buy = signals_in_conflict.get("buy")
            sell = signals_in_conflict.get("sell")

            if not buy and not sell:
                continue
            if buy and not sell:
                final.append(buy)
                continue
            if sell and not buy:
                final.append(sell)
                continue

            # Ambos existen → comparar score compuesto
            b_score = getattr(buy, "composite_score", buy.strength * buy.confidence)
            s_score = getattr(sell, "composite_score", sell.strength * sell.confidence)

            if self.keep_both_when_far and abs(b_score - s_score) > (self.conflict_tie_threshold * 5):
                # Señales muy divergentes → dejar ambas (para coberturas o pares).
                final.extend([buy, sell])
                continue

            # Elegir la más fuerte (si empate ~ tie_threshold, ganará la mayor)
            if (b_score - s_score) >= self.conflict_tie_threshold:
                final.append(buy)
            elif (s_score - b_score) >= self.conflict_tie_threshold:
                final.append(sell)
            else:
                # Empate técnico → nos quedamos con la de mayor confianza; si empatan, descartamos ambas (prudencia)
                if buy.confidence > sell.confidence:
                    final.append(buy)
                elif sell.confidence > buy.confidence:
                    final.append(sell)
                else:
                    logger.debug(f"Conflict tie for {symbol}; discarding both")

        logger.info(f"Final signals after composition and conflict resolution: {len(final)}")
        return final