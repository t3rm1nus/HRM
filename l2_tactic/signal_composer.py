# l2_tactic/signal_composer.py
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple, Iterable

from .config import L2Config
from .models import TacticalSignal

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
        self.conflict_tie_threshold = getattr(config, "conflict_tie_threshold", 0.03)  # si scores muy parecidos, gana el mayor
        self.keep_both_when_far = getattr(config, "conflict_keep_both_when_far", False)

        # Cómo promediar atributos continuos
        self.aggregate_price = getattr(config, "composer_price_agg", "first")  # 'first' | 'mean'
        self.aggregate_strength = getattr(config, "composer_strength_agg", "weighted")  # 'mean' | 'weighted'
        self.aggregate_conf = getattr(config, "composer_confidence_agg", "weighted")  # 'mean' | 'weighted'

        logger.info("SignalComposer initialized with config")

    # ---------- helpers ----------

    def _source_weight(self, source: str) -> float:
        source = (source or "").lower()
        base = {
            "ai": self.w_ai,
            "technical": self.w_tech,
            "pattern": self.w_pattern,
            "composite": 1.0,  # por si reenviamos resultados compuestos (no debería)
        }.get(source, 0.10)

        # Ajuste por performance histórica (si metrics está disponible)
        # Convención esperada de metrics (si existe):
        #   metrics.get_source_perf_weight(source: str) -> float   (1.0 = neutral)
        adj = 1.0
        if self.metrics and hasattr(self.metrics, "get_source_perf_weight"):
            try:
                adj = float(self.metrics.get_source_perf_weight(source))
            except Exception:  # defensivo
                adj = 1.0

        return max(0.0, base * adj)

    @staticmethod
    def _group_by_symbol_side(signals: Iterable[TacticalSignal]) -> Dict[Tuple[str, str], List[TacticalSignal]]:
        g: Dict[Tuple[str, str], List[TacticalSignal]] = {}
        for s in signals:
            key = (s.symbol, s.side)
            g.setdefault(key, []).append(s)
        return g

    def _aggregate_group(self, symbol: str, side: str, group: List[TacticalSignal]) -> Optional[TacticalSignal]:
        """
        Genera una señal compuesta a partir de un grupo de señales del mismo símbolo+lado.
        Usamos media/ponderación configurable sobre confidence y strength.
        """
        if not group:
            return None

        # Precio compuesto
        if self.aggregate_price == "mean":
            prices = [s.price for s in group if s.price]
            price = sum(prices) / len(prices) if prices else (group[0].price or 0.0)
        else:
            price = group[0].price

        # Pesos por fuente
        weights = [self._source_weight(s.source) for s in group]
        total_w = sum(weights) if any(w > 0 for w in weights) else len(group)
        norm_w = [w / total_w for w in (weights if total_w > 0 else [1.0] * len(group))]

        # Strength
        if self.aggregate_strength == "weighted" and total_w > 0:
            strength = sum(s.strength * w for s, w in zip(group, norm_w))
        else:
            strength = sum(s.strength for s in group) / len(group)

        # Confidence
        if self.aggregate_conf == "weighted" and total_w > 0:
            confidence = sum(s.confidence * w for s, w in zip(group, norm_w))
        else:
            confidence = sum(s.confidence for s in group) / len(group)

        # score compuesto para facilitar conflictos
        composite_score = strength * confidence

        # Usar el más reciente como base para mantener metadatos y timestamps
        base = max(group, key=lambda s: s.timestamp or 0)

        composite = replace(
            base,
            side=side,
            price=price,
            strength=strength,
            confidence=confidence,
            source="composite",
            composite_score=composite_score,
            notes={**(base.notes or {}), "sources": [s.source for s in group], "n_sources": len(group)},
        )

        # filtros de calidad
        if composite.confidence < self.min_conf or composite.strength < self.min_strength:
            return None

        return composite

    # ---------- API principal ----------

    async def compose_signals(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        """
        Entrada: señales de varias fuentes (IA, técnicas, patrones...).
        Salida: señales compuestas por símbolo (resolviendo conflictos buy/sell).
        """
        logger.info("Composing signals")
        if not signals:
            logger.warning("No signals provided for composition")
            return []

        # 1) Agrupar por (symbol, side) y componer
        by_key = self._group_by_symbol_side(signals)
        interim: Dict[str, Dict[str, TacticalSignal]] = {}  # symbol -> {"BUY": s?, "SELL": s?}

        for (symbol, side), group in by_key.items():
            comp = self._aggregate_group(symbol, side, group)
            if comp is None:
                continue
            interim.setdefault(symbol, {})[side] = comp

        if not interim:
            return []

        # 2) Resolver conflictos BUY vs SELL por símbolo
        final: List[TacticalSignal] = []
        for symbol, sides in interim.items():
            buy = sides.get("BUY")
            sell = sides.get("SELL")

            if buy and not sell:
                final.append(buy)
                continue
            if sell and not buy:
                final.append(sell)
                continue
            if not buy and not sell:
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

        logger.info(f"[L2] Composed {len(final)} signals from {len(signals)} candidates")
        return final
