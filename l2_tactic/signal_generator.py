# l2_tactic/signal_generator.py
# Orquestador L2: integra IA/tech/pattern -> composición -> sizing -> riesgo -> orden para L1

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import pandas as pd

from .config import L2Config
from .models import TacticalSignal, MarketFeatures
from .signal_composer import SignalComposer
from .position_sizer import PositionSizerManager
from .risk_controls import RiskControlManager
from .performance_optimizer import PerformanceOptimizer
from .ai_model_integration import AIModelIntegration

from l2_tactic.indicators.technical import TechnicalIndicators, IndicatorWindows
from l2_tactic.models import MarketFeatures

logger = logging.getLogger(__name__)


class L2TacticProcessor:
    """
    Flujo:
      1) Genera señales (IA + técnicas).
      2) Filtra por calidad.
      3) Ajusta pesos según régimen (L3).
      4) Devuelve lista de TacticalSignal.
    """

    def __init__(self, config: Optional[L2Config] = None, ai_model: Optional[AIModelIntegration] = None):
        self.config = config or L2Config()
        self.composer = SignalComposer(self.config)
        self.sizer = PositionSizerManager(self.config)
        self.risk = RiskControlManager(self.config)
        self.ai = ai_model or AIModelIntegration(config)
        self.optimizer = PerformanceOptimizer(config)
        self.market_data = None

    async def _generate_ai_signals(
    self,
    portfolio: Dict,
    market_data: Dict[str, "pd.DataFrame"],  # añadimos market_data aquí
) -> List[TacticalSignal]:
        """
        Genera señales AI enriquecidas con indicadores técnicos.
        - Calcula MarketFeatures por símbolo usando TechnicalIndicators
        - Llama al modelo AI para predicción
        - Devuelve lista de TacticalSignal
        """
        signals = []

        try:
            # --- calcular indicadores técnicos para todos los símbolos ---
            ti = TechnicalIndicators.compute_for_universe(
                market_data, IndicatorWindows(), as_features=True
            )

            features_by_symbol: Dict[str, MarketFeatures] = {}
            for sym, f in ti.items():
                features_by_symbol[sym] = MarketFeatures(
                    volatility=f.get("volatility", 0.0),
                    volume_ratio=f.get("volume_ratio", 0.0),
                    price_momentum=f.get("price_momentum", 0.0),
                    # aquí podrías mapear extras: RSI, MACD, etc.
                )

            # --- generar predicciones AI ---
            for symbol, features in features_by_symbol.items():
                try:
                    pred = await self.ai.predict(symbol, features)
                    if pred is None:
                        continue

                    sig = TacticalSignal(
                        symbol=symbol,
                        side="buy" if pred["direction"] == "LONG" else "sell",
                        strength=pred.get("strength", 0.5),
                        confidence=pred.get("confidence", 0.5),
                        price=market_data.get(symbol, {}).get("close", 0.0),
                        source="ai",
                        model_name=pred.get("model", "l2_ai_model"),
                        features_used=features.__dict__,
                        reasoning=pred.get("reasoning"),
                    )
                    signals.append(sig)

                except Exception as e:
                    logger.error(f"AI prediction failed for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Feature computation failed: {e}")

        return signals

    def _generate_technical_signals(
        self,
        features_by_symbol: Dict[str, MarketFeatures],
    ) -> List[TacticalSignal]:
        signals = []
        for symbol, features in features_by_symbol.items():
            price = self.market_data.get(symbol, {}).get("price", 0.0)
            # Ejemplo: MA crossover con momentum
            if features.price_momentum and features.price_momentum > self.config.MOMENTUM_THRESHOLD:
                signals.append(TacticalSignal(
                    symbol=symbol,
                    side="buy",
                    strength=features.price_momentum,
                    confidence=0.6,
                    price=price,
                    source="technical",
                    reasoning="Momentum breakout",
                ))
            elif features.price_momentum and features.price_momentum < -self.config.MOMENTUM_THRESHOLD:
                signals.append(TacticalSignal(
                    symbol=symbol,
                    side="sell",
                    strength=abs(features.price_momentum),
                    confidence=0.6,
                    price=price,
                    source="technical",
                    reasoning="Momentum breakdown",
                ))
            # RSI filter
            if features.rsi:
                if features.rsi < 30:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        side="buy",
                        strength=1 - features.rsi / 100,
                        confidence=0.5,
                        price=price,
                        source="technical",
                        reasoning="RSI oversold",
                    ))
                elif features.rsi > 70:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        side="sell",
                        strength=features.rsi / 100,
                        confidence=0.5,
                        price=price,
                        source="technical",
                        reasoning="RSI overbought",
                    ))
            # ATR breakout
            if features.atr and features.volatility:
                if features.volatility > 1.5 * features.atr:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        side="buy",
                        strength=features.volatility,
                        confidence=0.55,
                        price=price,
                        source="technical",
                        reasoning="ATR breakout",
                    ))
        return signals

    def _filter_signals(self, signals: List[TacticalSignal]) -> List[TacticalSignal]:
        filtered = []
        for s in signals:
            if s.confidence < getattr(self.config, "min_confidence", 0.4):
                continue
            if hasattr(s, "features_used"):
                vol = s.features_used.get("volatility")
                if vol is not None and vol < getattr(self.config, "min_volatility", 0.01):
                    continue
            filtered.append(s)
        return filtered

    def _apply_regime_weights(
        self, signals: List[TacticalSignal], regime: Optional[str]
    ) -> List[TacticalSignal]:
        if not regime:
            return signals
        adjusted = []
        for s in signals:
            if regime == "bull" and s.is_long():
                s.strength *= 1.2
            elif regime == "bear" and not s.is_long():
                s.strength *= 1.2
            elif regime == "neutral":
                s.strength *= 0.8
            adjusted.append(s)
        return adjusted

    async def _generate_signals(
        self,
        portfolio: Dict,
        market_data: Dict,
        features_by_symbol: Dict[str, MarketFeatures],
    ) -> List[TacticalSignal]:
        self.market_data = market_data

        ai_signals = await self._generate_ai_signals(portfolio, features_by_symbol)
        tech_signals = self._generate_technical_signals(features_by_symbol)

        combined = ai_signals + tech_signals
        filtered = self._filter_signals(combined)

        regime = portfolio.get("regime")
        adjusted = self._apply_regime_weights(filtered, regime)

        logger.info(f"Generated {len(adjusted)} signals (AI={len(ai_signals)}, Tech={len(tech_signals)})")
        return adjusted

    async def process(
        self,
        portfolio: Dict,
        market_data: Dict,
        features_by_symbol: Dict[str, MarketFeatures],
    ) -> List[TacticalSignal]:
        try:
            raw_signals = await self._generate_signals(
                portfolio=portfolio,
                market_data=market_data,
                features_by_symbol=features_by_symbol,
            )
            logger.debug(f"[L2] Señales generadas: {len(raw_signals)}")
            return raw_signals or []
        except Exception as e:
            logger.error(f"[L2] Error generando señales: {e}", exc_info=True)
            return []
    
    # ------------- STUBS ASYNC ----------------
    async def ai_signals(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera señales provenientes del modelo PPO (stub).
        Devuelve lista vacía hasta que tengas la lógica real.
        """
        return []

    async def technical_signals(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera señales técnicas (stub).
        """
        return []

    async def risk_overlay(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Capa de riesgo: puede devolver señales de reducción de exposición (stub).
        """
        return []

SignalGenerator = L2TacticProcessor