# l2_tactic/signal_generator.py
# Orquestador L2: integra IA/tech/pattern -> composición -> sizing -> riesgo -> orden para L1

from __future__ import annotations
import logging
from typing import Dict, List, Any, Optional

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

    def __init__(
        self,
        config: Optional[L2Config] = None,
        ai_model: Optional[AIModelIntegration] = None,
    ):
        self.config = config or L2Config()
        self.composer = SignalComposer(self.config)
        self.sizer = PositionSizerManager(self.config)
        self.risk = RiskControlManager(self.config)
        self.ai = ai_model or AIModelIntegration(config)
        self.tech_indicators = TechnicalIndicators()
        self.risk_manager = RiskControlManager(config)

    async def _generate_signals(
        self,
        portfolio: Dict,
        market_data: Dict,
        features_by_symbol: Dict[str, MarketFeatures],
    ) -> List[TacticalSignal]:
        raw_signals = []
        raw_signals.extend(await self.ai_signals(market_data))
        raw_signals.extend(await self.technical_signals(market_data))
        raw_signals.extend(await self.risk_overlay(market_data, portfolio))
        return raw_signals

    async def ai_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales provenientes del modelo PPO (ahora con valores de prueba).
        """
        signals = []
        if "BTCUSDT" in market_data:
            logger.debug("Generando señal de IA para BTCUSDT...")
            current_price = market_data["BTCUSDT"].get("close")
            if current_price:
                signals.append(
                    TacticalSignal(
                        symbol="BTCUSDT",
                        side="buy",
                        confidence=0.8,
                        strength=0.7,
                        source="ai",
                        price=current_price,
                        stop_loss=current_price * 0.98,
                    )
                )
        return signals

    async def technical_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales técnicas (ahora con valores de prueba).
        """
        signals = []
        if "ETHUSDT" in market_data:
            logger.debug("Generando señal técnica para ETHUSDT...")
            current_price = market_data["ETHUSDT"].get("close")
            if current_price:
                signals.append(
                    TacticalSignal(
                        symbol="ETHUSDT",
                        side="sell",
                        confidence=0.9,
                        strength=0.8,
                        source="technical",
                        price=current_price,
                        stop_loss=current_price * 1.02,
                    )
                )
        return signals

    async def risk_overlay(self, market_data: Dict[str, Any], portfolio: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales de riesgo.
        """
        exposicion_btc = portfolio.get("exposicion", {}).get("BTCUSDT", 0)
        
        signals = []
        if exposicion_btc > 0.5:
            current_price = market_data.get("BTCUSDT", {}).get("close")
            if current_price:
                signals.append(
                    TacticalSignal(
                        symbol="BTCUSDT",
                        side="sell",
                        confidence=1.0,
                        strength=1.0,
                        source="risk",
                        price=current_price,
                    )
                )
        return signals