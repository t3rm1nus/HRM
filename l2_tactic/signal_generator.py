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
        logger.info(f"[L2] Datos recibidos: {len(market_data)} símbolos")
        for symbol, data in market_data.items():
            logger.info(f"[L2] Último precio {symbol}: {data.get('close', 'N/A')}")
        raw_signals = []
        # 🔧 Forzar señales válidas sin riesgo
        btc_signal = TacticalSignal(
            symbol="BTCUSDT",
            side="buy",
            confidence=1.0,
            strength=1.0,
            source="ai",
            price=market_data["BTCUSDT"].get("close", 0),
            stop_loss=market_data["BTCUSDT"].get("close", 0) * 0.98,
        )
        raw_signals.append(btc_signal)
        return raw_signals

    def _prepare_features(self, market_data: Dict) -> np.ndarray:
        """Prepara features para FinRL"""
        # IMPLEMENTAR: conversión de market_data a features que espera FinRL
        return np.array([
            market_data.get('close', 0),
            market_data.get('volume', 0),
            # Añadir más features según entrenamiento FinRL
        ])

    async def ai_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales provenientes del modelo PPO (ahora con valores de prueba).
        """
        """USA EL MODELO FINRL REAL"""
        signals = []
        
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            if symbol in market_data:
                # USAR EL MODELO REAL, no hardcoded
                features = self._prepare_features(market_data[symbol])
                prediction = await self.ai.predict(features)  # Llamada real a FinRL
                
                confidence = prediction.confidence
                logger.info(f"[L2] AI score REAL para {symbol}: {confidence:.2f}")
                
                if confidence > self.config.ai_model.prediction_threshold:
                    signals.append(TacticalSignal(
                        symbol=symbol,
                        side=prediction.action,  # Del modelo real
                        confidence=confidence,
                        strength=prediction.strength,
                        source="ai",
                        price=market_data[symbol]["close"],
                        stop_loss=market_data[symbol]["close"] * (0.98 if prediction.action == "buy" else 1.02),
                    ))
        
        return signals

    async def technical_signals(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales técnicas (ahora con valores de prueba).
        """
        signals = []
        confidence = 0.9  # ✅ Valor que usas en la señal
        logger.info(f"[L2] Technical score para ETH: {confidence:.2f}, threshold: {self.config.signals.min_signal_strength}")
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


    def debug_signal_flow():
        from l1_operational.data_feed import DataFeed
        df = DataFeed().fetch_data('BTCUSDT')
        print(f"[DEBUG] Filas: {len(df)}")
        print(f"[DEBUG] Último close: {df['close'].iloc[-1]}")


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

async def debug_signal_flow():
    """Función temporal para debuggear flujo L2"""
    from l1_operational.data_feed import DataFeed
    from l2_tactic.config import L2Config
    from l2_tactic.signal_generator import L2TacticProcessor

    config = L2Config.from_env()
    processor = L2TacticProcessor(config)
    data_feed = DataFeed()

    df = data_feed.fetch_data('BTCUSDT')
    market_data = {"BTCUSDT": df.iloc[-1].to_dict() if not df.empty else {}}

    logger.info(f"[DEBUG] Filas BTC: {len(df)}")
    logger.info(f"[DEBUG] Último close BTC: {market_data['BTCUSDT'].get('close', 'N/A')}")

    signals = await processor._generate_signals(
        portfolio={"exposicion": {"BTCUSDT": 0.1}},
        market_data=market_data,
        features_by_symbol={}
    )

    logger.info(f"[DEBUG] Señales generadas: {len(signals)}")
    for s in signals:
        logger.info(f"[DEBUG] Señal: {s.symbol} | {s.side} | conf={s.confidence}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_signal_flow())