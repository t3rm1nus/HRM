# l2_tactic/signal_generator.py
# Orquestador L2: integra IA/tech/pattern -> composición -> sizing -> riesgo -> orden para L1

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import L2Config
from .models import TacticalSignal, MarketFeatures, PositionSize
from .signal_composer import SignalComposer  # asume que ya existe en tu L2
from .position_sizer import PositionSizerManager
from .risk_controls import RiskControlManager

logger = logging.getLogger(__name__)


class L2TacticProcessor:
    """
    Flujo:
      1) Recibe señales crudas (AI + técnicas + patrones).
      2) Compone/filtra con SignalComposer.
      3) Calcula tamaño de posición con PositionSizerManager.
      4) Asegura STOP-LOSS (y opcional TP) con RiskControlManager (pre-trade).
      5) Emite orden enriquecida (con SL/TP/size) para L1.
    """

    def __init__(self, config: Optional[L2Config] = None):
        self.config = config or L2Config()
        self.composer = SignalComposer(self.config)
        self.sizer = PositionSizerManager(self.config)
        self.risk = RiskControlManager(self.config)
        self.market_data = None 
        # relación riesgo/beneficio por defecto para TP si no viene en la señal
        self.default_rr = getattr(self.config, "default_rr", 2.0)

    async def _compose_signals(
        self, raw_signals: List[TacticalSignal]
    ) -> List[TacticalSignal]:
        # Composición/ensamble de señales (voting, weighted, etc.)
        final_signals = self.composer.compose(raw_signals, self.market_data)

        logger.info(f"Composed {len(final_signals)} tactical signals from {len(raw_signals)} candidates")
        return final_signals

    async def _ensure_stop_and_size(
        self,
        signal: TacticalSignal,
        market_features: MarketFeatures,
        portfolio_state: Dict,
        corr_matrix: Optional[pd.DataFrame] = None,
    ) -> Optional[Tuple[TacticalSignal, PositionSize]]:
        """
        Devuelve (signal_con_SL_y_TP, position_size_final) o None si se rechaza.
        """

        # 1) Sizing preliminar
        ps = await self.sizer.calculate_position_size(signal, market_features, portfolio_state)
        if ps is None:
            logger.info(f"Sizing rejected for {signal.symbol} (pre-checks)")
            return None

        # 2) Evaluación de riesgo pre-trade (aquí garantizamos SL si falta)
        allow, alerts, ps_adj = self.risk.evaluate_pre_trade_risk(
            signal=signal,
            position_size=ps,
            market_features=market_features,
            portfolio_state=portfolio_state,
            correlation_matrix=corr_matrix
        )

        if not allow or ps_adj is None:
            for a in alerts:
                logger.warning(f"[RISK] {a}")
            logger.warning(f"Trade rejected by risk for {signal.symbol}")
            return None

        # Si el riesgo no asignó SL (debería), asignamos aquí como fallback
        if ps_adj.stop_loss is None:
            # calcula SL inicial según riesgo dinámico; también lo escribe en signal.stop_loss
            from .risk_controls import DynamicStopLoss, RiskPosition
            dsl = DynamicStopLoss(self.config)
            dummy_pos = RiskPosition(
                symbol=signal.symbol, size=ps_adj.size if signal.is_long() else -ps_adj.size,
                entry_price=signal.price, current_price=signal.price, unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0
            )
            computed_sl = dsl.calculate_initial_stop(signal, market_features, dummy_pos)
            ps_adj.stop_loss = computed_sl
            signal.stop_loss = computed_sl
            logger.info(f"[FALLBACK] Added SL for {signal.symbol}: {computed_sl:.6f}")

        # 3) Asignar TP si no llegó en la señal y existe SL
        if (signal.take_profit is None) and (ps_adj.stop_loss is not None):
            dist = abs(signal.price - ps_adj.stop_loss)
            if signal.is_long():
                signal.take_profit = signal.price + self.default_rr * dist
            else:
                signal.take_profit = signal.price - self.default_rr * dist
            ps_adj.take_profit = signal.take_profit
            logger.info(f"Derived TP for {signal.symbol}: {ps_adj.take_profit:.6f} (RR={self.default_rr:.2f})")

        return signal, ps_adj

    def _to_l1_order(self, sig: TacticalSignal, ps: PositionSize) -> Dict:
        """
        Formato genérico de orden para L1. Ajusta las claves si tu L1 usa otras.
        """
        side = "buy" if sig.is_long() else "sell"
        order = {
            "id": f"l2_{sig.symbol}_{datetime.utcnow().timestamp()}",
            "symbol": sig.symbol,
            "side": side,
            "type": "market",   # o "limit" si prefieres
            "amount": float(ps.size),
            "price": float(sig.price),
            "stop_loss": float(ps.stop_loss) if ps.stop_loss else None,
            "take_profit": float(ps.take_profit) if ps.take_profit else None,
            "metadata": {
                "l2": {
                    "kelly_fraction": ps.kelly_fraction,
                    "vol_target_leverage": ps.vol_target_leverage,
                    "risk_amount": ps.risk_amount,
                    "notional": ps.notional,
                    "margin_required": ps.margin_required,
                },
                "signal": sig.as_dict() if hasattr(sig, "as_dict") else asdict(sig),
            }
        }
        return order


    async def _generate_signals(self, portfolio, market_data, features_by_symbol):
        """
        Genera señales brutas a partir de datos de mercado y features.
        """
        signals = []
        for symbol, features in features_by_symbol.items():
            # lógica de señal simple de ejemplo
            if features.momentum > self.config.MOMENTUM_THRESHOLD:
                signals.append(TacticalSignal(symbol=symbol, side="BUY", strength=features.momentum))
            elif features.momentum < -self.config.MOMENTUM_THRESHOLD:
                signals.append(TacticalSignal(symbol=symbol, side="SELL", strength=features.momentum))
        return signals


    async def process(
    self,
    portfolio: Dict,
    market_data: Dict,
    features_by_symbol: Dict[str, "MarketFeatures"],
) -> List["TacticalSignal"]:
        """
        Genera señales crudas a partir del portfolio, market_data y features.
        """

        self.market_data = market_data

        raw_signals = await self._generate_signals(
            portfolio=portfolio,
            market_data=market_data,
            features_by_symbol=features_by_symbol,
        )
        return raw_signals