# l2_tactic/risk_controls/stop_losses.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np

from ..config import L2Config
from ..models import TacticalSignal, MarketFeatures
from .positions import RiskPosition

from core.logging import logger


@dataclass
class StopLossOrder:
    symbol: str
    stop_price: float
    original_price: float
    entry_price: float
    position_size: float
    stop_type: str  # "fixed", "trailing", "atr", "volatility"
    last_updated: datetime
    trail_amount: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class DynamicStopLoss:
    """
    Calcula stop inicial (mix de fijo, ATR, volatilidad y S/R), TP por RR, trailing y breakeven.
    """

    def __init__(self, config: L2Config):
        self.config = config
        self.default_stop_pct = getattr(config, "default_stop_pct", 0.02)
        self.atr_multiplier = getattr(config, "atr_multiplier", 2.0)
        self.trailing_stop_pct = getattr(config, "trailing_stop_pct", 0.01)
        self.breakeven_threshold = getattr(config, "breakeven_threshold", 1.5)
        self.rr_min = getattr(config, "take_profit_rr_min", 1.5)
        self.rr_max = getattr(config, "take_profit_rr_max", 2.5)
        self.active_stops: Dict[str, StopLossOrder] = {}

    # ---------- componentes del stop inicial ----------

    def _calculate_fixed_stop_pct(self, signal: TacticalSignal, mf: MarketFeatures) -> float:
        base_stop = self.default_stop_pct
        confidence_adj = (1 - (signal.confidence or 0.0)) * 0.01
        vol_adj = 0.0
        if mf.volatility:
            vol_adj = max(0.0, min((mf.volatility - 0.2) * 0.5, 0.02))
        return base_stop + confidence_adj + vol_adj

    def _calculate_atr_stop(self, price: float, mf: MarketFeatures, side: str) -> Optional[float]:
        atr = getattr(mf, "atr", None)
        if not atr:
            return None
        dist = atr * self.atr_multiplier
        return price - dist if side == "buy" else price + dist

    def _calculate_volatility_stop(self, price: float, mf: MarketFeatures, side: str) -> Optional[float]:
        if not mf.volatility:
            return None
        daily_vol = mf.volatility / np.sqrt(252)
        pct = max(0.003, min(daily_vol * 2.0, 0.05))
        return price * (1 - pct) if side == "buy" else price * (1 + pct)

    def _calculate_support_resistance_stop(self, price: float, mf: MarketFeatures, side: str) -> Optional[float]:
        support = getattr(mf, "support", None)
        resistance = getattr(mf, "resistance", None)
        if side == "buy" and support:
            from .utils import safe_float
            return safe_float(support) * 0.995
        if side == "sell" and resistance:
            return safe_float(resistance) * 1.005
        return None

    # ---------- API pública ----------

    def calculate_initial_stop(self, signal: TacticalSignal, mf: MarketFeatures, position: RiskPosition) -> float:
        price = float(signal.price)
        side = signal.side

        fixed_pct = self._calculate_fixed_stop_pct(signal, mf)
        fixed_stop = price * (1 - fixed_pct) if side == "buy" else price * (1 + fixed_pct)
        atr_stop = self._calculate_atr_stop(price, mf, side)
        vol_stop = self._calculate_volatility_stop(price, mf, side)
        sr_stop = self._calculate_support_resistance_stop(price, mf, side)

        stops = [fixed_stop, atr_stop, vol_stop, sr_stop]
        weights = [0.3, 0.3, 0.2, 0.2]
        valid = [(s, w) for s, w in zip(stops, weights) if s is not None]

        if not valid:
            final_stop = fixed_stop
        else:
            tw = sum(w for _, w in valid)
            final_stop = sum(s * w for s, w in valid) / (tw or 1.0)

        # clamp distancia 0.5%–5%
        dist_pct = abs(final_stop - price) / price
        dist_pct = max(0.005, min(dist_pct, 0.05))
        final_stop = price * (1 - dist_pct) if side == "buy" else price * (1 + dist_pct)

        logger.info(f"Initial stop {signal.symbol}: price={price:.6f} stop={final_stop:.6f} ({dist_pct*100:.2f}%)")
        return float(final_stop)

    def suggest_take_profit(self, signal: TacticalSignal, stop_price: float) -> Optional[float]:
        """TP adaptativo basado en ratio riesgo/beneficio y side."""
        price = float(signal.price)
        side = signal.side
        risk = abs(price - float(stop_price))
        if risk <= 0:
            return None
        rr = max(self.rr_min, min(self.rr_max, 1.0 + (signal.confidence or 0.0)))
        tp = price + rr * risk if side == "buy" else price - rr * risk
        return float(tp)

    def update_trailing_stop(self, symbol: str, current_price: float, position: RiskPosition) -> Optional[float]:
        order = self.active_stops.get(symbol)
        if not order or order.stop_type != "trailing":
            return None

        old_stop = float(order.stop_price)
        trail_pct = float(self.trailing_stop_pct)
        is_long = position.size > 0

        if is_long and current_price > order.entry_price:
            new_stop = float(current_price) * (1 - trail_pct)
            if new_stop > old_stop:
                order.stop_price = new_stop
                order.last_updated = datetime.utcnow()
                logger.info(f"Trailing stop ↑ {symbol}: {old_stop:.6f} -> {new_stop:.6f}")
                return new_stop

        if (not is_long) and current_price < order.entry_price:
            new_stop = float(current_price) * (1 + trail_pct)
            if new_stop < old_stop:
                order.stop_price = new_stop
                order.last_updated = datetime.utcnow()
                logger.info(f"Trailing stop ↓ {symbol}: {old_stop:.6f} -> {new_stop:.6f}")
                return new_stop

        return None

    def should_move_to_breakeven(self, position: RiskPosition, current_price: float) -> bool:
        entry = float(position.entry_price)
        is_long = position.size > 0
        profit_pct = (current_price - entry) / entry if is_long else (entry - current_price) / entry
        return profit_pct >= (self.breakeven_threshold * self.default_stop_pct)
