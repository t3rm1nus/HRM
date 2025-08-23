# position_sizer.py - L2 position sizing (Kelly fraccional + Vol targeting)

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from .models import TacticalSignal, MarketFeatures, PositionSize
from .config import L2Config

logger = logging.getLogger(__name__)


@dataclass
class KellyInputs:
    win_prob: float           # probabilidad de acierto (0..1)
    win_loss_ratio: float     # beneficio medio / pérdida media (>0)


def _bounded(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PositionSizerManager:
    """
    Cálculo de tamaño de posición:
      1) Kelly fraccional (deriva de probabilidad/confianza y payoff)
      2) Vol targeting (ajuste por volatilidad objetivo)
      3) Límite por riesgo/heat de portfolio desde config/L3
    """

    def __init__(self, config: L2Config):
        self.config = config
        self.kelly_cap = getattr(config, "kelly_cap", 0.25)                # límite superior de Kelly
        self.kelly_fraction = getattr(config, "kelly_fraction", 0.5)       # fracción de Kelly a aplicar
        self.vol_target = getattr(config, "vol_target", 0.20)              # vol objetivo anualizada
        self.min_position_notional = getattr(config, "min_position_notional", 100.0)
        self.max_position_notional = getattr(config, "max_position_notional", 1_000_000.0)
        self.max_risk_per_trade = getattr(config, "max_risk_per_trade", 0.01)  # % capital

    # ---------- Kelly ----------

    def _kelly_from_signal(self, signal: TacticalSignal) -> KellyInputs:
        """
        Traducimos la confianza y "strength" a inputs de Kelly.
        - win_prob: mapea confidence (0..1) a [0.45..0.65] (centrado en 0.55)
        - win_loss_ratio: mapea strength (0..1) a [0.8..1.8]
        """
        win_prob = 0.45 + 0.20 * _bounded(signal.confidence, 0.0, 1.0)
        win_loss_ratio = 0.8 + 1.0 * _bounded(signal.strength, 0.0, 1.0)
        return KellyInputs(win_prob=win_prob, win_loss_ratio=win_loss_ratio)

    def _kelly_fraction(self, k: KellyInputs) -> float:
        """
        Kelly óptimo: f* = (b*p - q) / b
        donde b = win_loss_ratio, p = win_prob, q = 1 - p
        """
        b = max(1e-9, k.win_loss_ratio)
        p = _bounded(k.win_prob, 0.0, 1.0)
        q = 1.0 - p
        f_star = (b * p - q) / b
        f_star = _bounded(f_star, 0.0, self.kelly_cap)
        return f_star * self.kelly_fraction

    # ---------- Vol targeting ----------

    def _leverage_for_vol_target(self, realized_vol: Optional[float]) -> float:
        """
        Leverage recomendado según vol objetivo: lev = vol_target / realized_vol
        Si no hay realized_vol, devolvemos 1.0
        """
        if not realized_vol or realized_vol <= 0:
            return 1.0
        lev = self.vol_target / realized_vol
        # mantenemos límites razonables
        lev = _bounded(lev, 0.25, 5.0)
        return lev

    # ---------- API principal ----------

    async def calculate_position_size(
        self,
        signal: TacticalSignal,
        market_features: MarketFeatures,
        portfolio_state: Dict
    ) -> Optional[PositionSize]:
        """
        Devuelve un PositionSize o None si no pasa mínimos.
        """
        total_capital = float(portfolio_state.get("total_capital", 0.0) or 0.0)
        available_capital = float(portfolio_state.get("available_capital", total_capital))

        if total_capital <= 0 or signal.price is None or signal.price <= 0:
            logger.warning("Position sizing aborted: missing total_capital or price")
            return None

        # 1) Kelly fraccional
        kelly_inputs = self._kelly_from_signal(signal)
        f_kelly = self._kelly_fraction(kelly_inputs)  # fracción del capital

        # 2) Límite de riesgo por trade (cap a f_kelly)
        risk_pct_cap = _bounded(self.max_risk_per_trade, 0.001, 0.05)
        risk_fraction = min(f_kelly, risk_pct_cap)

        # 3) Vol targeting -> leverage recomendado
        realized_vol = market_features.volatility or self.vol_target
        vol_leverage = self._leverage_for_vol_target(realized_vol)

        # 4) Notional base y riesgos
        base_notional = total_capital * risk_fraction * 10.0  # multiplicador para convertir riesgo% en exposición base
        notional = base_notional * vol_leverage

        # bounds
        notional = _bounded(notional, self.min_position_notional, min(self.max_position_notional, available_capital))

        size = notional / signal.price

        # niveles SL/TP respetando los del signal si existen
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit

        # calculamos max_loss aproximado (si existe SL); si no, usamos riesgo fraccional sobre notional
        if stop_loss and stop_loss > 0:
            if signal.is_long():
                max_loss = max(0.0, (signal.price - stop_loss) * size)
            else:
                max_loss = max(0.0, (stop_loss - signal.price) * size)
        else:
            max_loss = notional * risk_fraction

        # margen aproximado si hay apalancamiento
        leverage = max(1.0, vol_leverage)
        margin_required = notional / leverage

        ps = PositionSize(
            symbol=signal.symbol,
            side=signal.side,
            price=signal.price,
            size=size,
            notional=notional,
            risk_amount=max_loss,
            kelly_fraction=f_kelly,
            vol_target_leverage=vol_leverage,
            max_loss=max_loss,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            margin_required=margin_required,
            metadata={
                "kelly_inputs": kelly_inputs.__dict__,
                "risk_fraction_cap": risk_fraction,
                "total_capital": total_capital,
                "available_capital": available_capital,
                "realized_vol": realized_vol,
            }
        )

        # sanity checks mínimos
        if ps.size <= 0 or ps.notional < self.min_position_notional:
            logger.info(f"Sizing rejected for {signal.symbol}: notional too small ({ps.notional:.2f})")
            return None

        return ps
