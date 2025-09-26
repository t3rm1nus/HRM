# l2_tactic/position_sizer.py - L2 position sizing (Kelly fraccional + Vol targeting + Liquidity + SL-aware)
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import TacticalSignal, MarketFeatures, PositionSize
from .config import L2Config

from core.logging import logger


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
      3) Ajuste por liquidez (cap por % de ADV/turnover)
      4) Límite por riesgo de la operación y límites absolutos
      5) Si hay SL -> sizing ajustado a pérdida máxima tolerada
    """

    def compute_position_size(self, *args, **kwargs):
        return self.calculate_position_size(*args, **kwargs)

    def __init__(self, config: L2Config):
        self.cfg = config

        # Kelly
        self.kelly_cap = getattr(config, "kelly_cap", 0.25)
        self.kelly_fraction = getattr(config, "kelly_fraction", 0.5)

        # Vol targeting
        self.vol_target = getattr(config, "vol_target", 0.20)

        # Liquidez
        self.max_notional_pct_of_adv = getattr(config, "max_notional_pct_of_adv", 0.02)
        self.min_adv_required = getattr(config, "min_adv_required", 10_000.0)

        # Límites absolutos
        self.min_position_notional = getattr(config, "min_position_notional", 100.0)
        self.max_position_notional = getattr(config, "max_position_notional", 1_000_000.0)
        self.max_risk_per_trade = getattr(config, "max_risk_per_trade", 0.01)

    # ---------- Kelly ----------

    def _kelly_from_signal(self, signal: TacticalSignal) -> KellyInputs:
        """
        Traduce confianza y 'strength' a inputs de Kelly.
        """
        win_prob = 0.45 + 0.20 * _bounded(signal.confidence, 0.0, 1.0)
        win_loss_ratio = 0.8 + 1.0 * _bounded(signal.strength, 0.0, 1.0)
        return KellyInputs(win_prob=win_prob, win_loss_ratio=win_loss_ratio)

    def _kelly_fraction(self, k: KellyInputs) -> float:
        """
        Kelly óptimo: f* = (b*p - q) / b
        """
        b = max(1e-9, k.win_loss_ratio)
        p = _bounded(k.win_prob, 0.0, 1.0)
        q = 1.0 - p
        f_star = (b * p - q) / b
        f_star = _bounded(f_star, 0.0, self.kelly_cap)
        return f_star * self.kelly_fraction

    # ---------- Vol targeting ----------

    def _leverage_for_vol_target(self, realized_vol: Optional[float]) -> float:
        if not realized_vol or realized_vol <= 0:
            return 1.0
        lev = self.vol_target / realized_vol
        return _bounded(lev, 0.25, 5.0)

    # ---------- Liquidez ----------

    def _cap_by_liquidity(self, desired_notional: float, features: Dict[str, Any]) -> float:
        adv = features.get("adv_notional") or features.get("liquidity")
        if adv and adv > 0:
            cap = float(self.max_notional_pct_of_adv) * float(adv)
            if desired_notional > cap:
                logger.info(f"Liquidity cap applied: desired={desired_notional:.2f}, cap={cap:.2f}")
            return min(desired_notional, cap)

        if features.get("volume") and features.get("price"):
            try:
                proxy_adv = float(features["volume"]) * float(features["price"])
                cap = float(self.max_notional_pct_of_adv) * proxy_adv
                return min(desired_notional, cap)
            except Exception:
                pass

        if adv is not None and adv < self.min_adv_required:
            logger.warning(f"⚠️ ADV={adv:.2f} < min required, forcing minimal position")
            return min(desired_notional, self.min_position_notional)

        return desired_notional

    # ---------- API principal ----------

    async def calculate_position_size(
        self,
        signal: TacticalSignal,
        market_features: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> Optional[PositionSize]:
        total_capital = float(portfolio_state.get("total_capital", 0.0) or 0.0)
        available_capital = float(portfolio_state.get("available_capital", total_capital))

        # Check if we have sufficient funds before proceeding
        if available_capital < self.min_position_notional:
            logger.warning(f"Insufficient available capital ({available_capital:.2f} < {self.min_position_notional} USDT) for {signal.symbol}")
            return None

        # Reservas de caja: hard floor 1%, soft reserve 5% (puede relajarse con alta confianza)
        hard_floor_pct = 0.01
        soft_reserve_pct = 0.05
        high_conf_threshold = 0.8
        hard_floor_usd = total_capital * hard_floor_pct
        soft_reserve_usd = total_capital * soft_reserve_pct

        # Si la confianza es alta, se permite usar hasta el hard floor; si no, respetar soft reserve
        confidence = float(getattr(signal, 'confidence', 0.0) or 0.0)
        min_cash_to_keep = hard_floor_usd if confidence >= high_conf_threshold else soft_reserve_usd
        effective_available = max(0.0, available_capital - min_cash_to_keep)
        if effective_available < available_capital:
            logger.info(
                f"Cash reserve aplicada ({(hard_floor_pct if confidence>=high_conf_threshold else soft_reserve_pct)*100:.0f}%): "
                f"available {available_capital:.2f} -> {effective_available:.2f}"
            )
        available_capital = effective_available

        if total_capital <= 0 or signal.price is None or signal.price <= 0:
            logger.warning("Position sizing aborted: missing total_capital or price")
            return None

        # 1) Kelly fraccional
        kelly_inputs = self._kelly_from_signal(signal)
        f_kelly = self._kelly_fraction(kelly_inputs)

        # 2) Límite de riesgo por trade
        risk_pct_cap = _bounded(self.max_risk_per_trade, 0.001, 0.05)
        risk_fraction = min(f_kelly, risk_pct_cap)

        # 3) Vol targeting
        realized_vol = market_features.get(signal.symbol, {}).get('volatility', self.vol_target)
        vol_leverage = self._leverage_for_vol_target(realized_vol)

        # 4) Notional inicial
        base_notional = total_capital * risk_fraction * 10.0
        notional = base_notional * vol_leverage

        # 5) Si hay stop_loss definido -> recalcular tamaño en base al riesgo real
        stop_loss = signal.stop_loss
        if stop_loss and stop_loss > 0:
            stop_distance = abs(signal.price - stop_loss)
            if stop_distance > 0:
                max_risk_amount = total_capital * risk_pct_cap
                size_sl_based = max_risk_amount / stop_distance
                notional_sl_based = size_sl_based * signal.price
                if notional_sl_based < notional:
                    logger.info(f"SL-based sizing applied: {notional_sl_based:.2f} vs {notional:.2f}")
                notional = min(notional, notional_sl_based)

        # 5.5) PYRAMIDING: Allow additional position for high confidence signals
        existing_position = portfolio_state.get(signal.symbol, {}).get("position", 0.0)
        if existing_position > 0 and signal.side == "buy" and signal.confidence > 0.8:
            # High confidence signal - allow pyramiding up to 50% of existing position
            max_pyramid_size = existing_position * 0.5
            pyramid_notional = max_pyramid_size * signal.price
            notional = min(notional, pyramid_notional)
            logger.info(f" pyramiding allowed: existing={existing_position:.4f}, max_add={max_pyramid_size:.4f}, notional={notional:.2f}")
        elif existing_position < 0 and signal.side == "sell" and signal.confidence > 0.8:
            # High confidence signal to reduce short position
            max_pyramid_size = abs(existing_position) * 0.5
            pyramid_notional = max_pyramid_size * signal.price
            notional = min(notional, pyramid_notional)
            logger.info(f" pyramiding short reduction: existing={existing_position:.4f}, max_add={max_pyramid_size:.4f}, notional={notional:.2f}")

        # 6) Liquidez
        notional = self._cap_by_liquidity(notional, market_features.get(signal.symbol, {}))

        # 7) Límites absolutos + disponibilidad de capital
        notional = _bounded(
            notional,
            self.min_position_notional,
            min(self.max_position_notional, available_capital),
        )

        size = notional / signal.price

        # 8) Estimar riesgo máximo
        if stop_loss and stop_loss > 0:
            if signal.is_long():
                max_loss = max(0.0, (signal.price - stop_loss) * size)
            else:
                max_loss = max(0.0, (stop_loss - signal.price) * size)
        else:
            max_loss = notional * risk_fraction

        # 9) Margen requerido
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
            take_profit=signal.take_profit,
            leverage=leverage,
            margin_required=margin_required,
            metadata={
                "kelly_inputs": kelly_inputs.__dict__,
                "risk_fraction_cap": risk_fraction,
                "total_capital": total_capital,
                "available_capital": available_capital,
                "realized_vol": realized_vol,
                "liquidity_cap_pct_of_adv": self.max_notional_pct_of_adv,
                "adv_notional": market_features.get(signal.symbol, {}).get("adv_notional")
                    or market_features.get(signal.symbol, {}).get("liquidity"),
                "sizing_method": "SL_based" if stop_loss else "heuristic",
            },
        )

        if ps.size <= 0 or ps.notional < self.min_position_notional:
            logger.info(f"Sizing rejected for {signal.symbol}: notional too small ({ps.notional:.2f})")
            return None

        return ps
