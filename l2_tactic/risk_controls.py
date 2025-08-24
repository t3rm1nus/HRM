# risk_controls.py - L2 Tactical Risk Management

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import TacticalSignal, PositionSize, RiskMetrics, MarketFeatures
from .config import L2Config

logger = logging.getLogger(__name__)


# ---------------------------
# Enums / DTOs de alertas
# ---------------------------
class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    CORRELATION_LIMIT = "correlation_limit"
    PORTFOLIO_HEAT = "portfolio_heat"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    POSITION_SIZE_LIMIT = "position_size_limit"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class RiskAlert:
    alert_type: AlertType
    severity: RiskLevel
    symbol: str
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.alert_type.value} for {self.symbol}: {self.message}"


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


@dataclass
class RiskPosition:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: float = 0.0
    time_in_position: timedelta = field(default_factory=lambda: timedelta())
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0


# ---------------------------
# Gestor de stop dinámico
# ---------------------------
class DynamicStopLoss:
    def __init__(self, config: L2Config):
        self.config = config
        self.default_stop_pct = getattr(config, "default_stop_pct", 0.02)
        self.atr_multiplier = getattr(config, "atr_multiplier", 2.0)
        self.trailing_stop_pct = getattr(config, "trailing_stop_pct", 0.01)
        self.breakeven_threshold = getattr(config, "breakeven_threshold", 1.5)
        self.active_stops: Dict[str, StopLossOrder] = {}

    def _calculate_fixed_stop_pct(self, signal: TacticalSignal, mf: MarketFeatures) -> float:
        base_stop = self.default_stop_pct
        confidence_adj = (1 - (signal.confidence or 0.0)) * 0.01
        vol_adj = 0.0
        if mf.volatility:
            vol_adj = max(0.0, min((mf.volatility - 0.2) * 0.5, 0.02))
        return base_stop + confidence_adj + vol_adj

    def _calculate_atr_stop(self, price: float, mf: MarketFeatures, side: str) -> Optional[float]:
        if not (hasattr(mf, "atr") and mf.atr):
            return None
        dist = mf.atr * self.atr_multiplier
        return price - dist if side == "buy" else price + dist

    def _calculate_volatility_stop(self, price: float, mf: MarketFeatures, side: str) -> Optional[float]:
        if not mf.volatility:
            return None
        daily_vol = mf.volatility / np.sqrt(252)
        pct = daily_vol * 2.0
        return price * (1 - pct) if side == "buy" else price * (1 + pct)

    def _calculate_support_resistance_stop(self, price: float, mf: MarketFeatures, side: str) -> Optional[float]:
        support = getattr(mf, "support", None)
        resistance = getattr(mf, "resistance", None)
        if side == "buy" and support:
            return support * 0.995
        if side == "sell" and resistance:
            return resistance * 1.005
        return None

    def calculate_initial_stop(self, signal: TacticalSignal, mf: MarketFeatures, position: RiskPosition) -> float:
        price = signal.price
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

        dist_pct = abs(final_stop - price) / price
        dist_pct = max(0.005, min(dist_pct, 0.05))
        final_stop = price * (1 - dist_pct) if side == "buy" else price * (1 + dist_pct)

        logger.info(f"Initial stop {signal.symbol}: price={price:.4f} stop={final_stop:.4f} ({dist_pct*100:.2f}%)")
        return final_stop

    def update_trailing_stop(self, symbol: str, current_price: float, position: RiskPosition) -> Optional[float]:
        order = self.active_stops.get(symbol)
        if not order or order.stop_type != "trailing":
            return None

        old_stop = order.stop_price
        trail_pct = self.trailing_stop_pct
        is_long = position.size > 0

        if is_long and current_price > order.entry_price:
            new_stop = current_price * (1 - trail_pct)
            if new_stop > old_stop:
                order.stop_price = new_stop
                order.last_updated = datetime.utcnow()
                logger.info(f"Trailing stop ↑ {symbol}: {old_stop:.4f} -> {new_stop:.4f}")
                return new_stop

        if (not is_long) and current_price < order.entry_price:
            new_stop = current_price * (1 + trail_pct)
            if new_stop < old_stop:
                order.stop_price = new_stop
                order.last_updated = datetime.utcnow()
                logger.info(f"Trailing stop ↓ {symbol}: {old_stop:.4f} -> {new_stop:.4f}")
                return new_stop

        return None

    def should_move_to_breakeven(self, position: RiskPosition, current_price: float) -> bool:
        entry = position.entry_price
        is_long = position.size > 0
        profit_pct = (current_price - entry) / entry if is_long else (entry - current_price) / entry
        return profit_pct >= (self.breakeven_threshold * self.default_stop_pct)


# ---------------------------
# Riesgo de portfolio
# ---------------------------
class PortfolioRiskManager:
    def __init__(self, config: L2Config):
        self.config = config
        self.max_correlation = getattr(config, "max_correlation", 0.7)
        self.max_portfolio_heat = getattr(config, "max_portfolio_heat", 0.8)
        self.daily_loss_limit = getattr(config, "daily_loss_limit", 0.05)
        self.max_drawdown_limit = getattr(config, "max_drawdown_limit", 0.15)
        self.max_positions = getattr(config, "max_positions", 5)

        self.risk_alerts: List[RiskAlert] = []
        self.daily_pnl_history: List[Tuple[datetime, float]] = []
        self.portfolio_value_history: List[Tuple[datetime, float]] = []

    def check_correlation_risk(
        self,
        new_signal: TacticalSignal,
        current_positions: Dict[str, RiskPosition],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, List[RiskAlert]]:
        alerts: List[RiskAlert] = []
        if correlation_matrix is None or new_signal.symbol not in correlation_matrix.index:
            return True, alerts

        for sym, pos in current_positions.items():
            if sym in correlation_matrix.columns:
                corr = float(abs(correlation_matrix.loc[new_signal.symbol, sym]))
                if corr > self.max_correlation:
                    alerts.append(
                        RiskAlert(
                            alert_type=AlertType.CORRELATION_LIMIT,
                            severity=RiskLevel.HIGH,
                            symbol=new_signal.symbol,
                            message=f"High correlation ({corr:.2f}) with {sym}",
                            current_value=corr,
                            threshold=self.max_correlation,
                            timestamp=datetime.utcnow(),
                            metadata={"correlated_symbol": sym},
                        )
                    )
        allow = all(a.severity != RiskLevel.HIGH for a in alerts)
        return allow, alerts

    def calculate_portfolio_heat(self, positions: Dict[str, RiskPosition], total_capital: float) -> float:
        total_risk = sum(max(0.0, pos.risk_amount) for pos in positions.values())
        heat = total_risk / total_capital if total_capital > 0 else 0.0
        return min(1.0, heat)

    def check_portfolio_limits(
        self,
        positions: Dict[str, RiskPosition],
        total_capital: float,
        daily_pnl: float
    ) -> List[RiskAlert]:
        alerts: List[RiskAlert] = []

        # Heat
        heat = self.calculate_portfolio_heat(positions, total_capital)
        if heat > self.max_portfolio_heat:
            alerts.append(
                RiskAlert(
                    alert_type=AlertType.PORTFOLIO_HEAT,
                    severity=RiskLevel.HIGH if heat > 0.9 else RiskLevel.MODERATE,
                    symbol="PORTFOLIO",
                    message=f"Portfolio heat too high: {heat:.2f}",
                    current_value=heat,
                    threshold=self.max_portfolio_heat,
                    timestamp=datetime.utcnow(),
                )
            )

        # Daily loss
        daily_loss_pct = abs(daily_pnl) / total_capital if daily_pnl < 0 else 0.0
        if daily_loss_pct > self.daily_loss_limit:
            alerts.append(
                RiskAlert(
                    alert_type=AlertType.DAILY_LOSS_LIMIT,
                    severity=RiskLevel.CRITICAL,
                    symbol="PORTFOLIO",
                    message=f"Daily loss limit exceeded: {daily_loss_pct:.2%}",
                    current_value=daily_loss_pct,
                    threshold=self.daily_loss_limit,
                    timestamp=datetime.utcnow(),
                )
            )

        # Num positions
        npos = len(positions)
        if npos >= self.max_positions:
            alerts.append(
                RiskAlert(
                    alert_type=AlertType.POSITION_SIZE_LIMIT,
                    severity=RiskLevel.MODERATE,
                    symbol="PORTFOLIO",
                    message=f"Maximum positions reached: {npos}/{self.max_positions}",
                    current_value=npos,
                    threshold=self.max_positions,
                    timestamp=datetime.utcnow(),
                )
            )
        return alerts

    def check_drawdown_limit(self, current_value: float, peak_value: float) -> Optional[RiskAlert]:
        if peak_value <= 0:
            return None
        dd = (peak_value - current_value) / peak_value
        if dd > self.max_drawdown_limit:
            return RiskAlert(
                alert_type=AlertType.DRAWDOWN_LIMIT,
                severity=RiskLevel.CRITICAL,
                symbol="PORTFOLIO",
                message=f"Drawdown limit exceeded: {dd:.2%}",
                current_value=dd,
                threshold=self.max_drawdown_limit,
                timestamp=datetime.utcnow(),
            )
        return None

    def update_daily_pnl(self, pnl: float):
        now = datetime.utcnow()
        self.daily_pnl_history.append((now, pnl))
        cutoff = now - timedelta(days=30)
        self.daily_pnl_history = [(t, v) for (t, v) in self.daily_pnl_history if t > cutoff]

    def update_portfolio_value(self, value: float):
        now = datetime.utcnow()
        self.portfolio_value_history.append((now, value))
        cutoff = now - timedelta(days=90)
        self.portfolio_value_history = [(t, v) for (t, v) in self.portfolio_value_history if t > cutoff]

    def _max_drawdown(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        peak = values[0]
        mdd = 0.0
        for v in values[1:]:
            peak = max(peak, v)
            mdd = max(mdd, (peak - v) / peak)
        return mdd

    def get_portfolio_metrics(self) -> Dict[str, float]:
        if len(self.portfolio_value_history) < 2:
            return {}
        values = [v for _, v in self.portfolio_value_history]
        rets = np.diff(values) / np.array(values[:-1])
        peak = max(values)
        cur = values[-1]
        dd = (peak - cur) / peak if peak > 0 else 0.0
        vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0
        mean_daily = float(np.mean(rets)) if len(rets) > 0 else 0.0
        sharpe = (mean_daily * 252) / vol if vol > 1e-12 else 0.0
        return {
            "current_drawdown": dd,
            "max_drawdown": self._max_drawdown(values),
            "volatility": vol,
            "sharpe_ratio": sharpe,
            "total_return": (cur - values[0]) / values[0] if values[0] > 0 else 0.0,
        }


# ---------------------------
# RiskControl Manager
# ---------------------------
class RiskControlManager:
    def __init__(self, config: L2Config):
        self.config = config
        self.stop_loss_manager = DynamicStopLoss(config)
        self.portfolio_manager = PortfolioRiskManager(config)
        self.current_positions: Dict[str, RiskPosition] = {}
        self.active_alerts: List[RiskAlert] = []
        logger.info("Initialized RiskControlManager")

    # ----- pre-trade -----
    def evaluate_pre_trade_risk(
        self,
        signal: TacticalSignal,
        position_size: PositionSize,
        market_features: MarketFeatures,
        portfolio_state: Dict,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, List[RiskAlert], Optional[PositionSize]]:
        alerts: List[RiskAlert] = []
        adjusted = position_size

        # 1) correlaciones
        allow_corr, corr_alerts = self.portfolio_manager.check_correlation_risk(
            signal, self.current_positions, correlation_matrix
        )
        alerts.extend(corr_alerts)

        # 2) límites de portfolio
        total_capital = float(portfolio_state.get("total_capital", 100_000.0))
        daily_pnl = float(portfolio_state.get("daily_pnl", 0.0))
        alerts.extend(self.portfolio_manager.check_portfolio_limits(self.current_positions, total_capital, daily_pnl))

        # 3) Asegurar STOP-LOSS si falta
        if adjusted.stop_loss is None:
            rp = RiskPosition(
                symbol=signal.symbol,
                size=adjusted.size if signal.is_long() else -adjusted.size,
                entry_price=signal.price,
                current_price=signal.price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
            )
            computed = self.stop_loss_manager.calculate_initial_stop(signal, market_features, rp)
            adjusted.stop_loss = computed
            signal.stop_loss = computed
            logger.info(f"[RISK] Assigned initial SL for {signal.symbol}: {computed:.6f}")

        # 4) ajuste por severidad
        if any(a.severity == RiskLevel.HIGH for a in alerts):
            adjusted.size *= 0.5
            adjusted.notional *= 0.5
            adjusted.risk_amount *= 0.5
            adjusted.metadata["risk_adjustment"] = "reduced_50pct_high_risk"
            logger.warning(f"Reduced size for {signal.symbol} due to high risk alerts")

        # 5) bloqueo por críticos
        allow_trade = allow_corr and not any(a.severity == RiskLevel.CRITICAL for a in alerts)
        self.active_alerts.extend(alerts)
        return allow_trade, alerts, (adjusted if allow_trade else None)

    # ----- on-trade / tracking -----
    def add_position(self, signal: TacticalSignal, position_size: PositionSize, mf: MarketFeatures):
        rp = RiskPosition(
            symbol=signal.symbol,
            size=position_size.size if signal.is_long() else -position_size.size,
            entry_price=signal.price,
            current_price=signal.price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_amount=position_size.risk_amount,
        )
        self.current_positions[signal.symbol] = rp

        # Stop inicial (si no vino en signal)
        if rp.stop_loss is None:
            rp.stop_loss = self.stop_loss_manager.calculate_initial_stop(signal, mf, rp)

        # Registrar stop dinámico
        self.stop_loss_manager.active_stops[signal.symbol] = StopLossOrder(
            symbol=signal.symbol,
            stop_price=rp.stop_loss,
            original_price=rp.stop_loss,
            entry_price=rp.entry_price,
            position_size=abs(rp.size),
            stop_type="trailing",
            last_updated=datetime.utcnow(),
        )
        logger.info(f"Position added to risk tracking: {signal.symbol} size={rp.size:.6f} SL={rp.stop_loss:.4f} TP={rp.take_profit}")

    def remove_position(self, symbol: str):
        self.current_positions.pop(symbol, None)
        self.stop_loss_manager.active_stops.pop(symbol, None)
        logger.info(f"Position removed from risk tracking: {symbol}")

    def monitor_existing_positions(self, price_data: Dict[str, float], portfolio_value: float) -> List[RiskAlert]:
        alerts: List[RiskAlert] = []
        for sym, pos in list(self.current_positions.items()):
            px = float(price_data.get(sym, 0.0) or 0.0)
            if px <= 0:
                continue

            pos.current_price = px
            if pos.size > 0:
                pos.unrealized_pnl = (px - pos.entry_price) * pos.size
            else:
                pos.unrealized_pnl = (pos.entry_price - px) * abs(pos.size)

            denom = max(1e-9, pos.entry_price * abs(pos.size))
            pos.unrealized_pnl_pct = pos.unrealized_pnl / denom

            # excursiones
            pos.max_adverse_excursion = min(pos.max_adverse_excursion, pos.unrealized_pnl)
            pos.max_favorable_excursion = max(pos.max_favorable_excursion, pos.unrealized_pnl)

            # stop loss
            sl_alert = self._check_stop_loss_trigger(sym, pos, px)
            if sl_alert:
                alerts.append(sl_alert)

            # trailing update
            self.stop_loss_manager.update_trailing_stop(sym, px, pos)

            # take profit
            if pos.take_profit:
                tp_hit = (pos.size > 0 and px >= pos.take_profit) or (pos.size < 0 and px <= pos.take_profit)
                if tp_hit:
                    alerts.append(
                        RiskAlert(
                            alert_type=AlertType.TAKE_PROFIT,
                            severity=RiskLevel.LOW,
                            symbol=sym,
                            message=f"Take profit triggered at {px:.4f}",
                            current_value=px,
                            threshold=float(pos.take_profit),
                            timestamp=datetime.utcnow(),
                        )
                    )

        # metrics de portfolio (opcional)
        self.portfolio_manager.update_portfolio_value(float(portfolio_value))
        return alerts

    def _check_stop_loss_trigger(self, symbol: str, position: RiskPosition, price: float) -> Optional[RiskAlert]:
        if position.stop_loss is None:
            return None
        trig = (position.size > 0 and price <= position.stop_loss) or (position.size < 0 and price >= position.stop_loss)
        if not trig:
            return None
        return RiskAlert(
            alert_type=AlertType.STOP_LOSS,
            severity=RiskLevel.HIGH,
            symbol=symbol,
            message=f"Stop loss triggered at {price:.4f}",
            current_value=price,
            threshold=float(position.stop_loss),
            timestamp=datetime.utcnow(),
            metadata={"position_size": position.size, "unrealized_pnl": position.unrealized_pnl},
        )
