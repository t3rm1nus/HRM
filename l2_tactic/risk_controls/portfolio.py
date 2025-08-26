# l2_tactic/risk_controls/portfolio.py

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import L2Config
from .alerts import AlertType, RiskAlert, RiskLevel
from .positions import RiskPosition
from ..models import TacticalSignal  # tipo para check_correlation_risk

logger = logging.getLogger(__name__)


class PortfolioRiskManager:
    """
    Riesgo de cartera: correlaciones, heat, pérdidas diarias, drawdown, número de posiciones
    y métricas agregadas.
    """

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

    # -------- correlación --------

    def check_correlation_risk(
        self,
        new_signal: TacticalSignal,
        current_positions: Dict[str, RiskPosition],
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> tuple[bool, List[RiskAlert]]:
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
                            metadata={"correlated_symbol": sym, "pos_size": pos.size},
                        )
                    )
        allow = all(a.severity != RiskLevel.HIGH for a in alerts)
        return allow, alerts

    # -------- límites de cartera --------

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

        # Pérdida diaria
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

        # Número de posiciones
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

    # -------- drawdown agregado + métricas --------

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

    # -------- series temporales --------

    def update_daily_pnl(self, pnl: float):
        now = datetime.utcnow()
        self.daily_pnl_history.append((now, float(pnl)))
        cutoff = now - timedelta(days=30)
        self.daily_pnl_history = [(t, v) for (t, v) in self.daily_pnl_history if t > cutoff]

    def update_portfolio_value(self, value: float):
        now = datetime.utcnow()
        self.portfolio_value_history.append((now, float(value)))
        cutoff = now - timedelta(days=90)
        self.portfolio_value_history = [(t, v) for (t, v) in self.portfolio_value_history if t > cutoff]

    def _max_drawdown(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        peak = values[0]
        mdd = 0.0
        for v in values[1:]:
            peak = max(peak, v)
            mdd = max(mdd, (peak - v) / peak if peak > 0 else 0.0)
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
