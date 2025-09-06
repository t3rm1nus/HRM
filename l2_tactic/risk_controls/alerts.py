# l2_tactic/risk_controls/alerts.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict

from core.logging import logger


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
    LIQUIDITY_INSUFFICIENT = "liquidity_insufficient"
    STRATEGY_DRAWDOWN = "strategy_drawdown"
    SIGNAL_DRAWDOWN = "signal_drawdown"


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
