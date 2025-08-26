# l2_tactic/risk_controls/positions.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class RiskPosition:
    symbol: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_amount: float = 0.0
    time_in_position: timedelta = field(default_factory=lambda: timedelta())
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
