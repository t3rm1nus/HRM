# l2_tactic/metrics.py
import time
import numpy as np
from typing import List, Dict
from .models import TacticalSignal, PositionSize

class L2Metrics:
    """
    Tracking de performance del módulo táctico (L2).
    Permite calcular métricas de calidad de señales y riesgo.
    """

    def __init__(self):
        self.signal_history: List[TacticalSignal] = []
        self.execution_history: List[Dict] = []
        self.latency_records: List[float] = []
        self.start_time = time.time()

    def record_signal(self, signal: TacticalSignal):
        self.signal_history.append(signal)

    def record_execution(self, order: PositionSize, result: Dict):
        """
        Guarda ejecución de una orden con resultado real (profit/loss, sl, tp).
        """
        self.execution_history.append({
            "order": order,
            "result": result
        })

    def record_latency(self, start_ts: float):
        self.latency_records.append(time.time() - start_ts)

    # === MÉTRICAS ===
    def hit_rate(self) -> float:
        if not self.execution_history:
            return 0.0
        wins = sum(1 for e in self.execution_history if e["result"].get("pnl", 0) > 0)
        return wins / len(self.execution_history)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        if not self.execution_history:
            return 0.0
        returns = [e["result"].get("pnl", 0) for e in self.execution_history]
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return (mean - risk_free_rate) / std

    def max_drawdown(self) -> float:
        if not self.execution_history:
            return 0.0
        cum_returns = np.cumsum([e["result"].get("pnl", 0) for e in self.execution_history])
        peak = np.maximum.accumulate(cum_returns)
        dd = (cum_returns - peak).min()
        return float(dd)

    def avg_latency(self) -> float:
        if not self.latency_records:
            return 0.0
        return float(np.mean(self.latency_records))

    def summary(self) -> Dict:
        return {
            "hit_rate": self.hit_rate(),
            "sharpe_ratio": self.sharpe_ratio(),
            "max_drawdown": self.max_drawdown(),
            "avg_latency": self.avg_latency(),
            "signals_total": len(self.signal_history),
            "executions_total": len(self.execution_history),
            "uptime_sec": time.time() - self.start_time,
        }
