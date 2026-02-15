# l2_tactic/metrics.py
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, DefaultDict, Any
from collections import defaultdict, deque
import numpy as np

from .models import TacticalSignal, PositionSize


@dataclass
class ExecRecord:
    order: PositionSize
    pnl: float
    ts: float
    symbol: str
    strategy_id: Optional[str] = None
    timeframe: Optional[str] = None
    side: Optional[str] = None  # "buy"/"sell"


class L2Metrics:
    """
    Tracking de performance del módulo táctico (L2).
    - Calidad de señales: hit rate, Sharpe, drawdown, latencia
    - Evaluación del modelo IA: precisión/recall/F1
    - Telemetría por timeframe y estrategia
    """
    def __init__(self, max_history: int = 10_000):
        self.total_return: float = 0.0
        self.sharpe: float = 0.0
        self.max_drawdown: float = 0.0

        # Señales y ejecuciones
        self.signal_history: List[TacticalSignal] = []
        self.execution_history: deque[ExecRecord] = deque(maxlen=max_history)

        # Latencia
        self.latency_records: List[float] = []

        # Predicciones y etiquetas reales para IA
        # key = (strategy_id, timeframe)
        self._preds: DefaultDict[Tuple[Optional[str], Optional[str]], List[int]] = defaultdict(list)
        self._truth: DefaultDict[Tuple[Optional[str], Optional[str]], List[int]] = defaultdict(list)
        self._scores: DefaultDict[Tuple[Optional[str], Optional[str]], List[float]] = defaultdict(list)

        # Telemetría por (strategy_id, timeframe)
        self._pnl_buckets: DefaultDict[Tuple[Optional[str], Optional[str]], List[float]] = defaultdict(list)
        self._wins_buckets: DefaultDict[Tuple[Optional[str], Optional[str]], int] = defaultdict(int)
        self._trades_buckets: DefaultDict[Tuple[Optional[str], Optional[str]], int] = defaultdict(int)

        self.start_time: float = time.time()

        
    # ==== Backward compatibility methods ====
    def update(self, success: bool, pnl: float, latency: float = 0.0):
        """Backward compatibility method for record_trade"""
        return self.record_trade(success=success, pnl=pnl, latency=latency)

    def update_prediction(self, y_true: int, y_pred: int, strategy_id: Optional[str] = None, timeframe: Optional[str] = None):
        """Backward compatibility method for recording predictions"""
        key = (strategy_id, timeframe)
        self._preds[key].append(int(y_pred))
        self._truth[key].append(int(y_true))

    def record_trade(self, success: bool, pnl: float, latency: float = 0.0, strategy_id: Optional[str] = None, timeframe: Optional[str] = None):
        """Record a trade outcome"""
        # Create a dummy ExecRecord for compatibility
        from datetime import datetime
        rec = ExecRecord(
            order=None,  # We don't have order info in this simplified call
            pnl=pnl,
            ts=time.time(),
            symbol="UNKNOWN",
            strategy_id=strategy_id,
            timeframe=timeframe,
            side="buy" if success else "sell"
        )
        self.execution_history.append(rec)
        
        if latency > 0:
            self.latency_records.append(latency)
            
        key = (strategy_id, timeframe)
        self._pnl_buckets[key].append(pnl)
        self._trades_buckets[key] += 1
        if success or pnl > 0:
            self._wins_buckets[key] += 1

    # ==== Registro ====
    def record_signal(self, signal: TacticalSignal):
        self.signal_history.append(signal)

    def record_execution(
        self,
        order: PositionSize,
        result: Dict,
        strategy_id: Optional[str] = None,
        timeframe: Optional[str] = None,
        side: Optional[str] = None,
    ):
        from .l2_utils import safe_float
        pnl = safe_float(result.get("pnl", 0.0))
        rec = ExecRecord(
            order=order,
            pnl=pnl,
            ts=time.time(),
            symbol=getattr(order, "symbol", None) or result.get("symbol", ""),
            strategy_id=strategy_id,
            timeframe=timeframe,
            side=side,
        )
        self.execution_history.append(rec)

        key = (strategy_id, timeframe)
        self._pnl_buckets[key].append(pnl)
        self._trades_buckets[key] += 1
        if pnl > 0:
            self._wins_buckets[key] += 1

    def record_latency(self, start_ts: float):
        self.latency_records.append(time.time() - start_ts)

    # ==== IA: registro de predicciones y outcomes ====
    def record_prediction(
        self,
        y_pred: int,
        y_score: float = 0.0,
        strategy_id: Optional[str] = None,
        timeframe: Optional[str] = None
    ):
        key = (strategy_id, timeframe)
        self._preds[key].append(int(y_pred))
        from .l2_utils import safe_float
        self._scores[key].append(safe_float(y_score))

    def record_outcome(
        self,
        y_true: int,
        strategy_id: Optional[str] = None,
        timeframe: Optional[str] = None
    ):
        key = (strategy_id, timeframe)
        self._truth[key].append(int(y_true))

    # ==== Métricas core ====
    def hit_rate(self) -> float:
        if not self.execution_history:
            return 0.0
        wins = sum(1 for e in self.execution_history if e.pnl > 0)
        return wins / len(self.execution_history)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        if not self.execution_history:
            return 0.0
        rets = [e.pnl for e in self.execution_history]
        from .l2_utils import safe_float
        mean = safe_float(np.mean(rets))
        std = safe_float(np.std(rets))
        if std == 0:
            return 0.0
        return (mean - risk_free_rate) / std

    def max_drawdown(self) -> float:
        if not self.execution_history:
            return 0.0
        cum = np.cumsum([e.pnl for e in self.execution_history])
        if len(cum) == 0:
            return 0.0
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak).min() if len(cum) else 0.0
        # devolver positivo como magnitud de DD
        from .l2_utils import safe_float
        return safe_float(abs(dd))

    def avg_latency(self) -> float:
        if not self.latency_records:
            return 0.0
        from .l2_utils import safe_float
        return safe_float(np.mean(self.latency_records))

    # ==== Métricas IA ====
    @staticmethod
    def _precision_recall(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
        if not y_true or len(y_true) != len(y_pred):
            return 0.0, 0.0, 0.0
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    def model_metrics(
        self,
        strategy_id: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> Dict[str, float]:
        key = (strategy_id, timeframe)
        y_pred = self._preds.get(key, [])
        y_true = self._truth.get(key, [])
        precision, recall, f1 = self._precision_recall(y_true, y_pred)
        from .l2_utils import safe_float
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_pairs": safe_float(min(len(y_true), len(y_pred))),
        }

    # ==== Telemetría por bucket ====
    def bucket_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Devuelve:
        {
          strategy_id: {
            timeframe: {
              trades, winrate, sharpe, pnl_mean, pnl_std
            }
          }
        }
        """
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for (strategy_id, timeframe), pnls in self._pnl_buckets.items():
            key_s = strategy_id or "default"
            key_t = timeframe or "any"
            out.setdefault(key_s, {})
            wins = self._wins_buckets[(strategy_id, timeframe)]
            n = self._trades_buckets[(strategy_id, timeframe)]
            from .l2_utils import safe_float
            mean = safe_float(np.mean(pnls)) if pnls else 0.0
            std = safe_float(np.std(pnls)) if pnls else 0.0
            sharpe = (mean / std) if std > 1e-12 else 0.0
            out[key_s][key_t] = {
                "trades": safe_float(n),
                "winrate": (wins / n) if n > 0 else 0.0,
                "sharpe": sharpe,
                "pnl_mean": mean,
                "pnl_std": std,
            }
        return out

    # ==== Resumen general ====
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

    def to_report_dict(self) -> Dict[str, Any]:
        """
        Exporta un payload compacto listo para PERFORMANCE_REPORT vía bus.
        Incluye métricas globales y bucketizadas.
        """
        return {
            "summary": self.summary(),
            "buckets": self.bucket_summary()
        }

    def to_dict(self):
        return {
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            # … cualquier otro atributo que quieras reportar
        }
