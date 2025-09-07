# /backtesting/performance_analyzer.py
"""
HRM Performance Analyzer - Analizador de Rendimiento
Analiza y calcula métricas de performance para el sistema HRM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from dataclasses import dataclass
from enum import Enum
import warnings

from core.logging import logger  # ✅ Logger centralizado

warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento calculadas"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    var_95: float
    var_99: float
    calmar_ratio: float
    sortino_ratio: float
    recovery_factor: float

@dataclass
class L1ModelMetrics:
    """Métricas específicas de modelos L1"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    profit_contribution: float
    signal_count: int
    avg_confidence: float
    latency_ms: float

@dataclass
class L2ModelMetrics:
    """Métricas específicas del modelo L2"""
    signal_quality: float
    sizing_efficiency: float
    risk_effectiveness: float
    hit_rate: float
    avg_signal_strength: float
    position_accuracy: float
    risk_adjusted_return: float

@dataclass
class L3ModelMetrics:
    """Métricas específicas de modelos L3"""
    decision_accuracy: float
    regime_detection_accuracy: float
    strategic_value: float
    allocation_efficiency: float
    risk_assessment_accuracy: float

class PerformanceAnalyzer:
    """Analizador principal de rendimiento del sistema HRM"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logger  # ✅ Usar logger centralizado
        
        # Configuración por defecto
        self.benchmark_symbol = config.get('benchmark', 'BTCUSDT')
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.var_confidence_levels = [0.95, 0.99]
    
    # El resto del código se mantiene igual, pero todos los self.logger.warning / error
    # ya apuntan al logger centralizado automáticamente.

    def analyze_results(self, testing_results: Dict, metrics: List[str]) -> Dict:
        """Calcula métricas básicas a partir de 'testing_results'.
        Espera una clave 'trades' con elementos que incluyan 'entry_price' y 'exit_price'.
        Devuelve un diccionario con sección 'overall' enriquecida.
        """
        try:
            trades = testing_results.get('trades', []) or []
            if not trades:
                self.logger.warning("No hay trades para analizar. Devolviendo resultados originales.")
                # Asegura estructura mínima
                overall = testing_results.get('overall', {})
                defaults = {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'max_drawdown_duration': 0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_trade_return': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'var_95': 0.0,
                    'var_99': 0.0,
                    'calmar_ratio': 0.0,
                    'sortino_ratio': 0.0,
                    'recovery_factor': 0.0,
                }
                defaults.update(overall)
                testing_results['overall'] = defaults
                return testing_results

            # Construir serie de retornos por trade
            trade_returns = []
            equity_curve = []
            equity = 1.0
            durations_days = []
            pnl_list = []
            timestamps = []

            for t in trades:
                entry = float(t.get('entry_price', 0) or 0)
                exitp = float(t.get('exit_price', 0) or 0)
                if entry <= 0 or exitp <= 0:
                    continue
                r = (exitp - entry) / entry
                trade_returns.append(r)
                equity *= (1.0 + r)
                equity_curve.append(equity)
                pnl_list.append(float(t.get('pnl', (exitp - entry)) or (exitp - entry)))
                # Duración
                et = t.get('entry_timestamp')
                xt = t.get('exit_timestamp')
                if et is not None and xt is not None:
                    try:
                        # Permite timestamps datetime o strings convertibles
                        if not hasattr(et, 'to_pydatetime') and isinstance(et, str):
                            from pandas import to_datetime
                            et = to_datetime(et)
                        if not hasattr(xt, 'to_pydatetime') and isinstance(xt, str):
                            from pandas import to_datetime
                            xt = to_datetime(xt)
                        delta_days = max(1e-9, (xt - et).total_seconds() / 86400.0)
                        durations_days.append(delta_days)
                        timestamps.append(xt)
                    except Exception:
                        pass

            if not trade_returns:
                self.logger.warning("No fue posible calcular retornos de trade. Devolviendo resultados originales.")
                return testing_results

            import numpy as np
            returns = np.array(trade_returns, dtype=float)

            # Métricas básicas
            total_trades = int(len(returns))
            winning_trades = int(np.sum(returns > 0))
            losing_trades = int(np.sum(returns < 0))
            win_rate = float(winning_trades) / total_trades if total_trades > 0 else 0.0
            gross_profit = float(np.sum(returns[returns > 0]))
            gross_loss = float(np.abs(np.sum(returns[returns < 0])))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
            avg_trade_return = float(np.mean(returns)) if total_trades > 0 else 0.0

            # Drawdown sobre equity compuesta
            eq = np.array(equity_curve, dtype=float)
            peaks = np.maximum.accumulate(eq)
            drawdowns = (eq - peaks) / peaks
            max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0
            # Duración aproximada de DD: conteo máximo consecutivo bajo pico
            dd_durations = []
            cur = 0
            for i in range(len(eq)):
                if eq[i] < peaks[i]:
                    cur += 1
                else:
                    if cur > 0:
                        dd_durations.append(cur)
                        cur = 0
            if cur > 0:
                dd_durations.append(cur)
            max_drawdown_duration = int(max(dd_durations) if dd_durations else 0)

            # Volatilidad y Sharpe por-trade; anualización basada en duración media
            vol = float(np.std(returns, ddof=1)) if total_trades > 1 else 0.0
            mean_ret = float(np.mean(returns)) if total_trades > 0 else 0.0
            # Estimar trades por año
            if durations_days:
                avg_days = float(np.mean(durations_days))
                periods_per_year = 365.0 / max(1e-9, avg_days)
            else:
                periods_per_year = 252.0  # suposición razonable
            sharpe = (mean_ret / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0

            # Retorno total compuesto y anualizado
            total_return = float(equity - 1.0)
            annualized_return = float((1.0 + mean_ret) ** periods_per_year - 1.0) if mean_ret != -1.0 else -1.0

            # VaR empírico sobre distribución de retornos de trade
            var_95 = float(np.percentile(returns, 5))
            var_99 = float(np.percentile(returns, 1))

            # Ratios complementarios
            calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0
            downside = np.std(returns[returns < 0], ddof=1) if np.sum(returns < 0) > 1 else 0.0
            sortino_ratio = (mean_ret / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0
            recovery_factor = (total_return / abs(max_drawdown)) if max_drawdown < 0 else 0.0

            overall = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade_return': avg_trade_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'var_95': var_95,
                'var_99': var_99,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'recovery_factor': recovery_factor,
            }

            # Integrar con resultados originales
            out = dict(testing_results)
            base_overall = out.get('overall', {})
            base_overall.update(overall)
            out['overall'] = base_overall

            # Log corto de resultados clave
            self.logger.info(
                f"Resumen: trades={total_trades}, win_rate={win_rate:.2%}, PF={profit_factor:.2f}, "
                f"Sharpe={sharpe:.2f}, MDD={max_drawdown:.2%}, TotalRet={total_return:.2%}"
            )

            return out
        except Exception as e:
            self.logger.error(f"Fallo al analizar resultados: {e}")
            return testing_results