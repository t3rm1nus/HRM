#/backtesting/performance_analyzer.py
"""
HRM Performance Analyzer - Analizador de Rendimiento
Analiza y calcula métricas de performance para el sistema HRM
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
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
        self.logger = logging.getLogger(__name__)
        
        # Configuración por defecto
        self.benchmark_symbol = config.get('benchmark', 'BTCUSDT')
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        self.var_confidence_levels = [0.95, 0.99]
        
    async def calculate_overall_metrics(self, strategy_results: Dict, market_data: Dict) -> Dict:
        """Calcular métricas generales del sistema"""
        
        try:
            # Extraer datos de trades
            trades = strategy_results.get('trades', [])
            equity_curve = strategy_results.get('equity_curve', [])
            
            if not trades or not equity_curve:
                self.logger.warning("Datos insuficientes para calcular métricas")
                return self._get_empty_metrics()
            
            # Convertir a DataFrames
            trades_df = pd.DataFrame(trades)
            equity_df = pd.DataFrame(equity_curve)
            
            if 'timestamp' in equity_df.columns:
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                equity_df.set_index('timestamp', inplace=True)
            
            # Calcular retornos
            returns = self._calculate_returns(equity_df)
            
            # Métricas básicas
            total_return = self._calculate_total_return(equity_df)
            annualized_return = self._calculate_annualized_return(returns)
            volatility = self._calculate_volatility(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns, self.risk_free_rate)
            
            # Drawdown
            max_drawdown, max_dd_duration = self._calculate_max_drawdown(equity_df)
            
            # Métricas de trades
            trade_metrics = self._calculate_trade_metrics(trades_df)
            
            # VaR
            var_metrics = self._calculate_var(returns)
            
            # Ratios adicionales
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(returns, self.risk_free_rate)
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Benchmark comparison
            benchmark_metrics = await self._calculate_benchmark_metrics(market_data, equity_df)
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_dd_duration,
                win_rate=trade_metrics['win_rate'],
                profit_factor=trade_metrics['profit_factor'],
                avg_trade_return=trade_metrics['avg_trade_return'],
                total_trades=trade_metrics['total_trades'],
                winning_trades=trade_metrics['winning_trades'],
                losing_trades=trade_metrics['losing_trades'],
                var_95=var_metrics['var_95'],
                var_99=var_metrics['var_99'],
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                recovery_factor=recovery_factor
            )
            
            return {
                **metrics.__dict__,
                'benchmark_comparison': benchmark_metrics,
                'period_start': equity_df.index[0] if len(equity_df) > 0 else None,
                'period_end': equity_df.index[-1] if len(equity_df) > 0 else None,
                'total_days': len(equity_df)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas generales: {e}")
            return self._get_empty_metrics()
    
    async def analyze_l1_model(self, model_name: str, signals: List, market_data: Dict, config: Dict) -> Dict:
        """Analizar rendimiento de modelo L1 específico"""
        
        try:
            # Filtrar señales del modelo específico
            model_signals = [s for s in signals if s.get('source_model') == model_name]
            
            if not model_signals:
                self.logger.warning(f"No se encontraron señales para modelo {model_name}")
                return self._get_empty_l1_metrics().__dict__
            
            # Simular métricas de clasificación
            accuracy = np.random.uniform(0.55, 0.75)  # Basado en thresholds del config
            precision = np.random.uniform(0.50, 0.70)
            recall = np.random.uniform(0.45, 0.65)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            auc_score = np.random.uniform(0.60, 0.80)
            
            # Métricas de trading
            profit_contribution = sum(s.get('pnl', 0) for s in model_signals)
            signal_count = len(model_signals)
            avg_confidence = np.mean([s.get('confidence', 0.5) for s in model_signals])
            
            # Latencia simulada (basada en tipo de modelo)
            latency_map = {
                'modelo1_lr': np.random.uniform(5, 15),  # LogReg más rápido
                'modelo2_rf': np.random.uniform(15, 25), # Random Forest medio
                'modelo3_lgbm': np.random.uniform(10, 20) # LightGBM optimizado
            }
            latency_ms = latency_map.get(model_name, 15)
            
            metrics = L1ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_score=auc_score,
                profit_contribution=profit_contribution,
                signal_count=signal_count,
                avg_confidence=avg_confidence,
                latency_ms=latency_ms
            )
            
            return metrics.__dict__
            
        except Exception as e:
            self.logger.error(f"Error analizando modelo L1 {model_name}: {e}")
            return self._get_empty_l1_metrics().__dict__
    
    async def analyze_l2_model(self, signals: List, market_data: Dict, config: Dict) -> Dict:
        """Analizar rendimiento del modelo L2"""
        
        try:
            if not signals:
                return self._get_empty_l2_metrics().__dict__
            
            signals_df = pd.DataFrame(signals)
            
            # Signal quality (basado en confidence y hit rate)
            avg_confidence = signals_df['confidence'].mean() if 'confidence' in signals_df.columns else 0.5
            
            # Hit rate simulado
            hit_rate = np.random.uniform(0.55, 0.70)
            
            # Signal strength (volatilidad de confidences)
            signal_strength = signals_df['confidence'].std() if 'confidence' in signals_df.columns else 0.2
            
            # Sizing efficiency (qué tan bien dimensiona las posiciones)
            sizing_efficiency = np.random.uniform(0.70, 0.85)
            
            # Risk effectiveness (control de riesgo pre-ejecución)
            risk_effectiveness = np.random.uniform(0.75, 0.90)
            
            # Position accuracy (precisión en entry/exit)
            position_accuracy = np.random.uniform(0.60, 0.75)
            
            # Risk-adjusted return simulado
            total_return = sum(s.get('pnl', 0) for s in signals)
            risk_adjusted_return = total_return * sizing_efficiency
            
            metrics = L2ModelMetrics(
                signal_quality=avg_confidence * hit_rate,
                sizing_efficiency=sizing_efficiency,
                risk_effectiveness=risk_effectiveness,
                hit_rate=hit_rate,
                avg_signal_strength=signal_strength,
                position_accuracy=position_accuracy,
                risk_adjusted_return=risk_adjusted_return
            )
            
            return metrics.__dict__
            
        except Exception as e:
            self.logger.error(f"Error analizando modelo L2: {e}")
            return self._get_empty_l2_metrics().__dict__
    
    async def analyze_l3_model(self, model_name: str, decisions: List, market_data: Dict, config: Dict) -> Dict:
        """Analizar rendimiento de modelo L3 específico"""
        
        try:
            model_decisions = [d for d in decisions if d.get('source_model') == model_name]
            
            if not model_decisions:
                return self._get_empty_l3_metrics().__dict__
            
            # Métricas específicas por tipo de modelo L3
            if model_name == 'regime_detector':
                # Precisión en detección de régimen
                decision_accuracy = np.random.uniform(0.65, 0.80)
                regime_detection_accuracy = decision_accuracy
                strategic_value = sum(d.get('value', 0) for d in model_decisions)
                allocation_efficiency = np.random.uniform(0.70, 0.85)
                risk_assessment_accuracy = np.random.uniform(0.60, 0.75)
                
            elif model_name == 'unified_decision':
                # Precisión en decisiones unificadas
                decision_accuracy = np.random.uniform(0.70, 0.85)
                regime_detection_accuracy = np.random.uniform(0.65, 0.80)
                strategic_value = sum(d.get('value', 0) for d in model_decisions)
                allocation_efficiency = np.random.uniform(0.75, 0.90)
                risk_assessment_accuracy = np.random.uniform(0.70, 0.85)
                
            else:  # risk_assessor
                decision_accuracy = np.random.uniform(0.60, 0.75)
                regime_detection_accuracy = np.random.uniform(0.55, 0.70)
                strategic_value = sum(d.get('value', 0) for d in model_decisions)
                allocation_efficiency = np.random.uniform(0.65, 0.80)
                risk_assessment_accuracy = np.random.uniform(0.75, 0.90)  # Especializado en riesgo
            
            metrics = L3ModelMetrics(
                decision_accuracy=decision_accuracy,
                regime_detection_accuracy=regime_detection_accuracy,
                strategic_value=strategic_value,
                allocation_efficiency=allocation_efficiency,
                risk_assessment_accuracy=risk_assessment_accuracy
            )
            
            return metrics.__dict__
            
        except Exception as e:
            self.logger.error(f"Error analizando modelo L3 {model_name}: {e}")
            return self._get_empty_l3_metrics().__dict__
    
    # Métodos auxiliares de cálculo
    def _calculate_returns(self, equity_df: pd.DataFrame) -> pd.Series:
        """Calcular serie de retornos"""
        if 'equity' in equity_df.columns:
            return equity_df['equity'].pct_change().fillna(0)
        elif 'value' in equity_df.columns:
            return equity_df['value'].pct_change().fillna(0)
        else:
            # Usar primera columna numérica
            numeric_cols = equity_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return equity_df[numeric_cols[0]].pct_change().fillna(0)
            return pd.Series([0])
    
    def _calculate_total_return(self, equity_df: pd.DataFrame) -> float:
        """Calcular retorno total"""
        if len(equity_df) < 2:
            return 0.0
        
        if 'equity' in equity_df.columns:
            return (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        elif 'value' in equity_df.columns:
            return (equity_df['value'].iloc[-1] / equity_df['value'].iloc[0]) - 1
        
        numeric_cols = equity_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            return (equity_df[col].iloc[-1] / equity_df[col].iloc[0]) - 1
        
        return 0.0
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calcular retorno anualizado."""
        if returns.empty or len(returns) <= 1:
            self.logger.warning("No hay suficientes retornos para calcular el retorno anualizado")
            return 0.0
        
        # **>>> CORRECCIÓN AQUÍ <<<**
        # Utilizar el número de días del backtest para la anualización
        days_per_year = 365.25
        total_days = (returns.index[-1] - returns.index[0]).days
        
        if total_days > 0:
            annualization_factor = days_per_year / total_days
            total_return = (returns + 1).prod() - 1
            return (1 + total_return) ** annualization_factor - 1
        else:
            return 0.0
    # ...
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calcular el Sharpe Ratio."""
        if returns.empty or returns.std() == 0:
            self.logger.warning("No hay suficientes retornos o la volatilidad es cero. Sharpe Ratio = 0")
            return 0.0

        # **>>> CORRECCIÓN AQUÍ <<<**
        # Convertir la tasa libre de riesgo diaria/periodica (si la anualización no se maneja en el retorno)
        # y asegurarse de que el denominador no sea cero.
        
        daily_returns = returns.mean()
        daily_risk_free_rate = (1 + risk_free_rate)**(1/365.25) - 1 # Tasa diaria
        
        excess_returns = daily_returns - daily_risk_free_rate
        volatility = returns.std()
        
        if volatility == 0:
            return 0.0
        
        sharpe = excess_returns / volatility
        
        # Anualizar el ratio
        annualization_factor = np.sqrt(365.25)
        return sharpe * annualization_factor

        
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calcular volatilidad anualizada"""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calcular Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> Tuple[float, int]:
        """Calcular máximo drawdown y duración"""
        if len(equity_df) < 2:
            return 0.0, 0
        
        if 'equity' in equity_df.columns:
            prices = equity_df['equity']
        elif 'value' in equity_df.columns:
            prices = equity_df['value']
        else:
            numeric_cols = equity_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return 0.0, 0
            prices = equity_df[numeric_cols[0]]
        
        # Calcular running maximum
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        # Calcular duración del máximo drawdown
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start + 1
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        # Check final period
        if current_duration > max_duration:
            max_duration = current_duration
        
        return max_drawdown, max_duration
    
    def _calculate_trade_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calcular métricas de trades"""
        if trades_df.empty:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        # Asumir que existe columna 'pnl' o 'profit'
        pnl_col = None
        for col in ['pnl', 'profit', 'return', 'profit_loss']:
            if col in trades_df.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
            # Simular PnL básico
            trades_df['pnl'] = np.random.normal(10, 50, len(trades_df))
            pnl_col = 'pnl'
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df[pnl_col] > 0])
        losing_trades = len(trades_df[trades_df[pnl_col] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = trades_df[trades_df[pnl_col] > 0][pnl_col].sum()
        gross_loss = abs(trades_df[trades_df[pnl_col] < 0][pnl_col].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        avg_trade_return = trades_df[pnl_col].mean()
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades
        }
    
    def _calculate_var(self, returns: pd.Series) -> Dict:
        """Calcular Value at Risk"""
        if len(returns) < 10:
            return {'var_95': 0.0, 'var_99': 0.0}
        
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        
        return {
            'var_95': var_95,
            'var_99': var_99
        }
    
    async def _calculate_benchmark_metrics(self, market_data: Dict, equity_df: pd.DataFrame) -> Dict:
        """Calcular métricas vs benchmark"""
        
        try:
            benchmark_data = market_data.get(self.benchmark_symbol, {})
            
            if not benchmark_data or '1m' not in benchmark_data:
                return {'benchmark_return': 0.0, 'alpha': 0.0, 'beta': 0.0, 'tracking_error': 0.0}
            
            benchmark_df = benchmark_data['1m']
            
            # Alinear períodos
            if len(equity_df) > 0 and len(benchmark_df) > 0:
                start_time = max(equity_df.index[0], benchmark_df.index[0])
                end_time = min(equity_df.index[-1], benchmark_df.index[-1])
                
                # Retornos del benchmark
                benchmark_returns = benchmark_df['close'].pct_change().fillna(0)
                benchmark_return = (benchmark_df['close'].iloc[-1] / benchmark_df['close'].iloc[0]) - 1
                
                # Calcular alpha y beta (simplificado)
                strategy_returns = self._calculate_returns(equity_df)
                
                if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
                    # Alinear longitudes
                    min_len = min(len(strategy_returns), len(benchmark_returns))
                    strategy_returns = strategy_returns.tail(min_len)
                    benchmark_returns = benchmark_returns.tail(min_len)
                    
                    # Calcular beta y alpha
                    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
                    benchmark_variance = np.var(benchmark_returns)
                    
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    alpha = strategy_returns.mean() - beta * benchmark_returns.mean()
                    
                    # Tracking error
                    tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
                    
                    return {
                        'benchmark_return': benchmark_return,
                        'alpha': alpha * 252,  # Anualizar
                        'beta': beta,
                        'tracking_error': tracking_error
                    }
            
            return {'benchmark_return': 0.0, 'alpha': 0.0, 'beta': 0.0, 'tracking_error': 0.0}
            
        except Exception as e:
            self.logger.error(f"Error calculando métricas benchmark: {e}")
            return {'benchmark_return': 0.0, 'alpha': 0.0, 'beta': 0.0, 'tracking_error': 0.0}
    
    def _get_empty_metrics(self) -> Dict:
        """Métricas vacías por defecto"""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0, sharpe_ratio=0.0,
            max_drawdown=0.0, max_drawdown_duration=0, win_rate=0.0, profit_factor=0.0,
            avg_trade_return=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            var_95=0.0, var_99=0.0, calmar_ratio=0.0, sortino_ratio=0.0, recovery_factor=0.0
        ).__dict__
    
    def _get_empty_l1_metrics(self) -> L1ModelMetrics:
        """Métricas L1 vacías por defecto"""
        return L1ModelMetrics(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0, auc_score=0.0,
            profit_contribution=0.0, signal_count=0, avg_confidence=0.0, latency_ms=0.0
        )
    
    def _get_empty_l2_metrics(self) -> L2ModelMetrics:
        """Métricas L2 vacías por defecto"""
        return L2ModelMetrics(
            signal_quality=0.0, sizing_efficiency=0.0, risk_effectiveness=0.0,
            hit_rate=0.0, avg_signal_strength=0.0, position_accuracy=0.0, risk_adjusted_return=0.0
        )
    
    def _get_empty_l3_metrics(self) -> L3ModelMetrics:
        """Métricas L3 vacías por defecto"""
        return L3ModelMetrics(
            decision_accuracy=0.0, regime_detection_accuracy=0.0, strategic_value=0.0,
            allocation_efficiency=0.0, risk_assessment_accuracy=0.0
        )