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
