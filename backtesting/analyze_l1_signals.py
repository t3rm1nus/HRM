#!/usr/bin/env python3
"""
Análisis de Señales L1 - Script de análisis post-backtest

Este script:
1. Carga los datos CSV generados por backtest_l1_standalone.py
2. Analiza la distribución de confidence scores vs win rate
3. Filtra trades por umbrales (0.6, 0.7, 0.8) y calcula métricas
4. Detecta sesgo de dirección (BUY vs SELL)
5. Calcula autocorrelación de señales consecutivas
6. Genera reporte HTML con gráficos de equity, distribución de confidence y heatmap por hora

Uso:
    python backtesting/analyze_l1_signals.py --signals-csv results/l1_backtest_signals_*.csv --trades-csv results/l1_backtest_trades_*.csv
    
    # O busca automáticamente los archivos más recientes
    python backtesting/analyze_l1_signals.py
"""

import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directorio de resultados
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
ANALYSIS_DIR = os.path.join(os.path.dirname(__file__), 'analysis')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConfidenceAnalysis:
    """Análisis de confidence scores"""
    threshold: float
    total_signals: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl: float
    total_pnl: float
    profit_factor: float
    sharpe_ratio: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'threshold': f"{self.threshold:.1f}",
            'total_signals': self.total_signals,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'avg_pnl': f"${self.avg_pnl:.2f}",
            'total_pnl': f"${self.total_pnl:.2f}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}"
        }


@dataclass
class DirectionBias:
    """Sesgo de dirección"""
    total_buy_signals: int
    total_sell_signals: int
    total_hold_signals: int
    buy_percentage: float
    sell_percentage: float
    hold_percentage: float
    buy_win_rate: float
    sell_win_rate: float
    buy_avg_pnl: float
    sell_avg_pnl: float
    bias_direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    bias_strength: str   # 'STRONG', 'MODERATE', 'WEAK'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_buy_signals': self.total_buy_signals,
            'total_sell_signals': self.total_sell_signals,
            'total_hold_signals': self.total_hold_signals,
            'buy_percentage': f"{self.buy_percentage:.2%}",
            'sell_percentage': f"{self.sell_percentage:.2%}",
            'hold_percentage': f"{self.hold_percentage:.2%}",
            'buy_win_rate': f"{self.buy_win_rate:.2%}",
            'sell_win_rate': f"{self.sell_win_rate:.2%}",
            'buy_avg_pnl': f"${self.buy_avg_pnl:.2f}",
            'sell_avg_pnl': f"${self.sell_avg_pnl:.2f}",
            'bias_direction': self.bias_direction,
            'bias_strength': self.bias_strength
        }


@dataclass
class AutocorrelationResult:
    """Resultado de autocorrelación de señales"""
    lag_1_correlation: float
    lag_2_correlation: float
    lag_3_correlation: float
    lag_5_correlation: float
    lag_10_correlation: float
    consecutive_same_signals: int
    total_transitions: int
    repetition_rate: float
    interpretation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lag_1_correlation': f"{self.lag_1_correlation:.3f}",
            'lag_2_correlation': f"{self.lag_2_correlation:.3f}",
            'lag_3_correlation': f"{self.lag_3_correlation:.3f}",
            'lag_5_correlation': f"{self.lag_5_correlation:.3f}",
            'lag_10_correlation': f"{self.lag_10_correlation:.3f}",
            'consecutive_same_signals': self.consecutive_same_signals,
            'total_transitions': self.total_transitions,
            'repetition_rate': f"{self.repetition_rate:.2%}",
            'interpretation': self.interpretation
        }


# =============================================================================
# CARGA DE DATOS
# =============================================================================

class DataLoader:
    """Cargador de datos de backtesting"""
    
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
    
    def find_latest_files(self) -> Tuple[Optional[str], Optional[str]]:
        """Encuentra los archivos CSV más recientes"""
        signals_pattern = os.path.join(self.results_dir, "l1_backtest_signals_*.csv")
        trades_pattern = os.path.join(self.results_dir, "l1_backtest_trades_*.csv")
        
        signals_files = glob.glob(signals_pattern)
        trades_files = glob.glob(trades_pattern)
        
        signals_csv = max(signals_files, key=os.path.getctime) if signals_files else None
        trades_csv = max(trades_files, key=os.path.getctime) if trades_files else None
        
        return signals_csv, trades_csv
    
    def load_signals_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        """Carga el CSV de señales"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"📁 Cargado signals CSV: {filepath} ({len(df)} registros)")
            
            # Normalizar columnas
            df.columns = [c.lower() for c in df.columns]
            
            # Convertir timestamp a datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            logger.error(f"❌ Error cargando {filepath}: {e}")
            return None
    
    def load_trades_csv(self, filepath: str) -> Optional[pd.DataFrame]:
        """Carga el CSV de trades"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"📁 Cargado trades CSV: {filepath} ({len(df)} trades)")
            
            # Normalizar columnas
            df.columns = [c.lower() for c in df.columns]
            
            # Convertir timestamps
            for col in ['timestamp', 'exit_timestamp']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df
        except Exception as e:
            logger.error(f"❌ Error cargando {filepath}: {e}")
            return None


# =============================================================================
# ANÁLISIS DE CONFIDENCE SCORES
# =============================================================================

class ConfidenceAnalyzer:
    """Analiza la relación entre confidence scores y rendimiento"""
    
    def __init__(self, trades_df: pd.DataFrame):
        self.trades_df = trades_df.copy()
    
    def analyze_by_thresholds(self, thresholds: List[float] = [0.6, 0.7, 0.8]) -> List[ConfidenceAnalysis]:
        """Analiza métricas para diferentes umbrales de confidence"""
        results = []
        
        for threshold in thresholds:
            analysis = self._analyze_threshold(threshold)
            results.append(analysis)
        
        return results
    
    def _analyze_threshold(self, threshold: float) -> ConfidenceAnalysis:
        """Analiza métricas para un umbral específico"""
        filtered = self.trades_df[self.trades_df['confidence'] >= threshold]
        
        if len(filtered) == 0:
            return ConfidenceAnalysis(
                threshold=threshold,
                total_signals=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_pnl=0.0,
                total_pnl=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0
            )
        
        total = len(filtered)
        winning = len(filtered[filtered['pnl'] > 0])
        losing = len(filtered[filtered['pnl'] <= 0])
        win_rate = winning / total if total > 0 else 0
        
        pnl_values = filtered['pnl'].values
        avg_pnl = np.mean(pnl_values)
        total_pnl = np.sum(pnl_values)
        
        # Profit factor
        gross_profit = np.sum(pnl_values[pnl_values > 0]) if winning > 0 else 0
        gross_loss = abs(np.sum(pnl_values[pnl_values <= 0])) if losing > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio simplificado
        sharpe = np.mean(pnl_values) / np.std(pnl_values) * np.sqrt(252) if len(pnl_values) > 1 and np.std(pnl_values) > 0 else 0
        
        return ConfidenceAnalysis(
            threshold=threshold,
            total_signals=total,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            total_pnl=total_pnl,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe
        )
    
    def get_confidence_distribution(self, bins: int = 10) -> pd.DataFrame:
        """Obtiene la distribución de confidence scores con win rate por bin"""
        if self.trades_df.empty:
            return pd.DataFrame()
        
        # Crear bins de confidence
        self.trades_df['confidence_bin'] = pd.cut(self.trades_df['confidence'], bins=bins)
        
        distribution = self.trades_df.groupby('confidence_bin').agg({
            'pnl': ['count', 'mean', 'sum'],
            'confidence': 'mean'
        }).reset_index()
        
        distribution.columns = ['confidence_range', 'total_trades', 'avg_pnl', 'total_pnl', 'avg_confidence']
        
        # Calcular win rate por bin
        win_rates = []
        for bin_range in distribution['confidence_range']:
            bin_trades = self.trades_df[self.trades_df['confidence_bin'] == bin_range]
            if len(bin_trades) > 0:
                win_rate = len(bin_trades[bin_trades['pnl'] > 0]) / len(bin_trades)
            else:
                win_rate = 0
            win_rates.append(win_rate)
        
        distribution['win_rate'] = win_rates
        
        return distribution
    
    def correlation_confidence_pnl(self) -> float:
        """Calcula la correlación entre confidence y PnL"""
        if len(self.trades_df) < 2:
            return 0.0
        return self.trades_df['confidence'].corr(self.trades_df['pnl'])


# =============================================================================
# ANÁLISIS DE SESGO DE DIRECCIÓN
# =============================================================================

class DirectionBiasAnalyzer:
    """Analiza el sesgo de dirección (BUY vs SELL)"""
    
    def __init__(self, signals_df: pd.DataFrame, trades_df: pd.DataFrame):
        self.signals_df = signals_df.copy() if signals_df is not None else None
        self.trades_df = trades_df.copy()
    
    def analyze(self) -> DirectionBias:
        """Analiza el sesgo de dirección"""
        if self.signals_df is not None and not self.signals_df.empty:
            # Contar todas las señales generadas
            total = len(self.signals_df)
            buy_signals = len(self.signals_df[self.signals_df['signal'] == 'BUY'])
            sell_signals = len(self.signals_df[self.signals_df['signal'] == 'SELL'])
            hold_signals = len(self.signals_df[self.signals_df['signal'] == 'HOLD'])
        else:
            # Usar trades como proxy
            total = len(self.trades_df) * 2  # Estimación
            buy_signals = len(self.trades_df[self.trades_df['side'] == 'BUY'])
            sell_signals = len(self.trades_df[self.trades_df['side'] == 'SELL'])
            hold_signals = 0
        
        # Calcular porcentajes
        buy_pct = buy_signals / total if total > 0 else 0
        sell_pct = sell_signals / total if total > 0 else 0
        hold_pct = hold_signals / total if total > 0 else 0
        
        # Analizar rendimiento por lado
        buy_trades = self.trades_df[self.trades_df['side'] == 'BUY']
        sell_trades = self.trades_df[self.trades_df['side'] == 'SELL']
        
        buy_win_rate = len(buy_trades[buy_trades['pnl'] > 0]) / len(buy_trades) if len(buy_trades) > 0 else 0
        sell_win_rate = len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades) if len(sell_trades) > 0 else 0
        
        buy_avg_pnl = buy_trades['pnl'].mean() if len(buy_trades) > 0 else 0
        sell_avg_pnl = sell_trades['pnl'].mean() if len(sell_trades) > 0 else 0
        
        # Determinar sesgo
        diff = abs(buy_pct - sell_pct)
        if buy_pct > sell_pct * 1.5:
            bias_direction = 'BUY'
            bias_strength = 'STRONG' if diff > 0.3 else 'MODERATE' if diff > 0.15 else 'WEAK'
        elif sell_pct > buy_pct * 1.5:
            bias_direction = 'SELL'
            bias_strength = 'STRONG' if diff > 0.3 else 'MODERATE' if diff > 0.15 else 'WEAK'
        else:
            bias_direction = 'NEUTRAL'
            bias_strength = 'NONE'
        
        return DirectionBias(
            total_buy_signals=buy_signals,
            total_sell_signals=sell_signals,
            total_hold_signals=hold_signals,
            buy_percentage=buy_pct,
            sell_percentage=sell_pct,
            hold_percentage=hold_pct,
            buy_win_rate=buy_win_rate,
            sell_win_rate=sell_win_rate,
            buy_avg_pnl=buy_avg_pnl,
            sell_avg_pnl=sell_avg_pnl,
            bias_direction=bias_direction,
            bias_strength=bias_strength
        )


# =============================================================================
# ANÁLISIS DE AUTOCORRELACIÓN
# =============================================================================

class AutocorrelationAnalyzer:
    """Analiza la autocorrelación de señales consecutivas"""
    
    def __init__(self, signals_df: pd.DataFrame):
        self.signals_df = signals_df.copy() if signals_df is not None else None
    
    def analyze(self) -> AutocorrelationResult:
        """Calcula la autocorrelación de señales"""
        if self.signals_df is None or self.signals_df.empty:
            return AutocorrelationResult(
                lag_1_correlation=0,
                lag_2_correlation=0,
                lag_3_correlation=0,
                lag_5_correlation=0,
                lag_10_correlation=0,
                consecutive_same_signals=0,
                total_transitions=0,
                repetition_rate=0,
                interpretation="No hay datos de señales disponibles"
            )
        
        # Convertir señales a numérico: BUY=1, HOLD=0, SELL=-1
        signal_map = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
        signals_numeric = self.signals_df['signal'].map(signal_map).fillna(0).values
        
        # Calcular autocorrelación para diferentes lags
        def autocorr(x, lag):
            if len(x) <= lag:
                return 0
            return np.corrcoef(x[lag:], x[:-lag])[0, 1] if np.std(x[lag:]) > 0 and np.std(x[:-lag]) > 0 else 0
        
        lag_1 = autocorr(signals_numeric, 1)
        lag_2 = autocorr(signals_numeric, 2)
        lag_3 = autocorr(signals_numeric, 3)
        lag_5 = autocorr(signals_numeric, 5)
        lag_10 = autocorr(signals_numeric, 10)
        
        # Contar señales consecutivas idénticas
        consecutive_same = sum(1 for i in range(1, len(signals_numeric)) if signals_numeric[i] == signals_numeric[i-1])
        total_transitions = len(signals_numeric) - 1
        repetition_rate = consecutive_same / total_transitions if total_transitions > 0 else 0
        
        # Interpretación
        if lag_1 > 0.5:
            interpretation = "ALTA AUTOCORRELACIÓN: Las señales tienden a repetirse consecutivamente. Considerar implementar cool-down entre señales."
        elif lag_1 > 0.2:
            interpretation = "MODERADA AUTOCORRELACIÓN: Algunas repeticiones de señales. Revisar lógica de generación."
        elif lag_1 < -0.2:
            interpretation = "AUTOCORRELACIÓN NEGATIVA: Las señales alternan frecuentemente. Posible sobreajuste o ruido."
        else:
            interpretation = "AUTOCORRELACIÓN BAJA: Las señales son relativamente independientes. Buena señal."
        
        return AutocorrelationResult(
            lag_1_correlation=lag_1,
            lag_2_correlation=lag_2,
            lag_3_correlation=lag_3,
            lag_5_correlation=lag_5,
            lag_10_correlation=lag_10,
            consecutive_same_signals=consecutive_same,
            total_transitions=total_transitions,
            repetition_rate=repetition_rate,
            interpretation=interpretation
        )


# =============================================================================
# GENERADOR DE REPORTE HTML
# =============================================================================

class HTMLReportGenerator:
    """Genera reporte HTML con gráficos"""
    
    def __init__(self, output_dir: str = ANALYSIS_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(
        self,
        signals_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        confidence_analyses: List[ConfidenceAnalysis],
        direction_bias: DirectionBias,
        autocorr_result: AutocorrelationResult,
        confidence_corr: float
    ) -> str:
        """Genera el reporte HTML completo"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"l1_signals_analysis_{timestamp}.html")
        
        # Generar datos para gráficos
        equity_data = self._prepare_equity_data(trades_df)
        confidence_dist = self._prepare_confidence_distribution(trades_df)
        hourly_data = self._prepare_hourly_heatmap(trades_df)
        
        html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análisis de Señales L1 - Reporte</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        :root {{
            --primary-color: #2563eb;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, var(--primary-color), #1e40af);
            color: white;
            border-radius: 16px;
            margin-bottom: 30px;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
        }}
        
        .metric-card.success {{ border-left-color: var(--success-color); }}
        .metric-card.danger {{ border-left-color: var(--danger-color); }}
        .metric-card.warning {{ border-left-color: var(--warning-color); }}
        
        .metric-card h3 {{
            font-size: 0.875em;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--text-color);
        }}
        
        .section {{
            background: var(--card-bg);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-color);
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.875em;
        }}
        
        tr:hover {{
            background: var(--bg-color);
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }}
        
        .alert {{
            padding: 16px 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .alert-warning {{
            background: #fef3c7;
            border: 1px solid #f59e0b;
            color: #92400e;
        }}
        
        .alert-danger {{
            background: #fee2e2;
            border: 1px solid #ef4444;
            color: #991b1b;
        }}
        
        .alert-success {{
            background: #d1fae5;
            border: 1px solid #10b981;
            color: #065f46;
        }}
        
        .heatmap-container {{
            display: grid;
            grid-template-columns: repeat(24, 1fr);
            gap: 2px;
            margin-top: 20px;
        }}
        
        .heatmap-cell {{
            aspect-ratio: 1;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            font-weight: 600;
        }}
        
        .heatmap-legend {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }}
        
        .heatmap-label {{
            font-size: 0.875em;
            color: var(--text-muted);
        }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        
        .progress-fill.buy {{ background: var(--success-color); }}
        .progress-fill.sell {{ background: var(--danger-color); }}
        .progress-fill.hold {{ background: var(--text-muted); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Análisis de Señales L1</h1>
            <p>Reporte generado el {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        </header>
        
        <!-- Métricas Principales -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Total Trades</h3>
                <div class="value">{len(trades_df)}</div>
            </div>
            <div class="metric-card {'success' if trades_df['pnl'].sum() > 0 else 'danger'}">
                <h3>PnL Total</h3>
                <div class="value">${trades_df['pnl'].sum():.2f}</div>
            </div>
            <div class="metric-card">
                <h3>Win Rate Global</h3>
                <div class="value">{len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100:.1f}%</div>
            </div>
            <div class="metric-card">
                <h3>Confidence-PnL Corr</h3>
                <div class="value">{confidence_corr:.3f}</div>
            </div>
        </div>
        
        <!-- Análisis por Umbrales -->
        <div class="section">
            <h2>🎯 Análisis por Umbrales de Confidence</h2>
            <table>
                <thead>
                    <tr>
                        <th>Umbral</th>
                        <th>Total Señales</th>
                        <th>Win Rate</th>
                        <th>PnL Total</th>
                        <th>PnL Promedio</th>
                        <th>Profit Factor</th>
                        <th>Sharpe Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_threshold_rows(confidence_analyses)}
                </tbody>
            </table>
        </div>
        
        <!-- Sesgo de Dirección -->
        <div class="section">
            <h2>⚖️ Sesgo de Dirección</h2>
            <div class="metrics-grid">
                <div class="metric-card success">
                    <h3>Señales BUY</h3>
                    <div class="value">{direction_bias.total_buy_signals}</div>
                    <div class="progress-bar">
                        <div class="progress-fill buy" style="width: {direction_bias.buy_percentage * 100}%"></div>
                    </div>
                    <small>{direction_bias.buy_percentage:.1%} del total</small>
                </div>
                <div class="metric-card danger">
                    <h3>Señales SELL</h3>
                    <div class="value">{direction_bias.total_sell_signals}</div>
                    <div class="progress-bar">
                        <div class="progress-fill sell" style="width: {direction_bias.sell_percentage * 100}%"></div>
                    </div>
                    <small>{direction_bias.sell_percentage:.1%} del total</small>
                </div>
                <div class="metric-card">
                    <h3>Señales HOLD</h3>
                    <div class="value">{direction_bias.total_hold_signals}</div>
                    <div class="progress-bar">
                        <div class="progress-fill hold" style="width: {direction_bias.hold_percentage * 100}%"></div>
                    </div>
                    <small>{direction_bias.hold_percentage:.1%} del total</small>
                </div>
            </div>
            
            <table style="margin-top: 30px;">
                <thead>
                    <tr>
                        <th>Dirección</th>
                        <th>Win Rate</th>
                        <th>PnL Promedio</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>BUY</td>
                        <td>{direction_bias.buy_win_rate:.2%}</td>
                        <td>${direction_bias.buy_avg_pnl:.2f}</td>
                    </tr>
                    <tr>
                        <td>SELL</td>
                        <td>{direction_bias.sell_win_rate:.2%}</td>
                        <td>${direction_bias.sell_avg_pnl:.2f}</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="alert {'alert-warning' if direction_bias.bias_strength in ['STRONG', 'MODERATE'] else 'alert-success'}">
                <strong>Sesgo Detectado:</strong> {direction_bias.bias_direction} ({direction_bias.bias_strength})
            </div>
        </div>
        
        <!-- Autocorrelación -->
        <div class="section">
            <h2>🔄 Autocorrelación de Señales</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Lag 1 Correlación</h3>
                    <div class="value">{autocorr_result.lag_1_correlation:.3f}</div>
                </div>
                <div class="metric-card">
                    <h3>Repetition Rate</h3>
                    <div class="value">{autocorr_result.repetition_rate:.1%}</div>
                </div>
                <div class="metric-card">
                    <h3>Señales Consecutivas</h3>
                    <div class="value">{autocorr_result.consecutive_same_signals}</div>
                </div>
            </div>
            
            <table style="margin-top: 20px;">
                <thead>
                    <tr>
                        <th>Lag</th>
                        <th>Correlación</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Lag 1</td>
                        <td>{autocorr_result.lag_1_correlation:.3f}</td>
                    </tr>
                    <tr>
                        <td>Lag 2</td>
                        <td>{autocorr_result.lag_2_correlation:.3f}</td>
                    </tr>
                    <tr>
                        <td>Lag 3</td>
                        <td>{autocorr_result.lag_3_correlation:.3f}</td>
                    </tr>
                    <tr>
                        <td>Lag 5</td>
                        <td>{autocorr_result.lag_5_correlation:.3f}</td>
                    </tr>
                    <tr>
                        <td>Lag 10</td>
                        <td>{autocorr_result.lag_10_correlation:.3f}</td>
                    </tr>
                </tbody>
            </table>
            
            <div class="alert {'alert-warning' if autocorr_result.lag_1_correlation > 0.5 else 'alert-success' if autocorr_result.lag_1_correlation < 0.2 else ''}">
                {autocorr_result.interpretation}
            </div>
        </div>
        
        <!-- Gráficos -->
        <div class="section">
            <h2>📈 Gráficos</h2>
            
            <div class="chart-grid">
                <div>
                    <h3>Equity Curve</h3>
                    <div class="chart-container">
                        <canvas id="equityChart"></canvas>
                    </div>
                </div>
                
                <div>
                    <h3>Distribución de Confidence</h3>
                    <div class="chart-container">
                        <canvas id="confidenceChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 30px;">
                <h3>Performance por Hora del Día</h3>
                <div class="chart-container">
                    <canvas id="hourlyChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Equity Curve
        const equityCtx = document.getElementById('equityChart').getContext('2d');
        new Chart(equityCtx, {{
            type: 'line',
            data: {{
                labels: {equity_data['labels']},
                datasets: [{{
                    label: 'Cumulative PnL',
                    data: {equity_data['values']},
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: true
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: false,
                        grid: {{
                            color: 'rgba(0,0,0,0.05)'
                        }}
                    }},
                    x: {{
                        grid: {{
                            display: false
                        }}
                    }}
                }}
            }}
        }});
        
        // Confidence Distribution
        const confCtx = document.getElementById('confidenceChart').getContext('2d');
        new Chart(confCtx, {{
            type: 'bar',
            data: {{
                labels: {confidence_dist['labels']},
                datasets: [
                    {{
                        label: 'Win Rate',
                        data: {confidence_dist['win_rates']},
                        backgroundColor: '#10b981',
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Total Trades',
                        data: {confidence_dist['counts']},
                        backgroundColor: 'rgba(37, 99, 235, 0.5)',
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Win Rate'
                        }},
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Total Trades'
                        }},
                        grid: {{
                            drawOnChartArea: false
                        }}
                    }}
                }}
            }}
        }});
        
        // Hourly Heatmap
        const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
        new Chart(hourlyCtx, {{
            type: 'bar',
            data: {{
                labels: {hourly_data['labels']},
                datasets: [
                    {{
                        label: 'Win Rate por Hora',
                        data: {hourly_data['win_rates']},
                        backgroundColor: {hourly_data['colors']},
                        borderColor: {hourly_data['border_colors']},
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Win Rate por Hora del Día (0-23h)'
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }},
                        title: {{
                            display: true,
                            text: 'Win Rate'
                        }}
                    }},
                    x: {{
                        title: {{
                            display: true,
                            text: 'Hora del Día'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Reporte HTML generado: {output_file}")
        return output_file
    
    def _generate_threshold_rows(self, analyses: List[ConfidenceAnalysis]) -> str:
        """Genera filas HTML para la tabla de umbrales"""
        rows = ""
        for analysis in analyses:
            rows += f"""
                    <tr>
                        <td><strong>≥ {analysis.threshold:.1f}</strong></td>
                        <td>{analysis.total_signals}</td>
                        <td>{analysis.win_rate:.1%}</td>
                        <td>${analysis.total_pnl:.2f}</td>
                        <td>${analysis.avg_pnl:.2f}</td>
                        <td>{analysis.profit_factor:.2f}</td>
                        <td>{analysis.sharpe_ratio:.2f}</td>
                    </tr>"""
        return rows
    
    def _prepare_equity_data(self, trades_df: pd.DataFrame) -> Dict:
        """Prepara datos para el equity curve"""
        if trades_df.empty:
            return {'labels': '[]', 'values': '[]'}
        
        # Ordenar por timestamp
        df = trades_df.sort_values('timestamp')
        
        # Calcular PnL acumulado
        cumulative = df['pnl'].cumsum().values.tolist()
        labels = [str(t) for t in df['timestamp'].values]
        
        return {
            'labels': str(labels).replace("'", '"'),
            'values': str(cumulative)
        }
    
    def _prepare_confidence_distribution(self, trades_df: pd.DataFrame) -> Dict:
        """Prepara datos para la distribución de confidence"""
        if trades_df.empty:
            return {'labels': '[]', 'win_rates': '[]', 'counts': '[]'}
        
        # Crear bins
        trades_df = trades_df.copy()
        trades_df['confidence_bin'] = pd.cut(trades_df['confidence'], bins=8)
        
        distribution = trades_df.groupby('confidence_bin').agg({
            'pnl': ['count', lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0]
        }).reset_index()
        
        distribution.columns = ['bin', 'count', 'win_rate']
        
        labels = [str(b) for b in distribution['bin'].values]
        win_rates = distribution['win_rate'].values.tolist()
        counts = distribution['count'].values.tolist()
        
        return {
            'labels': str(labels).replace("'", '"'),
            'win_rates': str(win_rates),
            'counts': str(counts)
        }
    
    def _prepare_hourly_heatmap(self, trades_df: pd.DataFrame) -> Dict:
        """Prepara datos para el heatmap por hora"""
        if trades_df.empty or 'timestamp' not in trades_df.columns:
            return {'labels': '[]', 'win_rates': '[]', 'colors': '[]', 'border_colors': '[]'}
        
        # Extraer hora
        trades_df = trades_df.copy()
        trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
        
        # Agrupar por hora
        hourly = trades_df.groupby('hour').agg({
            'pnl': ['count', lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0]
        }).reset_index()
        
        hourly.columns = ['hour', 'count', 'win_rate']
        
        # Asegurar todas las horas
        all_hours = pd.DataFrame({'hour': range(24)})
        hourly = all_hours.merge(hourly, on='hour', how='left').fillna(0)
        
        labels = [f"{h:02d}h" for h in range(24)]
        win_rates = hourly['win_rate'].values.tolist()
        
        # Generar colores basados en win rate
        colors = []
        border_colors = []
        for wr in win_rates:
            if wr >= 0.6:
                colors.append("'rgba(16, 185, 129, 0.8)'")
                border_colors.append("'rgba(16, 185, 129, 1)'")
            elif wr >= 0.5:
                colors.append("'rgba(245, 158, 11, 0.8)'")
                border_colors.append("'rgba(245, 158, 11, 1)'")
            else:
                colors.append("'rgba(239, 68, 68, 0.8)'")
                border_colors.append("'rgba(239, 68, 68, 1)'")
        
        return {
            'labels': str(labels).replace("'", '"'),
            'win_rates': str(win_rates),
            'colors': '[' + ','.join(colors) + ']',
            'border_colors': '[' + ','.join(border_colors) + ']'
        }


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def run_analysis(
    signals_csv: Optional[str] = None,
    trades_csv: Optional[str] = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Ejecuta el análisis completo
    
    Args:
        signals_csv: Ruta al CSV de señales (opcional, busca el más reciente)
        trades_csv: Ruta al CSV de trades (opcional, busca el más reciente)
        output_dir: Directorio para guardar el reporte
    
    Returns:
        Dict con resultados del análisis
    """
    output_dir = output_dir or ANALYSIS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🔍 INICIANDO ANÁLISIS DE SEÑALES L1")
    logger.info("=" * 80)
    
    # Cargar datos
    data_loader = DataLoader()
    
    # Buscar archivos si no se especificaron
    if not signals_csv or not trades_csv:
        latest_signals, latest_trades = data_loader.find_latest_files()
        signals_csv = signals_csv or latest_signals
        trades_csv = trades_csv or latest_trades
    
    if not signals_csv and not trades_csv:
        logger.error("❌ No se encontraron archivos CSV para analizar")
        return {}
    
    # Cargar DataFrames
    signals_df = data_loader.load_signals_csv(signals_csv) if signals_csv else None
    trades_df = data_loader.load_trades_csv(trades_csv) if trades_csv else None
    
    if trades_df is None or trades_df.empty:
        logger.error("❌ No hay datos de trades para analizar")
        return {}
    
    logger.info(f"📊 Datos cargados: {len(trades_df)} trades")
    
    # 1. Análisis de Confidence Scores
    logger.info("\n📈 Analizando distribución de confidence scores...")
    confidence_analyzer = ConfidenceAnalyzer(trades_df)
    confidence_analyses = confidence_analyzer.analyze_by_thresholds([0.6, 0.7, 0.8])
    confidence_corr = confidence_analyzer.correlation_confidence_pnl()
    
    logger.info(f"  Correlación Confidence-PnL: {confidence_corr:.3f}")
    for analysis in confidence_analyses:
        logger.info(f"  Umbral ≥{analysis.threshold:.1f}: {analysis.win_rate:.1%} win rate ({analysis.total_signals} trades)")
    
    # 2. Análisis de Sesgo de Dirección
    logger.info("\n⚖️ Analizando sesgo de dirección...")
    bias_analyzer = DirectionBiasAnalyzer(signals_df, trades_df)
    direction_bias = bias_analyzer.analyze()
    
    logger.info(f"  BUY: {direction_bias.buy_percentage:.1%} (Win rate: {direction_bias.buy_win_rate:.1%})")
    logger.info(f"  SELL: {direction_bias.sell_percentage:.1%} (Win rate: {direction_bias.sell_win_rate:.1%})")
    logger.info(f"  Sesgo: {direction_bias.bias_direction} ({direction_bias.bias_strength})")
    
    # 3. Análisis de Autocorrelación
    logger.info("\n🔄 Analizando autocorrelación de señales...")
    autocorr_analyzer = AutocorrelationAnalyzer(signals_df)
    autocorr_result = autocorr_analyzer.analyze()
    
    logger.info(f"  Lag-1 Correlación: {autocorr_result.lag_1_correlation:.3f}")
    logger.info(f"  Repetition Rate: {autocorr_result.repetition_rate:.1%}")
    logger.info(f"  {autocorr_result.interpretation}")
    
    # 4. Generar Reporte HTML
    logger.info("\n📄 Generando reporte HTML...")
    report_generator = HTMLReportGenerator(output_dir)
    report_file = report_generator.generate(
        signals_df=signals_df,
        trades_df=trades_df,
        confidence_analyses=confidence_analyses,
        direction_bias=direction_bias,
        autocorr_result=autocorr_result,
        confidence_corr=confidence_corr
    )
    
    # Guardar resumen JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'signals_file': signals_csv,
        'trades_file': trades_csv,
        'total_trades': len(trades_df),
        'total_pnl': float(trades_df['pnl'].sum()),
        'global_win_rate': float(len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)),
        'confidence_correlation': float(confidence_corr),
        'threshold_analysis': [a.to_dict() for a in confidence_analyses],
        'direction_bias': direction_bias.to_dict(),
        'autocorrelation': autocorr_result.to_dict(),
        'report_file': report_file
    }
    
    summary_file = os.path.join(output_dir, f"l1_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    import json
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"💾 Resumen guardado en: {summary_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ANÁLISIS COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"📄 Reporte HTML: {report_file}")
    
    return summary


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Análisis de Señales L1 - Análisis post-backtest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Analizar archivos más recientes automáticamente
  python backtesting/analyze_l1_signals.py

  # Especificar archivos manualmente
  python backtesting/analyze_l1_signals.py --signals-csv results/l1_backtest_signals_20260221_120000.csv --trades-csv results/l1_backtest_trades_20260221_120000.csv

  # Cambiar directorio de salida
  python backtesting/analyze_l1_signals.py --output-dir mi_analisis/
        """
    )
    
    parser.add_argument(
        '--signals-csv',
        help='Ruta al CSV de señales generado por backtest_l1_standalone.py'
    )
    parser.add_argument(
        '--trades-csv',
        help='Ruta al CSV de trades generado por backtest_l1_standalone.py'
    )
    parser.add_argument(
        '--output-dir',
        default=ANALYSIS_DIR,
        help=f'Directorio para guardar el reporte (default: {ANALYSIS_DIR})'
    )
    
    args = parser.parse_args()
    
    try:
        result = run_analysis(
            signals_csv=args.signals_csv,
            trades_csv=args.trades_csv,
            output_dir=args.output_dir
        )
        
        if result:
            print("\n✅ Análisis completado exitosamente")
            print(f"📄 Reporte: {result.get('report_file', 'N/A')}")
            return 0
        else:
            print("\n❌ El análisis falló")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Análisis interrumpido por el usuario")
        return 130
    except Exception as e:
        print(f"\n❌ Error en análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
