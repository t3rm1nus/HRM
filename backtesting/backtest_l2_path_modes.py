#!/usr/bin/env python3
"""
Backtesting L2 Path Modes - Análisis de rendimiento por modo de path

Este script:
1. Instancia l2_tactic/path_mode_generator.py y l2_tactic/path_modes.py
2. Para datos históricos de 6 meses, clasifica cada período en su path_mode correspondiente
3. Para cada path_mode, calcula métricas de trading independientemente
4. Identifica qué path_modes son rentables y cuáles destruyen valor
5. Genera una tabla: path_mode | avg_return | win_rate | num_periods | pct_time_active
6. Detecta si hay path_modes que nunca se activan (código muerto) o que se activan demasiado frecuentemente (posible bug)

IMPORTANTE: Aísla el comportamiento de cada PATH_MODE para entender su contribución individual
"""

import os
import sys
import argparse
import csv
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from enum import Enum

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importar componentes de path_mode
from l2_tactic.path_mode_generator import PathModeSignalGenerator
from l2_tactic.path_modes import (
    PATH2Processor,
    apply_pure_trend_following,
    apply_hybrid_mode,
    apply_full_l3_dominance
)

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

DEFAULT_INITIAL_BALANCE = 3000.0  # USDT
DEFAULT_FEE = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Path modes disponibles
PATH_MODES = ['PATH1', 'PATH2', 'PATH3']

# Umbrales para detección de anomalías
MIN_ACTIVATION_PCT = 5.0  # Mínimo 5% del tiempo activo para no ser "código muerto"
MAX_ACTIVATION_PCT = 80.0  # Máximo 80% para no ser "sobre-activo"

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PathModePeriod:
    """Representa un período clasificado en un path_mode"""
    start_time: datetime
    end_time: datetime
    path_mode: str
    regime: str
    l3_confidence: float
    
    # Señales durante el período
    l1_l2_signals: List[Dict] = field(default_factory=list)
    
    # Resultados
    trades: List[Dict] = field(default_factory=list)
    total_return: float = 0.0
    num_trades: int = 0
    winning_trades: int = 0


@dataclass
class PathModeMetrics:
    """Métricas para un path_mode específico"""
    path_mode: str
    num_periods: int = 0
    total_time_minutes: float = 0.0
    pct_time_active: float = 0.0
    
    # Métricas de trading
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    total_pnl: float = 0.0
    
    # Métricas adicionales
    avg_hold_time_minutes: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Señales
    signals_generated: int = 0
    signals_blocked: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'path_mode': self.path_mode,
            'num_periods': self.num_periods,
            'total_time_minutes': f"{self.total_time_minutes:.1f}",
            'pct_time_active': f"{self.pct_time_active:.2f}%",
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'avg_return': f"{self.avg_return:.4f}",
            'total_pnl': f"{self.total_pnl:.2f}",
            'avg_hold_time_minutes': f"{self.avg_hold_time_minutes:.1f}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'signals_generated': self.signals_generated,
            'signals_blocked': self.signals_blocked
        }


@dataclass
class TradeRecord:
    """Registro de una operación"""
    trade_id: int
    timestamp: datetime
    symbol: str
    path_mode: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    confidence: float = 0.5
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    hold_time_minutes: float = 0.0
    exit_timestamp: Optional[datetime] = None
    is_open: bool = True
    regime: str = "unknown"


@dataclass
class PathModeBacktestResult:
    """Resultado completo del backtest de path modes"""
    timestamp: datetime
    start_date: datetime
    end_date: datetime
    total_periods_analyzed: int
    
    # Métricas por path_mode
    path_mode_metrics: Dict[str, PathModeMetrics]
    
    # Resumen comparativo
    best_path_mode: str
    worst_path_mode: str
    most_frequent_path_mode: str
    least_frequent_path_mode: str
    
    # Anomalías detectadas
    dead_code_paths: List[str]
    overactive_paths: List[str]
    
    def to_summary_table(self) -> pd.DataFrame:
        """Genera tabla resumen de path_modes"""
        data = []
        for path_mode, metrics in self.path_mode_metrics.items():
            data.append({
                'path_mode': path_mode,
                'avg_return': metrics.avg_return,
                'win_rate': metrics.win_rate,
                'num_periods': metrics.num_periods,
                'pct_time_active': metrics.pct_time_active,
                'total_pnl': metrics.total_pnl,
                'total_trades': metrics.total_trades,
                'sharpe_ratio': metrics.sharpe_ratio
            })
        return pd.DataFrame(data)


# =============================================================================
# MOTOR DE BACKTESTING POR PATH MODE
# =============================================================================

class PathModeBacktestEngine:
    """Motor de backtesting para analizar rendimiento por path_mode"""
    
    def __init__(
        self,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE
    ):
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        
        # Inicializar componentes
        self.path_mode_generator = PathModeSignalGenerator()
        self.path2_processor = PATH2Processor()
        
        # Estado del backtest
        self.reset_state()
    
    def reset_state(self):
        """Resetea el estado del motor"""
        # Períodos clasificados por path_mode
        self.path_mode_periods: Dict[str, List[PathModePeriod]] = {
            'PATH1': [],
            'PATH2': [],
            'PATH3': []
        }
        
        # Métricas acumuladas por path_mode
        self.path_mode_stats: Dict[str, Dict] = {
            'PATH1': {
                'periods': [],
                'trades': [],
                'returns': [],
                'hold_times': []
            },
            'PATH2': {
                'periods': [],
                'trades': [],
                'returns': [],
                'hold_times': []
            },
            'PATH3': {
                'periods': [],
                'trades': [],
                'returns': [],
                'hold_times': []
            }
        }
        
        # Posiciones abiertas
        self.open_positions: Dict[str, TradeRecord] = {}
        
        # Balance y equity por path_mode
        self.balances: Dict[str, float] = {
            'PATH1': self.initial_balance,
            'PATH2': self.initial_balance,
            'PATH3': self.initial_balance
        }
        
        self.equity_curves: Dict[str, List[Tuple[datetime, float]]] = {
            'PATH1': [],
            'PATH2': [],
            'PATH3': []
        }
        
        # Contadores
        self.trade_counter = 0
        self.total_periods = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def classify_period(
        self,
        timestamp: datetime,
        symbol: str,
        l1_l2_signal: str,
        l1_l2_confidence: float,
        l3_regime: str,
        l3_confidence: float,
        l3_signal: str = 'hold',
        market_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Clasifica un período en su path_mode y genera señal correspondiente
        
        Returns:
            Dict con path_mode asignado y señal generada
        """
        # Construir contexto L3
        l3_context = {
            'regime': l3_regime.lower(),
            'l3_signal': l3_signal.lower(),
            'l3_confidence': l3_confidence,
            'setup_type': None,
            'allow_l2_signals': l3_confidence > 0.6,
            'allow_setup_trades': l3_confidence > 0.7,
            'setup_max_allocation': 0.10
        }
        
        # Determinar path_mode basado en el régimen y confianza
        path_mode = self._determine_path_mode(l3_regime, l3_confidence)
        
        # Generar señal usando el generador de path modes
        signal = self.path_mode_generator.generate_signal(
            symbol=symbol,
            l1_l2_signal=l1_l2_signal,
            l3_context=l3_context
        )
        
        # Actualizar estadísticas
        self.total_periods += 1
        if self.start_time is None:
            self.start_time = timestamp
        self.end_time = timestamp
        
        # Registrar período
        period = PathModePeriod(
            start_time=timestamp,
            end_time=timestamp,
            path_mode=path_mode,
            regime=l3_regime,
            l3_confidence=l3_confidence,
            l1_l2_signals=[{
                'timestamp': timestamp,
                'signal': l1_l2_signal,
                'confidence': l1_l2_confidence
            }]
        )
        
        self.path_mode_periods[path_mode].append(period)
        self.path_mode_stats[path_mode]['periods'].append({
            'timestamp': timestamp,
            'regime': l3_regime,
            'confidence': l3_confidence
        })
        
        return {
            'path_mode': path_mode,
            'signal': signal,
            'period': period
        }
    
    def _determine_path_mode(self, regime: str, l3_confidence: float) -> str:
        """
        Determina qué path_mode usar basado en el régimen y confianza L3
        
        Lógica:
        - PATH1: Régimen claro (bull/bear) con alta confianza (>0.7)
        - PATH2: Régimen mixto o confianza moderada (0.4-0.7)
        - PATH3: Condiciones extremas o baja confianza (<0.4)
        """
        regime_lower = regime.lower()
        
        # PATH3: Full L3 dominance para condiciones extremas
        if l3_confidence < 0.4 or regime_lower in ['extreme_volatility', 'crash', 'pump']:
            return 'PATH3'
        
        # PATH1: Pure trend following para tendencias claras con alta confianza
        if regime_lower in ['bull', 'bullish', 'bear', 'bearish'] and l3_confidence >= 0.7:
            return 'PATH1'
        
        # PATH2: Hybrid para todo lo demás (range, neutral, confianza moderada)
        return 'PATH2'
    
    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        path_mode: str,
        signal: Dict[str, Any],
        price: float,
        market_data: Optional[Dict] = None
    ) -> Optional[TradeRecord]:
        """
        Ejecuta un trade según la señal del path_mode
        
        Returns:
            TradeRecord si se ejecutó el trade, None si no
        """
        action = signal.get('action', 'HOLD').upper()
        
        if action == 'HOLD':
            return None
        
        # Actualizar equity curves
        self._update_equity_curves(timestamp)
        
        if action == 'BUY':
            # Verificar si ya hay posición abierta
            position_key = f"{symbol}_{path_mode}"
            if position_key in self.open_positions:
                return None
            
            # Calcular tamaño de posición
            confidence = signal.get('confidence', 0.5)
            size_multiplier = signal.get('size_multiplier', 1.0)
            
            notional = self.balances[path_mode] * 0.1 * confidence * size_multiplier
            quantity = notional / price if price > 0 else 0
            
            if quantity <= 0:
                return None
            
            # Aplicar slippage y fees
            execution_price = price * (1 + self.slippage)
            fees = notional * self.fee
            
            # Actualizar balance
            self.balances[path_mode] -= (notional + fees)
            
            # Crear trade
            self.trade_counter += 1
            trade = TradeRecord(
                trade_id=self.trade_counter,
                timestamp=timestamp,
                symbol=symbol,
                path_mode=path_mode,
                side='BUY',
                entry_price=execution_price,
                quantity=quantity,
                confidence=confidence,
                fees=fees,
                regime=signal.get('regime', 'unknown')
            )
            
            self.open_positions[position_key] = trade
            self.path_mode_stats[path_mode]['trades'].append({
                'type': 'open',
                'trade': trade
            })
            
            return trade
            
        elif action == 'SELL':
            # Buscar posición abierta
            position_key = f"{symbol}_{path_mode}"
            if position_key not in self.open_positions:
                return None
            
            position = self.open_positions[position_key]
            
            # Aplicar slippage y fees
            execution_price = price * (1 - self.slippage)
            exit_value = execution_price * position.quantity
            exit_fees = exit_value * self.fee
            
            # Calcular PnL
            entry_value = position.entry_price * position.quantity
            pnl = exit_value - entry_value - position.fees - exit_fees
            pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
            
            # Actualizar balance
            self.balances[path_mode] += (exit_value - exit_fees)
            
            # Cerrar posición
            hold_time = (timestamp - position.timestamp).total_seconds() / 60
            
            position.exit_price = execution_price
            position.pnl = pnl
            position.pnl_pct = pnl_pct
            position.exit_timestamp = timestamp
            position.hold_time_minutes = hold_time
            position.is_open = False
            position.fees += exit_fees
            
            # Registrar resultado
            self.path_mode_stats[path_mode]['returns'].append(pnl_pct)
            self.path_mode_stats[path_mode]['hold_times'].append(hold_time)
            self.path_mode_stats[path_mode]['trades'].append({
                'type': 'close',
                'trade': position,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            
            del self.open_positions[position_key]
            
            return position
        
        return None
    
    def _update_equity_curves(self, timestamp: datetime):
        """Actualiza las curvas de equity para cada path_mode"""
        for path_mode in PATH_MODES:
            equity = self.balances[path_mode]
            
            # Añadir valor de posiciones abiertas
            for key, position in self.open_positions.items():
                if key.endswith(f"_{path_mode}"):
                    # Usar último precio conocido o precio de entrada
                    position_value = position.quantity * position.entry_price
                    equity += position_value
            
            self.equity_curves[path_mode].append((timestamp, equity))
    
    def close_all_positions(self, timestamp: datetime, final_prices: Dict[str, float]):
        """Cierra todas las posiciones abiertas al final del backtest"""
        for key, position in list(self.open_positions.items()):
            symbol = position.symbol
            if symbol in final_prices:
                # Crear señal de SELL
                signal = {'action': 'SELL', 'confidence': 0.5}
                path_mode = position.path_mode
                
                self.execute_trade(
                    timestamp=timestamp,
                    symbol=symbol,
                    path_mode=path_mode,
                    signal=signal,
                    price=final_prices[symbol]
                )
    
    def calculate_metrics(self) -> PathModeBacktestResult:
        """Calcula métricas para cada path_mode"""
        path_mode_metrics = {}
        
        # Calcular tiempo total del backtest
        total_time_minutes = 0
        if self.start_time and self.end_time:
            total_time_minutes = (self.end_time - self.start_time).total_seconds() / 60
        
        for path_mode in PATH_MODES:
            stats = self.path_mode_stats[path_mode]
            periods = self.path_mode_periods[path_mode]
            
            # Métricas básicas
            num_periods = len(periods)
            
            # Tiempo activo (estimado basado en períodos)
            # Asumimos que cada período dura 1 hora (60 minutos) por defecto
            period_duration = 60  # minutos
            total_time_active = num_periods * period_duration
            pct_time_active = (total_time_active / total_time_minutes * 100) if total_time_minutes > 0 else 0
            
            # Métricas de trading
            trades = [t for t in stats['trades'] if t['type'] == 'close']
            total_trades = len(trades)
            
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            losing_trades = total_trades - winning_trades
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Retornos
            returns = stats['returns']
            avg_return = np.mean(returns) if returns else 0
            total_pnl = sum(t['pnl'] for t in trades) if trades else 0
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Max drawdown
            max_drawdown = 0
            if self.equity_curves[path_mode]:
                peak = self.initial_balance
                for _, equity in self.equity_curves[path_mode]:
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak if peak > 0 else 0
                    if dd > max_drawdown:
                        max_drawdown = dd
            
            # Sharpe ratio
            sharpe_ratio = 0
            if returns:
                returns_array = np.array(returns)
                if np.std(returns_array) > 0:
                    sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            
            # Hold time promedio
            hold_times = stats['hold_times']
            avg_hold_time = np.mean(hold_times) if hold_times else 0
            
            metrics = PathModeMetrics(
                path_mode=path_mode,
                num_periods=num_periods,
                total_time_minutes=total_time_active,
                pct_time_active=pct_time_active,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                avg_return=avg_return,
                total_pnl=total_pnl,
                avg_hold_time_minutes=avg_hold_time,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio
            )
            
            path_mode_metrics[path_mode] = metrics
        
        # Determinar mejores/peores
        sorted_by_pnl = sorted(
            path_mode_metrics.items(),
            key=lambda x: x[1].total_pnl,
            reverse=True
        )
        
        sorted_by_frequency = sorted(
            path_mode_metrics.items(),
            key=lambda x: x[1].pct_time_active,
            reverse=True
        )
        
        best_path_mode = sorted_by_pnl[0][0] if sorted_by_pnl else 'N/A'
        worst_path_mode = sorted_by_pnl[-1][0] if sorted_by_pnl else 'N/A'
        most_frequent_path_mode = sorted_by_frequency[0][0] if sorted_by_frequency else 'N/A'
        least_frequent_path_mode = sorted_by_frequency[-1][0] if sorted_by_frequency else 'N/A'
        
        # Detectar anomalías
        dead_code_paths = [
            pm for pm, metrics in path_mode_metrics.items()
            if metrics.pct_time_active < MIN_ACTIVATION_PCT
        ]
        
        overactive_paths = [
            pm for pm, metrics in path_mode_metrics.items()
            if metrics.pct_time_active > MAX_ACTIVATION_PCT
        ]
        
        return PathModeBacktestResult(
            timestamp=datetime.now(),
            start_date=self.start_time or datetime.now(),
            end_date=self.end_time or datetime.now(),
            total_periods_analyzed=self.total_periods,
            path_mode_metrics=path_mode_metrics,
            best_path_mode=best_path_mode,
            worst_path_mode=worst_path_mode,
            most_frequent_path_mode=most_frequent_path_mode,
            least_frequent_path_mode=least_frequent_path_mode,
            dead_code_paths=dead_code_paths,
            overactive_paths=overactive_paths
        )


# =============================================================================
# CARGA DE DATOS HISTÓRICOS
# =============================================================================

def load_historical_data(
    symbol: str = 'BTCUSDT',
    months: int = 6,
    data_dir: str = DATA_DIR
) -> pd.DataFrame:
    """
    Carga datos históricos de precios y genera señales simuladas
    
    Args:
        symbol: Símbolo a cargar
        months: Número de meses hacia atrás
        data_dir: Directorio de datos
    
    Returns:
        DataFrame con datos OHLCV y señales L1/L3 simuladas
    """
    logger.info(f"📊 Cargando datos históricos para {symbol} ({months} meses)")
    
    # Buscar archivos parquet disponibles
    cache_dir = os.path.join(data_dir, 'cache')
    
    if not os.path.exists(cache_dir):
        logger.warning(f"⚠️ Directorio de cache no encontrado: {cache_dir}")
        # Generar datos sintéticos para testing
        return generate_synthetic_data(symbol, months)
    
    # Intentar cargar archivos parquet
    parquet_files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet') and symbol in f]
    
    if not parquet_files:
        logger.warning(f"⚠️ No se encontraron archivos parquet para {symbol}")
        return generate_synthetic_data(symbol, months)
    
    # Cargar y combinar datos
    all_data = []
    for file in sorted(parquet_files):
        file_path = os.path.join(cache_dir, file)
        try:
            df = pd.read_parquet(file_path)
            # Reset index to make timestamp a column if it's the index
            if df.index.name == 'timestamp' or 'timestamp' not in df.columns:
                df = df.reset_index()
            all_data.append(df)
            logger.info(f"  ✅ Cargado: {file}")
        except Exception as e:
            logger.warning(f"  ⚠️ Error cargando {file}: {e}")
    
    if not all_data:
        return generate_synthetic_data(symbol, months)
    
    # Combinar datos
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ordenar por fecha
    if 'timestamp' in combined_df.columns:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp')
    elif 'open_time' in combined_df.columns:
        combined_df['timestamp'] = pd.to_datetime(combined_df['open_time'], unit='ms')
        combined_df = combined_df.sort_values('timestamp')
    
    # Filtrar últimos N meses
    end_date = combined_df['timestamp'].max()
    start_date = end_date - pd.DateOffset(months=months)
    combined_df = combined_df[combined_df['timestamp'] >= start_date]
    
    logger.info(f"✅ Datos cargados: {len(combined_df)} registros desde {start_date} hasta {end_date}")
    
    return combined_df


def generate_synthetic_data(symbol: str, months: int) -> pd.DataFrame:
    """Genera datos sintéticos para testing cuando no hay datos reales"""
    logger.info(f"🎲 Generando datos sintéticos para {symbol} ({months} meses)")
    
    # Generar datos de precios sintéticos
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(months=months)
    
    # Frecuencia horaria
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Precio inicial
    initial_price = 50000.0 if 'BTC' in symbol else 3000.0
    
    # Generar precios con tendencia y ruido
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.02, len(date_range))
    
    # Añadir algunos régimenes diferentes
    n = len(date_range)
    
    # Bull market (primer tercio)
    returns[:n//3] += 0.001
    
    # Bear market (segundo tercio)
    returns[n//3:2*n//3] -= 0.001
    
    # Range (último tercio)
    returns[2*n//3:] = np.random.normal(0, 0.01, n - 2*n//3)
    
    # Calcular precios
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Crear DataFrame OHLCV
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + abs(np.random.normal(0, 0.01, n))),
        'low': prices * (1 - abs(np.random.normal(0, 0.01, n))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    logger.info(f"✅ Datos sintéticos generados: {len(df)} registros")
    
    return df


def generate_l1_l3_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera señales L1/L2 y contexto L3 basado en indicadores técnicos
    
    Args:
        df: DataFrame con datos OHLCV
    
    Returns:
        DataFrame con columnas adicionales de señales
    """
    logger.info("📈 Generando señales L1/L2 y contexto L3...")
    
    df = df.copy()
    
    # Calcular indicadores básicos
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(365 * 24)
    
    # Generar señales L1/L2 basadas en cruces de medias
    df['l1_l2_signal'] = 'HOLD'
    df.loc[df['close'] > df['sma_20'], 'l1_l2_signal'] = 'BUY'
    df.loc[df['close'] < df['sma_20'], 'l1_l2_signal'] = 'SELL'
    
    # Confianza L1/L2 basada en distancia a la media
    df['l1_l2_confidence'] = abs(df['close'] - df['sma_20']) / df['sma_20']
    df['l1_l2_confidence'] = df['l1_l2_confidence'].clip(0.3, 0.9)
    
    # Determinar régimen L3
    df['l3_regime'] = 'neutral'
    
    # Bull: precio > SMA50 y RSI > 55
    bull_condition = (df['close'] > df['sma_50']) & (df['rsi'] > 55)
    df.loc[bull_condition, 'l3_regime'] = 'bull'
    
    # Bear: precio < SMA50 y RSI < 45
    bear_condition = (df['close'] < df['sma_50']) & (df['rsi'] < 45)
    df.loc[bear_condition, 'l3_regime'] = 'bear'
    
    # Range: RSI entre 45 y 55
    range_condition = (df['rsi'] >= 45) & (df['rsi'] <= 55)
    df.loc[range_condition, 'l3_regime'] = 'range'
    
    # Confianza L3 basada en tendencia del régimen
    df['l3_confidence'] = 0.5
    
    # Alta confianza para tendencias claras
    df.loc[bull_condition & (df['rsi'] > 65), 'l3_confidence'] = 0.85
    df.loc[bear_condition & (df['rsi'] < 35), 'l3_confidence'] = 0.85
    
    # Confianza moderada para tendencias moderadas
    df.loc[bull_condition & (df['rsi'] <= 65), 'l3_confidence'] = 0.70
    df.loc[bear_condition & (df['rsi'] >= 35), 'l3_confidence'] = 0.70
    
    # Baja confianza para range
    df.loc[range_condition, 'l3_confidence'] = 0.45
    
    # Señal L3 basada en el régimen
    df['l3_signal'] = 'hold'
    df.loc[bull_condition, 'l3_signal'] = 'buy'
    df.loc[bear_condition, 'l3_signal'] = 'sell'
    
    # Limpiar NaN
    df = df.dropna()
    
    logger.info(f"✅ Señales generadas: {len(df)} registros con señales")
    
    return df


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calcula RSI simple"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def run_path_mode_backtest(
    symbol: str = 'BTCUSDT',
    months: int = 6,
    output_dir: str = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE
) -> PathModeBacktestResult:
    """
    Ejecuta el backtest de path modes
    
    Args:
        symbol: Símbolo a analizar
        months: Meses de datos históricos
        output_dir: Directorio de salida
        initial_balance: Balance inicial
    
    Returns:
        PathModeBacktestResult con métricas por path_mode
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO BACKTEST DE PATH MODES")
    logger.info("=" * 80)
    logger.info(f"Símbolo: {symbol}")
    logger.info(f"Período: {months} meses")
    logger.info(f"Balance inicial: {initial_balance} USDT")
    
    # Cargar datos históricos
    df = load_historical_data(symbol, months)
    
    # Generar señales L1/L3
    df = generate_l1_l3_signals(df)
    
    # Inicializar motor
    engine = PathModeBacktestEngine(initial_balance=initial_balance)
    
    # Procesar cada período
    logger.info("⏳ Procesando períodos y clasificando path_modes...")
    
    progress_interval = max(1, len(df) // 10)
    
    for i, row in df.iterrows():
        timestamp = row['timestamp']
        price = row['close']
        
        # Clasificar período
        classification = engine.classify_period(
            timestamp=timestamp,
            symbol=symbol,
            l1_l2_signal=row['l1_l2_signal'],
            l1_l2_confidence=row['l1_l2_confidence'],
            l3_regime=row['l3_regime'],
            l3_confidence=row['l3_confidence'],
            l3_signal=row['l3_signal'],
            market_data={
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'volume': row['volume']
            }
        )
        
        path_mode = classification['path_mode']
        signal = classification['signal']
        
        # Ejecutar trade si hay señal
        engine.execute_trade(
            timestamp=timestamp,
            symbol=symbol,
            path_mode=path_mode,
            signal=signal,
            price=price,
            market_data=row.to_dict()
        )
        
        # Progreso
        if (i + 1) % progress_interval == 0:
            pct = (i + 1) / len(df) * 100
            logger.info(f"  Progreso: {pct:.1f}% ({i+1}/{len(df)})")
    
    # Cerrar posiciones abiertas
    logger.info("🔚 Cerrando posiciones abiertas...")
    final_prices = {symbol: df['close'].iloc[-1]}
    engine.close_all_positions(df['timestamp'].iloc[-1], final_prices)
    
    # Calcular métricas
    logger.info("📊 Calculando métricas por path_mode...")
    result = engine.calculate_metrics()
    
    # Mostrar resultados
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESULTADOS DEL BACKTEST DE PATH MODES")
    logger.info("=" * 80)
    
    logger.info(f"\n📅 Período analizado: {result.start_date} a {result.end_date}")
    logger.info(f"📈 Total períodos analizados: {result.total_periods_analyzed}")
    
    # Tabla resumen
    logger.info("\n📋 TABLA RESUMEN POR PATH MODE:")
    logger.info("-" * 100)
    summary_table = result.to_summary_table()
    
    # Formato de tabla
    header = f"{'Path Mode':<12} {'Avg Return':<12} {'Win Rate':<12} {'Periods':<10} {'% Time':<10} {'PnL':<12} {'Trades':<8} {'Sharpe':<10}"
    logger.info(header)
    logger.info("-" * 100)
    
    for _, row in summary_table.iterrows():
        line = f"{row['path_mode']:<12} {row['avg_return']:<12.4f} {row['win_rate']:<12.2%} {row['num_periods']:<10} {row['pct_time_active']:<10.2f}% {row['total_pnl']:<12.2f} {row['total_trades']:<8} {row['sharpe_ratio']:<10.2f}"
        logger.info(line)
    
    logger.info("-" * 100)
    
    # Análisis de rentabilidad
    logger.info("\n💰 ANÁLISIS DE RENTABILIDAD:")
    logger.info(f"  🥇 Mejor path_mode: {result.best_path_mode}")
    logger.info(f"  🥉 Peor path_mode: {result.worst_path_mode}")
    
    for path_mode, metrics in result.path_mode_metrics.items():
        if metrics.total_pnl > 0:
            logger.info(f"  ✅ {path_mode}: RENTABLE (+{metrics.total_pnl:.2f} USDT)")
        elif metrics.total_pnl < 0:
            logger.info(f"  ❌ {path_mode}: DESTRUYE VALOR ({metrics.total_pnl:.2f} USDT)")
        else:
            logger.info(f"  ⚪ {path_mode}: NEUTRO ({metrics.total_pnl:.2f} USDT)")
    
    # Análisis de frecuencia
    logger.info("\n📊 ANÁLISIS DE FRECUENCIA:")
    logger.info(f"  Más frecuente: {result.most_frequent_path_mode}")
    logger.info(f"  Menos frecuente: {result.least_frequent_path_mode}")
    
    # Detección de anomalías
    logger.info("\n🔍 DETECCIÓN DE ANOMALÍAS:")
    
    if result.dead_code_paths:
        logger.warning(f"  ⚠️  CÓDIGO MUERTO detectado en: {', '.join(result.dead_code_paths)}")
        logger.warning(f"      Estos path_modes están activos menos del {MIN_ACTIVATION_PCT}% del tiempo")
    else:
        logger.info(f"  ✅ No se detectó código muerto (todos activos >{MIN_ACTIVATION_PCT}%)")
    
    if result.overactive_paths:
        logger.warning(f"  🚨 POSIBLE BUG - Path modes sobre-activos: {', '.join(result.overactive_paths)}")
        logger.warning(f"      Estos path_modes están activos más del {MAX_ACTIVATION_PCT}% del tiempo")
    else:
        logger.info(f"  ✅ No se detectaron path modes sobre-activos (<{MAX_ACTIVATION_PCT}%)")
    
    # Recomendaciones
    logger.info("\n💡 RECOMENDACIONES:")
    
    # Recomendar basado en resultados
    best_metrics = result.path_mode_metrics[result.best_path_mode]
    worst_metrics = result.path_mode_metrics[result.worst_path_mode]
    
    if best_metrics.total_pnl > 0:
        logger.info(f"  → Priorizar {result.best_path_mode} (mejor rendimiento)")
    
    if worst_metrics.total_pnl < 0 and worst_metrics.pct_time_active > 20:
        logger.info(f"  → Considerar desactivar o ajustar {result.worst_path_mode} (pérdidas frecuentes)")
    
    if result.dead_code_paths:
        logger.info(f"  → Revisar implementación de: {', '.join(result.dead_code_paths)}")
    
    # Exportar resultados
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV de métricas
    csv_path = os.path.join(output_dir, f"path_mode_metrics_{symbol}_{timestamp_str}.csv")
    summary_table.to_csv(csv_path, index=False)
    logger.info(f"\n💾 Métricas exportadas a: {csv_path}")
    
    # JSON completo
    json_result = {
        'timestamp': result.timestamp.isoformat(),
        'symbol': symbol,
        'months_analyzed': months,
        'start_date': result.start_date.isoformat() if hasattr(result.start_date, 'isoformat') else str(result.start_date),
        'end_date': result.end_date.isoformat() if hasattr(result.end_date, 'isoformat') else str(result.end_date),
        'total_periods': result.total_periods_analyzed,
        'best_path_mode': result.best_path_mode,
        'worst_path_mode': result.worst_path_mode,
        'most_frequent_path_mode': result.most_frequent_path_mode,
        'least_frequent_path_mode': result.least_frequent_path_mode,
        'dead_code_paths': result.dead_code_paths,
        'overactive_paths': result.overactive_paths,
        'path_mode_metrics': {
            pm: metrics.to_dict() for pm, metrics in result.path_mode_metrics.items()
        },
        'summary_table': summary_table.to_dict(orient='records')
    }
    
    json_path = os.path.join(output_dir, f"path_mode_backtest_{symbol}_{timestamp_str}.json")
    with open(json_path, 'w') as f:
        json.dump(json_result, f, indent=2, default=str)
    logger.info(f"💾 Resultados completos guardados en: {json_path}")
    
    return result


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Backtesting de Path Modes L2 - Análisis por modo de path',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Backtest con configuración por defecto (BTCUSDT, 6 meses)
  python backtesting/backtest_l2_path_modes.py

  # Backtest con símbolo y período específicos
  python backtesting/backtest_l2_path_modes.py --symbol ETHUSDT --months 3

  # Backtest con balance inicial personalizado
  python backtesting/backtest_l2_path_modes.py --balance 5000

  # Exportar a directorio específico
  python backtesting/backtest_l2_path_modes.py --output-dir ./mis_resultados
        """
    )
    
    parser.add_argument(
        '--symbol',
        default='BTCUSDT',
        help='Símbolo a analizar (default: BTCUSDT)'
    )
    parser.add_argument(
        '--months',
        type=int,
        default=6,
        help='Meses de datos históricos (default: 6)'
    )
    parser.add_argument(
        '--balance',
        type=float,
        default=DEFAULT_INITIAL_BALANCE,
        help=f'Balance inicial en USDT (default: {DEFAULT_INITIAL_BALANCE})'
    )
    parser.add_argument(
        '--output-dir',
        default=RESULTS_DIR,
        help=f'Directorio de salida (default: {RESULTS_DIR})'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Iniciando backtest de path modes para {args.symbol}")
    print(f"   Período: {args.months} meses")
    print(f"   Balance inicial: {args.balance} USDT")
    
    # Ejecutar backtest
    try:
        import asyncio
        result = asyncio.run(run_path_mode_backtest(
            symbol=args.symbol,
            months=args.months,
            output_dir=args.output_dir,
            initial_balance=args.balance
        ))
        
        print("\n" + "=" * 80)
        print("✅ BACKTEST COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        
        # Resumen final
        print(f"\n📊 RESUMEN FINAL:")
        print(f"  Mejor path_mode: {result.best_path_mode}")
        print(f"  Peor path_mode: {result.worst_path_mode}")
        print(f"  Path_mode más frecuente: {result.most_frequent_path_mode}")
        
        if result.dead_code_paths:
            print(f"  ⚠️  Código muerto: {', '.join(result.dead_code_paths)}")
        if result.overactive_paths:
            print(f"  🚨 Sobre-activos: {', '.join(result.overactive_paths)}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Backtest interrumpido por el usuario")
        return 130
    except Exception as e:
        print(f"\n❌ Error en backtest: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
