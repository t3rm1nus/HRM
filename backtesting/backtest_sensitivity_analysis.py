#!/usr/bin/env python3
"""
Backtesting Sensitivity Analysis - Análisis de sensibilidad de parámetros del pipeline completo

Este script:
1. Ejecuta el pipeline completo L1→L2→L3 variando parámetros clave:
   - Umbral de confidence L1: [0.5, 0.6, 0.7, 0.8]
   - Umbral de bloqueo L3: [0.3, 0.5, 0.7]
   - Tamaño máximo de posición L2: [5%, 10%, 15%, 20%]
2. Para cada combinación, calcula: Sharpe, max drawdown, win rate, total trades
3. Genera heatmaps de Sharpe ratio vs cada par de parámetros
4. Identifica la combinación de parámetros más robusta (mayor Sharpe con menor varianza)
5. Detecta si el sistema es frágil (pequeños cambios producen grandes cambios) o robusto
6. Output: CSV con todas las combinaciones + heatmaps HTML

IMPORTANTE: Este script ejecuta 20 iteraciones por combinación para evaluar estabilidad
"""

import os
import sys
import asyncio
import argparse
import csv
import json
import time
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Suprimir warnings
warnings.filterwarnings('ignore', category=UserWarning)

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

# Importar componentes de los 3 niveles
from l1_operational.models import L1Model
from l1_operational.simulated_exchange_client import SimulatedExchangeClient
from l2_tactic.signal_generator_refactored import L2TacticProcessor
from l2_tactic.position_sizer import PositionSizerManager
from l2_tactic.risk_overlay import RiskOverlay
from l2_tactic.models import TacticalSignal
from l2_tactic.config import L2Config as L2ConfigClass
from core.l3_processor import get_l3_decision
from fix_l3_dominance import should_l3_block_l2_signals

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
DEFAULT_TIMEFRAME = '1m'
DEFAULT_INITIAL_BALANCE = 3000.0
DEFAULT_FEE = 0.001
DEFAULT_SLIPPAGE = 0.0005
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Parámetros para el análisis de sensibilidad
L1_CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
L3_BLOCK_THRESHOLDS = [0.3, 0.5, 0.7]
L2_POSITION_SIZES = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%

DEFAULT_ITERATIONS = 20  # Número de iteraciones por combinación

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ParameterCombination:
    """Combinación de parámetros para testear"""
    l1_confidence: float
    l3_block_threshold: float
    l2_position_size: float
    combination_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'combination_id': self.combination_id,
            'l1_confidence': self.l1_confidence,
            'l3_block_threshold': self.l3_block_threshold,
            'l2_position_size': self.l2_position_size,
            'l2_position_size_pct': f"{self.l2_position_size:.1%}"
        }
    
    def get_key(self) -> str:
        return f"L1_{self.l1_confidence:.1f}_L3_{self.l3_block_threshold:.1f}_L2_{self.l2_position_size:.2f}"


@dataclass
class IterationResult:
    """Resultado de una iteración individual"""
    combination_id: int
    iteration: int
    
    # Métricas de rendimiento
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    profit_factor: float = 0.0
    
    # Métricas de tiempo
    execution_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'combination_id': self.combination_id,
            'iteration': self.iteration,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'return_pct': self.return_pct,
            'profit_factor': self.profit_factor,
            'execution_time_seconds': self.execution_time_seconds
        }


@dataclass
class CombinationSummary:
    """Resumen estadístico de una combinación"""
    combination_id: int
    params: ParameterCombination
    
    # Estadísticas de Sharpe
    sharpe_mean: float = 0.0
    sharpe_std: float = 0.0
    sharpe_min: float = 0.0
    sharpe_max: float = 0.0
    sharpe_cv: float = 0.0  # Coefficient of variation
    
    # Estadísticas de Drawdown
    drawdown_mean: float = 0.0
    drawdown_std: float = 0.0
    
    # Estadísticas de Win Rate
    win_rate_mean: float = 0.0
    win_rate_std: float = 0.0
    
    # Estadísticas de Trades
    trades_mean: float = 0.0
    trades_std: float = 0.0
    
    # Estadísticas de PnL
    pnl_mean: float = 0.0
    pnl_std: float = 0.0
    
    # Score de robustez (mayor Sharpe, menor varianza)
    robustness_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'combination_id': self.combination_id,
            'l1_confidence': self.params.l1_confidence,
            'l3_block_threshold': self.params.l3_block_threshold,
            'l2_position_size': self.params.l2_position_size,
            'l2_position_size_pct': f"{self.params.l2_position_size:.1%}",
            'sharpe_mean': self.sharpe_mean,
            'sharpe_std': self.sharpe_std,
            'sharpe_min': self.sharpe_min,
            'sharpe_max': self.sharpe_max,
            'sharpe_cv': self.sharpe_cv,
            'drawdown_mean': self.drawdown_mean,
            'drawdown_std': self.drawdown_std,
            'win_rate_mean': self.win_rate_mean,
            'win_rate_std': self.win_rate_std,
            'trades_mean': self.trades_mean,
            'trades_std': self.trades_std,
            'pnl_mean': self.pnl_mean,
            'pnl_std': self.pnl_std,
            'robustness_score': self.robustness_score
        }


@dataclass
class TradeRecord:
    """Registro de una operación completada"""
    trade_id: int
    timestamp: datetime
    symbol: str
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


# =============================================================================
# CARGA DE DATOS
# =============================================================================

class DataLoader:
    """Cargador de datos históricos"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_from_csv(self, symbol: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        """Carga datos desde CSV local"""
        possible_names = [
            f"{symbol.lower()}_{timeframe}.csv",
            f"{symbol.upper()}_{timeframe}.csv",
            f"{symbol.replace('USDT', '').lower()}_1m.csv",
            f"{symbol.replace('USDT', '').lower()}_features_train.csv",
            f"{symbol.replace('USDT', '').lower()}_features_test.csv",
        ]
        
        for filename in possible_names:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                logger.info(f"📁 Cargando datos desde {filepath}")
                df = pd.read_csv(filepath)
                df = self._normalize_columns(df)
                return df
        
        return None
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas"""
        df.columns = [c.lower() for c in df.columns]
        
        column_mapping = {
            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close',
            'volume': 'volume', 'timestamp': 'timestamp', 'datetime': 'timestamp'
        }
        
        for old, new in column_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    async def download_from_binance(
        self, symbol: str, timeframe: str = '1m', days: int = 30, testnet: bool = True
    ) -> pd.DataFrame:
        """Descarga datos desde Binance"""
        try:
            import ccxt.async_support as ccxt
            
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
            })
            
            if testnet:
                exchange.set_sandbox_mode(True)
            
            logger.info(f"📡 Descargando datos de {symbol} desde Binance...")
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            since = int(start_time.timestamp() * 1000)
            
            all_ohlcv = []
            limit = 1000
            
            while since < int(end_time.timestamp() * 1000):
                ohlcv = await exchange.fetch_ohlcv(
                    symbol.replace('USDT', '/USDT'),
                    timeframe=timeframe, since=since, limit=limit
                )
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                await asyncio.sleep(0.1)
            
            await exchange.close()
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            output_file = os.path.join(self.data_dir, f"{symbol.lower()}_{timeframe}.csv")
            df.to_csv(output_file)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error descargando datos: {e}")
            raise
    
    async def load_data(self, symbols: List[str], timeframe: str = '1m', days: int = 30) -> Dict[str, pd.DataFrame]:
        """Carga datos para múltiples símbolos"""
        data = {}
        for symbol in symbols:
            df = self.load_from_csv(symbol, timeframe)
            if df is None:
                df = await self.download_from_binance(symbol, timeframe, days)
            if df is not None and not df.empty:
                data[symbol] = df
        return data


# =============================================================================
# MOTOR DE BACKTESTING SENSITIVITY
# =============================================================================

class SensitivityBacktestEngine:
    """Motor de backtesting para análisis de sensibilidad"""
    
    def __init__(
        self,
        params: ParameterCombination,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE
    ):
        self.params = params
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        
        # Inicializar componentes
        self._init_components()
        
        # Estado del backtest
        self.reset_state()
    
    def _init_components(self):
        """Inicializa los componentes L1, L2 y L3"""
        # L1
        self.l1_model = L1Model(config={})
        
        # L2
        l2_config = L2ConfigClass()
        l2_config.kelly_fraction = 0.5
        l2_config.kelly_cap = self.params.l2_position_size
        l2_config.vol_target = 0.20
        l2_config.max_risk_per_trade = self.params.l2_position_size * 0.1
        
        self.position_sizer = PositionSizerManager(config=l2_config)
        self.risk_overlay = RiskOverlay(config=l2_config)
    
    def reset_state(self):
        """Resetea el estado del motor"""
        # Exchange simulado
        SimulatedExchangeClient.reset_instance()
        self.exchange = SimulatedExchangeClient(
            initial_balances={'USDT': self.initial_balance, 'BTC': 0.0, 'ETH': 0.0},
            fee=self.fee, slippage=self.slippage
        )
        
        # Balance
        self.balance = self.initial_balance
        
        # Posiciones
        self.positions: Dict[str, TradeRecord] = {}
        
        # Trades cerrados
        self.closed_trades: List[TradeRecord] = []
        
        # Equity curve
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Contadores
        self.trade_counter = 0
        
        # Market data histórica
        self.market_data_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calcula tamaño de posición"""
        base_size = self.balance * self.params.l2_position_size
        scaled_size = base_size * confidence
        quantity = scaled_size / price if price > 0 else 0
        return round(quantity, 6)
    
    def _get_l1_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Genera señales L1"""
        try:
            l1_result = self.l1_model.predict(market_data)
            signals = l1_result.get('signals', [])
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales L1: {e}")
            return []
    
    async def _process_l2(
        self,
        timestamp: datetime,
        symbol: str,
        l1_signal: str,
        l1_confidence: float,
        l1_price: float
    ) -> Tuple[bool, Optional[Dict]]:
        """Procesa señal a través de L2. Retorna (aprobado, info_l2)"""
        
        if l1_signal.upper() == 'HOLD':
            return False, None
        
        if l1_confidence < self.params.l1_confidence:
            return False, {'rejection_reason': 'LOW_CONFIDENCE'}
        
        try:
            # Verificar filtros de Risk Overlay
            current_dd = self._calculate_current_drawdown()
            if current_dd > 0.12:
                return False, {'rejection_reason': 'RISK_DRAWDOWN'}
            
            # Crear TacticalSignal
            tactical_signal = TacticalSignal(
                symbol=symbol,
                side=l1_signal.lower(),
                strength=l1_confidence,
                confidence=l1_confidence,
                signal_type='l1_processed',
                source='l1_backtest',
                features=self._extract_features(symbol, l1_price),
                timestamp=timestamp
            )
            tactical_signal.price = l1_price
            
            # Calcular tamaño de posición
            market_features = self._build_market_features(symbol)
            
            position_size_result = await self.position_sizer.calculate_position_size(
                signal=tactical_signal,
                market_features=market_features,
                portfolio_state=self._get_portfolio_state()
            )
            
            if position_size_result is None:
                return False, {'rejection_reason': 'POSITION_SIZING_FAILED'}
            
            return True, {
                'position_size': position_size_result.size,
                'notional': position_size_result.notional,
                'stop_loss': position_size_result.stop_loss,
                'take_profit': position_size_result.take_profit,
                'leverage': position_size_result.leverage
            }
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento L2: {e}")
            return False, {'rejection_reason': f'ERROR_{str(e)[:30]}'}
    
    def _get_l3_decision(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Obtiene decisión L3"""
        try:
            l3_decision = get_l3_decision(market_data)
            return l3_decision
        except Exception as e:
            logger.error(f"❌ Error obteniendo decisión L3: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.5,
                'regime': 'neutral',
                'allow_l2_signals': True
            }
    
    async def process_candle(
        self,
        timestamp: datetime,
        symbol: str,
        candle_data: pd.Series,
        market_data: Dict[str, pd.DataFrame]
    ):
        """Procesa una vela a través del pipeline L1→L2→L3"""
        
        # Actualizar market data history
        self.market_data_history[symbol].append({
            'timestamp': timestamp,
            'close': candle_data['close'],
            'open': candle_data.get('open', candle_data['close']),
            'high': candle_data.get('high', candle_data['close']),
            'low': candle_data.get('low', candle_data['close']),
            'volume': candle_data.get('volume', 0)
        })
        
        # 1. GENERAR SEÑAL L1
        l1_signals = self._get_l1_signals(market_data)
        
        # Encontrar señal para este símbolo
        symbol_signal = None
        for signal in l1_signals:
            if symbol in getattr(signal, 'symbol', ''):
                symbol_signal = signal
                break
        
        if symbol_signal is None:
            self._update_equity(timestamp, {symbol: candle_data['close']})
            return
        
        l1_action = getattr(symbol_signal, 'direction', 'hold').upper()
        l1_confidence = getattr(symbol_signal, 'confidence', 0.5)
        l1_price = candle_data['close']
        
        if l1_action == 'HOLD' or l1_confidence < self.params.l1_confidence:
            self._update_equity(timestamp, {symbol: candle_data['close']})
            return
        
        # 2. PROCESAR L2
        l2_approved, l2_info = await self._process_l2(
            timestamp, symbol, l1_action, l1_confidence, l1_price
        )
        
        if not l2_approved:
            self._update_equity(timestamp, {symbol: candle_data['close']})
            return
        
        # 3. OBTENER DECISIÓN L3
        l3_decision = self._get_l3_decision(market_data)
        
        l3_signal = l3_decision.get('signal', 'hold')
        l3_confidence = l3_decision.get('confidence', 0.5)
        l3_regime = l3_decision.get('regime', 'neutral')
        l3_allow_l2 = l3_decision.get('allow_l2_signals', True)
        
        # 4. EVALUAR DOMINANCIA L3
        current_allocation = self._get_current_allocation()
        target_allocation = l3_decision.get('asset_allocation', {})
        
        l3_blocks = should_l3_block_l2_signals(
            l3_signal=l3_signal,
            l3_confidence=l3_confidence,
            regime=l3_regime,
            current_allocation=current_allocation,
            target_allocation=target_allocation,
            l2_signal_action=l1_action,
            has_position=(symbol in self.positions),
            allow_l2_signals=l3_allow_l2
        )
        
        # Aplicar umbral de bloqueo L3 del análisis de sensibilidad
        if l3_blocks and l3_confidence < self.params.l3_block_threshold:
            l3_blocks = False  # No bloquear si la confianza de L3 es baja
        
        # 5. EJECUTAR TRADE
        if not l3_blocks:
            await self._execute_trade(
                timestamp, symbol, l1_action, l1_price, l1_confidence, l2_info
            )
        
        # 6. ACTUALIZAR EQUITY
        self._update_equity(timestamp, {symbol: candle_data['close']})
    
    async def _execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        price: float,
        confidence: float,
        l2_info: Optional[Dict] = None
    ) -> bool:
        """Ejecuta un trade"""
        
        try:
            if side.upper() == 'BUY':
                if symbol in self.positions:
                    return False
                
                quantity = self._calculate_position_size(price, confidence)
                if quantity <= 0:
                    return False
                
                execution_price = price * (1 + self.slippage)
                notional = quantity * execution_price
                fees = notional * self.fee
                
                self.balance -= notional + fees
                
                self.trade_counter += 1
                
                self.positions[symbol] = TradeRecord(
                    trade_id=self.trade_counter,
                    timestamp=timestamp,
                    symbol=symbol,
                    side='BUY',
                    entry_price=execution_price,
                    quantity=quantity,
                    confidence=confidence,
                    fees=fees
                )
                
                return True
                
            elif side.upper() == 'SELL':
                if symbol not in self.positions:
                    return False
                
                position = self.positions[symbol]
                execution_price = price * (1 - self.slippage)
                
                entry_value = position.entry_price * position.quantity
                exit_value = execution_price * position.quantity
                exit_fees = exit_value * self.fee
                
                pnl = exit_value - entry_value - position.fees - exit_fees
                
                self.balance += exit_value - exit_fees
                
                hold_time = (timestamp - position.timestamp).total_seconds() / 60
                
                position.exit_price = execution_price
                position.pnl = pnl
                position.pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                position.exit_timestamp = timestamp
                position.hold_time_minutes = hold_time
                position.is_open = False
                position.fees += exit_fees
                
                self.closed_trades.append(position)
                del self.positions[symbol]
                
                return True
        
        except Exception as e:
            logger.error(f"❌ Error ejecutando trade: {e}")
        
        return False
    
    def _update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Actualiza la curva de equity"""
        equity = self.balance
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                equity += position.quantity * market_prices[symbol]
        self.equity_curve.append((timestamp, equity))
    
    def _calculate_current_drawdown(self) -> float:
        """Calcula drawdown actual"""
        if not self.equity_curve:
            return 0.0
        current = self.equity_curve[-1][1]
        peak = max(e[1] for e in self.equity_curve)
        return (peak - current) / peak if peak > 0 else 0.0
    
    def _get_current_allocation(self) -> Dict[str, float]:
        """Obtiene allocación actual"""
        total = self._get_portfolio_state().get('total_capital', self.initial_balance)
        if total == 0:
            return {'BTCUSDT': 0, 'ETHUSDT': 0, 'USDT': 1.0}
        
        allocation = {}
        for symbol, position in self.positions.items():
            allocation[symbol] = (position.quantity * position.entry_price) / total
        
        usdt_pct = self.balance / total
        allocation['USDT'] = usdt_pct
        
        return allocation
    
    def _get_portfolio_state(self) -> Dict:
        """Obtiene estado del portfolio"""
        equity = self.balance
        for symbol, position in self.positions.items():
            # Usar último precio conocido
            if self.equity_curve:
                equity = self.equity_curve[-1][1]
        
        return {
            'total_capital': equity,
            'available_capital': self.balance,
            'drawdown': self._calculate_current_drawdown(),
            'positions': {}
        }
    
    def _extract_features(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Extrae features para L2"""
        features = {'close': current_price, 'price': current_price}
        
        history = self.market_data_history.get(symbol, [])
        if len(history) >= 14:
            prices = [h['close'] for h in history[-20:]]
            
            if len(prices) >= 14:
                deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                gains = [d for d in deltas if d > 0]
                losses = [-d for d in deltas if d < 0]
                avg_gain = sum(gains) / len(gains) if gains else 0
                avg_loss = sum(losses) / len(losses) if losses else 0
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features['rsi'] = 100 - (100 / (1 + rs))
                else:
                    features['rsi'] = 50
            
            if len(prices) >= 2:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                features['volatility'] = np.std(returns) * np.sqrt(252) if returns else 0.2
            
            if history:
                features['volume'] = history[-1].get('volume', 0)
        
        return features
    
    def _build_market_features(self, symbol: str) -> Dict[str, Any]:
        """Construye market features para position sizing"""
        features = {}
        
        for sym in self.market_data_history:
            history = self.market_data_history[sym]
            if len(history) >= 2:
                prices = [h['close'] for h in history[-20:]]
                volumes = [h.get('volume', 0) for h in history[-20:]]
                
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                
                features[sym] = {
                    'volatility': np.std(returns) * np.sqrt(252) if returns else 0.2,
                    'price': prices[-1] if prices else 0,
                    'volume': volumes[-1] if volumes else 0,
                    'adv_notional': np.mean(volumes) * prices[-1] if volumes and prices else 1_000_000
                }
        
        return features
    
    def close_all_positions(self, timestamp: datetime, final_prices: Dict[str, float]):
        """Cierra todas las posiciones"""
        for symbol, price in final_prices.items():
            if symbol in self.positions:
                asyncio.create_task(self._execute_trade(timestamp, symbol, 'SELL', price, 0.5))
    
    def calculate_metrics(self) -> IterationResult:
        """Calcula métricas finales"""
        result = IterationResult(
            combination_id=self.params.combination_id,
            iteration=0  # Se actualizará externamente
        )
        
        if not self.closed_trades:
            return result
        
        # Métricas básicas
        result.total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        result.win_rate = winning_trades / result.total_trades if result.total_trades > 0 else 0
        
        pnl_values = [t.pnl for t in self.closed_trades]
        result.total_pnl = sum(pnl_values)
        result.return_pct = (result.total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        
        for timestamp, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0
            if drawdown > max_dd:
                max_dd = drawdown
        
        result.max_drawdown_pct = max_dd * 100
        
        # Sharpe ratio
        if len(pnl_values) > 1:
            returns = np.array(pnl_values)
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            result.sharpe_ratio = 0
        
        return result


# =============================================================================
# FUNCIÓN DE EJECUCIÓN DE UNA ITERACIÓN
# =============================================================================

async def run_single_iteration(
    params: ParameterCombination,
    iteration: int,
    market_data: Dict[str, pd.DataFrame],
    initial_balance: float = DEFAULT_INITIAL_BALANCE
) -> IterationResult:
    """Ejecuta una sola iteración del backtest"""
    
    start_time = time.time()
    
    # Inicializar motor
    engine = SensitivityBacktestEngine(
        params=params,
        initial_balance=initial_balance
    )
    
    # Procesar velas
    all_timestamps = set()
    for sym, df in market_data.items():
        all_timestamps.update(df.index)
    
    sorted_timestamps = sorted(all_timestamps)
    
    for timestamp in sorted_timestamps:
        current_market_data = {}
        for sym, df in market_data.items():
            mask = df.index <= timestamp
            if mask.any():
                current_market_data[sym] = df[mask].copy()
        
        if not current_market_data:
            continue
        
        for sym, df in current_market_data.items():
            if timestamp in df.index:
                candle = df.loc[timestamp]
                await engine.process_candle(timestamp, sym, candle, current_market_data)
    
    # Cerrar posiciones
    final_prices = {sym: df['close'].iloc[-1] for sym, df in market_data.items()}
    
    # Cerrar manualmente
    for symbol, price in final_prices.items():
        if symbol in engine.positions:
            position = engine.positions[symbol]
            execution_price = price * (1 - engine.slippage)
            entry_value = position.entry_price * position.quantity
            exit_value = execution_price * position.quantity
            exit_fees = exit_value * engine.fee
            pnl = exit_value - entry_value - position.fees - exit_fees
            
            engine.balance += exit_value - exit_fees
            
            position.exit_price = execution_price
            position.pnl = pnl
            position.pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
            position.exit_timestamp = datetime.now()
            position.hold_time_minutes = (datetime.now() - position.timestamp).total_seconds() / 60
            position.is_open = False
            position.fees += exit_fees
            
            engine.closed_trades.append(position)
    
    engine.positions.clear()
    
    # Calcular métricas
    result = engine.calculate_metrics()
    result.iteration = iteration
    result.execution_time_seconds = time.time() - start_time
    
    return result


# =============================================================================
# GENERADOR DE COMBINACIONES
# =============================================================================

def generate_parameter_combinations() -> List[ParameterCombination]:
    """Genera todas las combinaciones de parámetros"""
    combinations = []
    combination_id = 0
    
    for l1_conf in L1_CONFIDENCE_THRESHOLDS:
        for l3_block in L3_BLOCK_THRESHOLDS:
            for l2_size in L2_POSITION_SIZES:
                combination_id += 1
                combinations.append(ParameterCombination(
                    l1_confidence=l1_conf,
                    l3_block_threshold=l3_block,
                    l2_position_size=l2_size,
                    combination_id=combination_id
                ))
    
    return combinations


# =============================================================================
# EJECUCIÓN DEL ANÁLISIS DE SENSIBILIDAD
# =============================================================================

async def run_sensitivity_analysis(
    symbols: List[str] = None,
    timeframe: str = '1m',
    days: int = 30,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    iterations: int = DEFAULT_ITERATIONS,
    output_dir: str = None,
    parallel: bool = False
) -> Dict[str, Any]:
    """
    Ejecuta el análisis de sensibilidad completo
    
    Args:
        symbols: Lista de símbolos a testear
        timeframe: Timeframe de las velas
        days: Días de datos históricos
        initial_balance: Balance inicial
        iterations: Número de iteraciones por combinación
        output_dir: Directorio de salida
        parallel: Si se debe ejecutar en paralelo (no implementado en esta versión)
    
    Returns:
        Dict con resultados completos del análisis
    """
    symbols = symbols or DEFAULT_SYMBOLS
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🔬 INICIANDO ANÁLISIS DE SENSIBILIDAD")
    logger.info("=" * 80)
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Días: {days}")
    logger.info(f"Balance inicial: ${initial_balance:.2f}")
    logger.info(f"Iteraciones por combinación: {iterations}")
    
    # Generar combinaciones
    combinations = generate_parameter_combinations()
    total_combinations = len(combinations)
    
    logger.info(f"\n📊 Total de combinaciones a testear: {total_combinations}")
    logger.info(f"   - Umbral L1 confidence: {L1_CONFIDENCE_THRESHOLDS}")
    logger.info(f"   - Umbral L3 bloqueo: {L3_BLOCK_THRESHOLDS}")
    logger.info(f"   - Tamaño L2 posición: {[f'{s:.0%}' for s in L2_POSITION_SIZES]}")
    logger.info(f"\n🔄 Total de iteraciones: {total_combinations * iterations}")
    
    # Cargar datos
    logger.info("\n📊 Cargando datos históricos...")
    data_loader = DataLoader()
    market_data = await data_loader.load_data(symbols, timeframe, days)
    
    if not market_data:
        logger.error("❌ No se pudieron cargar datos")
        return {}
    
    logger.info(f"✅ Datos cargados: {list(market_data.keys())}")
    for sym, df in market_data.items():
        logger.info(f"  - {sym}: {len(df)} velas")
    
    # Ejecutar todas las iteraciones
    all_iteration_results: List[IterationResult] = []
    combination_summaries: List[CombinationSummary] = []
    
    total_iterations = total_combinations * iterations
    completed_iterations = 0
    
    start_time_total = time.time()
    
    for combo in combinations:
        combo_start_time = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Combinación {combo.combination_id}/{total_combinations}")
        logger.info(f"   L1 confidence: {combo.l1_confidence:.1f}")
        logger.info(f"   L3 block threshold: {combo.l3_block_threshold:.1f}")
        logger.info(f"   L2 position size: {combo.l2_position_size:.1%}")
        logger.info(f"{'='*60}")
        
        combo_results: List[IterationResult] = []
        
        for i in range(iterations):
            completed_iterations += 1
            
            logger.info(f"  🔄 Iteración {i+1}/{iterations} "
                       f"(Total: {completed_iterations}/{total_iterations})")
            
            try:
                result = await run_single_iteration(
                    params=combo,
                    iteration=i+1,
                    market_data=market_data,
                    initial_balance=initial_balance
                )
                combo_results.append(result)
                all_iteration_results.append(result)
                
                logger.info(f"     ✓ Sharpe: {result.sharpe_ratio:.2f}, "
                           f"PnL: ${result.total_pnl:.2f}, "
                           f"Trades: {result.total_trades}")
                
            except Exception as e:
                logger.error(f"     ❌ Error en iteración {i+1}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calcular resumen de la combinación
        if combo_results:
            summary = calculate_combination_summary(combo, combo_results)
            combination_summaries.append(summary)
            
            logger.info(f"\n  📊 Resumen Combinación {combo.combination_id}:")
            logger.info(f"     Sharpe: {summary.sharpe_mean:.2f} ± {summary.sharpe_std:.2f}")
            logger.info(f"     Win Rate: {summary.win_rate_mean:.1%} ± {summary.win_rate_std:.1%}")
            logger.info(f"     Drawdown: {summary.drawdown_mean:.1f}% ± {summary.drawdown_std:.1f}%")
            logger.info(f"     PnL: ${summary.pnl_mean:.2f} ± ${summary.pnl_std:.2f}")
            logger.info(f"     Robustness Score: {summary.robustness_score:.2f}")
            logger.info(f"     Tiempo: {time.time() - combo_start_time:.1f}s")
    
    total_time = time.time() - start_time_total
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ ANÁLISIS COMPLETADO en {total_time:.1f} segundos")
    logger.info(f"{'='*80}")
    
    # Encontrar mejor combinación
    best_combo = max(combination_summaries, key=lambda x: x.robustness_score)
    
    logger.info(f"\n🏆 MEJOR COMBINACIÓN DE PARÁMETROS:")
    logger.info(f"   ID: {best_combo.combination_id}")
    logger.info(f"   L1 Confidence: {best_combo.params.l1_confidence:.1f}")
    logger.info(f"   L3 Block Threshold: {best_combo.params.l3_block_threshold:.1f}")
    logger.info(f"   L2 Position Size: {best_combo.params.l2_position_size:.1%}")
    logger.info(f"   Sharpe Promedio: {best_combo.sharpe_mean:.2f}")
    logger.info(f"   Sharpe Std: {best_combo.sharpe_std:.2f}")
    logger.info(f"   Robustness Score: {best_combo.robustness_score:.2f}")
    
    # Análisis de fragilidad/robustez
    fragility_analysis = analyze_fragility(combination_summaries)
    
    logger.info(f"\n{'='*80}")
    logger.info("📈 ANÁLISIS DE FRAGILIDAD/ROBUSTEZ")
    logger.info(f"{'='*80}")
    logger.info(f"   Sistema: {fragility_analysis['classification']}")
    logger.info(f"   Score de Robustez Global: {fragility_analysis['global_robustness_score']:.2f}")
    logger.info(f"   Sensibilidad a L1: {fragility_analysis['sensitivity_l1']:.3f}")
    logger.info(f"   Sensibilidad a L3: {fragility_analysis['sensitivity_l3']:.3f}")
    logger.info(f"   Sensibilidad a L2: {fragility_analysis['sensitivity_l2']:.3f}")
    
    # Exportar resultados
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV de todas las iteraciones
    iterations_csv = os.path.join(output_dir, f"sensitivity_iterations_{timestamp_str}.csv")
    iterations_df = pd.DataFrame([r.to_dict() for r in all_iteration_results])
    
    # Añadir parámetros a cada fila
    for result in all_iteration_results:
        combo = next(c for c in combinations if c.combination_id == result.combination_id)
        for key, value in combo.to_dict().items():
            if key not in iterations_df.columns:
                iterations_df[key] = None
            iterations_df.loc[iterations_df['combination_id'] == result.combination_id, key] = value
    
    iterations_df.to_csv(iterations_csv, index=False)
    logger.info(f"\n💾 Iteraciones guardadas en: {iterations_csv}")
    
    # CSV de resúmenes por combinación
    summary_csv = os.path.join(output_dir, f"sensitivity_summary_{timestamp_str}.csv")
    summary_df = pd.DataFrame([s.to_dict() for s in combination_summaries])
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"💾 Resúmenes guardados en: {summary_csv}")
    
    # JSON completo
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'symbols': symbols,
            'timeframe': timeframe,
            'days': days,
            'initial_balance': initial_balance,
            'iterations': iterations,
            'total_combinations': total_combinations,
            'total_iterations': total_iterations,
            'execution_time_seconds': total_time
        },
        'best_combination': best_combo.to_dict(),
        'fragility_analysis': fragility_analysis,
        'combinations': [s.to_dict() for s in combination_summaries],
        'all_iterations': [r.to_dict() for r in all_iteration_results],
        'files': {
            'iterations_csv': iterations_csv,
            'summary_csv': summary_csv
        }
    }
    
    results_json = os.path.join(output_dir, f"sensitivity_analysis_{timestamp_str}.json")
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Resultados JSON guardados en: {results_json}")
    
    # Generar heatmaps HTML
    logger.info("\n📊 Generando heatmaps HTML...")
    generate_heatmaps_html(combination_summaries, output_dir, timestamp_str)
    
    return results


def calculate_combination_summary(
    params: ParameterCombination,
    results: List[IterationResult]
) -> CombinationSummary:
    """Calcula el resumen estadístico de una combinación"""
    
    summary = CombinationSummary(
        combination_id=params.combination_id,
        params=params
    )
    
    if not results:
        return summary
    
    # Extraer arrays de métricas
    sharpes = [r.sharpe_ratio for r in results]
    drawdowns = [r.max_drawdown_pct for r in results]
    win_rates = [r.win_rate for r in results]
    trades = [r.total_trades for r in results]
    pnls = [r.total_pnl for r in results]
    
    # Estadísticas de Sharpe
    summary.sharpe_mean = np.mean(sharpes)
    summary.sharpe_std = np.std(sharpes)
    summary.sharpe_min = np.min(sharpes)
    summary.sharpe_max = np.max(sharpes)
    summary.sharpe_cv = summary.sharpe_std / abs(summary.sharpe_mean) if summary.sharpe_mean != 0 else float('inf')
    
    # Estadísticas de Drawdown
    summary.drawdown_mean = np.mean(drawdowns)
    summary.drawdown_std = np.std(drawdowns)
    
    # Estadísticas de Win Rate
    summary.win_rate_mean = np.mean(win_rates)
    summary.win_rate_std = np.std(win_rates)
    
    # Estadísticas de Trades
    summary.trades_mean = np.mean(trades)
    summary.trades_std = np.std(trades)
    
    # Estadísticas de PnL
    summary.pnl_mean = np.mean(pnls)
    summary.pnl_std = np.std(pnls)
    
    # Score de robustez: mayor Sharpe, menor varianza
    # Fórmula: Sharpe_mean / (1 + sharpe_std) - penalización por drawdown
    sharpe_score = summary.sharpe_mean / (1 + summary.sharpe_std)
    drawdown_penalty = summary.drawdown_mean / 100  # Normalizar
    summary.robustness_score = sharpe_score - drawdown_penalty
    
    return summary


def analyze_fragility(summaries: List[CombinationSummary]) -> Dict[str, Any]:
    """Analiza la fragilidad/robustez del sistema"""
    
    if not summaries:
        return {}
    
    # Calcular varianza de Sharpe entre combinaciones similares
    sharpe_values = [s.sharpe_mean for s in summaries]
    global_sharpe_std = np.std(sharpe_values)
    global_sharpe_mean = np.mean(sharpe_values)
    
    # Sensibilidad a cada parámetro
    sensitivity_l1 = calculate_parameter_sensitivity(summaries, 'l1_confidence')
    sensitivity_l3 = calculate_parameter_sensitivity(summaries, 'l3_block_threshold')
    sensitivity_l2 = calculate_parameter_sensitivity(summaries, 'l2_position_size')
    
    # Score de robustez global
    global_robustness = 1.0 / (1 + global_sharpe_std) if global_sharpe_std > 0 else 1.0
    
    # Clasificación
    if global_sharpe_std < 0.5:
        classification = "ROBUSTO"
    elif global_sharpe_std < 1.0:
        classification = "MODERADAMENTE ROBUSTO"
    elif global_sharpe_std < 2.0:
        classification = "MODERADAMENTE FRÁGIL"
    else:
        classification = "FRÁGIL"
    
    return {
        'classification': classification,
        'global_robustness_score': global_robustness,
        'global_sharpe_std': global_sharpe_std,
        'global_sharpe_mean': global_sharpe_mean,
        'sensitivity_l1': sensitivity_l1,
        'sensitivity_l3': sensitivity_l3,
        'sensitivity_l2': sensitivity_l2,
        'most_sensitive_parameter': max([
            ('L1', sensitivity_l1),
            ('L3', sensitivity_l3),
            ('L2', sensitivity_l2)
        ], key=lambda x: x[1])[0]
    }


def calculate_parameter_sensitivity(
    summaries: List[CombinationSummary],
    param_name: str
) -> float:
    """Calcula la sensibilidad del Sharpe a un parámetro específico"""
    
    # Agrupar por valores del parámetro
    param_values = defaultdict(list)
    
    for s in summaries:
        value = getattr(s.params, param_name)
        param_values[value].append(s.sharpe_mean)
    
    # Calcular varianza entre medias de grupos
    group_means = [np.mean(values) for values in param_values.values()]
    
    if len(group_means) > 1:
        return np.std(group_means)
    return 0.0


# =============================================================================
# GENERACIÓN DE HEATMAPS HTML
# =============================================================================

def generate_heatmaps_html(
    summaries: List[CombinationSummary],
    output_dir: str,
    timestamp_str: str
):
    """Genera heatmaps HTML interactivos"""
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        
        # Crear DataFrame para heatmaps
        df = pd.DataFrame([s.to_dict() for s in summaries])
        
        # Heatmap 1: L1 Confidence vs L3 Block Threshold (colored by Sharpe)
        fig1 = create_heatmap_figure(
            df, 
            x_col='l1_confidence',
            y_col='l3_block_threshold', 
            z_col='sharpe_mean',
            title='Sharpe Ratio: L1 Confidence vs L3 Block Threshold',
            x_title='L1 Confidence Threshold',
            y_title='L3 Block Threshold'
        )
        
        # Heatmap 2: L1 Confidence vs L2 Position Size (colored by Sharpe)
        fig2 = create_heatmap_figure(
            df,
            x_col='l1_confidence',
            y_col='l2_position_size',
            z_col='sharpe_mean',
            title='Sharpe Ratio: L1 Confidence vs L2 Position Size',
            x_title='L1 Confidence Threshold',
            y_title='L2 Position Size'
        )
        
        # Heatmap 3: L3 Block Threshold vs L2 Position Size (colored by Sharpe)
        fig3 = create_heatmap_figure(
            df,
            x_col='l3_block_threshold',
            y_col='l2_position_size',
            z_col='sharpe_mean',
            title='Sharpe Ratio: L3 Block Threshold vs L2 Position Size',
            x_title='L3 Block Threshold',
            y_title='L2 Position Size'
        )
        
        # Heatmap 4: Robustness Score
        fig4 = create_heatmap_figure(
            df,
            x_col='l1_confidence',
            y_col='l3_block_threshold',
            z_col='robustness_score',
            title='Robustness Score: L1 Confidence vs L3 Block Threshold',
            x_title='L1 Confidence Threshold',
            y_title='L3 Block Threshold',
            colorscale='RdYlGn'
        )
        
        # Combinar todos los heatmaps en un HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Sensitivity Analysis Heatmaps</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .heatmap-container {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>🔬 Sensitivity Analysis Heatmaps</h1>
    <p style="text-align: center; color: #666;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="heatmap-container">
        <h2>Sharpe Ratio: L1 Confidence vs L3 Block Threshold</h2>
        {fig1.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div class="heatmap-container">
        <h2>Sharpe Ratio: L1 Confidence vs L2 Position Size</h2>
        {fig2.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div class="heatmap-container">
        <h2>Sharpe Ratio: L3 Block Threshold vs L2 Position Size</h2>
        {fig3.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div class="heatmap-container">
        <h2>Robustness Score</h2>
        {fig4.to_html(full_html=False, include_plotlyjs=False)}
    </div>
    
    <div class="summary">
        <h2>📊 Summary Statistics</h2>
        {generate_summary_table(summaries)}
    </div>
</body>
</html>
"""
        
        html_path = os.path.join(output_dir, f"sensitivity_heatmaps_{timestamp_str}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"💾 Heatmaps HTML guardados en: {html_path}")
        
    except ImportError:
        logger.warning("⚠️ plotly no instalado, generando heatmaps estáticos con matplotlib")
        generate_heatmaps_matplotlib(summaries, output_dir, timestamp_str)
    except Exception as e:
        logger.error(f"❌ Error generando heatmaps: {e}")
        import traceback
        traceback.print_exc()


def create_heatmap_figure(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    title: str,
    x_title: str,
    y_title: str,
    colorscale: str = 'RdYlGn'
) -> go.Figure:
    """Crea una figura de heatmap con plotly"""
    
    import plotly.graph_objects as go
    
    # Crear matriz pivot
    pivot_df = df.pivot_table(
        values=z_col,
        index=y_col,
        columns=x_col,
        aggfunc='mean'
    )
    
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=[f"{valor:.2f}" for valor in pivot_df.columns],
            y=[f"{valor:.2f}" for valor in pivot_df.index],
            colorscale=colorscale,
            colorbar=dict(title=z_col.replace("_", " ").title()),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig
