#!/usr/bin/env python3
"""
Backtesting L3 Override - Evaluación de la contribución de L3 al sistema

Este script:
1. Ejecuta el pipeline completo L1→L2→L3 en modo backtest con datos históricos
2. Registra cada vez que L3 bloquea o modifica una señal de L2
3. Para cada override de L3, calcula si fue correcto (¿hubiera sido ganadora/perdedora?)
4. Métricas finales:
   - % de veces que L3 bloqueó correctamente señales perdedoras de L2
   - % de veces que L3 bloqueó erróneamente señales ganadoras de L2
   - Contribución neta de L3 al PnL total (alpha de L3)
5. Genera gráfico de "decisiones de L3" superpuesto sobre equity curve
6. Alerta si L3 tiene contribución neta negativa

IMPORTANTE: Este script usa fix_l3_dominance.py para la lógica de dominancia L3
"""

import os
import sys
import asyncio
import argparse
import csv
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import warnings

# Suprimir warnings de matplotlib
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
from fix_l3_dominance import should_l3_block_l2_signals, apply_l3_push_to_signals

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

# =============================================================================
# DATA CLASSES PARA RESULTADOS
# =============================================================================

@dataclass
class L3OverrideDecision:
    """Registro de una decisión de override de L3"""
    timestamp: datetime
    symbol: str
    
    # Señal L2 original
    l2_signal: str
    l2_confidence: float
    l2_price: float
    
    # Decisión L3
    l3_signal: str
    l3_confidence: float
    l3_regime: str
    l3_allow_l2: bool
    
    # Override realizado
    override_type: str  # 'BLOCK', 'MODIFY', 'ALLOW'
    blocked: bool  # True si L3 bloqueó la señal
    
    # Resultado (calculado retrospectivamente)
    would_be_winner: Optional[bool] = None  # ¿Hubiera sido ganadora la señal L2?
    would_pnl: Optional[float] = None  # PnL que hubiera tenido L2
    realized_pnl: Optional[float] = None  # PnL real con L3
    
    # Métricas de resultado
    override_correct: Optional[bool] = None  # ¿Fue correcto el override?
    alpha_contribution: float = 0.0  # Contribución al alpha (positivo o negativo)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'l2_signal': self.l2_signal,
            'l2_confidence': self.l2_confidence,
            'l2_price': self.l2_price,
            'l3_signal': self.l3_signal,
            'l3_confidence': self.l3_confidence,
            'l3_regime': self.l3_regime,
            'l3_allow_l2': self.l3_allow_l2,
            'override_type': self.override_type,
            'blocked': self.blocked,
            'would_be_winner': self.would_be_winner,
            'would_pnl': self.would_pnl,
            'realized_pnl': self.realized_pnl,
            'override_correct': self.override_correct,
            'alpha_contribution': self.alpha_contribution
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
    l1_signal_type: str = "unknown"
    l2_approved: bool = False
    l3_override: bool = False
    l3_override_type: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    hold_time_minutes: float = 0.0
    exit_timestamp: Optional[datetime] = None
    is_open: bool = True
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class L3BacktestMetrics:
    """Métricas de rendimiento del backtest L3"""
    # Métricas generales
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    
    # Métricas de L3 overrides
    total_overrides: int = 0
    blocks_correct: int = 0  # Bloqueos de señales perdedoras
    blocks_incorrect: int = 0  # Bloqueos de señales ganadoras
    allows_correct: int = 0  # Permisos de señales ganadoras
    allows_incorrect: int = 0  # Permisos de señales perdedoras
    
    # Porcentajes
    pct_blocks_correct: float = 0.0  # % bloqueos correctos
    pct_blocks_incorrect: float = 0.0  # % bloqueos incorrectos
    pct_allows_correct: float = 0.0  # % permisos correctos
    pct_allows_incorrect: float = 0.0  # % permisos incorrectos
    
    # Alpha de L3
    l3_alpha_contribution: float = 0.0  # Contribución neta de L3
    l3_alpha_pct: float = 0.0  # Como porcentaje del PnL total
    
    # Métricas de protección
    losses_prevented: float = 0.0  # Pérdidas evitadas por L3
    gains_prevented: float = 0.0  # Ganancias evitadas por L3 (error L3)
    net_protection: float = 0.0  # Protección neta (positivo = bueno)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'total_pnl': f"{self.total_pnl:.2f}",
            'return_pct': f"{self.return_pct:.2%}",
            'total_overrides': self.total_overrides,
            'blocks_correct': self.blocks_correct,
            'blocks_incorrect': self.blocks_incorrect,
            'allows_correct': self.allows_correct,
            'allows_incorrect': self.allows_incorrect,
            'pct_blocks_correct': f"{self.pct_blocks_correct:.2%}",
            'pct_blocks_incorrect': f"{self.pct_blocks_incorrect:.2%}",
            'pct_allows_correct': f"{self.pct_allows_correct:.2%}",
            'pct_allows_incorrect': f"{self.pct_allows_incorrect:.2%}",
            'l3_alpha_contribution': f"{self.l3_alpha_contribution:.2f}",
            'l3_alpha_pct': f"{self.l3_alpha_pct:.2f}%",
            'losses_prevented': f"{self.losses_prevented:.2f}",
            'gains_prevented': f"{self.gains_prevented:.2f}",
            'net_protection': f"{self.net_protection:.2f}"
        }


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
# MOTOR DE BACKTESTING L3
# =============================================================================

class L3BacktestEngine:
    """Motor de backtesting para evaluar contribución de L3"""
    
    def __init__(
        self,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE,
        position_size_pct: float = 0.1,
        min_l1_confidence: float = 0.6,
        min_l2_confidence: float = 0.6,
        forward_bars: int = 20  # Barras para calcular resultado de override
    ):
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        self.min_l1_confidence = min_l1_confidence
        self.min_l2_confidence = min_l2_confidence
        self.forward_bars = forward_bars
        
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
        l2_config.kelly_cap = 0.25
        l2_config.vol_target = 0.20
        l2_config.max_risk_per_trade = 0.01
        
        self.position_sizer = PositionSizerManager(config=l2_config)
        self.risk_overlay = RiskOverlay(config=l2_config)
        
        logger.info("✅ Componentes L1, L2, L3 inicializados")
    
    def reset_state(self):
        """Resetea el estado del motor"""
        # Exchange simulado
        SimulatedExchangeClient.reset_instance()
        self.exchange = SimulatedExchangeClient(
            initial_balances={'USDT': self.initial_balance, 'BTC': 0.0, 'ETH': 0.0},
            fee=self.fee, slippage=self.slippage
        )
        
        # Balances
        self.balance_l2_only = self.initial_balance  # Solo L1+L2 (sin L3)
        self.balance_l3_filtered = self.initial_balance  # L1+L2+L3
        
        # Posiciones
        self.positions_l2: Dict[str, TradeRecord] = {}
        self.positions_l3: Dict[str, TradeRecord] = {}
        
        # Trades cerrados
        self.closed_trades_l2: List[TradeRecord] = []
        self.closed_trades_l3: List[TradeRecord] = []
        
        # Equity curves
        self.equity_curve_l2: List[Tuple[datetime, float]] = []
        self.equity_curve_l3: List[Tuple[datetime, float]] = []
        
        # Decisiones de L3
        self.l3_decisions: List[L3OverrideDecision] = []
        
        # Historial de precios para cálculo retrospectivo
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Contadores
        self.trade_counter = 0
        self.pending_overrides: Dict[str, L3OverrideDecision] = {}  # Para cálculo retrospectivo
        
        # Portfolio state para L2/L3
        self.portfolio_state = {
            'total_capital': self.initial_balance,
            'available_capital': self.initial_balance,
            'drawdown': 0.0,
            'positions': {}
        }
        
        # Market data histórica
        self.market_data_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def _calculate_position_size(self, price: float, confidence: float, balance: float) -> float:
        """Calcula tamaño de posición"""
        base_size = balance * self.position_size_pct
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
        
        if l1_confidence < self.min_l2_confidence:
            return False, {'rejection_reason': 'LOW_CONFIDENCE'}
        
        try:
            # Verificar filtros de Risk Overlay
            current_dd = self._calculate_current_drawdown()
            if current_dd > 0.12:  # max_drawdown_limit
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
                portfolio_state=self.portfolio_state
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
        
        # Guardar historial de precios
        self.price_history[symbol].append((timestamp, candle_data['close']))
        
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
        
        if l1_action == 'HOLD' or l1_confidence < self.min_l1_confidence:
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
            has_position=(symbol in self.positions_l3),
            allow_l2_signals=l3_allow_l2
        )
        
        # Crear registro de decisión L3
        override_decision = L3OverrideDecision(
            timestamp=timestamp,
            symbol=symbol,
            l2_signal=l1_action,
            l2_confidence=l1_confidence,
            l2_price=l1_price,
            l3_signal=l3_signal,
            l3_confidence=l3_confidence,
            l3_regime=l3_regime,
            l3_allow_l2=l3_allow_l2,
            override_type='BLOCK' if l3_blocks else 'ALLOW',
            blocked=l3_blocks
        )
        
        self.l3_decisions.append(override_decision)
        
        # Guardar para análisis retrospectivo
        decision_key = f"{symbol}_{timestamp.isoformat()}"
        self.pending_overrides[decision_key] = override_decision
        
        # 5. EJECUTAR TRADES
        # L2 only (sin filtro L3)
        self._execute_trade(
            timestamp, symbol, l1_action, l1_price, l1_confidence,
            is_l3_filtered=False, l2_info=l2_info
        )
        
        # L3 filtered (con filtro L3)
        if not l3_blocks:
            self._execute_trade(
                timestamp, symbol, l1_action, l1_price, l1_confidence,
                is_l3_filtered=True, l2_info=l2_info, l3_decision=override_decision
            )
        
        # 6. ACTUALIZAR EQUITY
        self._update_equity(timestamp, {symbol: candle_data['close']})
        
        # 7. EVALUAR RESULTADOS RETROSPECTIVAMENTE
        self._evaluate_pending_overrides(timestamp, symbol, candle_data['close'])
    
    def _execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        price: float,
        confidence: float,
        is_l3_filtered: bool,
        l2_info: Optional[Dict] = None,
        l3_decision: Optional[L3OverrideDecision] = None
    ) -> bool:
        """Ejecuta un trade"""
        
        positions = self.positions_l3 if is_l3_filtered else self.positions_l2
        balance = self.balance_l3_filtered if is_l3_filtered else self.balance_l2_only
        
        try:
            if side.upper() == 'BUY':
                if symbol in positions:
                    return False
                
                quantity = self._calculate_position_size(price, confidence, balance)
                if quantity <= 0:
                    return False
                
                execution_price = price * (1 + self.slippage)
                notional = quantity * execution_price
                fees = notional * self.fee
                
                new_balance = balance - notional - fees
                if is_l3_filtered:
                    self.balance_l3_filtered = new_balance
                else:
                    self.balance_l2_only = new_balance
                
                self.trade_counter += 1
                
                positions[symbol] = TradeRecord(
                    trade_id=self.trade_counter,
                    timestamp=timestamp,
                    symbol=symbol,
                    side='BUY',
                    entry_price=execution_price,
                    quantity=quantity,
                    confidence=confidence,
                    l2_approved=True,
                    l3_override=(l3_decision is not None),
                    l3_override_type=l3_decision.override_type if l3_decision else None,
                    fees=fees,
                    stop_loss=l2_info.get('stop_loss') if l2_info else None,
                    take_profit=l2_info.get('take_profit') if l2_info else None
                )
                
                return True
                
            elif side.upper() == 'SELL':
                if symbol not in positions:
                    return False
                
                position = positions[symbol]
                execution_price = price * (1 - self.slippage)
                
                entry_value = position.entry_price * position.quantity
                exit_value = execution_price * position.quantity
                exit_fees = exit_value * self.fee
                
                pnl = exit_value - entry_value - position.fees - exit_fees
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                
                new_balance = (self.balance_l3_filtered if is_l3_filtered else self.balance_l2_only) + exit_value - exit_fees
                if is_l3_filtered:
                    self.balance_l3_filtered = new_balance
                else:
                    self.balance_l2_only = new_balance
                
                hold_time = (timestamp - position.timestamp).total_seconds() / 60
                
                position.exit_price = execution_price
                position.pnl = pnl
                position.pnl_pct = pnl_pct
                position.exit_timestamp = timestamp
                position.hold_time_minutes = hold_time
                position.is_open = False
                position.fees += exit_fees
                
                if is_l3_filtered:
                    self.closed_trades_l3.append(position)
                    # Actualizar PnL realizado en la decisión L3
                    if l3_decision:
                        l3_decision.realized_pnl = pnl
                else:
                    self.closed_trades_l2.append(position)
                
                del positions[symbol]
                return True
        
        except Exception as e:
            logger.error(f"❌ Error ejecutando trade: {e}")
        
        return False
    
    def _evaluate_pending_overrides(self, current_timestamp: datetime, symbol: str, current_price: float):
        """Evalúa retrospectivamente las decisiones de L3"""
        
        keys_to_remove = []
        
        for key, decision in self.pending_overrides.items():
            if not key.startswith(symbol):
                continue
            
            # Verificar si han pasado suficientes barras
            decision_time = decision.timestamp
            elapsed_bars = 0
            
            for ts, price in self.price_history[symbol]:
                if ts > decision_time:
                    elapsed_bars += 1
            
            if elapsed_bars >= self.forward_bars:
                # Calcular resultado hipotético de L2
                entry_price = decision.l2_price
                
                # Simular posición larga/corta
                if decision.l2_signal == 'BUY':
                    would_pnl = (current_price - entry_price) / entry_price
                else:  # SELL
                    would_pnl = (entry_price - current_price) / entry_price
                
                # Aplicar fees
                would_pnl *= 0.998  # Aproximación de fees
                
                decision.would_pnl = would_pnl
                decision.would_be_winner = (would_pnl > 0)
                
                # Determinar si el override fue correcto
                if decision.blocked:
                    # Bloqueo fue correcto si la señal hubiera sido perdedora
                    decision.override_correct = not decision.would_be_winner
                    decision.alpha_contribution = -would_pnl if would_pnl < 0 else -would_pnl
                else:
                    # Permiso fue correcto si la señal hubiera sido ganadora
                    decision.override_correct = decision.would_be_winner
                    decision.alpha_contribution = would_pnl if would_pnl > 0 else would_pnl
                
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pending_overrides[key]
    
    def _update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Actualiza las curvas de equity"""
        # L2 only
        equity_l2 = self.balance_l2_only
        for symbol, position in self.positions_l2.items():
            if symbol in market_prices:
                equity_l2 += position.quantity * market_prices[symbol]
        self.equity_curve_l2.append((timestamp, equity_l2))
        
        # L3 filtered
        equity_l3 = self.balance_l3_filtered
        for symbol, position in self.positions_l3.items():
            if symbol in market_prices:
                equity_l3 += position.quantity * market_prices[symbol]
        self.equity_curve_l3.append((timestamp, equity_l3))
        
        # Actualizar portfolio state
        self.portfolio_state['total_capital'] = equity_l3
        self.portfolio_state['available_capital'] = self.balance_l3_filtered
        self.portfolio_state['drawdown'] = self._calculate_current_drawdown()
    
    def _calculate_current_drawdown(self) -> float:
        """Calcula drawdown actual"""
        if not self.equity_curve_l3:
            return 0.0
        current = self.equity_curve_l3[-1][1]
        peak = max(e[1] for e in self.equity_curve_l3)
        return (peak - current) / peak if peak > 0 else 0.0
    
    def _get_current_allocation(self) -> Dict[str, float]:
        """Obtiene allocación actual"""
        total = self.portfolio_state.get('total_capital', self.initial_balance)
        if total == 0:
            return {'BTCUSDT': 0, 'ETHUSDT': 0, 'USDT': 1.0}
        
        allocation = {}
        for symbol, position in self.positions_l3.items():
            allocation[symbol] = (position.quantity * position.entry_price) / total
        
        usdt_pct = self.balance_l3_filtered / total
        allocation['USDT'] = usdt_pct
        
        return allocation
    
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
            if symbol in self.positions_l2:
                self._execute_trade(timestamp, symbol, 'SELL', price, 0.5, False)
            if symbol in self.positions_l3:
                self._execute_trade(timestamp, symbol, 'SELL', price, 0.5, True)
    
    def calculate_metrics(self) -> L3BacktestMetrics:
        """Calcula métricas finales de L3"""
        metrics = L3BacktestMetrics()
        
        # Métricas básicas
        metrics.total_trades = len(self.closed_trades_l3)
        metrics.winning_trades = sum(1 for t in self.closed_trades_l3 if t.pnl > 0)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        
        pnl_values_l3 = [t.pnl for t in self.closed_trades_l3]
        metrics.total_pnl = sum(pnl_values_l3)
        metrics.return_pct = (metrics.total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        # Métricas de overrides
        evaluated_decisions = [d for d in self.l3_decisions if d.would_be_winner is not None]
        metrics.total_overrides = len(evaluated_decisions)
        
        blocks = [d for d in evaluated_decisions if d.blocked]
        allows = [d for d in evaluated_decisions if not d.blocked]
        
        # Bloqueos correctos (bloquearon señales perdedoras)
        metrics.blocks_correct = sum(1 for d in blocks if not d.would_be_winner)
        metrics.blocks_incorrect = sum(1 for d in blocks if d.would_be_winner)
        
        # Permisos correctos (permitieron señales ganadoras)
        metrics.allows_correct = sum(1 for d in allows if d.would_be_winner)
        metrics.allows_incorrect = sum(1 for d in allows if not d.would_be_winner)
        
        # Porcentajes
        total_blocks = len(blocks)
        total_allows = len(allows)
        
        metrics.pct_blocks_correct = metrics.blocks_correct / total_blocks if total_blocks > 0 else 0
        metrics.pct_blocks_incorrect = metrics.blocks_incorrect / total_blocks if total_blocks > 0 else 0
        metrics.pct_allows_correct = metrics.allows_correct / total_allows if total_allows > 0 else 0
        metrics.pct_allows_incorrect = metrics.allows_incorrect / total_allows if total_allows > 0 else 0
        
        # Calcular alpha de L3
        pnl_l2 = sum(t.pnl for t in self.closed_trades_l2)
        pnl_l3 = sum(t.pnl for t in self.closed_trades_l3)
        
        metrics.l3_alpha_contribution = pnl_l3 - pnl_l2
        metrics.l3_alpha_pct = (metrics.l3_alpha_contribution / abs(pnl_l2) * 100) if pnl_l2 != 0 else 0
        
        # Protección neta
        metrics.losses_prevented = sum(abs(d.would_pnl) for d in blocks if not d.would_be_winner and d.would_pnl)
        metrics.gains_prevented = sum(abs(d.would_pnl) for d in blocks if d.would_be_winner and d.would_pnl)
        metrics.net_protection = metrics.losses_prevented - metrics.gains_prevented
        
        return metrics


# =============================================================================
# GENERACIÓN DE GRÁFICOS
# =============================================================================

def generate_l3_analysis_chart(
    engine: L3BacktestEngine,
    output_path: str,
    metrics: L3BacktestMetrics
):
    """Genera gráfico de análisis de decisiones L3"""
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. Equity Curves
        ax1 = axes[0]
        
        if engine.equity_curve_l2:
            times_l2, values_l2 = zip(*engine.equity_curve_l2)
            ax1.plot(times_l2, values_l2, label='L1+L2 (sin L3)', color='blue', alpha=0.7, linewidth=1.5)
        
        if engine.equity_curve_l3:
            times_l3, values_l3 = zip(*engine.equity_curve_l3)
            ax1.plot(times_l3, values_l3, label='L1+L2+L3 (con filtro)', color='green', linewidth=2)
        
        ax1.axhline(y=engine.initial_balance, color='gray', linestyle='--', alpha=0.5, label='Balance inicial')
        ax1.set_ylabel('Equity (USDT)')
        ax1.set_title('Comparación de Equity Curves: L1+L2 vs L1+L2+L3')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Decisiones de L3 sobre el tiempo
        ax2 = axes[1]
        
        if engine.l3_decisions:
            decision_times = [d.timestamp for d in engine.l3_decisions]
            decision_types = [1 if d.blocked else 0 for d in engine.l3_decisions]
            decision_colors = ['red' if d.blocked else 'green' for d in engine.l3_decisions]
            
            ax2.scatter(decision_times, decision_types, c=decision_colors, alpha=0.6, s=30)
            ax2.set_ylim(-0.5, 1.5)
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['ALLOW', 'BLOCK'])
            ax2.set_ylabel('Decisión L3')
            ax2.set_title('Decisiones de L3 a lo largo del tiempo (Rojo=Bloqueo, Verde=Permiso)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Métricas de override correcto/incorrecto
        ax3 = axes[2]
        
        categories = ['Bloqueos\nCorrectos', 'Bloqueos\nIncorrectos', 'Permisos\nCorrectos', 'Permisos\nIncorrectos']
        values = [
            metrics.blocks_correct,
            metrics.blocks_incorrect,
            metrics.allows_correct,
            metrics.allows_incorrect
        ]
        colors = ['darkgreen', 'darkred', 'green', 'red']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Número de Decisiones')
        ax3.set_title('Efectividad de Decisiones de L3')
        
        # Añadir valores sobre las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Gráfico guardado en: {output_path}")
        
    except ImportError:
        logger.warning("⚠️ matplotlib no instalado, no se generó gráfico")
    except Exception as e:
        logger.error(f"❌ Error generando gráfico: {e}")


def generate_alpha_chart(
    engine: L3BacktestEngine,
    output_path: str,
    metrics: L3BacktestMetrics
):
    """Genera gráfico de contribución alpha de L3"""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Contribución Alpha
        ax1 = axes[0]
        
        labels = ['Pérdidas\nEvitadas', 'Ganancias\nEvitadas', 'Protección\nNeta']
        values = [metrics.losses_prevented, -metrics.gains_prevented, metrics.net_protection]
        colors = ['green', 'red', 'blue' if metrics.net_protection > 0 else 'orange']
        
        bars = ax1.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Valor (USDT)')
        ax1.set_title(f'Contribución de L3 (Alpha: {metrics.l3_alpha_contribution:+.2f} USDT)')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
        
        # 2. Porcentajes de efectividad
        ax2 = axes[1]
        
        labels_pct = ['Bloqueos\nCorrectos %', 'Permisos\nCorrectos %']
        values_pct = [metrics.pct_blocks_correct * 100, metrics.pct_allows_correct * 100]
        colors_pct = ['darkgreen', 'green']
        
        bars = ax2.bar(labels_pct, values_pct, color=colors_pct, alpha=0.7, edgecolor='black')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Porcentaje (%)')
        ax2.set_title('Efectividad de Decisiones de L3')
        
        for bar, value in zip(bars, values_pct):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Gráfico Alpha guardado en: {output_path}")
        
    except ImportError:
        logger.warning("⚠️ matplotlib no instalado, no se generó gráfico alpha")
    except Exception as e:
        logger.error(f"❌ Error generando gráfico alpha: {e}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def run_l3_backtest(
    symbols: List[str] = None,
    timeframe: str = '1m',
    days: int = 30,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    output_dir: str = None
) -> Dict[str, Any]:
    """Ejecuta backtest completo de L3"""
    
    symbols = symbols or DEFAULT_SYMBOLS
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO BACKTEST L3 OVERRIDE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Días: {days}")
    logger.info(f"Balance inicial: ${initial_balance:.2f}")
    
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
    
    # Inicializar motor
    engine = L3BacktestEngine(initial_balance=initial_balance)
    
    # Procesar velas
    logger.info("\n⏳ Procesando pipeline L1→L2→L3...")
    
    all_timestamps = set()
    for sym, df in market_data.items():
        all_timestamps.update(df.index)
    
    sorted_timestamps = sorted(all_timestamps)
    logger.info(f"Total de velas a procesar: {len(sorted_timestamps)}")
    
    progress_interval = max(1, len(sorted_timestamps) // 10)
    
    for i, timestamp in enumerate(sorted_timestamps):
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
        
        if (i + 1) % progress_interval == 0:
            pct = (i + 1) / len(sorted_timestamps) * 100
            logger.info(f"  Progreso: {pct:.1f}% ({i+1}/{len(sorted_timestamps)})")
    
    # Cerrar posiciones
    logger.info("\n🔚 Cerrando posiciones abiertas...")
    final_prices = {sym: df['close'].iloc[-1] for sym, df in market_data.items()}
    engine.close_all_positions(datetime.now(), final_prices)
    
    # Calcular métricas
    logger.info("\n📈 Calculando métricas de L3...")
    metrics = engine.calculate_metrics()
    
    # Mostrar resultados
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESULTADOS DEL BACKTEST L3")
    logger.info("=" * 80)
    
    logger.info("\n📈 Métricas Generales:")
    logger.info(f"  Total trades (L3):      {metrics.total_trades}")
    logger.info(f"  Win rate:               {metrics.win_rate:.2%}")
    logger.info(f"  PnL total (L3):         ${metrics.total_pnl:.2f}")
    logger.info(f"  Retorno:                {metrics.return_pct:.2f}%")
    
    logger.info("\n🎯 Decisiones de L3:")
    logger.info(f"  Total overrides evaluados: {metrics.total_overrides}")
    logger.info(f"  Bloqueos correctos:        {metrics.blocks_correct} ({metrics.pct_blocks_correct:.1%})")
    logger.info(f"  Bloqueos incorrectos:      {metrics.blocks_incorrect} ({metrics.pct_blocks_incorrect:.1%})")
    logger.info(f"  Permisos correctos:        {metrics.allows_correct} ({metrics.pct_allows_correct:.1%})")
    logger.info(f"  Permisos incorrectos:      {metrics.allows_incorrect} ({metrics.pct_allows_incorrect:.1%})")
    
    logger.info("\n💎 Alpha de L3:")
    logger.info(f"  Contribución neta:    ${metrics.l3_alpha_contribution:+.2f}")
    logger.info(f"  Como % del PnL L2:    {metrics.l3_alpha_pct:+.2f}%")
    logger.info(f"  Pérdidas evitadas:    ${metrics.losses_prevented:.2f}")
    logger.info(f"  Ganancias evitadas:   ${metrics.gains_prevented:.2f}")
    logger.info(f"  Protección neta:      ${metrics.net_protection:+.2f}")
    
    # Alertas si contribución negativa
    logger.info("\n" + "=" * 80)
    if metrics.l3_alpha_contribution < 0:
        logger.warning("🚨🚨🚨 ALERTA CRÍTICA 🚨🚨🚨")
        logger.warning("=" * 80)
        logger.warning("L3 TIENE CONTRIBUCIÓN NEGATIVA AL SISTEMA")
        logger.warning(f"Alpha de L3: ${metrics.l3_alpha_contribution:.2f}")
        logger.warning(f"Esto significa que el sistema funcionaría MEJOR sin L3")
        logger.warning("\nPosibles causas:")
        logger.warning("  - L3 está bloqueando demasiadas señales ganadoras")
        logger.warning("  - Los umbrales de confianza de L3 son demasiado agresivos")
        logger.warning("  - L3 está en régimen incorrecto frecuentemente")
        logger.warning("\nRecomendaciones:")
        logger.warning("  - Revisar fix_l3_dominance.py")
        logger.warning("  - Ajustar thresholds de bloqueo en should_l3_block_l2_signals()")
        logger.warning("  - Considerar reducir la confianza mínima para permitir L2")
    else:
        logger.info("✅ L3 tiene contribución positiva al sistema")
        logger.info(f"   Alpha generado: ${metrics.l3_alpha_contribution:.2f}")
    logger.info("=" * 80)
    
    # Exportar resultados
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV de decisiones L3
    decisions_csv = os.path.join(output_dir, f"l3_decisions_{timestamp_str}.csv")
    decisions_df = pd.DataFrame([d.to_dict() for d in engine.l3_decisions])
    decisions_df.to_csv(decisions_csv, index=False)
    logger.info(f"\n💾 Decisiones L3 exportadas a: {decisions_csv}")
    
    # JSON de métricas
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'symbols': symbols,
            'timeframe': timeframe,
            'days': days,
            'initial_balance': initial_balance
        },
        'metrics': metrics.to_dict(),
        'summary': {
            'l3_contribution': 'POSITIVE' if metrics.l3_alpha_contribution > 0 else 'NEGATIVE',
            'alpha_usdt': metrics.l3_alpha_contribution,
            'block_accuracy': metrics.pct_blocks_correct,
            'allow_accuracy': metrics.pct_allows_correct
        }
    }
    
    results_json = os.path.join(output_dir, f"l3_backtest_{timestamp_str}.json")
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Resultados JSON guardados en: {results_json}")
    
    # Generar gráficos
    logger.info("\n📊 Generando gráficos...")
    chart_path = os.path.join(output_dir, f"l3_analysis_{timestamp_str}.png")
    generate_l3_analysis_chart(engine, chart_path, metrics)
    
    alpha_chart_path = os.path.join(output_dir, f"l3_alpha_{timestamp_str}.png")
    generate_alpha_chart(engine, alpha_chart_path, metrics)
    
    return results


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Backtesting L3 Override - Evaluación de contribución de L3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Backtest con configuración por defecto
  python backtesting/backtest_l3_override.py

  # Backtest con parámetros personalizados
  python backtesting/backtest_l3_override.py --symbols BTCUSDT --days 60 --balance 5000

  # Backtest con más datos para mejor evaluación
  python backtesting/backtest_l3_override.py --days 90
        """
    )
    
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS,
                       help=f'Símbolos a testear (default: {DEFAULT_SYMBOLS})')
    parser.add_argument('--timeframe', default=DEFAULT_TIMEFRAME,
                       help=f'Timeframe de las velas (default: {DEFAULT_TIMEFRAME})')
    parser.add_argument('--days', type=int, default=30,
                       help='Días de datos históricos (default: 30)')
    parser.add_argument('--balance', type=float, default=DEFAULT_INITIAL_BALANCE,
                       help=f'Balance inicial en USDT (default: {DEFAULT_INITIAL_BALANCE})')
    parser.add_argument('--output-dir', default=RESULTS_DIR,
                       help=f'Directorio de salida (default: {RESULTS_DIR})')
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(run_l3_backtest(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            initial_balance=args.balance,
            output_dir=args.output_dir
        ))
        
        if result:
            print("\n✅ Backtest L3 completado exitosamente")
            return 0
        else:
            print("\n❌ El backtest falló")
            return 1
            
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
