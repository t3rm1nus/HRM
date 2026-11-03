#!/usr/bin/env python3
"""
Backtesting L2 Standalone - Script independiente para backtesting de filtros L2

Este script:
1. Carga el CSV de señales L1 generado por backtest_l1_standalone.py
2. Instancia los componentes de l2_tactic/: L2TacticProcessor (con modelo RL real),
   position_sizer.py, risk_overlay.py
3. Para cada señal L1 en el CSV, la pasa a L2 y observa cómo L2 modifica:
   - Timing de entrada
   - Tamaño de posición
   - Stop loss dinámico
4. Compara dos equity curves: "L1 solo" vs "L1 + filtro L2"
5. Métricas adicionales de L2: señales rechazadas, mejora/empeora Sharpe, reduce drawdown
6. Test de robustez: ejecuta el mismo backtest con 3 conjuntos de parámetros L2 distintos

IMPORTANTE: NO importa nada de l3_strategy/ - L2 completamente aislado
FIX: Ahora usa L2TacticProcessor con el modelo RL real (claude.zip) para filtrado
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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import copy

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path para importar módulos de l2_tactic
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importar SOLO de l2_tactic - NADA de l3_strategy
from l2_tactic.signal_generator_refactored import L2TacticProcessor
from l2_tactic.position_sizer import PositionSizerManager
from l2_tactic.risk_overlay import RiskOverlay
from l2_tactic.models import TacticalSignal, PositionSize
from l2_tactic.config import L2Config as L2ConfigClass
from l2_tactic.ai_model_integration import AIModelWrapper

# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

DEFAULT_INITIAL_BALANCE = 3000.0  # USDT
DEFAULT_FEE = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Path del modelo L2 - se determina dinámicamente según configuración
L2_MODEL_PATH = None

def get_l2_model_path(model_key: Optional[str] = None) -> Optional[str]:
    """
    Obtiene el path del modelo L2 activo.
    
    Orden de prioridad:
    1. Modelo especificado por key (parámetro)
    2. Modelo configurado en L2Config.ai_model.model_path
    3. Fallback: primer modelo existente en lista hardcodeada
    
    Args:
        model_key: Clave del modelo (claude, deepseek, gemini, gpt, grok, kimi)
    
    Returns:
        Path al archivo del modelo o None si no se encuentra
    """
    # Prioridad 1: Modelo especificado por key
    if model_key:
        config = L2ConfigClass()
        if model_key in config.ai_model.available_models:
            model_path = config.ai_model.available_models[model_key]
            if os.path.exists(model_path):
                logger.info(f"✅ Usando modelo especificado: {model_key} -> {model_path}")
                return model_path
            else:
                logger.warning(f"⚠️ Modelo {model_key} no encontrado en: {model_path}")
        else:
            logger.warning(f"⚠️ Modelo '{model_key}' no disponible. Disponibles: {list(config.ai_model.available_models.keys())}")
    
    # Prioridad 2: Modelo configurado en L2Config
    try:
        config = L2ConfigClass()
        config_path = config.ai_model.model_path
        if os.path.exists(config_path):
            logger.info(f"✅ Usando modelo de configuración L2Config: {config_path}")
            return config_path
        else:
            logger.warning(f"⚠️ Modelo en configuración no encontrado: {config_path}")
    except Exception as e:
        logger.warning(f"⚠️ Error leyendo configuración: {e}")
    
    # Prioridad 3: Fallback a lista hardcodeada
    fallback_candidates = [
        os.path.join(REPO_ROOT, 'models', 'L2', 'claude.zip'),
        os.path.join(REPO_ROOT, 'models', 'L2', 'deepseek.zip'),
        os.path.join(REPO_ROOT, 'models', 'L2', 'gemini.zip'),
        os.path.join(REPO_ROOT, 'models', 'L2', 'gpt.zip'),
        os.path.join(REPO_ROOT, 'models', 'L2', 'grok.zip'),
        os.path.join(REPO_ROOT, 'models', 'L2', 'kimi.zip'),
    ]
    for path in fallback_candidates:
        if os.path.exists(path):
            logger.info(f"✅ Usando modelo fallback: {path}")
            return path
    
    logger.error("❌ No se encontró ningún modelo L2 disponible")
    return None

def get_active_model_name() -> str:
    """Obtiene el nombre del modelo activo basado en el path"""
    if L2_MODEL_PATH:
        return os.path.basename(L2_MODEL_PATH).replace('.zip', '')
    return 'none'

# =============================================================================
# DATA CLASSES PARA RESULTADOS
# =============================================================================

@dataclass
class L2BacktestConfig:
    """Configuración para backtest L2"""
    name: str
    # Parámetros de Position Sizer
    kelly_fraction: float = 0.5
    kelly_cap: float = 0.25
    vol_target: float = 0.20
    max_risk_per_trade: float = 0.01
    min_position_notional: float = 100.0
    max_position_notional: float = 1_000_000.0
    max_notional_pct_of_adv: float = 0.02

    # Parámetros de Risk Overlay
    max_drawdown_limit: float = 0.12
    max_expected_vol: float = 0.20
    correlation_limit: float = 0.95

    # Parámetros de filtrado L2
    min_l2_confidence: float = 0.6
    use_volatility_filter: bool = True
    use_correlation_filter: bool = True
    use_drawdown_filter: bool = True

    # Usar modelo RL para filtrado (True = usa L2TacticProcessor con claude.zip)
    use_rl_model: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class L2Decision:
    """Decisión de L2 sobre una señal L1"""
    timestamp: datetime
    symbol: str
    l1_signal: str
    l1_confidence: float
    l1_price: float

    # Decision L2
    l2_approved: bool
    l2_rejection_reason: Optional[str] = None
    l2_modified_side: Optional[str] = None
    l2_modified_confidence: Optional[float] = None

    # Ajustes de posición
    l2_position_size: Optional[float] = None
    l2_notional: Optional[float] = None
    l2_stop_loss: Optional[float] = None
    l2_take_profit: Optional[float] = None
    l2_leverage: Optional[float] = None

    # Métricas
    processing_time_ms: float = 0.0
    filtered_by_rl: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'l1_signal': self.l1_signal,
            'l1_confidence': self.l1_confidence,
            'l1_price': self.l1_price,
            'l2_approved': self.l2_approved,
            'l2_rejection_reason': self.l2_rejection_reason,
            'l2_modified_side': self.l2_modified_side,
            'l2_modified_confidence': self.l2_modified_confidence,
            'l2_position_size': self.l2_position_size,
            'l2_notional': self.l2_notional,
            'l2_stop_loss': self.l2_stop_loss,
            'l2_take_profit': self.l2_take_profit,
            'l2_leverage': self.l2_leverage,
            'processing_time_ms': self.processing_time_ms,
            'filtered_by_rl': self.filtered_by_rl
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
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    hold_time_minutes: float = 0.0
    exit_timestamp: Optional[datetime] = None
    is_open: bool = True
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class BacktestMetrics:
    """Métricas de rendimiento del backtest"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    avg_hold_time_minutes: float = 0.0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    start_balance: float = 0.0
    end_balance: float = 0.0
    return_pct: float = 0.0

    # Métricas específicas de L2
    l1_signals_processed: int = 0
    l2_signals_approved: int = 0
    l2_signals_rejected: int = 0
    l2_rejection_rate: float = 0.0
    avg_position_size_l1: float = 0.0
    avg_position_size_l2: float = 0.0
    avg_leverage_l2: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2f}",
            'max_drawdown_pct': f"{self.max_drawdown_pct:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'total_pnl': f"{self.total_pnl:.2f}",
            'avg_hold_time_minutes': f"{self.avg_hold_time_minutes:.1f}",
            'avg_trade_pnl': f"{self.avg_trade_pnl:.2f}",
            'best_trade': f"{self.best_trade:.2f}",
            'worst_trade': f"{self.worst_trade:.2f}",
            'start_balance': f"{self.start_balance:.2f}",
            'end_balance': f"{self.end_balance:.2f}",
            'return_pct': f"{self.return_pct:.2%}",
            'l1_signals_processed': self.l1_signals_processed,
            'l2_signals_approved': self.l2_signals_approved,
            'l2_signals_rejected': self.l2_signals_rejected,
            'l2_rejection_rate': f"{self.l2_rejection_rate:.2%}",
            'avg_position_size_l1': f"{self.avg_position_size_l1:.4f}",
            'avg_position_size_l2': f"{self.avg_position_size_l2:.4f}",
            'avg_leverage_l2': f"{self.avg_leverage_l2:.2f}x"
        }


# =============================================================================
# CONFIGURACIONES DE ROBUSTEZ
# =============================================================================

ROBUSTNESS_CONFIGS = {
    'conservative': L2BacktestConfig(
        name='Conservador',
        kelly_fraction=0.3,
        kelly_cap=0.15,
        vol_target=0.15,
        max_risk_per_trade=0.005,
        max_drawdown_limit=0.08,
        max_expected_vol=0.15,
        min_l2_confidence=0.7,
        use_volatility_filter=True,
        use_correlation_filter=True,
        use_drawdown_filter=True,
        use_rl_model=True
    ),
    'balanced': L2BacktestConfig(
        name='Balanceado',
        kelly_fraction=0.5,
        kelly_cap=0.25,
        vol_target=0.20,
        max_risk_per_trade=0.01,
        max_drawdown_limit=0.12,
        max_expected_vol=0.20,
        min_l2_confidence=0.6,
        use_volatility_filter=True,
        use_correlation_filter=True,
        use_drawdown_filter=True,
        use_rl_model=True
    ),
    'aggressive': L2BacktestConfig(
        name='Agresivo',
        kelly_fraction=0.7,
        kelly_cap=0.35,
        vol_target=0.25,
        max_risk_per_trade=0.02,
        max_drawdown_limit=0.18,
        max_expected_vol=0.25,
        min_l2_confidence=0.5,
        use_volatility_filter=False,
        use_correlation_filter=False,
        use_drawdown_filter=False,
        use_rl_model=True
    )
}


# =============================================================================
# MOTOR DE BACKTESTING L2
# =============================================================================

class L2BacktestEngine:
    """Motor de backtesting para filtros L2 - usa modelo RL real (claude.zip)"""

    def __init__(
        self,
        config: L2BacktestConfig,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE,
        position_size_pct: float = 0.1
    ):
        self.config = config
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        self.position_size_pct = position_size_pct

        # Inicializar componentes L2
        self._init_l2_components()

        # Estado del backtest
        self.reset_state()

    def _init_l2_components(self):
        """Inicializa los componentes L2 con la configuración dada"""
        # Crear configuración L2
        l2_config = L2ConfigClass()

        # Aplicar configuraciones personalizadas
        l2_config.kelly_fraction = self.config.kelly_fraction
        l2_config.kelly_cap = self.config.kelly_cap
        l2_config.vol_target = self.config.vol_target
        l2_config.max_risk_per_trade = self.config.max_risk_per_trade
        l2_config.min_position_notional = self.config.min_position_notional
        l2_config.max_position_notional = self.config.max_position_notional
        l2_config.max_notional_pct_of_adv = self.config.max_notional_pct_of_adv

        # Inicializar componentes estáticos
        self.position_sizer = PositionSizerManager(config=l2_config)
        self.risk_overlay = RiskOverlay(config=l2_config)

        # Ajustar parámetros de RiskOverlay
        self.risk_overlay.max_drawdown_limit = self.config.max_drawdown_limit
        self.risk_overlay.max_expected_vol = self.config.max_expected_vol
        self.risk_overlay.correlation_limit = self.config.correlation_limit

        # ===================================================================
        # FIX CRÍTICO: Inicializar modelo RL real usando AIModelWrapper
        # ===================================================================
        self.l2_processor = None
        self.ai_model_wrapper = None
        if self.config.use_rl_model and L2_MODEL_PATH:
            try:
                # Crear configuración temporal con el modelo seleccionado
                temp_config = L2ConfigClass()
                temp_config.ai_model.model_path = L2_MODEL_PATH
                
                # Inicializar el wrapper del modelo RL
                self.ai_model_wrapper = AIModelWrapper(temp_config)
                
                if self.ai_model_wrapper.model_loaded:
                    model_name = os.path.basename(L2_MODEL_PATH)
                    logger.info(f"✅ Modelo RL cargado: {model_name}")
                    
                    # Mostrar info del modelo
                    model_info = self.ai_model_wrapper.get_model_info()
                    logger.info(f"🧠 Modelo RL | Tipo: {model_info.get('model_type', 'unknown')}")
                    logger.info(f"🧠 Modelo RL | Observation Space: {model_info.get('observation_space', 'unknown')}")
                else:
                    logger.warning("⚠️ No se pudo cargar el modelo RL")
                    self.ai_model_wrapper = None

            except Exception as e:
                logger.warning(f"⚠️ Error inicializando modelo RL: {e}")
                logger.warning("   Continuando sin filtro RL - solo reglas estáticas")
                self.ai_model_wrapper = None
        else:
            if not L2_MODEL_PATH:
                logger.warning("⚠️ No se encontró modelo L2 (claude.zip / deepseek.zip)")
            logger.info("ℹ️ Usando solo filtros estáticos (sin modelo RL)")

        logger.info(f"✅ Componentes L2 inicializados - Config: {self.config.name}")

    def reset_state(self):
        """Resetea el estado del motor"""
        # Balance y equity
        self.balance_l1_only = self.initial_balance
        self.balance_l2_filtered = self.initial_balance

        # Posiciones
        self.positions_l1: Dict[str, TradeRecord] = {}
        self.positions_l2: Dict[str, TradeRecord] = {}

        # Trades cerrados
        self.closed_trades_l1: List[TradeRecord] = []
        self.closed_trades_l2: List[TradeRecord] = []

        # Equity curves
        self.equity_curve_l1: List[Tuple[datetime, float]] = []
        self.equity_curve_l2: List[Tuple[datetime, float]] = []

        # Decisiones L2
        self.l2_decisions: List[L2Decision] = []

        # Contadores
        self.trade_counter = 0
        self.l1_signals_count = 0
        self.l2_approved_count = 0
        self.l2_rejected_count = 0
        self.rl_rejections = 0

        # Portfolio state para L2
        self.portfolio_state = {
            'total_capital': self.initial_balance,
            'available_capital': self.initial_balance,
            'drawdown': 0.0,
            'positions': {}
        }

        # Market data histórica para cálculos
        self.market_data_history: Dict[str, List[Dict]] = defaultdict(list)

        # Buffer OHLCV para L2TacticProcessor (necesita DataFrame)
        self.ohlcv_buffer: Dict[str, List[Dict]] = defaultdict(list)

    async def _query_rl_model(
        self,
        symbol: str,
        l1_signal: str,
        l1_confidence: float,
        l1_price: float,
        timestamp: datetime
    ) -> Optional[Tuple[str, float]]:
        """
        Consulta el modelo RL (AIModelWrapper) para validar/modificar la señal L1.

        Returns:
            Tuple (action, confidence) si el modelo aprueba, None si rechaza
        """
        if self.ai_model_wrapper is None or not self.ai_model_wrapper.model_loaded:
            return None

        try:
            # Construir DataFrame OHLCV mínimo para el procesador
            buffer = self.ohlcv_buffer.get(symbol, [])
            if len(buffer) < 30:
                # No hay suficientes datos históricos para el modelo
                return None

            # Preparar features para el modelo RL
            df = pd.DataFrame(buffer[-200:])  # últimas 200 velas máximo
            if len(df.columns) >= 6:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Extraer últimos valores como features
            latest = df.iloc[-1] if len(df) > 0 else {}
            
            # Calcular indicadores básicos
            if len(df) >= 14:
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['rsi'] = self._calculate_rsi(df['close'], 14)
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                df['macd'] = df['ema_12'] - df['ema_26']
            
            # Helper para convertir valores numpy a tipos Python nativos
            def to_native(val):
                if hasattr(val, 'item'):  # numpy scalar
                    return val.item()
                return float(val) if val is not None else 0.0
            
            # Preparar features para el modelo (convertir numpy a tipos nativos)
            features = {
                'symbol': symbol,
                'market_data': {
                    'open': to_native(latest.get('open', l1_price)),
                    'high': to_native(latest.get('high', l1_price)),
                    'low': to_native(latest.get('low', l1_price)),
                    'close': to_native(latest.get('close', l1_price)),
                    'volume': to_native(latest.get('volume', 0)),
                    'sma_20': to_native(df['sma_20'].iloc[-1]) if 'sma_20' in df else float(l1_price),
                    'sma_50': to_native(df['sma_50'].iloc[-1]) if 'sma_50' in df else float(l1_price),
                    'rsi': to_native(df['rsi'].iloc[-1]) if 'rsi' in df else 50.0,
                    'ema_12': to_native(df['ema_12'].iloc[-1]) if 'ema_12' in df else float(l1_price),
                    'ema_26': to_native(df['ema_26'].iloc[-1]) if 'ema_26' in df else float(l1_price),
                    'macd': to_native(df['macd'].iloc[-1]) if 'macd' in df else 0.0,
                }
            }

            # Consultar el modelo RL con manejo de errores robusto
            try:
                prediction = await self.ai_model_wrapper.predict_async(features)
            except Exception as model_error:
                logger.debug(f"🤖 RL Model error (continuing without filter): {model_error}")
                return None

            if prediction is None:
                return None

            # El modelo RL devuelve una predicción (-1 a 1) y confianza (0 a 1)
            # Convertir predicción a side
            pred_value = prediction.prediction  # -1 a 1
            pred_confidence = prediction.confidence  # 0 a 1
            
            # Mapear predicción a side
            if pred_value < -0.2:  # Umbral para SELL
                rl_side = 'SELL'
            elif pred_value > 0.2:  # Umbral para BUY
                rl_side = 'BUY'
            else:
                rl_side = 'HOLD'

            logger.debug(f"🤖 RL Model | {symbol}: {rl_side} (pred={pred_value:.3f}, conf={pred_confidence:.3f})")

            # El modelo RL confirma si su acción coincide con L1
            if rl_side == l1_signal.upper():
                # Combinar confianza de L1 con la del modelo RL
                combined_confidence = (l1_confidence + pred_confidence) / 2
                return (rl_side, combined_confidence)
            else:
                # Modelo RL discrepa con L1 → rechazar
                logger.debug(f"🤖 RL Model rechazó {symbol}: L1={l1_signal}, RL={rl_side}")
                return None

        except Exception as e:
            logger.debug(f"RL model query error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI simple"""
        try:
            deltas = prices.diff()
            gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
            loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50.0] * len(prices), index=prices.index)

    def _update_ohlcv_buffer(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        market_data: Optional[Dict] = None
    ):
        """Actualiza el buffer OHLCV para el modelo RL"""
        entry = {
            'timestamp': timestamp,
            'open': market_data.get('open', price) if market_data else price,
            'high': market_data.get('high', price) if market_data else price,
            'low': market_data.get('low', price) if market_data else price,
            'close': price,
            'volume': market_data.get('volume', 0) if market_data else 0
        }
        self.ohlcv_buffer[symbol].append(entry)
        # Limitar buffer a 500 velas
        if len(self.ohlcv_buffer[symbol]) > 500:
            self.ohlcv_buffer[symbol] = self.ohlcv_buffer[symbol][-500:]

    async def process_l1_signal(
        self,
        timestamp: datetime,
        symbol: str,
        l1_signal: str,
        l1_confidence: float,
        l1_price: float,
        market_data: Optional[Dict] = None
    ) -> L2Decision:
        """
        Procesa una señal L1 a través de los filtros L2.
        Ahora incluye validación con el modelo RL real.

        Returns:
            L2Decision con la decisión de L2 sobre la señal
        """
        start_time = time.time()
        self.l1_signals_count += 1

        # Actualizar buffer OHLCV
        self._update_ohlcv_buffer(symbol, timestamp, l1_price, market_data)

        decision = L2Decision(
            timestamp=timestamp,
            symbol=symbol,
            l1_signal=l1_signal,
            l1_confidence=l1_confidence,
            l1_price=l1_price,
            l2_approved=False
        )

        # Si es HOLD, no hay nada que procesar
        if l1_signal.upper() == 'HOLD':
            decision.l2_approved = False
            decision.l2_rejection_reason = 'L1_HOLD_SIGNAL'
            decision.processing_time_ms = (time.time() - start_time) * 1000
            return decision

        try:
            # Actualizar market data history
            if market_data:
                self.market_data_history[symbol].append({
                    'timestamp': timestamp,
                    'close': l1_price,
                    'open': market_data.get('open', l1_price),
                    'high': market_data.get('high', l1_price),
                    'low': market_data.get('low', l1_price),
                    'volume': market_data.get('volume', 0)
                })

            # 1. Verificar filtros de Risk Overlay
            if self.config.use_drawdown_filter:
                current_dd = self._calculate_current_drawdown()
                if current_dd > self.config.max_drawdown_limit:
                    decision.l2_rejection_reason = f'RISK_DRAWDOWN_{current_dd:.2%}'
                    self.l2_rejected_count += 1
                    decision.processing_time_ms = (time.time() - start_time) * 1000
                    self.l2_decisions.append(decision)
                    return decision

            # 2. Verificar confianza mínima L2
            if l1_confidence < self.config.min_l2_confidence:
                decision.l2_rejection_reason = f'LOW_CONFIDENCE_{l1_confidence:.2f}'
                self.l2_rejected_count += 1
                decision.processing_time_ms = (time.time() - start_time) * 1000
                self.l2_decisions.append(decision)
                return decision

            # ===================================================================
            # FIX: Consultar modelo RL real ANTES del position sizing
            # ===================================================================
            rl_action = None
            rl_confidence = l1_confidence

            if self.config.use_rl_model and self.ai_model_wrapper is not None:
                rl_result = await self._query_rl_model(
                    symbol=symbol,
                    l1_signal=l1_signal,
                    l1_confidence=l1_confidence,
                    l1_price=l1_price,
                    timestamp=timestamp
                )

                if rl_result is not None:
                    rl_action, rl_confidence = rl_result
                    logger.info(f"✅ RL Model aprobó {symbol}: {rl_action} (conf: {rl_confidence:.3f})")
                elif len(self.ohlcv_buffer.get(symbol, [])) >= 30:
                    # Modelo RL tiene datos suficientes pero rechazó → señal bloqueada
                    decision.l2_rejection_reason = 'RL_MODEL_REJECTED'
                    decision.filtered_by_rl = True
                    self.l2_rejected_count += 1
                    self.rl_rejections += 1
                    decision.processing_time_ms = (time.time() - start_time) * 1000
                    self.l2_decisions.append(decision)
                    logger.info(f"🚫 RL Model rechazó {symbol}: {l1_signal} -> NO TRADE")
                    return decision
                else:
                    logger.info(f"⚠️ RL Model no aplicado para {symbol}: datos insuficientes ({len(self.ohlcv_buffer.get(symbol, []))} < 30)")
                # Si no hay suficientes datos para el RL, continuar con filtros estáticos

            # 3. Crear TacticalSignal para procesamiento L2
            tactical_signal = TacticalSignal(
                symbol=symbol,
                side=l1_signal.lower(),
                strength=rl_confidence,
                confidence=rl_confidence,
                signal_type='l1_processed',
                source='l1_backtest',
                features=self._extract_features(symbol, l1_price),
                timestamp=timestamp
            )
            tactical_signal.price = l1_price
            tactical_signal.stop_loss = None
            tactical_signal.take_profit = None

            # 4. Calcular tamaño de posición con L2
            market_features = self._build_market_features(symbol)

            position_size_result = await self.position_sizer.calculate_position_size(
                signal=tactical_signal,
                market_features=market_features,
                portfolio_state=self.portfolio_state
            )

            if position_size_result is None:
                decision.l2_rejection_reason = 'POSITION_SIZING_FAILED'
                self.l2_rejected_count += 1
                decision.processing_time_ms = (time.time() - start_time) * 1000
                self.l2_decisions.append(decision)
                return decision

            # 5. Aprobar señal con modificaciones L2
            decision.l2_approved = True
            decision.l2_modified_side = l1_signal.lower()
            decision.l2_modified_confidence = rl_confidence * 0.95
            decision.l2_position_size = position_size_result.size
            decision.l2_notional = position_size_result.notional
            decision.l2_stop_loss = getattr(position_size_result, 'stop_loss', None)
            decision.l2_take_profit = getattr(position_size_result, 'take_profit', None)
            decision.l2_leverage = getattr(position_size_result, 'leverage', 1.0)

            self.l2_approved_count += 1

        except Exception as e:
            logger.error(f"❌ Error procesando señal L2: {e}")
            decision.l2_rejection_reason = f'ERROR_{str(e)[:50]}'
            self.l2_rejected_count += 1

        decision.processing_time_ms = (time.time() - start_time) * 1000
        self.l2_decisions.append(decision)
        return decision

    def _extract_features(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Extrae features de la historia de mercado"""
        features = {
            'close': current_price,
            'price': current_price
        }

        history = self.market_data_history.get(symbol, [])
        if len(history) >= 14:
            prices = [h['close'] for h in history[-20:]]

            # Calcular RSI simplificado
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

            # Calcular volatilidad
            if len(prices) >= 2:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                features['volatility'] = np.std(returns) * np.sqrt(252) if returns else 0.2

            # Último volumen
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

    def _calculate_current_drawdown(self) -> float:
        """Calcula el drawdown actual basado en la equity curve"""
        if not self.equity_curve_l2:
            return 0.0

        current_equity = self.equity_curve_l2[-1][1]
        peak = max(e[1] for e in self.equity_curve_l2)

        if peak > 0:
            return (peak - current_equity) / peak
        return 0.0

    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        price: float,
        confidence: float,
        is_l2_filtered: bool,
        l2_decision: Optional[L2Decision] = None
    ) -> bool:
        """Ejecuta un trade en la estrategia correspondiente"""

        if is_l2_filtered and l2_decision and not l2_decision.l2_approved:
            return False

        positions = self.positions_l2 if is_l2_filtered else self.positions_l1
        closed_trades = self.closed_trades_l2 if is_l2_filtered else self.closed_trades_l1
        balance_attr = 'balance_l2_filtered' if is_l2_filtered else 'balance_l1_only'
        balance = getattr(self, balance_attr)

        try:
            if side.upper() == 'BUY':
                # Verificar si ya hay posición abierta
                if symbol in positions:
                    return False

                # Calcular tamaño de posición
                if is_l2_filtered and l2_decision:
                    quantity = l2_decision.l2_position_size or 0
                    notional = l2_decision.l2_notional or 0
                else:
                    # Tamaño base para L1
                    notional = balance * self.position_size_pct * confidence
                    quantity = notional / price if price > 0 else 0

                if quantity <= 0 or notional <= 0:
                    return False

                # Aplicar slippage
                execution_price = price * (1 + self.slippage)

                # Calcular fees
                fees = notional * self.fee

                # Actualizar balance
                new_balance = balance - notional - fees
                setattr(self, balance_attr, new_balance)

                # Determinar stop loss
                if l2_decision and l2_decision.l2_stop_loss:
                    stop_loss = l2_decision.l2_stop_loss
                    take_profit = l2_decision.l2_take_profit
                else:
                    stop_loss = execution_price * 0.98  # 2% por defecto
                    take_profit = None

                # Crear trade
                self.trade_counter += 1
                positions[symbol] = TradeRecord(
                    trade_id=self.trade_counter,
                    timestamp=timestamp,
                    symbol=symbol,
                    side='BUY',
                    entry_price=execution_price,
                    quantity=quantity,
                    confidence=confidence,
                    l1_signal_type='l1_signal',
                    l2_approved=is_l2_filtered,
                    fees=fees,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

                return True

            elif side.upper() == 'SELL':
                # Verificar si hay posición abierta
                if symbol not in positions:
                    return False

                position = positions[symbol]

                # Aplicar slippage
                execution_price = price * (1 - self.slippage)

                # Calcular PnL
                entry_value = position.entry_price * position.quantity
                exit_value = execution_price * position.quantity
                exit_fees = exit_value * self.fee

                pnl = exit_value - entry_value - position.fees - exit_fees
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                # Actualizar balance
                new_balance = getattr(self, balance_attr) + exit_value - exit_fees
                setattr(self, balance_attr, new_balance)

                # Cerrar posición
                hold_time = (timestamp - position.timestamp).total_seconds() / 60

                position.exit_price = execution_price
                position.pnl = pnl
                position.pnl_pct = pnl_pct
                position.exit_timestamp = timestamp
                position.hold_time_minutes = hold_time
                position.is_open = False
                position.fees += exit_fees

                closed_trades.append(position)
                del positions[symbol]

                return True

        except Exception as e:
            logger.error(f"❌ Error ejecutando trade: {e}")

        return False

    def check_stop_losses(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Comprueba y ejecuta stop-losses activos"""
        for strategy_positions, is_l2 in [(self.positions_l1, False), (self.positions_l2, True)]:
            symbols_to_close = []
            for symbol, position in strategy_positions.items():
                if symbol in market_prices and position.stop_loss:
                    current_price = market_prices[symbol]
                    if position.side == 'BUY' and current_price <= position.stop_loss:
                        symbols_to_close.append(symbol)

            for symbol in symbols_to_close:
                self.execute_trade(
                    timestamp=timestamp,
                    symbol=symbol,
                    side='SELL',
                    price=market_prices[symbol],
                    confidence=0.5,
                    is_l2_filtered=is_l2
                )

    def update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Actualiza las curvas de equity"""
        # Verificar stop losses primero
        self.check_stop_losses(timestamp, market_prices)

        # L1 only
        equity_l1 = self.balance_l1_only
        for symbol, position in self.positions_l1.items():
            if symbol in market_prices:
                position_value = position.quantity * market_prices[symbol]
                equity_l1 += position_value
        self.equity_curve_l1.append((timestamp, equity_l1))

        # L2 filtered
        equity_l2 = self.balance_l2_filtered
        for symbol, position in self.positions_l2.items():
            if symbol in market_prices:
                position_value = position.quantity * market_prices[symbol]
                equity_l2 += position_value
        self.equity_curve_l2.append((timestamp, equity_l2))

        # Actualizar portfolio state para L2
        self.portfolio_state['total_capital'] = equity_l2
        self.portfolio_state['available_capital'] = self.balance_l2_filtered
        self.portfolio_state['drawdown'] = self._calculate_current_drawdown()

    def close_all_positions(self, timestamp: datetime, final_prices: Dict[str, float]):
        """Cierra todas las posiciones al final del backtest"""
        for symbol, price in final_prices.items():
            # Cerrar L1
            if symbol in self.positions_l1:
                self.execute_trade(timestamp, symbol, 'SELL', price, 0.5, False)

            # Cerrar L2
            if symbol in self.positions_l2:
                self.execute_trade(timestamp, symbol, 'SELL', price, 0.5, True)

    def calculate_metrics(self, is_l2: bool = False) -> BacktestMetrics:
        """Calcula métricas de rendimiento"""
        closed_trades = self.closed_trades_l2 if is_l2 else self.closed_trades_l1
        equity_curve = self.equity_curve_l2 if is_l2 else self.equity_curve_l1

        if not closed_trades:
            return BacktestMetrics(start_balance=self.initial_balance, end_balance=self.initial_balance)

        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades

        pnl_values = [t.pnl for t in closed_trades]
        total_pnl = sum(pnl_values)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        max_drawdown = 0
        max_drawdown_pct = 0
        peak = self.initial_balance

        for timestamp, equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct

        # Sharpe ratio
        if len(pnl_values) > 1:
            returns = np.array(pnl_values)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        hold_times = [t.hold_time_minutes for t in closed_trades if t.hold_time_minutes > 0]
        avg_hold_time = np.mean(hold_times) if hold_times else 0

        best_trade = max(pnl_values) if pnl_values else 0
        worst_trade = min(pnl_values) if pnl_values else 0

        end_balance = self.initial_balance + total_pnl
        # FIX: return_pct correcto - no multiplicar por 100 aquí (to_dict usa :.2%)
        return_pct = (total_pnl / self.initial_balance) if self.initial_balance > 0 else 0

        # Métricas L2 específicas
        metrics = BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            total_pnl=total_pnl,
            avg_hold_time_minutes=avg_hold_time,
            avg_trade_pnl=total_pnl / total_trades if total_trades > 0 else 0,
            best_trade=best_trade,
            worst_trade=worst_trade,
            start_balance=self.initial_balance,
            end_balance=end_balance,
            return_pct=return_pct
        )

        if is_l2:
            metrics.l1_signals_processed = self.l1_signals_count
            metrics.l2_signals_approved = self.l2_approved_count
            metrics.l2_signals_rejected = self.l2_rejected_count
            metrics.l2_rejection_rate = self.l2_rejected_count / self.l1_signals_count if self.l1_signals_count > 0 else 0

            # Calcular tamaños promedio
            l1_sizes = [t.quantity for t in self.closed_trades_l1]
            l2_sizes = [t.quantity for t in self.closed_trades_l2]
            l2_leverages = [dec.l2_leverage for dec in self.l2_decisions if dec.l2_leverage]

            metrics.avg_position_size_l1 = np.mean(l1_sizes) if l1_sizes else 0
            metrics.avg_position_size_l2 = np.mean(l2_sizes) if l2_sizes else 0
            metrics.avg_leverage_l2 = np.mean(l2_leverages) if l2_leverages else 1.0

        return metrics


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def run_l2_backtest(
    l1_signals_csv: str,
    config: L2BacktestConfig,
    output_dir: str = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE
) -> Dict[str, Any]:
    """
    Ejecuta un backtest L2 con una configuración específica

    Args:
        l1_signals_csv: Ruta al CSV de señales L1
        config: Configuración L2 a usar
        output_dir: Directorio de salida
        initial_balance: Balance inicial

    Returns:
        Dict con métricas y resultados
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"🚀 INICIANDO BACKTEST L2 - Config: {config.name}")
    logger.info("=" * 80)

    # Log modelo usado
    if L2_MODEL_PATH:
        logger.info(f"🤖 Modelo L2: {os.path.basename(L2_MODEL_PATH)}")
    else:
        logger.warning("⚠️ Sin modelo L2 - solo filtros estáticos")

    # Cargar señales L1
    logger.info(f"📁 Cargando señales L1 desde: {l1_signals_csv}")
    df_l1 = pd.read_csv(l1_signals_csv)
    logger.info(f"✅ Cargadas {len(df_l1)} señales L1")

    # Convertir timestamp
    if 'timestamp' in df_l1.columns:
        df_l1['timestamp'] = pd.to_datetime(df_l1['timestamp'])

    # Inicializar motor
    engine = L2BacktestEngine(
        config=config,
        initial_balance=initial_balance
    )

    # Procesar señales
    logger.info("⏳ Procesando señales L1 a través de L2...")

    progress_interval = max(1, len(df_l1) // 10)

    for i, row in df_l1.iterrows():
        timestamp = row.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        symbol = row.get('symbol', 'BTCUSDT')
        l1_signal = row.get('signal', 'HOLD')
        l1_confidence = row.get('confidence', 0.5)
        l1_price = row.get('price', 0)

        # Market data opcional
        market_data = {
            'open': row.get('open', l1_price),
            'high': row.get('high', l1_price),
            'low': row.get('low', l1_price),
            'volume': row.get('volume', 0)
        }

        # Procesar señal L2
        l2_decision = await engine.process_l1_signal(
            timestamp=timestamp,
            symbol=symbol,
            l1_signal=l1_signal,
            l1_confidence=l1_confidence,
            l1_price=l1_price,
            market_data=market_data
        )

        # Ejecutar trades
        if l1_signal.upper() in ['BUY', 'SELL']:
            # L1 solo (sin filtro)
            engine.execute_trade(
                timestamp=timestamp,
                symbol=symbol,
                side=l1_signal,
                price=l1_price,
                confidence=l1_confidence,
                is_l2_filtered=False
            )

            # L2 filtrado
            engine.execute_trade(
                timestamp=timestamp,
                symbol=symbol,
                side=l1_signal,
                price=l1_price,
                confidence=l1_confidence,
                is_l2_filtered=True,
                l2_decision=l2_decision
            )

        # Actualizar equity
        engine.update_equity(timestamp, {symbol: l1_price})

        # Progreso
        if (i + 1) % progress_interval == 0:
            pct = (i + 1) / len(df_l1) * 100
            logger.info(f"  Progreso: {pct:.1f}% ({i+1}/{len(df_l1)})")

    # Cerrar posiciones
    logger.info("🔚 Cerrando posiciones abiertas...")
    final_prices = {}
    for symbol in df_l1['symbol'].unique():
        last_price = df_l1[df_l1['symbol'] == symbol]['price'].iloc[-1]
        final_prices[symbol] = last_price

    engine.close_all_positions(datetime.now(), final_prices)

    # Calcular métricas
    logger.info("📈 Calculando métricas...")
    metrics_l1 = engine.calculate_metrics(is_l2=False)
    metrics_l2 = engine.calculate_metrics(is_l2=True)

    # Mostrar resultados
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESULTADOS DEL BACKTEST L2")
    logger.info("=" * 80)
    logger.info(f"\nConfiguración: {config.name}")
    logger.info(f"Modelo RL usado: {os.path.basename(L2_MODEL_PATH) if L2_MODEL_PATH else 'Ninguno'}")
    logger.info(f"Rechazos por RL: {engine.rl_rejections}")
    logger.info("-" * 40)

    logger.info("\n📈 ESTRATEGIA L1 (sin filtro):")
    for key, value in metrics_l1.to_dict().items():
        if not key.startswith('l2_') and not key.startswith('avg_pos') and not key.startswith('avg_lev'):
            logger.info(f"  {key:25s}: {value}")

    logger.info("\n🛡️ ESTRATEGIA L1 + L2 (con filtro RL):")
    for key, value in metrics_l2.to_dict().items():
        logger.info(f"  {key:25s}: {value}")

    logger.info("\n📊 COMPARACIÓN L1 vs L1+L2:")
    pnl_diff = metrics_l2.total_pnl - metrics_l1.total_pnl
    pnl_diff_pct = (pnl_diff / abs(metrics_l1.total_pnl) * 100) if metrics_l1.total_pnl != 0 else 0
    sharpe_diff = metrics_l2.sharpe_ratio - metrics_l1.sharpe_ratio
    dd_diff = metrics_l2.max_drawdown_pct - metrics_l1.max_drawdown_pct

    logger.info(f"  Diferencia PnL:          {pnl_diff:+.2f} USDT ({pnl_diff_pct:+.1f}%)")
    logger.info(f"  Diferencia Sharpe:       {sharpe_diff:+.2f}")
    logger.info(f"  Diferencia Max DD:       {dd_diff:+.2f}%")
    logger.info(f"  Señales L2 rechazadas:   {metrics_l2.l2_signals_rejected} ({metrics_l2.l2_rejection_rate})")
    logger.info(f"  Rechazos por modelo RL:  {engine.rl_rejections}")

    # Exportar resultados
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_slug = config.name.lower().replace(' ', '_')

    # CSV de decisiones L2
    decisions_csv = os.path.join(output_dir, f"l2_decisions_{config_slug}_{timestamp_str}.csv")
    decisions_df = pd.DataFrame([d.to_dict() for d in engine.l2_decisions])
    decisions_df.to_csv(decisions_csv, index=False)
    logger.info(f"\n💾 Decisiones L2 exportadas a: {decisions_csv}")

    # JSON de métricas
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config.to_dict(),
        'model_used': os.path.basename(L2_MODEL_PATH) if L2_MODEL_PATH else 'none',
        'l1_signals_csv': l1_signals_csv,
        'metrics_l1': metrics_l1.to_dict(),
        'metrics_l2': metrics_l2.to_dict(),
        'comparison': {
            'pnl_diff': pnl_diff,
            'pnl_diff_pct': pnl_diff_pct,
            'sharpe_diff': sharpe_diff,
            'drawdown_diff': dd_diff,
            'rejection_rate': metrics_l2.l2_rejection_rate,
            'rl_rejections': engine.rl_rejections
        },
        'files': {
            'decisions_csv': decisions_csv
        }
    }

    results_json = os.path.join(output_dir, f"l2_backtest_{config_slug}_{timestamp_str}.json")
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Resultados JSON guardados en: {results_json}")

    return results


async def run_robustness_test(
    l1_signals_csv: str,
    output_dir: str = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE
) -> Dict[str, Any]:
    """
    Ejecuta el test de robustez con 3 configuraciones diferentes
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("🧪 INICIANDO TEST DE ROBUSTEZ L2")
    logger.info("=" * 80)

    all_results = {}

    for config_name, config in ROBUSTNESS_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Ejecutando configuración: {config.name}")
        logger.info(f"{'='*60}")

        result = await run_l2_backtest(
            l1_signals_csv=l1_signals_csv,
            config=config,
            output_dir=output_dir,
            initial_balance=initial_balance
        )

        all_results[config_name] = result

    # Comparar resultados
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESUMEN DE ROBUSTEZ - COMPARACIÓN DE CONFIGURACIONES")
    logger.info("=" * 80)

    comparison_data = []
    for config_name, result in all_results.items():
        comparison_data.append({
            'Config': result['config']['name'],
            'PnL L1': result['metrics_l1']['total_pnl'],
            'PnL L2': result['metrics_l2']['total_pnl'],
            'Sharpe L1': result['metrics_l1']['sharpe_ratio'],
            'Sharpe L2': result['metrics_l2']['sharpe_ratio'],
            'MaxDD L1': result['metrics_l1']['max_drawdown_pct'],
            'MaxDD L2': result['metrics_l2']['max_drawdown_pct'],
            'Rechazo L2': result['metrics_l2']['l2_rejection_rate'],
            'PnL Diff': result['comparison']['pnl_diff'],
            'Sharpe Diff': result['comparison']['sharpe_diff'],
            'RL Rechazos': result['comparison'].get('rl_rejections', 0)
        })

    comparison_df = pd.DataFrame(comparison_data)
    logger.info("\n" + comparison_df.to_string(index=False))

    # Análisis de sensibilidad
    logger.info("\n📈 ANÁLISIS DE SENSIBILIDAD:")

    pnl_values = [r['comparison']['pnl_diff'] for r in all_results.values()]
    sharpe_values = [r['comparison']['sharpe_diff'] for r in all_results.values()]

    pnl_std = np.std(pnl_values)
    sharpe_std = np.std(sharpe_values)

    logger.info(f"  Desviación estándar PnL Diff:    {pnl_std:.2f} USDT")
    logger.info(f"  Desviación estándar Sharpe Diff: {sharpe_std:.2f}")
    logger.info(f"  Rango PnL Diff:                  {min(pnl_values):.2f} a {max(pnl_values):.2f}")
    logger.info(f"  Rango Sharpe Diff:               {min(sharpe_values):.2f} a {max(sharpe_values):.2f}")

    if pnl_std < abs(np.mean(pnl_values)) * 0.5:
        logger.info("  ✅ Resultados ESTABLES bajo diferentes configuraciones")
    else:
        logger.info("  ⚠️ Resultados VARIABLES - alta sensibilidad a parámetros")

    # Guardar resumen de robustez
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    robustness_summary = {
        'timestamp': datetime.now().isoformat(),
        'model_used': os.path.basename(L2_MODEL_PATH) if L2_MODEL_PATH else 'none',
        'l1_signals_csv': l1_signals_csv,
        'configs_tested': list(ROBUSTNESS_CONFIGS.keys()),
        'comparison_table': comparison_data,
        'sensitivity_analysis': {
            'pnl_diff_std': pnl_std,
            'sharpe_diff_std': sharpe_std,
            'pnl_diff_range': [min(pnl_values), max(pnl_values)],
            'sharpe_diff_range': [min(sharpe_values), max(sharpe_values)],
            'stability_score': 'STABLE' if pnl_std < abs(np.mean(pnl_values)) * 0.5 else 'VARIABLE'
        },
        'detailed_results': all_results
    }

    robustness_json = os.path.join(output_dir, f"l2_robustness_summary_{timestamp_str}.json")
    with open(robustness_json, 'w') as f:
        json.dump(robustness_summary, f, indent=2, default=str)
    logger.info(f"\n💾 Resumen de robustez guardado en: {robustness_json}")

    return robustness_summary


def find_latest_l1_signals() -> Optional[str]:
    """Encuentra el archivo CSV de señales L1 más reciente"""
    import glob

    pattern = os.path.join(RESULTS_DIR, "l1_backtest_signals_*.csv")
    files = glob.glob(pattern)

    if not files:
        pattern = os.path.join(".", "l1_backtest_signals_*.csv")
        files = glob.glob(pattern)

    if files:
        return max(files, key=os.path.getctime)

    return None


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Backtesting L2 Standalone - Filtros L2 sobre señales L1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Backtest L2 con CSV de L1 más reciente
  python backtesting/backtest_l2_standalone.py

  # Backtest L2 con CSV específico
  python backtesting/backtest_l2_standalone.py --l1-csv results/l1_backtest_signals_20260221_120000.csv

  # Test de robustez con 3 configuraciones
  python backtesting/backtest_l2_standalone.py --robustness-test

  # Configuración personalizada
  python backtesting/backtest_l2_standalone.py --l1-csv signals.csv --balance 5000 --config balanced
        """
    )

    parser.add_argument(
        '--l1-csv',
        help='Ruta al CSV de señales L1 (default: más reciente en results/)'
    )
    parser.add_argument(
        '--balance',
        type=float,
        default=DEFAULT_INITIAL_BALANCE,
        help=f'Balance inicial en USDT (default: {DEFAULT_INITIAL_BALANCE})'
    )
    parser.add_argument(
        '--config',
        choices=['conservative', 'balanced', 'aggressive'],
        default='balanced',
        help='Configuración L2 a usar (default: balanced)'
    )
    parser.add_argument(
        '--robustness-test',
        action='store_true',
        help='Ejecutar test de robustez con 3 configuraciones'
    )
    parser.add_argument(
        '--output-dir',
        default=RESULTS_DIR,
        help=f'Directorio de salida (default: {RESULTS_DIR})'
    )
    parser.add_argument(
        '--no-rl',
        action='store_true',
        help='Desactivar filtro de modelo RL (solo reglas estáticas)'
    )
    parser.add_argument(
        '--model',
        choices=['claude', 'deepseek', 'gemini', 'gpt', 'grok', 'kimi'],
        default=None,
        help='Modelo L2 a usar (default: usar el configurado en L2Config o el primero disponible)'
    )

    args = parser.parse_args()

    # Inicializar L2_MODEL_PATH global basado en el argumento o configuración
    global L2_MODEL_PATH
    L2_MODEL_PATH = get_l2_model_path(args.model)

    # Encontrar CSV de L1
    l1_csv = args.l1_csv
    if not l1_csv:
        l1_csv = find_latest_l1_signals()
        if not l1_csv:
            print("❌ No se encontró archivo CSV de señales L1")
            print("   Ejecuta primero: python backtesting/backtest_l1_standalone.py")
            return 1

    if not os.path.exists(l1_csv):
        print(f"❌ Archivo no encontrado: {l1_csv}")
        return 1

    print(f"📁 Usando señales L1: {l1_csv}")
    print(f"🤖 Modelo L2: {os.path.basename(L2_MODEL_PATH) if L2_MODEL_PATH else 'No encontrado'}")

    # Ajustar use_rl si se pasó --no-rl
    if args.no_rl:
        for cfg in ROBUSTNESS_CONFIGS.values():
            cfg.use_rl_model = False

    # Ejecutar backtest
    try:
        if args.robustness_test:
            result = asyncio.run(run_robustness_test(
                l1_signals_csv=l1_csv,
                output_dir=args.output_dir,
                initial_balance=args.balance
            ))
        else:
            config = ROBUSTNESS_CONFIGS[args.config]
            if args.no_rl:
                config.use_rl_model = False
            result = asyncio.run(run_l2_backtest(
                l1_signals_csv=l1_csv,
                config=config,
                output_dir=args.output_dir,
                initial_balance=args.balance
            ))

        print("\n✅ Backtest L2 completado exitosamente")
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
    import asyncio
    sys.exit(main())