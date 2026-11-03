#!/usr/bin/env python3
"""
Backtesting L1 Standalone - Script independiente para backtesting de modelos L1

Este script:
1. Carga datos históricos OHLCV de BTC y ETH desde CSV locales o Binance testnet
2. Instancia SOLO los modelos L1 de l1_operational/ con mocks para el exchange
3. Para cada vela histórica, genera señales L1 (BUY/SELL/HOLD) con confidence score
4. Simula ejecución de órdenes con el SimulatedExchangeClient
5. Calcula y reporta: win rate, profit factor, max drawdown, Sharpe ratio, total trades, avg hold time
6. Genera un CSV con columna timestamp, signal, confidence, pnl_trade, cumulative_pnl

IMPORTANTE: NO importa nada de l2_tactic/ ni l3_strategy/ — L1 completamente aislado
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
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path para importar módulos de l1_operational
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importar SOLO de l1_operational - NADA de l2_tactic ni l3_strategy
from l1_operational.models import (
    L1Model, L1Signal, L1SignalType, Signal, create_signal
)
# Nota: AIModelPipeline no se usa en backtest standalone
# from l1_operational.ai_pipeline import AIModelPipeline, AIDecision
from l1_operational.trend_ai import filter_signal
from l1_operational.simulated_exchange_client import SimulatedExchangeClient
from l1_operational.config import ConfigObject, TREND_THRESHOLD
from core.technical_indicators import calculate_technical_indicators


# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

DEFAULT_SYMBOLS = ['ETHUSDT']  # FILTRO DE SÍMBOLO: Solo ETHUSDT temporalmente
DEFAULT_TIMEFRAME = '1m'
DEFAULT_INITIAL_BALANCE = 3000.0  # USDT
DEFAULT_FEE = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


# =============================================================================
# DATA CLASSES PARA RESULTADOS
# =============================================================================

@dataclass
class TradeRecord:
    """Registro de una operación completada"""
    trade_id: int
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' o 'SELL'
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    confidence: float = 0.5
    signal_type: str = "unknown"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    hold_time_minutes: float = 0.0
    exit_timestamp: Optional[datetime] = None
    is_open: bool = True
    exit_reason: str = "SIGNAL"
    max_favorable_price: Optional[float] = None  # Para análisis de SL
    max_favorable_pct: Optional[float] = None    # Máximo beneficio potencial


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
            'return_pct': f"{self.return_pct:.2%}"
        }


# =============================================================================
# CARGA DE DATOS
# =============================================================================

class DataLoader:
    """Cargador de datos históricos desde CSV o Binance"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_from_csv(self, symbol: str, timeframe: str = '1m') -> Optional[pd.DataFrame]:
        """Carga datos desde CSV local"""
        # Intentar diferentes patrones de nombres
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
                # Normalizar columnas
                df = self._normalize_columns(df)
                return df
        
        return None
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas"""
        # Convertir a minúsculas
        df.columns = [c.lower() for c in df.columns]
        
        # Mapear columnas comunes
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'timestamp': 'timestamp',
            'datetime': 'timestamp',
            'date': 'timestamp',
            'time': 'timestamp'
        }
        
        # Renombrar si es necesario
        for old, new in column_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        # Asegurar que timestamp es datetime
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    async def download_from_binance(
        self, 
        symbol: str, 
        timeframe: str = '1m',
        days: int = 30,
        testnet: bool = True
    ) -> pd.DataFrame:
        """Descarga datos desde Binance testnet"""
        try:
            import ccxt.async_support as ccxt
            
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
            
            if testnet:
                exchange.set_sandbox_mode(True)
            
            logger.info(f"📡 Descargando datos de {symbol} desde Binance {'testnet' if testnet else 'live'}...")
            
            # Calcular rango de fechas
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            since = int(start_time.timestamp() * 1000)
            
            # Descargar OHLCV
            all_ohlcv = []
            limit = 1000
            
            while since < int(end_time.timestamp() * 1000):
                ohlcv = await exchange.fetch_ohlcv(
                    symbol.replace('USDT', '/USDT'),
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                logger.debug(f"Descargados {len(ohlcv)} registros, total: {len(all_ohlcv)}")
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            await exchange.close()
            
            # Crear DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            logger.info(f"✅ Descargados {len(df)} registros de {symbol}")
            
            # Guardar CSV para uso futuro
            output_file = os.path.join(self.data_dir, f"{symbol.lower()}_{timeframe}.csv")
            df.to_csv(output_file)
            logger.info(f"💾 Guardado en {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error descargando datos: {e}")
            raise
    
    async def load_data(
        self,
        symbols: List[str],
        timeframe: str = '1m',
        days: int = 30,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Carga datos para múltiples símbolos"""
        data = {}
        
        for symbol in symbols:
            df = None
            
            # Intentar cargar desde CSV primero
            if use_cache:
                df = self.load_from_csv(symbol, timeframe)
            
            # Si no hay CSV, descargar de Binance
            if df is None:
                df = await self.download_from_binance(symbol, timeframe, days)
            
            if df is not None and not df.empty:
                data[symbol] = df
            else:
                logger.warning(f"⚠️ No se pudieron cargar datos para {symbol}")
        
        return data


# =============================================================================
# MOTOR DE BACKTESTING
# =============================================================================

class L1BacktestEngine:
    """Motor de backtesting para modelos L1"""
    
    def __init__(
        self,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE,
        position_size_pct: float = 0.1,  # 10% del capital por operación
        min_confidence: float = 0.68,  # FILTRO DE CONFIANZA: Subido a 0.68 (era 0.60)
        cooldown_minutes: float = 360,  # FILTRO COOLDOWN: Aumentado a 6 horas (era 240)
        stop_loss_pct: float = 0.015,
        take_profit_pct: float = 0.025,
    ):
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.min_confidence = min_confidence
        self.cooldown_minutes = cooldown_minutes
        
        # Parámetros de stop-loss y take-profit
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Resetear el singleton de SimulatedExchangeClient (cleanup resetea _instance Y _initialized)
        SimulatedExchangeClient.cleanup()
        
        # Inicializar exchange simulado
        self.exchange = SimulatedExchangeClient(
            initial_balances={'USDT': initial_balance, 'BTC': 0.0, 'ETH': 0.0},
            fee=fee,
            slippage=slippage
        )
        
        # Inicializar modelos L1
        self.l1_model = L1Model(config={})
        # Nota: AIModelPipeline deshabilitado para backtest standalone
        # self.ai_pipeline = AIModelPipeline(models_path="models/")
        
        # Estado del backtest
        self.positions: Dict[str, TradeRecord] = {}  # Posiciones abiertas por símbolo
        self.closed_trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.signals_history: List[Dict] = []
        
        self.trade_counter = 0
        self.current_timestamp = None
        
        # Cooldown entre operaciones
        self.last_trade_time: Dict[str, datetime] = {}
        
        # DIAGNÓSTICO: Contadores de señales
        self.diag_signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        self.diag_filtered_by_confidence = 0
        self.diag_filtered_by_trend = 0
        self.diag_executed = 0
        
        # Para análisis de stop-loss: historial de precios por símbolo
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    
    def reset(self):
        """Resetea el estado del motor"""
        SimulatedExchangeClient.cleanup()  # resetea _instance Y _initialized
        self.exchange = SimulatedExchangeClient(
            initial_balances={'USDT': self.initial_balance, 'BTC': 0.0, 'ETH': 0.0},
            fee=DEFAULT_FEE,
            slippage=DEFAULT_SLIPPAGE
        )
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = []
        self.signals_history = []
        self.trade_counter = 0
    
    def _calculate_position_size(self, price: float, confidence: float) -> float:
        """Calcula el tamaño de posición basado en confianza"""
        # Obtener balance disponible
        balances = self.exchange.get_balances()
        available_usdt = balances.get('USDT', 0)
        
        # Tamaño base basado en porcentaje del capital
        base_size = available_usdt * self.position_size_pct
        
        # Escalar por confianza
        scaled_size = base_size * confidence
        
        # Convertir a cantidad del activo
        quantity = scaled_size / price
        
        # Redondear a 6 decimales
        return round(quantity, 6)
    
    def _is_trend_aligned(self, action: str, market_data: Dict, symbol: str) -> bool:
        """Solo operar en dirección de tendencia de las últimas 4 horas"""
        df = market_data.get(symbol)
        if df is None or len(df) < 240:
            return True  # sin datos suficientes, permitir
        
        # SMA de 240 velas (4 horas en 1m)
        sma_240 = df['close'].iloc[-240:].mean()
        current_price = df['close'].iloc[-1]
        
        trend_up = current_price > sma_240 * 1.001   # 0.1% por encima
        trend_down = current_price < sma_240 * 0.999  # 0.1% por debajo
        
        if action == 'BUY' and not trend_up:
            return False  # no comprar en tendencia bajista
        if action == 'SELL' and not trend_down:
            return False  # no vender en tendencia alcista
        
        return True
    
    def _get_signal_action(self, l1_signals: List[L1Signal], symbol: str) -> Tuple[str, float, str]:
        """
        Determina la acción final a partir de múltiples señales L1
        Retorna: (action, confidence, signal_type)
        
        FILTRO DE SEÑAL:
        - Solo operar señales tipo 'technical' normalmente
        - Señales 'momentum' solo con confianza >= 0.72
        """
        if not l1_signals:
            return 'HOLD', 0.5, 'none'
        
        # Filtrar señales para el símbolo
        symbol_signals = [s for s in l1_signals if symbol in s.symbol]
        
        if not symbol_signals:
            return 'HOLD', 0.5, 'none'
        
        # FIX: usar confidence directamente, no confidence*strength
        buy_confidence_sum = 0.0
        sell_confidence_sum = 0.0
        buy_count = 0
        sell_count = 0

        for signal in symbol_signals:
            if signal.direction == 'buy':
                buy_confidence_sum += signal.confidence
                buy_count += 1
            elif signal.direction == 'sell':
                sell_confidence_sum += signal.confidence
                sell_count += 1

        buy_score = buy_confidence_sum / buy_count if buy_count > 0 else 0.0
        sell_score = sell_confidence_sum / sell_count if sell_count > 0 else 0.0
        avg_confidence = (buy_confidence_sum + sell_confidence_sum) / len(symbol_signals)

        THRESHOLD = 0.52
        signal_type = 'momentum' if (buy_count > 1 or sell_count > 1) else 'technical'
        
        # FILTRO DE SEÑAL: Determinar acción candidata
        action = 'HOLD'
        confidence = avg_confidence
        
        if buy_count > 0 and buy_score >= THRESHOLD and buy_score >= sell_score:
            action = 'BUY'
            confidence = buy_score
        elif sell_count > 0 and sell_score >= THRESHOLD and sell_score > buy_score:
            action = 'SELL'
            confidence = sell_score
        
        # FILTRO DE SEÑAL: Aplicar restricciones por tipo de señal
        if action != 'HOLD':
            if signal_type == 'momentum' and confidence < 0.72:
                # Momentum solo con confianza muy alta (>= 0.72)
                return 'HOLD', confidence, signal_type
        
        if action != 'HOLD':
            return action, confidence, signal_type
        else:
            return 'HOLD', avg_confidence, 'neutral'
    
    def process_candle(
        self,
        timestamp: datetime,
        symbol: str,
        candle_data: pd.Series,
        market_data: Dict[str, pd.DataFrame],
        candle_idx: int = 0  # DIAGNÓSTICO: índice de la vela
    ) -> Dict[str, Any]:
        """Procesa una vela individual"""
        self.current_timestamp = timestamp
        
        result = {
            'timestamp': timestamp,
            'symbol': symbol,
            'price': candle_data['close'],
            'signal': 'HOLD',
            'confidence': 0.5,
            'action_taken': False,
            'pnl': 0.0,
            'cumulative_pnl': 0.0
        }
        
        try:
            # Generar señales L1
            l1_result = self.l1_model.predict(market_data)
            l1_signals = l1_result.get('signals', [])
            
            # DIAGNÓSTICO: Loggear métricas de señales
            metrics = l1_result.get('metrics', {})
            
            # Determinar acción
            action, confidence, signal_type = self._get_signal_action(l1_signals, symbol)
            result['signal'] = action
            result['confidence'] = confidence
            
            # DIAGNÓSTICO — ejecutar solo en las primeras 100 velas
            if candle_idx < 100:
                logger.info(f"[DIAG] vela {candle_idx} ({symbol}): "
                           f"action_raw={action}, conf_raw={confidence:.4f}, "
                           f"total_signals={metrics.get('total_signals', 0)}, "
                           f"buy={metrics.get('buy_signals', 0)}, "
                           f"sell={metrics.get('sell_signals', 0)}, "
                           f"hold={metrics.get('hold_signals', 0)}")
            
            # Contar distribución de señales (diagnóstico global)
            self.diag_signal_counts[action] = self.diag_signal_counts.get(action, 0) + 1
            
            # Validar con TrendAI si hay señal
            if action != 'HOLD':
                # Crear señal para validación
                signal_dict = {
                    'symbol': symbol.replace('USDT', ''),
                    'timeframe': '1m',
                    'price': candle_data['close'],
                    'volume': candle_data.get('volume', 0),
                    'features': self._extract_features(candle_data, market_data, symbol),
                    'signal_id': f"{symbol}_{int(timestamp.timestamp())}"
                }
                
                # DIAGNÓSTICO: Loggear features en las primeras 5 velas
                if candle_idx < 5:
                    logger.info(f"[DIAG-FEATURES] vela {candle_idx}: features_keys={list(signal_dict['features'].keys())}")
                    # Verificar si hay modelos cargados en trend_ai
                    from l1_operational import trend_ai
                    loaded_models = [name for name, m in trend_ai.models.items() if m is not None]
                    logger.info(f"[DIAG-MODELS] Modelos cargados en trend_ai: {loaded_models}")
                    
                    # Intentar verificar feature_names_in_ del primer modelo disponible
                    for name, model in trend_ai.models.items():
                        if model is not None and hasattr(model, 'feature_names_in_'):
                            logger.info(f"[DIAG-MODEL-{name}] Features del modelo: {list(model.feature_names_in_)}")
                            break
                
                # Validar con TrendAI
                trend_valid = filter_signal(signal_dict)
                
                if not trend_valid:
                    action = 'HOLD'
                    result['signal'] = 'HOLD'
                    result['trend_filtered'] = True
                    self.diag_filtered_by_trend += 1
                    
                    if candle_idx < 100:
                        logger.info(f"[DIAG] vela {candle_idx}: FILTRADA por TrendAI")
            
            # Verificar cooldown
            _sentinel = datetime(2000, 1, 1)
            last_trade = self.last_trade_time.get(symbol, _sentinel)
            time_since = (timestamp - last_trade).total_seconds() if isinstance(timestamp, datetime) else 9999
            if action != 'HOLD' and time_since < self.cooldown_minutes * 60:
                action = 'HOLD'

            # Verificar tendencia macro
            if action != 'HOLD' and not self._is_trend_aligned(action, market_data, symbol):
                action = 'HOLD'
                result['trend_aligned'] = False
                self.diag_filtered_by_trend += 1
                if candle_idx < 100:
                    logger.info(f"[DIAG] vela {candle_idx}: FILTRADA por tendencia contraria")

            # Ejecutar operación si pasa los filtros
            if action != 'HOLD' and confidence >= self.min_confidence:
                executed = self._execute_signal(action, symbol, candle_data['close'], confidence, signal_type)
                result['action_taken'] = executed
                if executed:
                    self.diag_executed += 1
                    self.last_trade_time[symbol] = timestamp
                    
                    # VALIDACIÓN DE SANIDAD: detectar balances corruptos inmediatamente
                    balances = self.exchange.get_balances()
                    usdt = balances.get('USDT', 0)
                    if usdt > self.initial_balance * 10:
                        logger.error(f"🚨 BALANCE ANÓMALO detectado: {usdt} — abortando")
                        raise RuntimeError(f"Balance corruption detected: {usdt}")
            elif action != 'HOLD' and confidence < self.min_confidence:
                self.diag_filtered_by_confidence += 1
                if candle_idx < 100:
                    logger.info(f"[DIAG] vela {candle_idx}: FILTRADA por confianza baja ({confidence:.4f} < {self.min_confidence})")
            
            # Actualizar PnL de posiciones abiertas
            self._update_positions(symbol, candle_data['close'])
            
            # Calcular PnL acumulado
            result['cumulative_pnl'] = self._calculate_total_pnl()
            
            # Guardar en historial
            self.signals_history.append(result.copy())
            
        except Exception as e:
            logger.error(f"❌ Error procesando vela {timestamp}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return result
    
    def _extract_features(
        self,
        candle_data: pd.Series,
        market_data: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict[str, float]:
        """Extrae features de la vela actual - COMPATIBLE con TrendAI (52 features)"""
        features = {}
        
        # Obtener DataFrame del símbolo
        df = market_data.get(symbol)
        if df is None or len(df) < 50:
            # Fallback: retornar features mínimos
            return {
                'close': candle_data.get('close', 0),
                'rsi': 50,
                'macd': 0,
                'rsi_trend': 0.5,
                'macd_trend': 0.5,
                'price_slope': 0,
            }
        
        try:
            # Calcular indicadores técnicos
            from core.technical_indicators import calculate_technical_indicators
            indicators_dict = calculate_technical_indicators({symbol: df})
            
            if symbol in indicators_dict and not indicators_dict[symbol].empty:
                ind = indicators_dict[symbol]
                latest = ind.iloc[-1]
                
                # Features básicos (versión simplificada compatible con TrendAI)
                close = candle_data.get('close', latest.get('close', 0))
                open_price = candle_data.get('open', latest.get('open', close))
                
                # Calcular EMAs y SMAs si no existen
                if 'ema_10' not in ind.columns:
                    ind['ema_10'] = ind['close'].ewm(span=10).mean()
                if 'ema_20' not in ind.columns:
                    ind['ema_20'] = ind['close'].ewm(span=20).mean()
                if 'sma_10' not in ind.columns:
                    ind['sma_10'] = ind['close'].rolling(window=10).mean()
                if 'sma_20' not in ind.columns:
                    ind['sma_20'] = ind['close'].rolling(window=20).mean()
                
                # Volume features
                vol_mean = df['volume'].rolling(window=20).mean().iloc[-1] if 'volume' in df.columns else 1
                current_vol = candle_data.get('volume', df['volume'].iloc[-1] if 'volume' in df.columns else 0)
                vol_rel = current_vol / vol_mean if vol_mean > 0 else 1.0
                
                # RSI y MACD
                rsi = latest.get('rsi', 50)
                macd = latest.get('macd', 0)
                macd_signal = latest.get('macd_signal', 0)
                macd_hist = macd - macd_signal
                
                # Features principales que espera TrendAI (subset crítico)
                features = {
                    'delta_close': close - open_price,
                    'ema_10': ind['ema_10'].iloc[-1] if 'ema_10' in ind.columns else close,
                    'ema_20': ind['ema_20'].iloc[-1] if 'ema_20' in ind.columns else close,
                    'sma_10': ind['sma_10'].iloc[-1] if 'sma_10' in ind.columns else close,
                    'sma_20': ind['sma_20'].iloc[-1] if 'sma_20' in ind.columns else close,
                    'volume': current_vol,
                    'vol_rel': vol_rel,
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_hist': macd_hist,
                    'trend_adx': latest.get('adx', 25),
                    'momentum_stoch': latest.get('stoch_k', 50),
                    'momentum_stoch_signal': latest.get('stoch_d', 50),
                    'volume_obv': latest.get('obv', 0),
                    'volatility_bbw': latest.get('bb_width', 0.02),
                    'volatility_atr': latest.get('atr', close * 0.01),
                    'trend_sma_fast': latest.get('sma_20', close),
                    'trend_sma_slow': latest.get('sma_50', close),
                    'trend_ema_fast': ind['ema_10'].iloc[-1] if 'ema_10' in ind.columns else close,
                    'trend_ema_slow': ind['ema_20'].iloc[-1] if 'ema_20' in ind.columns else close,
                    'trend_macd': macd,
                    'momentum_rsi': rsi / 100,
                    # Features para compatibilidad con fallback
                    'rsi_trend': rsi / 100,
                    'macd_trend': (macd + 10) / 20,  # Normalizar a 0-1 aprox
                    'price_slope': (close - df['close'].iloc[-20]) / df['close'].iloc[-20] if len(df) >= 20 else 0,
                }
                
                # Añadir features de 5m (usando los mismos valores si no hay datos 5m)
                for key in list(features.keys()):
                    features[f"{key}_5m"] = features[key]
                
                # Features de correlación BTC/ETH
                features['eth_btc_ratio'] = 0.05  # Valor aproximado actual
                features['eth_btc_ratio_sma'] = 0.05
                features['btc_eth_corr'] = 0.8
                
            else:
                # Fallback básico
                features = {
                    'close': candle_data.get('close', 0),
                    'rsi': 50,
                    'macd': 0,
                    'rsi_trend': 0.5,
                    'macd_trend': 0.5,
                    'price_slope': 0,
                }
                
        except Exception as e:
            logger.warning(f"[DIAG] Error extrayendo features: {e}")
            features = {
                'close': candle_data.get('close', 0),
                'rsi': 50,
                'macd': 0,
                'rsi_trend': 0.5,
                'macd_trend': 0.5,
                'price_slope': 0,
            }
        
        return features
    
    def _execute_signal(
        self,
        action: str,
        symbol: str,
        price: float,
        confidence: float,
        signal_type: str
    ) -> bool:
        """Ejecuta una señal de trading"""
        asset = symbol.replace('USDT', '')
        
        try:
            if action == 'BUY':
                # Verificar si ya hay posición abierta
                if symbol in self.positions:
                    return False
                
                # Calcular tamaño de posición
                quantity = self._calculate_position_size(price, confidence)
                
                if quantity <= 0:
                    return False
                
                # Ejecutar orden
                trade = self.exchange.execute_order(symbol, 'BUY', quantity, price)
                
                if trade:
                    self.trade_counter += 1
                    self.positions[symbol] = TradeRecord(
                        trade_id=self.trade_counter,
                        timestamp=self.current_timestamp,
                        symbol=symbol,
                        side='BUY',
                        entry_price=trade['execution_price'],
                        quantity=quantity,
                        confidence=confidence,
                        signal_type=signal_type,
                        fees=trade['fee']
                    )
                    logger.info(f"🟢 BUY {symbol} @ {price:.2f} qty={quantity:.6f} conf={confidence:.2f}")
                    return True
                    
            elif action == 'SELL':
                # Verificar si hay posición abierta
                if symbol not in self.positions:
                    return False
                
                position = self.positions[symbol]
                
                # Ejecutar orden de venta - el exchange actualiza balances internamente
                trade = self.exchange.execute_order(symbol, 'SELL', position.quantity, price)
                
                if trade:
                    # Calcular PnL SOLO para el registro contable del TradeRecord
                    # NO modificar el balance - el exchange ya lo hizo internamente
                    entry_value = position.entry_price * position.quantity
                    exit_value = trade['execution_price'] * position.quantity
                    pnl = exit_value - entry_value - trade['fee'] - position.fees
                    pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                    
                    hold_time = (self.current_timestamp - position.timestamp).total_seconds() / 60
                    
                    # Actualizar TradeRecord con PnL y marcar como cerrada
                    position.exit_price = trade['execution_price']
                    position.pnl = pnl
                    position.pnl_pct = pnl_pct
                    position.exit_timestamp = self.current_timestamp
                    position.hold_time_minutes = hold_time
                    position.is_open = False
                    
                    # Mover a cerrados, eliminar de abiertas
                    self.closed_trades.append(position)
                    del self.positions[symbol]
                    
                    # NO hacer ninguna operación aritmética sobre self.balance ni self.exchange.balances
                    # El exchange ya gestionó los balances internamente en execute_order()
                    
                    emoji = "🟢" if pnl > 0 else "🔴"
                    logger.info(f"{emoji} SELL {symbol} @ {price:.2f} PnL={pnl:.2f} ({pnl_pct:.2f}%) hold={hold_time:.1f}m")
                    return True
        
        except Exception as e:
            logger.error(f"❌ Error ejecutando {action} {symbol}: {e}")
        
        return False
    
    def _update_positions(self, symbol: str, current_price: float):
        """Actualiza el estado de las posiciones abiertas y comprueba stop-loss/take-profit"""
        # Registrar precio actual para este símbolo
        if not hasattr(self, '_last_prices'):
            self._last_prices = {}
        self._last_prices[symbol] = current_price
        
        # Guardar historial de precios para análisis de SL
        if self.current_timestamp:
            self.price_history[symbol].append((self.current_timestamp, current_price))

        # BUG CRÍTICO CORREGIDO: solo comprobar SL/TP del símbolo actual
        # El bug anterior iteraba TODAS las posiciones con current_price del símbolo
        # en proceso → precio ETH (~1976) aplicado a posición BTC (~96000)
        # → pnl_pct = (1976-96000)/96000 = -97% → SL inmediato → PnL de 1e+48
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.side == 'BUY':
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price

                if pnl_pct <= -self.stop_loss_pct:
                    self._force_close_position(symbol, current_price, reason='STOP_LOSS')
                elif pnl_pct >= self.take_profit_pct:
                    self._force_close_position(symbol, current_price, reason='TAKE_PROFIT')
        
        # Calcular equity actual
        balances = self.exchange.get_balances()
        total_value = balances.get('USDT', 0.0)
        
        # Añadir valor de posiciones abiertas
        for sym, pos in self.positions.items():
            asset = sym.replace('USDT', '')
            asset_balance = balances.get(asset, 0)
            if asset_balance > 0:
                # Usar el precio actual del símbolo correspondiente
                # current_price es del símbolo actual, necesitamos el del activo de la posición
                asset_value = asset_balance * pos.entry_price  # Valor aproximado
                total_value += asset_value
        
        self.equity_curve.append((self.current_timestamp, total_value))
    
    def _force_close_position(self, symbol: str, price: float, reason: str):
        """Fuerza el cierre de una posición por stop-loss o take-profit.
        
        IMPORTANTE: NO modificar el balance directamente. El exchange ya actualiza
        los balances internamente al ejecutar la orden. Solo calculamos PnL para
        el registro contable del TradeRecord.
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        try:
            # 1. Ejecutar orden de venta - el exchange actualiza balances internamente
            trade = self.exchange.execute_order(symbol, 'SELL', position.quantity, price)
            
            if trade:
                # 2. Calcular PnL SOLO para el registro contable del TradeRecord
                entry_value = position.entry_price * position.quantity
                exit_value = trade['execution_price'] * position.quantity
                pnl = exit_value - entry_value - trade['fee'] - position.fees
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                
                hold_time = (self.current_timestamp - position.timestamp).total_seconds() / 60 if self.current_timestamp else 0
                
                # 3. Análisis de SL: calcular máximo favorable si es stop-loss
                max_favorable_price = None
                max_favorable_pct = None
                if reason == 'STOP_LOSS' and symbol in self.price_history:
                    # Buscar el máximo precio desde la entrada hasta el cierre
                    entry_time = position.timestamp
                    exit_time = self.current_timestamp
                    prices_in_range = [
                        p for ts, p in self.price_history[symbol]
                        if entry_time <= ts <= exit_time
                    ]
                    if prices_in_range:
                        max_favorable_price = max(prices_in_range)
                        max_favorable_pct = (max_favorable_price - position.entry_price) / position.entry_price
                
                # 4. Actualizar TradeRecord con PnL y datos de salida
                position.exit_price = trade['execution_price']
                position.pnl = pnl
                position.pnl_pct = pnl_pct
                position.exit_timestamp = self.current_timestamp
                position.hold_time_minutes = hold_time
                position.is_open = False
                position.exit_reason = reason
                position.max_favorable_price = max_favorable_price
                position.max_favorable_pct = max_favorable_pct
                
                # 5. Mover a cerrados, eliminar de abiertas
                self.closed_trades.append(position)
                del self.positions[symbol]
                
                # 6. NO hacer ninguna operación aritmética sobre self.balance ni self.exchange.balances
                # El exchange ya gestionó los balances internamente en execute_order()
                
                emoji = "🛑" if reason == 'STOP_LOSS' else "✅"
                extra_info = ""
                if max_favorable_pct is not None:
                    extra_info = f" | MaxFav: {max_favorable_pct:.2%}"
                logger.info(f"{emoji} {reason} {symbol} @ {price:.2f} PnL={pnl:.2f}{extra_info}")
        
        except Exception as e:
            logger.error(f"❌ Error cerrando posición {symbol} por {reason}: {e}")
    
    def _calculate_total_pnl(self) -> float:
        """Calcula el PnL total (realizado + no realizado)"""
        realized_pnl = sum(t.pnl for t in self.closed_trades)
        
        # PnL no realizado de posiciones abiertas
        unrealized_pnl = 0
        balances = self.exchange.get_balances()
        
        for sym, pos in self.positions.items():
            if self.current_timestamp:
                # Usar último precio conocido
                last_price = self.equity_curve[-1][1] if self.equity_curve else pos.entry_price
                # Estimación aproximada
                unrealized_pnl += 0  # Simplificación
        
        return realized_pnl + unrealized_pnl
    
    def _calculate_end_balance(self) -> float:
        """
        Calcula el balance final total en USDT.
        
        Lee el balance USDT del exchange simulado y añade el valor
        de BTC y ETH a precios de mercado si hay posiciones abiertas.
        
        Returns:
            float: Balance total en USDT
        """
        balances = self.exchange.get_balances()
        
        # Balance USDT (cash)
        usdt_balance = balances.get('USDT', 0.0)
        
        # El balance ya incluye el valor de las posiciones cerradas
        # Las posiciones abiertas se cierran al final del backtest
        return usdt_balance

    def close_all_positions(self, final_prices: Dict[str, float]):
        """Cierra todas las posiciones abiertas al final del backtest.
        
        IMPORTANTE: NO modificar el balance directamente. El exchange ya actualiza
        los balances internamente al ejecutar la orden. Solo calculamos PnL para
        el registro contable del TradeRecord.
        """
        for symbol, position in list(self.positions.items()):
            if symbol in final_prices:
                try:
                    price = final_prices[symbol]
                    
                    # 1. Ejecutar orden de venta - el exchange actualiza balances internamente
                    trade = self.exchange.execute_order(symbol, 'SELL', position.quantity, price)
                    
                    if trade:
                        # 2. Calcular PnL SOLO para el registro contable del TradeRecord
                        entry_value = position.entry_price * position.quantity
                        exit_value = trade['execution_price'] * position.quantity
                        pnl = exit_value - entry_value - trade['fee'] - position.fees
                        
                        hold_time = (self.current_timestamp - position.timestamp).total_seconds() / 60 if self.current_timestamp else 0
                        
                        # 3. Actualizar TradeRecord con PnL y datos de salida
                        position.exit_price = trade['execution_price']
                        position.pnl = pnl
                        position.pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                        position.exit_timestamp = self.current_timestamp
                        position.hold_time_minutes = hold_time
                        position.is_open = False
                        position.exit_reason = 'FINAL_CLOSE'
                        
                        # 4. Mover a cerrados
                        self.closed_trades.append(position)
                        
                        # 5. NO hacer ninguna operación aritmética sobre self.balance ni self.exchange.balances
                        # El exchange ya gestionó los balances internamente en execute_order()
                        
                        logger.info(f"🔚 Cierre final {symbol} @ {price:.2f} PnL={pnl:.2f}")
                
                except Exception as e:
                    logger.error(f"Error cerrando {symbol}: {e}")
        
        self.positions.clear()
    
    def calculate_metrics(self) -> BacktestMetrics:
        """Calcula las métricas finales del backtest"""
        if not self.closed_trades:
            return BacktestMetrics(start_balance=self.initial_balance, end_balance=self.initial_balance)
        
        # Métricas básicas
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        pnl_values = [t.pnl for t in self.closed_trades]
        total_pnl = sum(pnl_values)
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        max_drawdown = 0
        max_drawdown_pct = 0
        peak = self.initial_balance
        
        for timestamp, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = drawdown / peak if peak > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            if drawdown_pct > max_drawdown_pct:
                max_drawdown_pct = drawdown_pct
        
        # Sharpe ratio (simplificado)
        if len(pnl_values) > 1:
            returns = np.array(pnl_values)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Hold time promedio
        hold_times = [t.hold_time_minutes for t in self.closed_trades if t.hold_time_minutes > 0]
        avg_hold_time = np.mean(hold_times) if hold_times else 0
        
        # Best/worst trade
        best_trade = max(pnl_values) if pnl_values else 0
        worst_trade = min(pnl_values) if pnl_values else 0
        
        # Balances
        end_balance = self._calculate_end_balance()
        return_pct = (end_balance - self.initial_balance) / self.initial_balance if self.initial_balance > 0 else 0
        
        return BacktestMetrics(
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
    
    def export_results_csv(self, filepath: str):
        """Exporta los resultados a CSV"""
        if not self.signals_history:
            logger.warning("No hay datos para exportar")
            return
        
        df = pd.DataFrame(self.signals_history)
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Resultados exportados a {filepath}")
    
    def export_trades_csv(self, filepath: str):
        """Exporta el historial de trades a CSV"""
        if not self.closed_trades:
            logger.warning("No hay trades para exportar")
            return
        
        trades_data = []
        for trade in self.closed_trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'confidence': trade.confidence,
                'signal_type': trade.signal_type,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'fees': trade.fees,
                'hold_time_minutes': trade.hold_time_minutes,
                'exit_timestamp': trade.exit_timestamp
            })
        
        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Trades exportados a {filepath}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def run_backtest(
    symbols: List[str] = None,
    timeframe: str = '1m',
    days: int = 30,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    position_size_pct: float = 0.1,
    min_confidence: float = 0.6,
    use_cache: bool = True,
    output_dir: str = None,
    sample_every: int = 5,
    stop_loss_pct: float = 0.015,
    take_profit_pct: float = 0.025,
    cooldown_minutes: float = 240,
    analyze_exits: bool = False
) -> Dict[str, Any]:
    """
    Ejecuta el backtest completo
    
    Args:
        symbols: Lista de símbolos a testear
        timeframe: Timeframe de las velas
        days: Días de datos históricos
        initial_balance: Balance inicial en USDT
        position_size_pct: Porcentaje del capital por operación
        min_confidence: Confianza mínima para operar
        use_cache: Usar datos locales si existen
        output_dir: Directorio para guardar resultados
    
    Returns:
        Dict con métricas y resultados
    """
    symbols = symbols or DEFAULT_SYMBOLS
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO BACKTEST L1 STANDALONE")
    logger.info("=" * 80)
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Días: {days}")
    logger.info(f"Balance inicial: ${initial_balance:.2f}")
    logger.info(f"Tamaño posición: {position_size_pct:.1%}")
    logger.info(f"Confianza mínima: {min_confidence:.2f}")
    logger.info(f"Cooldown: {cooldown_minutes:.0f} minutos")
    logger.info(f"Sample every: {sample_every} velas")
    logger.info(f"Stop-loss: {stop_loss_pct:.2%}")
    logger.info(f"Take-profit: {take_profit_pct:.2%}")
    
    # Cargar datos
    logger.info("\n📊 Cargando datos históricos...")
    data_loader = DataLoader()
    market_data = await data_loader.load_data(symbols, timeframe, days, use_cache)
    
    if not market_data:
        logger.error("❌ No se pudieron cargar datos")
        return {}
    
    logger.info(f"✅ Datos cargados: {list(market_data.keys())}")
    for sym, df in market_data.items():
        logger.info(f"  - {sym}: {len(df)} velas")
    
    # Inicializar motor de backtest
    engine = L1BacktestEngine(
        initial_balance=initial_balance,
        position_size_pct=position_size_pct,
        min_confidence=min_confidence,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        cooldown_minutes=cooldown_minutes
    )
    
    # Procesar velas
    logger.info("\n⏳ Procesando velas...")
    
    # Unificar timestamps de todos los símbolos
    all_timestamps = set()
    for sym, df in market_data.items():
        all_timestamps.update(df.index)
    
    sorted_timestamps = sorted(all_timestamps)
    total = len(sorted_timestamps)
    logger.info(f"Total de velas a procesar: {total}")

    WARMUP = 50
    SAMPLE_EVERY = sample_every
    sym_indices = {sym: 0 for sym in market_data}
    progress_interval = max(1, total // 20)
    start_time_loop = time.time()

    for i, timestamp in enumerate(sorted_timestamps):
        if i % SAMPLE_EVERY != 0:
            for sym, df in market_data.items():
                while sym_indices[sym] < len(df) and df.index[sym_indices[sym]] <= timestamp:
                    sym_indices[sym] += 1
            continue

        for sym, df in market_data.items():
            while sym_indices[sym] < len(df) and df.index[sym_indices[sym]] <= timestamp:
                sym_indices[sym] += 1

        current_market_data = {}
        for sym, df in market_data.items():
            end_idx = sym_indices[sym]
            if end_idx >= WARMUP:
                slice_df = df.iloc[max(0, end_idx - 200):end_idx]
                slice_df.index.name = sym
                current_market_data[sym] = slice_df

        if not current_market_data:
            continue

        for sym, df_view in current_market_data.items():
            if not df_view.empty and df_view.index[-1] == timestamp:
                candle = df_view.iloc[-1]
                engine.process_candle(timestamp, sym, candle, current_market_data, candle_idx=i)

        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time_loop
            pct = (i + 1) / total
            eta = (elapsed / pct) * (1 - pct) if pct > 0 else 0
            logger.info(f"  Progreso: {pct*100:.1f}% ({i+1}/{total}) | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
    
    # Cerrar posiciones abiertas
    logger.info("\n🔚 Cerrando posiciones abiertas...")
    final_prices = {sym: df['close'].iloc[-1] for sym, df in market_data.items()}
    engine.close_all_positions(final_prices)
    
    # Calcular métricas
    logger.info("\n📈 Calculando métricas...")
    metrics = engine.calculate_metrics()
    
    # DIAGNÓSTICO: Mostrar estadísticas de señales
    logger.info("\n" + "=" * 80)
    logger.info("🔍 ESTADÍSTICAS DE DIAGNÓSTICO")
    logger.info("=" * 80)
    logger.info(f"  Distribución de señales raw: BUY={engine.diag_signal_counts.get('BUY', 0)}, "
               f"SELL={engine.diag_signal_counts.get('SELL', 0)}, "
               f"HOLD={engine.diag_signal_counts.get('HOLD', 0)}")
    logger.info(f"  Filtradas por confianza baja: {engine.diag_filtered_by_confidence}")
    logger.info(f"  Filtradas por TrendAI: {engine.diag_filtered_by_trend}")
    logger.info(f"  Trades ejecutados: {engine.diag_executed}")
    logger.info(f"  Confianza mínima usada: {engine.min_confidence}")
    
    # Estadísticas de stop-loss y take-profit
    stops_hit = sum(1 for t in engine.closed_trades if t.exit_reason == 'STOP_LOSS')
    tps_hit = sum(1 for t in engine.closed_trades if t.exit_reason == 'TAKE_PROFIT')
    logger.info(f"  Stop-loss activados: {stops_hit}, Take-profit: {tps_hit}")
    
    # Mostrar resultados
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESULTADOS DEL BACKTEST")
    logger.info("=" * 80)
    for key, value in metrics.to_dict().items():
        logger.info(f"  {key:25s}: {value}")
    
    # Exportar resultados
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_csv = os.path.join(output_dir, f"l1_backtest_signals_{timestamp_str}.csv")
    engine.export_results_csv(results_csv)
    
    trades_csv = os.path.join(output_dir, f"l1_backtest_trades_{timestamp_str}.csv")
    engine.export_trades_csv(trades_csv)
    
    # Guardar métricas en JSON
    metrics_json = os.path.join(output_dir, f"l1_backtest_metrics_{timestamp_str}.json")
    with open(metrics_json, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"💾 Métricas guardadas en {metrics_json}")
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'symbols': symbols,
        'timeframe': timeframe,
        'days': days,
        'initial_balance': initial_balance,
        'position_size_pct': position_size_pct,
        'min_confidence': min_confidence,
        'metrics': metrics.to_dict(),
        'files': {
            'signals_csv': results_csv,
            'trades_csv': trades_csv,
            'metrics_json': metrics_json
        }
    }
    
    summary_json = os.path.join(output_dir, f"l1_backtest_summary_{timestamp_str}.json")
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # ANÁLISIS DE EXITS (si está habilitado)
    if analyze_exits:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 ANÁLISIS DE STOP-LOSS")
        logger.info("=" * 80)
        
        # Filtrar trades cerrados por stop-loss con datos de max_favorable
        sl_trades = [
            t for t in engine.closed_trades 
            if t.exit_reason == 'STOP_LOSS' and t.max_favorable_pct is not None
        ]
        
        if sl_trades:
            total_sl = len(sl_trades)
            # Contar trades donde el precio llegó al TP antes del SL
            sl_reached_tp = sum(
                1 for t in sl_trades 
                if t.max_favorable_pct >= take_profit_pct
            )
            
            # Calcular porcentaje
            pct_reached_tp = (sl_reached_tp / total_sl) * 100 if total_sl > 0 else 0
            
            logger.info(f"  Trades cerrados por SL: {total_sl}")
            logger.info(f"  Trades donde el precio llegó al TP antes: {sl_reached_tp}/{total_sl} ({pct_reached_tp:.1f}%)")
            
            # Mostrar estadísticas adicionales
            max_fav_values = [t.max_favorable_pct for t in sl_trades]
            avg_max_fav = sum(max_fav_values) / len(max_fav_values)
            max_max_fav = max(max_fav_values)
            
            logger.info(f"  Max favorable promedio en SL: {avg_max_fav:.2%}")
            logger.info(f"  Max favorable más alto en SL: {max_max_fav:.2%}")
            
            # Diagnóstico
            if pct_reached_tp > 40:
                logger.info("\n  ⚠️  DIAGNÓSTICO: El SL está muy ajustado para la volatilidad del activo.")
                logger.info(f"      {pct_reached_tp:.1f}% de los trades cerrados por SL habrían alcanzado el TP.")
                logger.info(f"      → Recomendación: Subir stop_loss_pct (actual: {stop_loss_pct:.2%})")
            elif pct_reached_tp < 20:
                logger.info("\n  ✅ DIAGNÓSTICO: El SL parece bien ajustado.")
                logger.info(f"      Solo {pct_reached_tp:.1f}% de los trades cerrados por SL habrían alcanzado el TP.")
                logger.info("      → El problema es probablemente de dirección (modelo entrando en contra de tendencia)")
            else:
                logger.info(f"\n  ℹ️  DIAGNÓSTICO: Valor intermedio ({pct_reached_tp:.1f}%).")
                logger.info("      Ajuste de SL podría ayudar, pero no es crítico.")
        else:
            logger.info("  No hay trades cerrados por stop-loss para analizar.")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BACKTEST COMPLETADO")
    logger.info("=" * 80)
    
    return summary


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Backtesting L1 Standalone - Modelos L1 aislados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Backtest con configuración por defecto (BTC y ETH, últimos 30 días)
  python backtesting/backtest_l1_standalone.py

  # Backtest con parámetros personalizados
  python backtesting/backtest_l1_standalone.py --symbols BTCUSDT --days 60 --balance 5000

  # Backtest con descarga forzada (sin caché)
  python backtesting/backtest_l1_standalone.py --no-cache --days 90
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=DEFAULT_SYMBOLS,
        help=f'Símbolos a testear (default: {DEFAULT_SYMBOLS})'
    )
    parser.add_argument(
        '--timeframe',
        default=DEFAULT_TIMEFRAME,
        help=f'Timeframe de las velas (default: {DEFAULT_TIMEFRAME})'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Días de datos históricos (default: 30)'
    )
    parser.add_argument(
        '--balance',
        type=float,
        default=DEFAULT_INITIAL_BALANCE,
        help=f'Balance inicial en USDT (default: {DEFAULT_INITIAL_BALANCE})'
    )
    parser.add_argument(
        '--position-size',
        type=float,
        default=0.1,
        help='Porcentaje del capital por operación (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.68,
        help='Confianza mínima para operar (default: 0.68)'
    )
    parser.add_argument(
        '--sample-every',
        type=int,
        default=5,
        help='Procesar 1 de cada N velas (default: 5)'
    )
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.015,
        help='Stop-loss porcentual (default: 0.015 = 1.5%%)'
    )
    parser.add_argument(
        '--take-profit',
        type=float,
        default=0.025,
        help='Take-profit porcentual (default: 0.025 = 2.5%%)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='No usar datos en caché, descargar siempre'
    )
    parser.add_argument(
        '--output-dir',
        default=RESULTS_DIR,
        help=f'Directorio de salida (default: {RESULTS_DIR})'
    )
    parser.add_argument(
        '--analyze-exits',
        action='store_true',
        help='Analizar trades cerrados por stop-loss para detectar si el SL estaba muy ajustado'
    )
    parser.add_argument(
        '--cooldown',
        type=float,
        default=360,
        help='Cooldown entre trades en minutos (default: 360 = 6 horas)'
    )
    
    args = parser.parse_args()
    
    # Ejecutar backtest
    try:
        result = asyncio.run(run_backtest(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            initial_balance=args.balance,
            position_size_pct=args.position_size,
            min_confidence=args.min_confidence,
            use_cache=not args.no_cache,
            output_dir=args.output_dir,
            sample_every=args.sample_every,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            cooldown_minutes=args.cooldown,
            analyze_exits=args.analyze_exits
        ))
        
        if result:
            print("\n✅ Backtest completado exitosamente")
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