#!/usr/bin/env python3
"""
Backtesting L3 Regime Classifier - Validación del clasificador de régimen de mercado

Este script:
1. Carga datos históricos OHLCV de BTC y ETH desde CSV locales o Binance (mínimo 1 año)
2. Instancia SOLO l3_strategy/regime_classifier.py y l3_strategy/range_detector.py
3. Para cada período, clasifica el régimen de mercado (bull/bear/sideways/volatile)
4. Valida la clasificación comparando con labels manuales conocidos (crashes de 2022, bull de 2023, etc.)
5. Calcula la confusion matrix del clasificador de régimen
6. Mide el lag del clasificador: ¿cuántas velas tarda en detectar un cambio de régimen real?
7. Compara: "trading con régimen L3 correcto" vs "trading ignorando L3" → ¿L3 añade alpha real?

IMPORTANTE: NO importa nada de l1_operational/ ni l2_tactic/ — L3 completamente aislado
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
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import logging
from enum import Enum

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path para importar módulos de l3_strategy
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Importar SOLO de l3_strategy - NADA de l1_operational ni l2_tactic
from l3_strategy.regime_classifier import MarketRegimeClassifier
from l3_strategy.range_detector import classify_regime, detect_range_market
from core.technical_indicators import calculate_technical_indicators

# Suprimir logs verbose de matplotlib si se usa
logging.getLogger('matplotlib').setLevel(logging.WARNING)


# =============================================================================
# CONFIGURACIÓN Y CONSTANTES
# =============================================================================

DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
DEFAULT_TIMEFRAME = '1h'  # 1h para análisis de régimen (más estable)
DEFAULT_DAYS = 730  # 2 años de datos por defecto
MINIMUM_DAYS = 365  # Mínimo 1 año requerido
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

# Régimenes posibles
class RegimeType(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"

# Mapeo de régimenes del clasificador a régimenes simplificados
REGIME_MAPPING = {
    'TRENDING': {
        'STRONG_BULL': RegimeType.BULL,
        'MODERATE_BULL': RegimeType.BULL,
        'WEAK_BULL': RegimeType.BULL,
        'STRONG_BEAR': RegimeType.BEAR,
        'MODERATE_BEAR': RegimeType.BEAR,
        'WEAK_BEAR': RegimeType.BEAR,
        None: RegimeType.UNKNOWN
    },
    'RANGE': {
        'TIGHT_RANGE': RegimeType.SIDEWAYS,
        'NORMAL_RANGE': RegimeType.SIDEWAYS,
        'WIDE_RANGE': RegimeType.SIDEWAYS,
        'OVERSOLD_SETUP': RegimeType.SIDEWAYS,
        'OVERBOUGHT_SETUP': RegimeType.SIDEWAYS,
        None: RegimeType.SIDEWAYS
    },
    'VOLATILE': {
        'HIGH_VOLATILITY': RegimeType.VOLATILE,
        None: RegimeType.VOLATILE
    },
    'BREAKOUT': {
        'BULL_BREAKOUT': RegimeType.BREAKOUT,
        'BEAR_BREAKOUT': RegimeType.BREAKOUT,
        None: RegimeType.BREAKOUT
    },
    'ERROR': {
        None: RegimeType.UNKNOWN
    }
}


def map_regime_to_simplified(primary_regime: str, subtype: Optional[str]) -> RegimeType:
    """Mapea el régimen del clasificador a un tipo simplificado"""
    if primary_regime in REGIME_MAPPING:
        if subtype in REGIME_MAPPING[primary_regime]:
            return REGIME_MAPPING[primary_regime][subtype]
        return REGIME_MAPPING[primary_regime][None]
    return RegimeType.UNKNOWN


# =============================================================================
# LABELS MANUALES DE VALIDACIÓN (Ground Truth)
# =============================================================================

# Eventos históricos conocidos para validación
HISTORICAL_EVENTS = {
    'BTCUSDT': [
        # 2022 - Crypto Winter / Crashes
        {'start': '2022-01-01', 'end': '2022-03-15', 'regime': 'bear', 'label': 'Early 2022 Correction'},
        {'start': '2022-03-15', 'end': '2022-06-30', 'regime': 'bear', 'label': 'Terra/UST Collapse'},
        {'start': '2022-06-30', 'end': '2022-09-15', 'regime': 'sideways', 'label': 'Summer Consolidation'},
        {'start': '2022-09-15', 'end': '2022-11-30', 'regime': 'bear', 'label': 'FTX Collapse'},
        {'start': '2022-11-30', 'end': '2023-01-31', 'regime': 'sideways', 'label': 'End of 2022 Accumulation'},
        
        # 2023 - Recovery & Bull Market
        {'start': '2023-02-01', 'end': '2023-04-15', 'regime': 'bull', 'label': 'Q1 2023 Rally'},
        {'start': '2023-04-15', 'end': '2023-06-30', 'regime': 'sideways', 'label': 'Mid-2023 Consolidation'},
        {'start': '2023-07-01', 'end': '2023-10-31', 'regime': 'bull', 'label': 'Q3-Q4 2023 Bull'},
        {'start': '2023-11-01', 'end': '2024-01-31', 'regime': 'bull', 'label': 'ETF Approval Rally'},
        
        # 2024 - Continued Bull & Halving
        {'start': '2024-02-01', 'end': '2024-03-15', 'regime': 'bull', 'label': 'Pre-Halving Rally'},
        {'start': '2024-03-15', 'end': '2024-07-31', 'regime': 'sideways', 'label': 'Post-Halving Consolidation'},
        {'start': '2024-08-01', 'end': '2024-12-31', 'regime': 'volatile', 'label': 'Late 2024 Volatility'},
    ],
    'ETHUSDT': [
        # 2022
        {'start': '2022-01-01', 'end': '2022-03-15', 'regime': 'bear', 'label': 'Early 2022 Correction'},
        {'start': '2022-03-15', 'end': '2022-06-30', 'regime': 'bear', 'label': 'DeFi Contagion'},
        {'start': '2022-06-30', 'end': '2022-09-15', 'regime': 'sideways', 'label': 'Summer Consolidation'},
        {'start': '2022-09-15', 'end': '2022-12-31', 'regime': 'bear', 'label': 'Merge Sell-off & FTX'},
        
        # 2023
        {'start': '2023-01-01', 'end': '2023-04-15', 'regime': 'bull', 'label': 'Q1 2023 Rally'},
        {'start': '2023-04-15', 'end': '2023-10-31', 'regime': 'sideways', 'label': 'Long Consolidation'},
        {'start': '2023-11-01', 'end': '2024-03-15', 'regime': 'bull', 'label': 'ETF Narrative Rally'},
        
        # 2024
        {'start': '2024-03-15', 'end': '2024-05-31', 'regime': 'bull', 'label': 'ETH ETF Speculation'},
        {'start': '2024-06-01', 'end': '2024-12-31', 'regime': 'sideways', 'label': 'Post-ETF Consolidation'},
    ]
}


# =============================================================================
# DATA CLASSES PARA RESULTADOS
# =============================================================================

@dataclass
class RegimeClassification:
    """Clasificación de régimen en un punto temporal"""
    timestamp: datetime
    symbol: str
    primary_regime: str  # TRENDING, RANGE, VOLATILE, BREAKOUT
    subtype: Optional[str]
    simplified_regime: RegimeType
    confidence: float
    price: float
    price_change_24h: float
    price_change_7d: float
    volatility: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class RegimeChange:
    """Cambio de régimen detectado"""
    timestamp: datetime
    symbol: str
    from_regime: RegimeType
    to_regime: RegimeType
    price_at_change: float
    detection_lag_candles: int  # Velas de retraso en detectar
    confidence: float


@dataclass
class ConfusionMatrix:
    """Matriz de confusión para evaluación del clasificador"""
    # Matriz: predicho vs actual
    matrix: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    def add(self, actual: str, predicted: str):
        self.matrix[actual][predicted] += 1
    
    def accuracy(self) -> float:
        total = sum(sum(row.values()) for row in self.matrix.values())
        correct = sum(self.matrix[r].get(r, 0) for r in self.matrix.keys())
        return correct / total if total > 0 else 0.0
    
    def precision(self, regime: str) -> float:
        tp = self.matrix[regime].get(regime, 0)
        fp = sum(self.matrix[r].get(regime, 0) for r in self.matrix.keys() if r != regime)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def recall(self, regime: str) -> float:
        tp = self.matrix[regime].get(regime, 0)
        fn = sum(self.matrix[regime].get(r, 0) for r in self.matrix.keys() if r != regime)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def f1_score(self, regime: str) -> float:
        p = self.precision(regime)
        r = self.recall(regime)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def to_dict(self) -> Dict:
        regimes = list(self.matrix.keys())
        return {
            'matrix': {k: dict(v) for k, v in self.matrix.items()},
            'accuracy': self.accuracy(),
            'per_regime': {
                r: {
                    'precision': self.precision(r),
                    'recall': self.recall(r),
                    'f1_score': self.f1_score(r)
                }
                for r in regimes
            }
        }


@dataclass
class LagAnalysis:
    """Análisis de lag del clasificador"""
    total_changes: int = 0
    detected_changes: int = 0
    avg_lag_candles: float = 0.0
    median_lag_candles: float = 0.0
    max_lag_candles: int = 0
    min_lag_candles: int = 0
    lag_distribution: List[int] = field(default_factory=list)
    
    def add_lag(self, lag: int):
        self.lag_distribution.append(lag)
        self.total_changes += 1
        self.avg_lag_candles = np.mean(self.lag_distribution)
        self.median_lag_candles = np.median(self.lag_distribution)
        self.max_lag_candles = max(self.lag_distribution)
        self.min_lag_candles = min(self.lag_distribution)
    
    def to_dict(self) -> Dict:
        return {
            'total_regime_changes': self.total_changes,
            'detected_changes': self.detected_changes,
            'avg_lag_candles': self.avg_lag_candles,
            'median_lag_candles': self.median_lag_candles,
            'min_lag_candles': self.min_lag_candles,
            'max_lag_candles': self.max_lag_candles,
            'lag_distribution': self.lag_distribution
        }


@dataclass
class TradingPerformance:
    """Métricas de trading"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    
    def add_trade(self, entry_price: float, exit_price: float, side: str, timestamp: datetime):
        if side == 'BUY':
            pnl = exit_price - entry_price
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SELL/SHORT
            pnl = entry_price - exit_price
            pnl_pct = (entry_price - exit_price) / entry_price
        
        self.trades.append({
            'timestamp': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'side': side,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        self.avg_pnl = self.total_pnl / self.total_trades if self.total_trades > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy_name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'total_pnl': f"{self.total_pnl:.2f}",
            'avg_pnl': f"{self.avg_pnl:.2f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}"
        }


# =============================================================================
# CARGA DE DATOS
# =============================================================================

class DataLoader:
    """Cargador de datos históricos desde CSV o Binance"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_from_csv(self, symbol: str, timeframe: str = '1h') -> Optional[pd.DataFrame]:
        """Carga datos desde CSV local"""
        possible_names = [
            f"{symbol.lower()}_{timeframe}.csv",
            f"{symbol.upper()}_{timeframe}.csv",
            f"{symbol.replace('USDT', '').lower()}_1h.csv",
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
        
        for old, new in column_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    async def download_from_binance(
        self, 
        symbol: str, 
        timeframe: str = '1h',
        days: int = 730,
        testnet: bool = True
    ) -> pd.DataFrame:
        """Descarga datos desde Binance"""
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
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            since = int(start_time.timestamp() * 1000)
            
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
                
                if len(all_ohlcv) % 10000 == 0:
                    logger.info(f"  Descargados {len(all_ohlcv)} registros...")
                
                await asyncio.sleep(0.1)
            
            await exchange.close()
            
            df = pd.DataFrame(
                all_ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            logger.info(f"✅ Descargados {len(df)} registros de {symbol} ({len(df)/24:.1f} días)")
            
            # Guardar CSV
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
        timeframe: str = '1h',
        days: int = 730,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Carga datos para múltiples símbolos"""
        data = {}
        
        for symbol in symbols:
            df = None
            
            if use_cache:
                df = self.load_from_csv(symbol, timeframe)
            
            if df is None:
                df = await self.download_from_binance(symbol, timeframe, days)
            
            if df is not None and not df.empty:
                # Verificar que tenemos al menos 1 año de datos
                if len(df) < MINIMUM_DAYS * 24:  # 1 año de datos horarios
                    logger.warning(f"⚠️ {symbol}: Solo {len(df)/24:.0f} días de datos, se recomienda al menos 365")
                data[symbol] = df
            else:
                logger.warning(f"⚠️ No se pudieron cargar datos para {symbol}")
        
        return data


# =============================================================================
# MOTOR DE BACKTESTING DE RÉGIMEN
# =============================================================================

class L3RegimeBacktestEngine:
    """Motor de backtesting para clasificador de régimen L3"""
    
    def __init__(
        self,
        window_size: int = 168,  # 7 días de datos horarios
        classification_interval: int = 6,  # Clasificar cada 6 horas
    ):
        self.window_size = window_size
        self.classification_interval = classification_interval
        
        # Inicializar clasificadores
        self.regime_classifier = MarketRegimeClassifier()
        
        # Estado
        self.classifications: List[RegimeClassification] = []
        self.regime_changes: List[RegimeChange] = []
        self.ground_truth: List[Dict] = []
        self.confusion_matrix = ConfusionMatrix()
        self.lag_analysis = LagAnalysis()
        
        # Tracking de cambios reales
        self._actual_regime_changes: List[Dict] = []
        self._last_known_regime: Dict[str, RegimeType] = {}
    
    def classify_regime(
        self,
        timestamp: datetime,
        symbol: str,
        window_df: pd.DataFrame
    ) -> Optional[RegimeClassification]:
        """Clasifica el régimen actual"""
        try:
            # Usar el clasificador de régimen
            result = self.regime_classifier.classify_market_regime(
                df=window_df,
                symbol=symbol,
                cycle_id=int(timestamp.timestamp())
            )
            
            if result['primary_regime'] == 'ERROR':
                return None
            
            # Calcular cambios de precio
            current_price = window_df['close'].iloc[-1]
            price_24h_ago = window_df['close'].iloc[-min(24, len(window_df))] if len(window_df) >= 24 else window_df['close'].iloc[0]
            price_7d_ago = window_df['close'].iloc[-min(168, len(window_df))] if len(window_df) >= 168 else window_df['close'].iloc[0]
            
            price_change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0
            price_change_7d = (current_price - price_7d_ago) / price_7d_ago if price_7d_ago > 0 else 0
            
            # Calcular volatilidad
            returns = window_df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24) if len(returns) > 1 else 0
            
            # Mapear a régimen simplificado
            simplified = map_regime_to_simplified(
                result['primary_regime'],
                result['subtype']
            )
            
            classification = RegimeClassification(
                timestamp=timestamp,
                symbol=symbol,
                primary_regime=result['primary_regime'],
                subtype=result['subtype'],
                simplified_regime=simplified,
                confidence=result['confidence'],
                price=current_price,
                price_change_24h=price_change_24h,
                price_change_7d=price_change_7d,
                volatility=volatility,
                metadata=result.get('metadata', {})
            )
            
            return classification
            
        except Exception as e:
            logger.error(f"❌ Error clasificando régimen: {e}")
            return None
    
    def detect_actual_regime_changes(
        self,
        df: pd.DataFrame,
        symbol: str,
        lookback: int = 72
    ) -> List[Dict]:
        """Detecta cambios de régimen reales basados en movimientos de precio"""
        changes = []
        
        if len(df) < lookback * 2:
            return changes
        
        for i in range(lookback, len(df) - lookback):
            window_before = df.iloc[i-lookback:i]
            window_after = df.iloc[i:i+lookback]
            
            price_before = window_before['close'].iloc[-1]
            price_after = window_after['close'].iloc[-1]
            
            returns_before = window_before['close'].pct_change().std()
            returns_after = window_after['close'].pct_change().std()
            
            # Detectar cambio significativo
            price_change = (price_after - price_before) / price_before
            
            # Clasificar el régimen antes y después
            regime_before = self._classify_simple_regime(window_before)
            regime_after = self._classify_simple_regime(window_after)
            
            if regime_before != regime_after and abs(price_change) > 0.05:  # 5% de cambio mínimo
                changes.append({
                    'timestamp': df.index[i],
                    'symbol': symbol,
                    'from_regime': regime_before,
                    'to_regime': regime_after,
                    'price': df['close'].iloc[i],
                    'price_change': price_change
                })
        
        return changes
    
    def _classify_simple_regime(self, df: pd.DataFrame) -> str:
        """Clasificación simple de régimen basada en movimiento de precio"""
        if len(df) < 24:
            return 'unknown'
        
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        price_change = (end_price - start_price) / start_price
        
        volatility = df['close'].pct_change().std() * np.sqrt(len(df))
        
        if volatility > 0.03:  # Alta volatilidad
            return 'volatile'
        elif price_change > 0.10:  # +10%
            return 'bull'
        elif price_change < -0.10:  # -10%
            return 'bear'
        else:
            return 'sideways'
    
    def validate_against_ground_truth(
        self,
        classifications: List[RegimeClassification],
        symbol: str
    ) -> None:
        """Valida clasificaciones contra labels manuales"""
        if symbol not in HISTORICAL_EVENTS:
            return
        
        events = HISTORICAL_EVENTS[symbol]
        
        for event in events:
            event_start = pd.to_datetime(event['start'])
            event_end = pd.to_datetime(event['end'])
            actual_regime = event['regime']
            
            # Encontrar clasificaciones en este período
            period_classifications = [
                c for c in classifications
                if event_start <= c.timestamp <= event_end
            ]
            
            if not period_classifications:
                continue
            
            # La clasificación predominante
            regime_votes = Counter([c.simplified_regime.value for c in period_classifications])
            predicted_regime = regime_votes.most_common(1)[0][0]
            
            # Añadir a matriz de confusión
            self.confusion_matrix.add(actual_regime, predicted_regime)
            
            # Guardar para análisis
            self.ground_truth.append({
                'event_label': event['label'],
                'period': f"{event['start']} to {event['end']}",
                'actual_regime': actual_regime,
                'predicted_regime': predicted_regime,
                'confidence': np.mean([c.confidence for c in period_classifications]),
                'correct': actual_regime == predicted_regime,
                'classifications_count': len(period_classifications)
            })
    
    def calculate_lag(
        self,
        actual_changes: List[Dict],
        detected_changes: List[RegimeChange]
    ) -> LagAnalysis:
        """Calcula el lag del clasificador en detectar cambios"""
        lag_analysis = LagAnalysis()
        
        for actual in actual_changes:
            actual_time = actual['timestamp']
            
            # Buscar detección correspondiente
            for detected in detected_changes:
                time_diff_hours = abs((detected.timestamp - actual_time).total_seconds() / 3600)
                
                # Si está dentro de 7 días (168 horas), considerarlo detectado
                if time_diff_hours <= 168:
                    lag_candles = int(time_diff_hours / self.classification_interval)
                    lag_analysis.add_lag(lag_candles)
                    lag_analysis.detected_changes += 1
                    break
        
        return lag_analysis
    
    def run_backtest(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Ejecuta el backtest completo"""
        logger.info("=" * 80)
        logger.info("🚀 INICIANDO BACKTEST L3 RÉGIMEN")
        logger.info("=" * 80)
        
        for symbol, df in market_data.items():
            logger.info(f"\n📊 Procesando {symbol}...")
            logger.info(f"   Datos: {len(df)} velas ({len(df)/24:.1f} días)")
            
            # Detectar cambios reales de régimen
            actual_changes = self.detect_actual_regime_changes(df, symbol)
            self._actual_regime_changes.extend(actual_changes)
            logger.info(f"   Cambios de régimen reales detectados: {len(actual_changes)}")
            
            # Clasificar régimen en intervalos
            classifications = []
            last_regime: Optional[RegimeType] = None
            
            for i in range(self.window_size, len(df), self.classification_interval):
                window_df = df.iloc[i-self.window_size:i].copy()
                timestamp = df.index[i]
                
                classification = self.classify_regime(timestamp, symbol, window_df)
                
                if classification:
                    classifications.append(classification)
                    
                    # Detectar cambio de régimen
                    if last_regime is not None and classification.simplified_regime != last_regime:
                        regime_change = RegimeChange(
                            timestamp=timestamp,
                            symbol=symbol,
                            from_regime=last_regime,
                            to_regime=classification.simplified_regime,
                            price_at_change=classification.price,
                            detection_lag_candles=0,  # Se calculará después
                            confidence=classification.confidence
                        )
                        self.regime_changes.append(regime_change)
                    
                    last_regime = classification.simplified_regime
                
                # Progreso
                if (i - self.window_size) % (self.classification_interval * 100) == 0:
                    pct = (i - self.window_size) / (len(df) - self.window_size) * 100
                    logger.info(f"   Progreso: {pct:.1f}%")
            
            self.classifications.extend(classifications)
            
            # Validar contra ground truth
            logger.info(f"   Validando contra labels manuales...")
            self.validate_against_ground_truth(classifications, symbol)
        
        # Calcular lag
        logger.info(f"\n⏱️ Calculando lag del clasificador...")
        self.lag_analysis = self.calculate_lag(
            self._actual_regime_changes,
            self.regime_changes
        )
        
        logger.info("=" * 80)
        logger.info("✅ BACKTEST COMPLETADO")
        logger.info("=" * 80)
        
        return self._compile_results()
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compila todos los resultados"""
        return {
            'total_classifications': len(self.classifications),
            'total_regime_changes': len(self.regime_changes),
            'confusion_matrix': self.confusion_matrix.to_dict(),
            'lag_analysis': self.lag_analysis.to_dict(),
            'ground_truth_validation': self.ground_truth,
            'classifications_summary': self._summarize_classifications(),
            'regime_distribution': self._calculate_regime_distribution()
        }
    
    def _summarize_classifications(self) -> Dict:
        """Resumen de clasificaciones por símbolo"""
        summary = {}
        
        for symbol in set(c.symbol for c in self.classifications):
            symbol_classifications = [c for c in self.classifications if c.symbol == symbol]
            
            summary[symbol] = {
                'total': len(symbol_classifications),
                'avg_confidence': np.mean([c.confidence for c in symbol_classifications]),
                'regimes': dict(Counter([c.simplified_regime.value for c in symbol_classifications]))
            }
        
        return summary
    
    def _calculate_regime_distribution(self) -> Dict:
        """Distribución de régimenes clasificados"""
        regimes = [c.simplified_regime.value for c in self.classifications]
        total = len(regimes)
        
        distribution = Counter(regimes)
        return {
            regime: {
                'count': count,
                'percentage': count / total if total > 0 else 0
            }
            for regime, count in distribution.items()
        }


# =============================================================================
# COMPARACIÓN DE ESTRATEGIAS DE TRADING
# =============================================================================

class TradingComparator:
    """Compara trading con y sin información de régimen L3"""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        position_size: float = 0.1,
        fee: float = 0.001
    ):
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.fee = fee
    
    def simulate_trading_with_l3(
        self,
        df: pd.DataFrame,
        classifications: List[RegimeClassification],
        symbol: str
    ) -> TradingPerformance:
        """Simula trading usando información de régimen L3"""
        perf = TradingPerformance(strategy_name="Trading con L3")
        
        # Crear lookup de clasificaciones por timestamp
        regime_lookup = {c.timestamp: c for c in classifications}
        
        in_position = False
        entry_price = 0.0
        entry_time = None
        
        for i in range(1, len(df)):
            timestamp = df.index[i]
            price = df['close'].iloc[i]
            
            # Obtener régimen actual
            regime = regime_lookup.get(timestamp)
            if not regime:
                continue
            
            # Estrategia basada en régimen
            if not in_position:
                # Solo entrar en bull o breakout
                if regime.simplified_regime in [RegimeType.BULL, RegimeType.BREAKOUT]:
                    if regime.confidence > 0.6:
                        entry_price = price
                        entry_time = timestamp
                        in_position = True
                        
            else:
                # Salir en bear o si hay señal de reversión
                if regime.simplified_regime in [RegimeType.BEAR]:
                    # Calcular PnL
                    pnl = (price - entry_price) / entry_price - 2 * self.fee
                    perf.add_trade(entry_price, price, 'BUY', timestamp)
                    in_position = False
                    
                # También salir si el cambio es muy negativo
                elif price < entry_price * 0.95:  # Stop loss del 5%
                    perf.add_trade(entry_price, price, 'BUY', timestamp)
                    in_position = False
                    
                # Take profit del 10%
                elif price > entry_price * 1.10:
                    perf.add_trade(entry_price, price, 'BUY', timestamp)
                    in_position = False
        
        # Cerrar posición abierta
        if in_position and entry_time:
            final_price = df['close'].iloc[-1]
            perf.add_trade(entry_price, final_price, 'BUY', df.index[-1])
        
        return perf
    
    def simulate_trading_without_l3(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> TradingPerformance:
        """Simula trading ignorando régimen (solo momentum simple)"""
        perf = TradingPerformance(strategy_name="Trading sin L3 (Momentum)")
        
        in_position = False
        entry_price = 0.0
        entry_time = None
        
        for i in range(24, len(df)):
            timestamp = df.index[i]
            price = df['close'].iloc[i]
            
            # Calcular momentum simple (24h)
            price_24h_ago = df['close'].iloc[i-24]
            momentum = (price - price_24h_ago) / price_24h_ago
            
            if not in_position:
                # Entrar en momentum positivo
                if momentum > 0.02:  # +2% en 24h
                    entry_price = price
                    entry_time = timestamp
                    in_position = True
                    
            else:
                # Salir en momentum negativo
                if momentum < -0.01:  # -1% en 24h
                    perf.add_trade(entry_price, price, 'BUY', timestamp)
                    in_position = False
                    
                # Stop loss
                elif price < entry_price * 0.95:
                    perf.add_trade(entry_price, price, 'BUY', timestamp)
                    in_position = False
                    
                # Take profit
                elif price > entry_price * 1.10:
                    perf.add_trade(entry_price, price, 'BUY', timestamp)
                    in_position = False
        
        # Cerrar posición abierta
        if in_position and entry_time:
            final_price = df['close'].iloc[-1]
            perf.add_trade(entry_price, final_price, 'BUY', df.index[-1])
        
        return perf
    
    def compare_strategies(
        self,
        market_data: Dict[str, pd.DataFrame],
        classifications: List[RegimeClassification]
    ) -> Dict[str, Any]:
        """Compara ambas estrategias"""
        logger.info("\n" + "=" * 80)
        logger.info("📈 COMPARACIÓN DE ESTRATEGIAS DE TRADING")
        logger.info("=" * 80)
        
        results = {
            'with_l3': {},
            'without_l3': {},
            'comparison': {}
        }
        
        for symbol, df in market_data.items():
            logger.info(f"\n🔄 Comparando estrategias para {symbol}...")
            
            symbol_classifications = [c for c in classifications if c.symbol == symbol]
            
            # Simular con L3
            perf_with_l3 = self.simulate_trading_with_l3(df, symbol_classifications, symbol)
            results['with_l3'][symbol] = perf_with_l3.to_dict()
            
            # Simular sin L3
            perf_without_l3 = self.simulate_trading_without_l3(df, symbol)
            results['without_l3'][symbol] = perf_without_l3.to_dict()
            
            # Comparar
            pnl_with = perf_with_l3.total_pnl
            pnl_without = perf_without_l3.total_pnl
            alpha = pnl_with - pnl_without
            
            results['comparison'][symbol] = {
                'pnl_with_l3': pnl_with,
                'pnl_without_l3': pnl_without,
                'alpha_added': alpha,
                'alpha_percentage': (alpha / abs(pnl_without) * 100) if pnl_without != 0 else 0,
                'win_rate_improvement': perf_with_l3.win_rate - perf_without_l3.win_rate,
                'l3_adds_alpha': alpha > 0
            }
            
            logger.info(f"   Trading CON L3: PnL={pnl_with:.2f}, Win Rate={perf_with_l3.win_rate:.1%}")
            logger.info(f"   Trading SIN L3: PnL={pnl_without:.2f}, Win Rate={perf_without_l3.win_rate:.1%}")
            logger.info(f"   Alpha añadido por L3: {alpha:.2f} ({alpha/abs(pnl_without)*100:.1f}%)" if pnl_without != 0 else "   Alpha añadido por L3: N/A")
        
        return results


# =============================================================================
# REPORTING Y VISUALIZACIÓN
# =============================================================================

class ReportGenerator:
    """Genera reportes del backtest"""
    
    def __init__(self, output_dir: str = RESULTS_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self,
        regime_results: Dict[str, Any],
        trading_results: Dict[str, Any],
        timestamp_str: str
    ) -> str:
        """Genera reporte completo"""
        report_file = os.path.join(self.output_dir, f"l3_regime_backtest_report_{timestamp_str}.txt")
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BACKTEST L3 RÉGIMEN - REPORTE DE RESULTADOS\n")
            f.write(f"Generado: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Resumen de clasificaciones
            f.write("📊 RESUMEN DE CLASIFICACIONES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total de clasificaciones: {regime_results['total_classifications']}\n")
            f.write(f"Cambios de régimen detectados: {regime_results['total_regime_changes']}\n\n")
            
            # Distribución de régimenes
            f.write("📈 DISTRIBUCIÓN DE RÉGIMENES\n")
            f.write("-" * 80 + "\n")
            for regime, stats in regime_results['regime_distribution'].items():
                f.write(f"  {regime:12s}: {stats['count']:4d} ({stats['percentage']:.1%})\n")
            f.write("\n")
            
            # Confusion Matrix
            f.write("🎯 MATRIZ DE CONFUSIÓN\n")
            f.write("-" * 80 + "\n")
            cm = regime_results['confusion_matrix']
            f.write(f"Accuracy global: {cm['accuracy']:.2%}\n\n")
            
            f.write("Matriz (Actual vs Predicho):\n")
            regimes = list(cm['matrix'].keys())
            f.write(f"{'Actual/Pred':12s}")
            for r in regimes:
                f.write(f"{r:10s}")
            f.write("\n")
            
            for actual in regimes:
                f.write(f"{actual:12s}")
                for predicted in regimes:
                    count = cm['matrix'][actual].get(predicted, 0)
                    f.write(f"{count:10d}")
                f.write("\n")
            
            f.write("\nMétricas por régimen:\n")
            for regime, metrics in cm['per_regime'].items():
                f.write(f"  {regime:12s}: Precision={metrics['precision']:.2%}, ")
                f.write(f"Recall={metrics['recall']:.2%}, F1={metrics['f1_score']:.2f}\n")
            f.write("\n")
            
            # Lag Analysis
            f.write("⏱️ ANÁLISIS DE LAG\n")
            f.write("-" * 80 + "\n")
            lag = regime_results['lag_analysis']
            f.write(f"Cambios de régimen totales: {lag['total_regime_changes']}\n")
            f.write(f"Cambios detectados: {lag['detected_changes']}\n")
            f.write(f"Lag promedio: {lag['avg_lag_candles']:.1f} velas\n")
            f.write(f"Lag mediana: {lag['median_lag_candles']:.1f} velas\n")
            f.write(f"Lag mínimo: {lag['min_lag_candles']} velas\n")
            f.write(f"Lag máximo: {lag['max_lag_candles']} velas\n\n")
            
            # Ground Truth Validation
            f.write("✅ VALIDACIÓN CONTRA GROUND TRUTH\n")
            f.write("-" * 80 + "\n")
            correct = sum(1 for g in regime_results['ground_truth_validation'] if g['correct'])
            total = len(regime_results['ground_truth_validation'])
            f.write(f"Eventos correctamente clasificados: {correct}/{total} ({correct/total:.1%})\n\n")
            
            for gt in regime_results['ground_truth_validation']:
                status = "✅" if gt['correct'] else "❌"
                f.write(f"{status} {gt['event_label'][:40]:40s} | ")
                f.write(f"Actual: {gt['actual_regime']:8s} | Pred: {gt['predicted_regime']:8s} | ")
                f.write(f"Conf: {gt['confidence']:.2f}\n")
            f.write("\n")
            
            # Trading Comparison
            f.write("💰 COMPARACIÓN DE ESTRATEGIAS DE TRADING\n")
            f.write("-" * 80 + "\n")
            
            for symbol, comp in trading_results['comparison'].items():
                f.write(f"\n{symbol}:\n")
                f.write(f"  Trading CON L3:     PnL={comp['pnl_with_l3']:.2f}\n")
                f.write(f"  Trading SIN L3:     PnL={comp['pnl_without_l3']:.2f}\n")
                f.write(f"  Alpha añadido:      {comp['alpha_added']:.2f} ({comp['alpha_percentage']:.1f}%)\n")
                f.write(f"  Mejora Win Rate:    {comp['win_rate_improvement']:.1%}\n")
                f.write(f"  L3 añade alpha:     {'SÍ' if comp['l3_adds_alpha'] else 'NO'}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("FIN DEL REPORTE\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"📄 Reporte guardado en: {report_file}")
        return report_file
    
    def export_classifications_csv(
        self,
        classifications: List[RegimeClassification],
        timestamp_str: str
    ) -> str:
        """Exporta clasificaciones a CSV"""
        csv_file = os.path.join(self.output_dir, f"l3_regime_classifications_{timestamp_str}.csv")
        
        data = []
        for c in classifications:
            data.append({
                'timestamp': c.timestamp,
                'symbol': c.symbol,
                'primary_regime': c.primary_regime,
                'subtype': c.subtype,
                'simplified_regime': c.simplified_regime.value,
                'confidence': c.confidence,
                'price': c.price,
                'price_change_24h': c.price_change_24h,
                'price_change_7d': c.price_change_7d,
                'volatility': c.volatility
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 Clasificaciones exportadas a: {csv_file}")
        return csv_file


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

async def run_backtest(
    symbols: List[str] = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    days: int = DEFAULT_DAYS,
    use_cache: bool = True,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Ejecuta el backtest completo del clasificador de régimen L3
    
    Args:
        symbols: Lista de símbolos a testear
        timeframe: Timeframe de las velas
        days: Días de datos históricos
        use_cache: Usar datos locales si existen
        output_dir: Directorio para guardar resultados
    
    Returns:
        Dict con métricas y resultados
    """
    symbols = symbols or DEFAULT_SYMBOLS
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO BACKTEST L3 RÉGIMEN")
    logger.info("=" * 80)
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Días: {days}")
    
    # Cargar datos
    logger.info("\n📊 Cargando datos históricos...")
    data_loader = DataLoader()
    market_data = await data_loader.load_data(symbols, timeframe, days, use_cache)
    
    if not market_data:
        logger.error("❌ No se pudieron cargar datos")
        return {}
    
    logger.info(f"✅ Datos cargados: {list(market_data.keys())}")
    for sym, df in market_data.items():
        logger.info(f"  - {sym}: {len(df)} velas ({len(df)/24:.1f} días)")
    
    # Ejecutar backtest de régimen
    logger.info("\n⏳ Ejecutando backtest de clasificación de régimen...")
    engine = L3RegimeBacktestEngine()
    regime_results = engine.run_backtest(market_data)
    
    # Comparar estrategias de trading
    logger.info("\n💰 Comparando estrategias de trading...")
    comparator = TradingComparator()
    trading_results = comparator.compare_strategies(market_data, engine.classifications)
    
    # Generar reportes
    logger.info("\n📄 Generando reportes...")
    report_gen = ReportGenerator(output_dir)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Reporte de texto
    report_file = report_gen.generate_report(regime_results, trading_results, timestamp_str)
    
    # CSV de clasificaciones
    csv_file = report_gen.export_classifications_csv(engine.classifications, timestamp_str)
    
    # Guardar resultados en JSON
    results_json = os.path.join(output_dir, f"l3_regime_backtest_results_{timestamp_str}.json")
    with open(results_json, 'w') as f:
        json.dump({
            'regime_results': regime_results,
            'trading_results': trading_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'symbols': symbols,
                'timeframe': timeframe,
                'days': days
            }
        }, f, indent=2, default=str)
    logger.info(f"💾 Resultados JSON guardados en: {results_json}")
    
    # Mostrar resumen final
    logger.info("\n" + "=" * 80)
    logger.info("📊 RESUMEN FINAL")
    logger.info("=" * 80)
    logger.info(f"Total clasificaciones: {regime_results['total_classifications']}")
    logger.info(f"Accuracy matriz confusión: {regime_results['confusion_matrix']['accuracy']:.2%}")
    logger.info(f"Lag promedio: {regime_results['lag_analysis']['avg_lag_candles']:.1f} velas")
    
    # Alpha añadido
    total_alpha = sum(comp['alpha_added'] for comp in trading_results['comparison'].values())
    logger.info(f"Alpha total añadido por L3: {total_alpha:.2f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BACKTEST COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"📄 Reporte: {report_file}")
    logger.info(f"💾 CSV: {csv_file}")
    logger.info(f"📊 JSON: {results_json}")
    
    return {
        'regime_results': regime_results,
        'trading_results': trading_results,
        'files': {
            'report': report_file,
            'csv': csv_file,
            'json': results_json
        }
    }


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Backtesting L3 Regime Classifier - Validación del clasificador de régimen',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Backtest con configuración por defecto (BTC y ETH, últimos 2 años)
  python backtesting/backtest_l3_regime.py

  # Backtest con parámetros personalizados
  python backtesting/backtest_l3_regime.py --symbols BTCUSDT --days 365

  # Backtest con descarga forzada (sin caché)
  python backtesting/backtest_l3_regime.py --no-cache --days 500
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
        default=DEFAULT_DAYS,
        help=f'Días de datos históricos (default: {DEFAULT_DAYS}, mínimo: {MINIMUM_DAYS})'
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
    
    args = parser.parse_args()
    
    # Validar días mínimos
    if args.days < MINIMUM_DAYS:
        logger.warning(f"⚠️ Se recomiendan al menos {MINIMUM_DAYS} días de datos. Usando {args.days} días.")
    
    # Ejecutar backtest
    try:
        result = asyncio.run(run_backtest(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            use_cache=not args.no_cache,
            output_dir=args.output_dir
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
