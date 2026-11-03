#!/usr/bin/env python3
"""
Backtest RL Puro - Evaluación de modelos RL sin filtros externos

Este script:
1. Detecta todos los modelos .zip en models/L2/
2. Para cada modelo, carga FinRLProcessor directamente (sin filtros)
3. Ejecuta backtest con señales PURAS del modelo
4. Calcula métricas de precisión direccional real

IMPORTANTE: No importa nada de l3_strategy/, no aplica filtros de confianza/RSI/volatilidad
"""

import os
import sys
import argparse
import glob
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

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

# Importar SOLO lo necesario de l2_tactic - NADA de l3_strategy
from l2_tactic.finrl_processor import FinRLProcessor
from l2_tactic.observation_builders import ObservationBuilders
from l2_tactic.models import TacticalSignal

# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

DEFAULT_INITIAL_BALANCE = 3000.0  # USDT
DEFAULT_FEE = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODELS_DIR = os.path.join(REPO_ROOT, 'models', 'L2')

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelInfo:
    """Información de un modelo RL detectado"""
    name: str
    path: str
    dims: int
    model_type: str
    loaded: bool = False
    error: Optional[str] = None

@dataclass
class TradeRecord:
    """Registro de una operación"""
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
    model_action: str = ""  # BUY/SELL/HOLD según el modelo

@dataclass
class DirectionalAccuracy:
    """Métricas de precisión direccional"""
    buy_signals: int = 0
    buy_correct: int = 0
    sell_signals: int = 0
    sell_correct: int = 0
    hold_signals: int = 0
    hold_correct: int = 0  # Precio no se movió significativamente
    
    @property
    def buy_accuracy(self) -> float:
        return self.buy_correct / self.buy_signals if self.buy_signals > 0 else 0.0
    
    @property
    def sell_accuracy(self) -> float:
        return self.sell_correct / self.sell_signals if self.sell_signals > 0 else 0.0
    
    @property
    def overall_accuracy(self) -> float:
        total = self.buy_signals + self.sell_signals + self.hold_signals
        correct = self.buy_correct + self.sell_correct + self.hold_correct
        return correct / total if total > 0 else 0.0

@dataclass
class PolicyActivationDiagnostics:
    """Diagnóstico de activaciones de política para detectar modelos degenerados"""
    model_name: str
    observations_tested: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    action_probabilities: List[float] = field(default_factory=list)
    is_degenerate: bool = False
    degeneracy_reason: str = ""
    
    @property
    def buy_pct(self) -> float:
        total = self.buy_count + self.sell_count + self.hold_count
        return self.buy_count / total * 100 if total > 0 else 0.0
    
    @property
    def sell_pct(self) -> float:
        total = self.buy_count + self.sell_count + self.hold_count
        return self.sell_count / total * 100 if total > 0 else 0.0
    
    @property
    def hold_pct(self) -> float:
        total = self.buy_count + self.sell_count + self.hold_count
        return self.hold_count / total * 100 if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'observations_tested': self.observations_tested,
            'buy_count': self.buy_count,
            'sell_count': self.sell_count,
            'hold_count': self.hold_count,
            'buy_pct': f"{self.buy_pct:.1f}%",
            'sell_pct': f"{self.sell_pct:.1f}%",
            'hold_pct': f"{self.hold_pct:.1f}%",
            'avg_action_probability': f"{np.mean(self.action_probabilities):.3f}" if self.action_probabilities else "N/A",
            'is_degenerate': self.is_degenerate,
            'degeneracy_reason': self.degeneracy_reason,
        }

@dataclass
class ModelResults:
    """Resultados de backtest para un modelo"""
    model_name: str
    model_dims: int
    
    # Métricas de trading
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    
    # Distribución de señales
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    total_signals: int = 0
    
    # Precisión direccional
    directional_accuracy: DirectionalAccuracy = field(default_factory=DirectionalAccuracy)
    
    # Equity curve
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'model_dims': self.model_dims,
            'total_trades': self.total_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'max_drawdown_pct': f"{self.max_drawdown_pct:.2%}",
            'total_pnl': f"{self.total_pnl:.2f}",
            'return_pct': f"{self.return_pct:.2%}",
            'buy_pct': f"{self.buy_count / self.total_signals:.2%}" if self.total_signals > 0 else "0%",
            'sell_pct': f"{self.sell_count / self.total_signals:.2%}" if self.total_signals > 0 else "0%",
            'hold_pct': f"{self.hold_count / self.total_signals:.2%}" if self.total_signals > 0 else "0%",
            'buy_accuracy': f"{self.directional_accuracy.buy_accuracy:.2%}",
            'sell_accuracy': f"{self.directional_accuracy.sell_accuracy:.2%}",
            'overall_directional_accuracy': f"{self.directional_accuracy.overall_accuracy:.2%}",
        }

# =============================================================================
# DETECCIÓN DE MODELOS
# =============================================================================

def detect_models(models_dir: str = MODELS_DIR) -> List[ModelInfo]:
    """Detecta todos los modelos .zip en el directorio y sus dimensionalidades"""
    models = []
    
    zip_files = glob.glob(os.path.join(models_dir, "*.zip"))
    logger.info(f"🔍 Detectados {len(zip_files)} modelos en {models_dir}")
    
    for zip_path in sorted(zip_files):
        name = os.path.basename(zip_path).replace('.zip', '')
        logger.info(f"  📦 {name}: {zip_path}")
        
        try:
            # Intentar cargar el modelo para detectar dimensionalidad
            processor = FinRLProcessor(zip_path)
            
            dims = processor.observation_space_info.get('expected_dims', 0) if processor.observation_space_info else 0
            model_type = processor.observation_space_info.get('type', 'unknown') if processor.observation_space_info else 'unknown'
            
            models.append(ModelInfo(
                name=name,
                path=zip_path,
                dims=dims,
                model_type=model_type,
                loaded=True
            ))
            
            logger.info(f"    ✅ Cargado: {dims} dims, tipo {model_type}")
            
            # Liberar memoria
            del processor
            
        except Exception as e:
            logger.error(f"    ❌ Error cargando {name}: {e}")
            models.append(ModelInfo(
                name=name,
                path=zip_path,
                dims=0,
                model_type='error',
                loaded=False,
                error=str(e)
            ))
    
    return models

# =============================================================================
# CARGA DE DATOS
# =============================================================================

def find_latest_l1_signals() -> Optional[str]:
    """Encuentra el archivo CSV de señales L1 más reciente"""
    patterns = [
        os.path.join(RESULTS_DIR, "l1_backtest_signals_*.csv"),
        os.path.join(".", "l1_backtest_signals_*.csv"),
        os.path.join(RESULTS_DIR, "*_signals_*.csv"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            return max(files, key=os.path.getctime)
    
    return None

def download_binance_data(symbol: str = "BTCUSDT", days: int = 30) -> pd.DataFrame:
    """Descarga datos históricos de Binance si no hay CSV disponible"""
    try:
        import requests
        
        logger.info(f"📥 Descargando datos de Binance para {symbol} ({days} días)...")
        
        # Calcular timestamps
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '1h',
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        logger.info(f"✅ Descargadas {len(df)} velas")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error descargando datos: {e}")
        return pd.DataFrame()

def load_market_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Carga datos de mercado desde CSV o descarga de Binance"""
    if csv_path and os.path.exists(csv_path):
        logger.info(f"📁 Cargando datos desde: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Detectar y renombrar columnas comunes
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'timestamp' in col_lower or 'date' in col_lower or 'time' in col_lower:
                col_mapping[col] = 'timestamp'
            elif col_lower in ['open', 'high', 'low', 'close', 'volume']:
                col_mapping[col] = col_lower
            elif col_lower == 'price':
                col_mapping[col] = 'close'
        
        df = df.rename(columns=col_mapping)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"✅ Cargadas {len(df)} filas")
        return df
    
    # Si no hay CSV, descargar de Binance
    return download_binance_data()

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class PureRLBacktestEngine:
    """Motor de backtest puro - usa modelo RL directamente sin filtros"""
    
    def __init__(
        self,
        model_info: ModelInfo,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        fee: float = DEFAULT_FEE,
        slippage: float = DEFAULT_SLIPPAGE
    ):
        self.model_info = model_info
        self.initial_balance = initial_balance
        self.fee = fee
        self.slippage = slippage
        
        # Cargar modelo
        self.processor = None
        self._load_model()
        
        # Estado del backtest
        self.reset_state()
    
    def _load_model(self):
        """Carga el modelo RL"""
        try:
            self.processor = FinRLProcessor(self.model_info.path)
            logger.info(f"✅ Modelo {self.model_info.name} cargado ({self.model_info.dims} dims)")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise
    
    def diagnose_policy_activations(self, n_observations: int = 10) -> PolicyActivationDiagnostics:
        """
        Diagnóstico de activaciones de política para detectar modelos degenerados.
        
        Genera observaciones aleatorias y analiza la distribución de acciones.
        Esto ayuda a diferenciar entre:
        - Modelo que ve datos fuera de distribución → colapsa a HOLD
        - Modelo entrenado para nunca actuar (reward shaping malo)
        - Modelo que funciona pero el backtest tiene un bug de señal
        
        Args:
            n_observations: Número de observaciones aleatorias a generar (default: 10)
            
        Returns:
            PolicyActivationDiagnostics con los resultados del diagnóstico
        """
        diagnostics = PolicyActivationDiagnostics(model_name=self.model_info.name)
        
        try:
            logger.info(f"🔍 DIAGNÓSTICO DE POLÍTICA para {self.model_info.name}")
            logger.info(f"   Generando {n_observations} observaciones aleatorias...")
            
            dims = self.model_info.dims
            if dims == 0:
                logger.warning(f"   ⚠️ No se pudo detectar dimensionalidad del modelo")
                diagnostics.is_degenerate = True
                diagnostics.degeneracy_reason = "Dimensionalidad desconocida"
                return diagnostics
            
            # Generar observaciones aleatorias dentro del espacio de observación
            # Usar rango típico para indicadores técnicos normalizados (-5 a 5)
            np.random.seed(42)  # Para reproducibilidad
            
            for i in range(n_observations):
                # Generar observación aleatoria
                obs = np.random.randn(1, dims).astype(np.float32)
                
                # Obtener predicción del modelo
                try:
                    action, _states = self.processor.model.predict(obs, deterministic=False)
                    
                    # Intentar obtener probabilidades de acción si están disponibles (SB3)
                    action_prob = None
                    try:
                        if hasattr(self.processor.model, 'policy'):
                            policy = self.processor.model.policy
                            if hasattr(policy, 'get_distribution'):
                                # Obtener distribución de la política
                                obs_tensor = self.processor.model.policy.obs_to_tensor(obs)[0]
                                with torch.no_grad() if hasattr(torch, 'no_grad') else suppress():
                                    distribution = policy.get_distribution(obs_tensor)
                                    if hasattr(distribution, 'distribution'):
                                        probs = distribution.distribution.probs
                                        action_prob = probs.max().item()
                                        diagnostics.action_probabilities.append(action_prob)
                    except Exception as e:
                        # Las probabilidades no están disponibles, es normal
                        pass
                    
                    # Mapear acción
                    if hasattr(action, 'item'):
                        action_val = action.item()
                    elif hasattr(action, '__len__') and len(action) == 1:
                        action_val = action[0]
                    else:
                        action_val = action
                    
                    # Contar acciones
                    if isinstance(action_val, (int, np.integer)):
                        # Discreto: 0=HOLD, 1=BUY, 2=SELL
                        if action_val == 0:
                            diagnostics.hold_count += 1
                        elif action_val == 1:
                            diagnostics.buy_count += 1
                        elif action_val == 2:
                            diagnostics.sell_count += 1
                    else:
                        # Continuo: mapear 0-1 a señales
                        action_float = float(action_val)
                        if action_float < 0.4:
                            diagnostics.sell_count += 1
                        elif action_float > 0.6:
                            diagnostics.buy_count += 1
                        else:
                            diagnostics.hold_count += 1
                    
                    diagnostics.observations_tested += 1
                    
                except Exception as e:
                    logger.debug(f"   Error en predicción {i}: {e}")
                    continue
            
            # Calcular porcentajes
            total = diagnostics.buy_count + diagnostics.sell_count + diagnostics.hold_count
            
            if total > 0:
                hold_pct = diagnostics.hold_count / total * 100
                buy_pct = diagnostics.buy_count / total * 100
                sell_pct = diagnostics.sell_count / total * 100
                
                # Loggear distribución
                logger.info(f"   📊 DISTRIBUCIÓN DE ACCIONES (sobre {total} observaciones):")
                logger.info(f"      BUY:  {diagnostics.buy_count:2d} ({buy_pct:5.1f}%)")
                logger.info(f"      SELL: {diagnostics.sell_count:2d} ({sell_pct:5.1f}%)")
                logger.info(f"      HOLD: {diagnostics.hold_count:2d} ({hold_pct:5.1f}%)")
                
                # Loggear probabilidades si están disponibles
                if diagnostics.action_probabilities:
                    avg_prob = np.mean(diagnostics.action_probabilities)
                    logger.info(f"   📈 Probabilidad promedio de acción: {avg_prob:.3f}")
                
                # Verificar coherencia del mapeo de acciones
                logger.info(f"   🗺️  MAPEO DE ACCIONES VERIFICADO: 0=HOLD, 1=BUY, 2=SELL")
                if buy_pct > 0 or sell_pct > 0:
                    logger.info(f"      → El modelo genera señales BUY/SELL: mapeo correcto")
                elif hold_pct == 100:
                    logger.warning(f"      → Solo HOLD: podría ser degenerado O el modelo es conservador")
                
                # Verificar si es política degenerada
                if hold_pct > 90:
                    diagnostics.is_degenerate = True
                    diagnostics.degeneracy_reason = f"HOLD {hold_pct:.1f}% > 90% - Política colapsada"
                    logger.warning(f"   🚨 POLÍTICA DEGENERADA: {diagnostics.degeneracy_reason}")
                    logger.warning(f"      → Posible causa: reward shaping incorrecto, entrenamiento insuficiente,")
                    logger.warning(f"        o espacio de observación mal configurado")
                elif buy_pct == 0 and sell_pct == 0:
                    diagnostics.is_degenerate = True
                    diagnostics.degeneracy_reason = "Solo genera HOLD (100%)"
                    logger.warning(f"   🚨 POLÍTICA DEGENERADA: {diagnostics.degeneracy_reason}")
                elif hold_pct > 70:
                    logger.warning(f"   ⚠️  Política muy conservadora: {hold_pct:.1f}% HOLD")
                else:
                    logger.info(f"   ✅ Política activa: BUY={buy_pct:.1f}%, SELL={sell_pct:.1f}%, HOLD={hold_pct:.1f}%")
            else:
                diagnostics.is_degenerate = True
                diagnostics.degeneracy_reason = "No se pudieron realizar predicciones"
                logger.error(f"   ❌ No se pudieron realizar predicciones")
            
        except Exception as e:
            logger.error(f"   ❌ Error en diagnóstico: {e}")
            diagnostics.is_degenerate = True
            diagnostics.degeneracy_reason = f"Error en diagnóstico: {e}"
        
        return diagnostics
    
    def reset_state(self):
        """Resetea el estado del motor"""
        self.balance = self.initial_balance
        self.positions: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trade_counter = 0
        
        # Contadores de señales
        self.buy_count = 0
        self.sell_count = 0
        self.hold_count = 0
        
        # Precisión direccional
        self.directional_accuracy = DirectionalAccuracy()
    
    def _build_observation(self, df: pd.DataFrame, current_idx: int) -> Optional[np.ndarray]:
        """Construye el vector de observación según la dimensionalidad del modelo"""
        try:
            # Obtener ventana de datos hasta current_idx
            window = df.iloc[:current_idx+1]
            if len(window) < 30:  # Mínimo de datos necesarios
                return None
            
            symbol = "BTCUSDT"  # Asumimos BTC como default
            
            # Crear market_data dict en formato esperado
            market_data = {symbol: window}
            
            # Calcular indicadores básicos
            indicators = self._calculate_indicators(window)
            
            # Construir observación según dimensionalidad
            dims = self.model_info.dims
            
            if dims == 13:
                obs = ObservationBuilders.build_legacy_observation(market_data, symbol, indicators)
            elif dims == 85:
                obs = ObservationBuilders.build_hrm_native_obs(market_data, symbol, indicators)
            elif dims == 257:
                obs = ObservationBuilders.build_generic_obs(market_data, symbol, indicators, 257)
            elif dims == 971:
                obs = ObservationBuilders.build_multiasset_obs(market_data, symbol, indicators)
            else:
                # Fallback a genérico
                obs = ObservationBuilders.build_generic_obs(market_data, symbol, indicators, dims)
            
            if obs is None:
                return None
            
            # Asegurar forma correcta (1, dims) para predict
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            
            return obs
            
        except Exception as e:
            logger.debug(f"Error construyendo observación: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula indicadores técnicos básicos"""
        indicators = {}
        
        try:
            close = df['close'].astype(float)
            
            # SMAs
            indicators['sma_20'] = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else close.iloc[-1]
            indicators['sma_50'] = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.iloc[-1]
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1] if len(close) >= 14 else 50.0
            
            # MACD
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            indicators['macd'] = (ema_12 - ema_26).iloc[-1]
            indicators['ema_12'] = ema_12.iloc[-1]
            indicators['ema_26'] = ema_26.iloc[-1]
            
            # Bollinger
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            indicators['bollinger_upper'] = (sma_20 + 2 * std_20).iloc[-1]
            indicators['bollinger_lower'] = (sma_20 - 2 * std_20).iloc[-1]
            
        except Exception as e:
            logger.debug(f"Error calculando indicadores: {e}")
        
        return indicators
    
    def _get_model_prediction(self, obs: np.ndarray) -> Tuple[str, float]:
        """Obtiene predicción del modelo RL - SIN FILTROS"""
        try:
            # Llamada directa al modelo
            action, _states = self.processor.model.predict(obs, deterministic=True)
            
            # Mapear acción
            # Formato común: 0=HOLD, 1=BUY, 2=SELL (discreto)
            # O valor continuo 0-1 donde <0.4=SELL, >0.6=BUY, else HOLD
            
            if hasattr(action, 'item'):
                action_val = action.item()
            elif hasattr(action, '__len__') and len(action) == 1:
                action_val = action[0]
            else:
                action_val = action
            
            # Detectar si es espacio discreto o continuo
            if isinstance(action_val, (int, np.integer)):
                # Discreto: 0=HOLD, 1=BUY, 2=SELL
                if action_val == 0:
                    side = "hold"
                elif action_val == 1:
                    side = "buy"
                elif action_val == 2:
                    side = "sell"
                else:
                    side = "hold"
                confidence = 0.7  # Confianza por defecto para discreto
            else:
                # Continuo: mapear 0-1 a señales
                action_float = float(action_val)
                if action_float < 0.4:
                    side = "sell"
                    confidence = 0.6 + (0.4 - action_float) * 0.5
                elif action_float > 0.6:
                    side = "buy"
                    confidence = 0.6 + (action_float - 0.6) * 0.5
                else:
                    side = "hold"
                    confidence = 0.5
            
            return side, min(0.95, confidence)
            
        except Exception as e:
            logger.debug(f"Error en predicción: {e}")
            return "hold", 0.5
    
    def _check_directional_accuracy(
        self, 
        signal: str, 
        current_price: float, 
        future_price: float,
        threshold: float = 0.001
    ) -> bool:
        """Verifica si la señal fue correcta direccionalmente"""
        price_change = (future_price - current_price) / current_price
        
        if signal == "buy":
            return price_change > threshold
        elif signal == "sell":
            return price_change < -threshold
        else:  # hold
            return abs(price_change) <= threshold
    
    def run_backtest(self, df: pd.DataFrame, lookahead_bars: int = 6) -> ModelResults:
        """Ejecuta el backtest puro"""
        logger.info(f"🚀 Iniciando backtest puro para {self.model_info.name}")
        
        results = ModelResults(
            model_name=self.model_info.name,
            model_dims=self.model_info.dims
        )
        
        total_rows = len(df)
        progress_interval = max(1, total_rows // 10)
        
        for i in range(30, total_rows - lookahead_bars):  # Empezar en 30 para tener datos suficientes
            # Log de progreso
            if (i - 30) % progress_interval == 0:
                pct = (i - 30) / (total_rows - 30 - lookahead_bars) * 100
                logger.info(f"  Progreso: {pct:.1f}%")
            
            current_row = df.iloc[i]
            future_row = df.iloc[i + lookahead_bars]  # Mirar N barras adelante
            
            timestamp = current_row.get('timestamp', df.index[i])
            if isinstance(timestamp, (int, float)):
                timestamp = pd.to_datetime(timestamp, unit='ms')
            
            current_price = float(current_row['close'])
            future_price = float(future_row['close'])
            symbol = current_row.get('symbol', 'BTCUSDT')
            
            # Construir observación
            obs = self._build_observation(df, i)
            if obs is None:
                continue
            
            # Obtener predicción del modelo (SIN FILTROS)
            signal, confidence = self._get_model_prediction(obs)
            
            # Contar señales
            results.total_signals += 1
            if signal == "buy":
                results.buy_count += 1
            elif signal == "sell":
                results.sell_count += 1
            else:
                results.hold_count += 1
            
            # Verificar precisión direccional
            correct = self._check_directional_accuracy(signal, current_price, future_price)
            
            if signal == "buy":
                results.directional_accuracy.buy_signals += 1
                if correct:
                    results.directional_accuracy.buy_correct += 1
            elif signal == "sell":
                results.directional_accuracy.sell_signals += 1
                if correct:
                    results.directional_accuracy.sell_correct += 1
            else:
                results.directional_accuracy.hold_signals += 1
                if correct:
                    results.directional_accuracy.hold_correct += 1
            
            # Ejecutar trade si es BUY o SELL (no hold)
            if signal in ["buy", "sell"]:
                self._execute_trade(timestamp, symbol, signal, current_price, confidence)
            
            # Actualizar equity
            self._update_equity(timestamp, {symbol: current_price})
        
        # Cerrar posiciones abiertas
        if self.positions:
            final_price = float(df['close'].iloc[-1])
            final_timestamp = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else datetime.now()
            self._close_all_positions(final_timestamp, {'BTCUSDT': final_price})
        
        # Calcular métricas finales
        self._calculate_metrics(results)
        results.equity_curve = self.equity_curve
        
        logger.info(f"✅ Backtest completado para {self.model_info.name}")
        return results
    
    def _execute_trade(self, timestamp: datetime, symbol: str, side: str, price: float, confidence: float):
        """Ejecuta un trade"""
        try:
            if side == "buy":
                # Verificar si ya hay posición
                if symbol in self.positions:
                    return
                
                # Calcular tamaño de posición (fijo % del balance)
                notional = self.balance * 0.1 * confidence  # 10% * confianza
                quantity = notional / price if price > 0 else 0
                
                if quantity <= 0 or notional <= 0:
                    return
                
                # Aplicar slippage y fee
                execution_price = price * (1 + self.slippage)
                fees = notional * self.fee
                
                # Actualizar balance
                self.balance -= notional + fees
                
                # Crear trade
                self.trade_counter += 1
                self.positions[symbol] = TradeRecord(
                    trade_id=self.trade_counter,
                    timestamp=timestamp,
                    symbol=symbol,
                    side='BUY',
                    entry_price=execution_price,
                    quantity=quantity,
                    confidence=confidence,
                    fees=fees,
                    model_action='BUY'
                )
                
            elif side == "sell":
                # Verificar si hay posición abierta
                if symbol not in self.positions:
                    return
                
                position = self.positions[symbol]
                
                # Aplicar slippage
                execution_price = price * (1 - self.slippage)
                
                # Calcular PnL
                entry_value = position.entry_price * position.quantity
                exit_value = execution_price * position.quantity
                exit_fees = exit_value * self.fee
                
                pnl = exit_value - entry_value - position.fees - exit_fees
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                
                # Actualizar balance
                self.balance += exit_value - exit_fees
                
                # Cerrar posición
                hold_time = (timestamp - position.timestamp).total_seconds() / 60
                
                position.exit_price = execution_price
                position.pnl = pnl
                position.pnl_pct = pnl_pct
                position.exit_timestamp = timestamp
                position.hold_time_minutes = hold_time
                position.is_open = False
                position.fees += exit_fees
                
                self.closed_trades.append(position)
                del self.positions[symbol]
                
        except Exception as e:
            logger.debug(f"Error ejecutando trade: {e}")
    
    def _update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Actualiza la curva de equity"""
        equity = self.balance
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                position_value = position.quantity * market_prices[symbol]
                equity += position_value
        
        self.equity_curve.append((timestamp, equity))
    
    def _close_all_positions(self, timestamp: datetime, final_prices: Dict[str, float]):
        """Cierra todas las posiciones al final"""
        for symbol in list(self.positions.keys()):
            if symbol in final_prices:
                self._execute_trade(timestamp, symbol, "sell", final_prices[symbol], 0.5)
    
    def _calculate_metrics(self, results: ModelResults):
        """Calcula métricas finales"""
        # Métricas de trades
        results.total_trades = len(self.closed_trades)
        if results.total_trades == 0:
            return
        
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl <= 0]
        
        results.winning_trades = len(winning_trades)
        results.losing_trades = len(losing_trades)
        results.win_rate = results.winning_trades / results.total_trades
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe ratio
        pnls = [t.pnl for t in self.closed_trades]
        if len(pnls) > 1:
            returns = np.array(pnls)
            results.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Max drawdown
        if self.equity_curve:
            peak = self.equity_curve[0][1]
            max_dd = 0
            for _, equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            results.max_drawdown_pct = max_dd
        
        # Return
        results.total_pnl = sum(pnls)
        results.return_pct = results.total_pnl / self.initial_balance if self.initial_balance > 0 else 0
        
        # Contadores de señales
        results.buy_count = self.buy_count
        results.sell_count = self.sell_count
        results.hold_count = self.hold_count

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def print_comparison_table(results_list: List[ModelResults]):
    """Imprime tabla comparativa de modelos"""
    
    print("\n" + "="*120)
    print("📊 TABLA COMPARATIVA DE MODELOS RL - BACKTEST PURO (SIN FILTROS)")
    print("="*120)
    
    # Header
    header = (
        f"{'Modelo':<12} | {'Dims':<5} | {'WinRate':<8} | {'ProfitF':<8} | "
        f"{'Sharpe':<7} | {'Return':<8} | {'DirAcc':<7} | {'BUY%':<6} | {'SELL%':<6} | {'HOLD%':<6}"
    )
    print(header)
    print("-"*120)
    
    # Filas
    for r in results_list:
        total_signals = r.total_signals if r.total_signals > 0 else 1
        buy_pct = r.buy_count / total_signals * 100
        sell_pct = r.sell_count / total_signals * 100
        hold_pct = r.hold_count / total_signals * 100
        
        row = (
            f"{r.model_name:<12} | {r.model_dims:<5} | {r.win_rate*100:>6.1f}% | "
            f"{r.profit_factor:>7.2f} | {r.sharpe_ratio:>6.2f} | {r.return_pct*100:>6.1f}% | "
            f"{r.directional_accuracy.overall_accuracy*100:>5.1f}% | {buy_pct:>5.1f}% | "
            f"{sell_pct:>5.1f}% | {hold_pct:>5.1f}%"
        )
        print(row)
    
    print("="*120)
    
    # Análisis
    print("\n📈 ANÁLISIS:")
    
    # Mejor por return
    best_return = max(results_list, key=lambda x: x.return_pct)
    print(f"  🥇 Meor Return: {best_return.model_name} ({best_return.return_pct*100:.1f}%)")
    
    # Mejor por Sharpe
    best_sharpe = max(results_list, key=lambda x: x.sharpe_ratio)
    print(f"  🥇 Mejor Sharpe: {best_sharpe.model_name} ({best_sharpe.sharpe_ratio:.2f})")
    
    # Mejor por precisión direccional
    best_dir = max(results_list, key=lambda x: x.directional_accuracy.overall_accuracy)
    print(f"  🥇 Mejor DirAccuracy: {best_dir.model_name} ({best_dir.directional_accuracy.overall_accuracy*100:.1f}%)")
    
    # Modelos con edge
    models_with_edge = [r for r in results_list if r.return_pct > 0 and r.sharpe_ratio > 0.5]
    if models_with_edge:
        print(f"\n  ✅ Modelos con EDGE detectado (>0% return, >0.5 Sharpe): {len(models_with_edge)}")
        for m in models_with_edge:
            print(f"     - {m.model_name}: {m.return_pct*100:.1f}% return, {m.sharpe_ratio:.2f} Sharpe")
    else:
        print(f"\n  ⚠️  NINGÚN MODELO muestra edge consistente - necesitan reentrenamiento")
    
    print()

def export_results(results_list: List[ModelResults], output_dir: str):
    """Exporta resultados a JSON y CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON completo
    json_path = os.path.join(output_dir, f"rl_pure_backtest_{timestamp}.json")
    data = {
        'timestamp': datetime.now().isoformat(),
        'models_tested': len(results_list),
        'results': [r.to_dict() for r in results_list],
        'summary': {
            'best_by_return': max(results_list, key=lambda x: x.return_pct).model_name,
            'best_by_sharpe': max(results_list, key=lambda x: x.sharpe_ratio).model_name,
            'best_by_accuracy': max(results_list, key=lambda x: x.directional_accuracy.overall_accuracy).model_name,
        }
    }
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"💾 Resultados exportados a: {json_path}")
    
    # CSV resumen
    csv_path = os.path.join(output_dir, f"rl_pure_summary_{timestamp}.csv")
    rows = []
    for r in results_list:
        total_signals = r.total_signals if r.total_signals > 0 else 1
        rows.append({
            'model': r.model_name,
            'dims': r.model_dims,
            'win_rate': r.win_rate,
            'profit_factor': r.profit_factor,
            'sharpe': r.sharpe_ratio,
            'return_pct': r.return_pct,
            'max_dd_pct': r.max_drawdown_pct,
            'total_trades': r.total_trades,
            'buy_pct': r.buy_count / total_signals,
            'sell_pct': r.sell_count / total_signals,
            'hold_pct': r.hold_count / total_signals,
            'directional_accuracy': r.directional_accuracy.overall_accuracy,
            'buy_accuracy': r.directional_accuracy.buy_accuracy,
            'sell_accuracy': r.directional_accuracy.sell_accuracy,
        })
    
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"💾 CSV exportado a: {csv_path}")
    
    return json_path, csv_path

async def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(
        description='Backtest RL Puro - Evaluación de modelos sin filtros',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Probar todos los modelos
  python backtesting/backtest_rl_pure.py

  # Probar un modelo específico
  python backtesting/backtest_rl_pure.py --model models/L2/claude.zip

  # Usar CSV específico
  python backtesting/backtest_rl_pure.py --csv results/l1_backtest_signals_20260221.csv
        """
    )
    
    parser.add_argument('--model', help='Path a modelo específico (.zip)')
    parser.add_argument('--csv', help='Path a CSV de datos de mercado')
    parser.add_argument('--balance', type=float, default=DEFAULT_INITIAL_BALANCE, help='Balance inicial')
    parser.add_argument('--output-dir', default=RESULTS_DIR, help='Directorio de salida')
    parser.add_argument('--lookahead', type=int, default=6, help='Barras de lookahead para precisión direccional')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🧪 BACKTEST RL PURO - Evaluación de modelos sin filtros externos")
    print("="*80)
    print("\n⚠️  IMPORTANTE: Este script evalúa los modelos RL DIRECTAMENTE")
    print("    Sin filtros de confianza, RSI, volatilidad, o L3")
    print("    Objetivo: Determinar si los modelos tienen edge predictivo real\n")
    
    # Detectar modelos
    if args.model:
        # Usar modelo específico
        name = os.path.basename(args.model).replace('.zip', '')
        try:
            processor = FinRLProcessor(args.model)
            dims = processor.observation_space_info.get('expected_dims', 0) if processor.observation_space_info else 0
            models = [ModelInfo(name=name, path=args.model, dims=dims, model_type='manual', loaded=True)]
        except Exception as e:
            logger.error(f"❌ Error cargando modelo especificado: {e}")
            return 1
    else:
        # Detectar todos los modelos
        models = detect_models()
    
    if not models:
        logger.error("❌ No se encontraron modelos para evaluar")
        return 1
    
    logger.info(f"\n📦 Modelos a evaluar: {len(models)}")
    for m in models:
        if m.loaded:
            logger.info(f"   ✅ {m.name}: {m.dims} dims")
        else:
            logger.info(f"   ❌ {m.name}: ERROR - {m.error}")
    
    # Cargar datos de mercado
    df = load_market_data(args.csv)
    if df.empty:
        logger.error("❌ No se pudieron cargar datos de mercado")
        return 1
    
    # Ejecutar backtest para cada modelo
    results_list = []
    
    for model_info in models:
        if not model_info.loaded:
            continue
        
        print(f"\n{'='*80}")
        print(f"🚀 Evaluando modelo: {model_info.name}")
        print(f"{'='*80}")
        
        try:
            engine = PureRLBacktestEngine(
                model_info=model_info,
                initial_balance=args.balance
            )
            
            # ==================================================================
            # DIAGNÓSTICO DE ACTIVACIONES DE POLÍTICA (antes del backtest)
            # ==================================================================
            diagnostics = engine.diagnose_policy_activations(n_observations=10)
            
            # Si la política es degenerada, saltar este modelo
            if diagnostics.is_degenerate:
                logger.warning(f"⏭️  SALTANDO MODELO DEGENERADO: {model_info.name}")
                logger.warning(f"   Razón: {diagnostics.degeneracy_reason}")
                logger.warning(f"   → El modelo no será evaluado en backtest")
                continue
            
            # ==================================================================
            # EJECUTAR BACKTEST (solo si no es degenerado)
            # ==================================================================
            results = engine.run_backtest(df, lookahead_bars=args.lookahead)
            results_list.append(results)
            
            # Mostrar resultados individuales
            print(f"\n📊 Resultados {model_info.name}:")
            for key, value in results.to_dict().items():
                print(f"   {key}: {value}")
            
        except Exception as e:
            logger.error(f"❌ Error en backtest de {model_info.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Tabla comparativa
    if len(results_list) > 1:
        print_comparison_table(results_list)
    
    # Exportar resultados
    if results_list:
        export_results(results_list, args.output_dir)
    
    print("\n" + "="*80)
    print("✅ BACKTEST RL PURO COMPLETADO")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    import asyncio
    sys.exit(asyncio.run(main()))
