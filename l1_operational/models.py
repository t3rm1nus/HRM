# l1_operational/models.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import time
import pandas as pd
import numpy as np
from datetime import datetime

from core.logging import logger
from core.technical_indicators import calculate_technical_indicators

# ============================================================================
# SIGNAL GENERATION FUNCTIONS
# ============================================================================

def generate_momentum_signals(indicators):
    """Generate momentum signals based on MACD and momentum indicators"""
    signals = []

    # Positive MACD crossover + positive momentum
    if indicators['macd_diff'] > 0 and indicators['momentum_5p'] > 0:
        signals.append({
            "action": "buy",
            "confidence": min(0.6 + (indicators['macd_diff'] / 10), 0.85),
            "source": "l1_momentum"
        })

    # Negative MACD crossover + negative momentum
    elif indicators['macd_diff'] < -2 and indicators['momentum_5p'] < -0.01:
        signals.append({
            "action": "sell",
            "confidence": 0.65,
            "source": "l1_momentum"
        })

    else:
        signals.append({"action": "hold", "confidence": 0.5})

    return signals

def generate_technical_signals(indicators):
    """Generate technical signals based on RSI"""
    signals = []

    # Oversold - potential BUY
    if indicators['rsi'] < 35:
        signals.append({
            "action": "buy",
            "confidence": 0.7,
            "source": "l1_technical"
        })

    # Overbought - potential SELL
    elif indicators['rsi'] > 70:
        signals.append({
            "action": "sell",
            "confidence": 0.7,
            "source": "l1_technical"
        })

    return signals  # Can return empty list if neutral

def generate_volume_signals(indicators, market_data):
    """Generate volume signals based on volume spikes"""
    signals = []

    current_volume = market_data['volume'].iloc[-1]
    avg_volume = market_data['volume'].iloc[-20:].mean()

    # Volume spike + price increase
    if current_volume > avg_volume * 1.5 and indicators['momentum_5p'] > 0:
        signals.append({
            "action": "buy",
            "confidence": 0.65,
            "source": "l1_volume"
        })

    # Volume spike + price decrease
    elif current_volume > avg_volume * 1.5 and indicators['momentum_5p'] < -0.01:
        signals.append({
            "action": "sell",
            "confidence": 0.65,
            "source": "l1_volume"
        })

    else:
        signals.append({"action": "hold", "confidence": 0.7})

    return signals

# ============================================================================
# NUEVAS CLASES AGREGADAS (para compatibilidad con imports)
# ============================================================================

class L1SignalType(Enum):
    """Tipos de señales L1"""
    MOMENTUM_SHORT = "momentum_short"
    MOMENTUM_MEDIUM = "momentum_medium"
    TECHNICAL_RSI = "technical_rsi"
    TECHNICAL_MACD = "technical_macd"
    TECHNICAL_BOLLINGER = "technical_bollinger"
    VOLUME_FLOW = "volume_flow"
    VOLUME_LIQUIDITY = "volume_liquidity"

@dataclass
class L1Signal:
    """Señal generada por modelo L1"""
    symbol: str
    signal_type: L1SignalType
    direction: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    features: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseL1Model:
    """Base class for all L1 models"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[L1Signal]:
        """Generate signals for all symbols in market_data"""
        raise NotImplementedError

    def _validate_market_data(self, df: pd.DataFrame, min_periods: int = 20) -> bool:
        """Validate that market data has sufficient history"""
        if df is None or df.empty:
            return False
        if len(df) < min_periods:
            logger.warning(f"{self.name}: Insufficient data points: {len(df)} < {min_periods}")
            return False
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"{self.name}: Missing required columns: {required_cols}")
            return False
        return True

class MomentumModel(BaseL1Model):
    """Modelo de momentum técnico - tendencias de corto/medio plazo"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.short_period = self.config.get('short_period', 5)
        self.medium_period = self.config.get('medium_period', 20)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.5)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[L1Signal]:
        signals = []

        for symbol, df in market_data.items():
            if not self._validate_market_data(df, self.medium_period):
                continue

            try:
                # Calcular indicadores técnicos
                indicators = calculate_technical_indicators({symbol: df})
                if symbol not in indicators or indicators[symbol].empty:
                    continue

                df_ind = indicators[symbol]

                # Señales de momentum corto plazo
                short_signal = self._calculate_short_momentum(df_ind)
                if short_signal:
                    signals.append(short_signal)

                # Señales de momentum medio plazo
                medium_signal = self._calculate_medium_momentum(df_ind)
                if medium_signal:
                    signals.append(medium_signal)

            except Exception as e:
                logger.error(f"Error generating momentum signals for {symbol}: {e}")

        return signals

    def _calculate_short_momentum(self, df: pd.DataFrame) -> Optional[L1Signal]:
        """Calcular señal de momentum corto plazo"""
        try:
            if len(df) < self.short_period + 1:
                return None

            # Momentum basado en retornos recientes
            recent_returns = df['close'].pct_change(self.short_period).iloc[-1]
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-self.short_period-1]

            # Calcular momentum score
            momentum_score = (current_price - prev_price) / prev_price

            # Determinar dirección y confianza
            if abs(momentum_score) < self.momentum_threshold:
                direction = 'hold'
                confidence = 0.5
            else:
                direction = 'buy' if momentum_score > 0 else 'sell'
                confidence = min(0.9, 0.5 + abs(momentum_score) * 2)

            strength = min(1.0, abs(momentum_score) * 3)

            features = {
                'momentum_score': momentum_score,
                'recent_returns': recent_returns,
                'current_price': current_price,
                'prev_price': prev_price,
                'period': self.short_period
            }

            return L1Signal(
                symbol=df.index.name or 'UNKNOWN',
                signal_type=L1SignalType.MOMENTUM_SHORT,
                direction=direction,
                strength=strength,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                metadata={'model': 'MomentumModel', 'period_type': 'short'}
            )

        except Exception as e:
            logger.error(f"Error calculating short momentum: {e}")
            return None

    def _calculate_medium_momentum(self, df: pd.DataFrame) -> Optional[L1Signal]:
        """Calcular señal de momentum medio plazo"""
        try:
            if len(df) < self.medium_period + 1:
                return None

            # Momentum basado en SMA crossover
            sma_short = df['close'].rolling(window=self.short_period).mean()
            sma_medium = df['close'].rolling(window=self.medium_period).mean()

            current_short = sma_short.iloc[-1]
            current_medium = sma_medium.iloc[-1]
            prev_short = sma_short.iloc[-2]
            prev_medium = sma_medium.iloc[-2]

            # Detectar crossover
            prev_diff = prev_short - prev_medium
            current_diff = current_short - current_medium

            momentum_score = current_diff / current_medium  # Normalizado

            # Señales basadas en crossover
            if prev_diff <= 0 and current_diff > 0:
                direction = 'buy'
                confidence = 0.7
            elif prev_diff >= 0 and current_diff < 0:
                direction = 'sell'
                confidence = 0.7
            else:
                direction = 'hold'
                confidence = 0.5

            strength = min(1.0, abs(momentum_score) * 2)

            features = {
                'momentum_score': momentum_score,
                'sma_short': current_short,
                'sma_medium': current_medium,
                'crossover_detected': prev_diff * current_diff < 0,
                'period_short': self.short_period,
                'period_medium': self.medium_period
            }

            return L1Signal(
                symbol=df.index.name or 'UNKNOWN',
                signal_type=L1SignalType.MOMENTUM_MEDIUM,
                direction=direction,
                strength=strength,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                metadata={'model': 'MomentumModel', 'period_type': 'medium'}
            )

        except Exception as e:
            logger.error(f"Error calculating medium momentum: {e}")
            return None

class TechnicalIndicatorsModel(BaseL1Model):
    """Modelo de indicadores técnicos - RSI, MACD, Bandas de Bollinger"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.macd_threshold = self.config.get('macd_threshold', 0.1)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[L1Signal]:
        signals = []

        for symbol, df in market_data.items():
            if not self._validate_market_data(df, 50):  # Necesitamos suficientes datos para MACD
                continue

            try:
                # Calcular indicadores técnicos
                indicators = calculate_technical_indicators({symbol: df})
                if symbol not in indicators or indicators[symbol].empty:
                    continue

                df_ind = indicators[symbol]

                # Señales RSI
                rsi_signals = self._calculate_rsi_signals(df_ind)
                signals.extend(rsi_signals)

                # Señales MACD
                macd_signals = self._calculate_macd_signals(df_ind)
                signals.extend(macd_signals)

                # Señales Bollinger Bands
                bollinger_signals = self._calculate_bollinger_signals(df_ind)
                signals.extend(bollinger_signals)

            except Exception as e:
                logger.error(f"Error generating technical indicator signals for {symbol}: {e}")

        return signals

    def _calculate_rsi_signals(self, df: pd.DataFrame) -> List[L1Signal]:
        """Calcular señales basadas en RSI"""
        signals = []

        try:
            rsi = df['rsi'].iloc[-1]
            rsi_prev = df['rsi'].iloc[-2] if len(df) > 1 else 50

            # Señales de sobrecompra/sobreventa
            if rsi <= self.rsi_oversold and rsi_prev > self.rsi_oversold:
                direction = 'buy'
                confidence = 0.75
                reason = 'rsi_oversold_crossover'
            elif rsi >= self.rsi_overbought and rsi_prev < self.rsi_overbought:
                direction = 'sell'
                confidence = 0.75
                reason = 'rsi_overbought_crossover'
            else:
                return signals  # No signal

            strength = min(1.0, abs(50 - rsi) / 30)  # Más fuerte cuando más extremo

            features = {
                'rsi': rsi,
                'rsi_prev': rsi_prev,
                'rsi_overbought': self.rsi_overbought,
                'rsi_oversold': self.rsi_oversold,
                'reason': reason
            }

            signal = L1Signal(
                symbol=df.index.name or 'UNKNOWN',
                signal_type=L1SignalType.TECHNICAL_RSI,
                direction=direction,
                strength=strength,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                metadata={'model': 'TechnicalIndicatorsModel', 'indicator': 'rsi', 'reason': reason}
            )
            signals.append(signal)

        except Exception as e:
            logger.error(f"Error calculating RSI signals: {e}")

        return signals

    def _calculate_macd_signals(self, df: pd.DataFrame) -> List[L1Signal]:
        """Calcular señales basadas en MACD"""
        signals = []

        try:
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_prev = df['macd'].iloc[-2] if len(df) > 1 else 0
            macd_signal_prev = df['macd_signal'].iloc[-2] if len(df) > 1 else 0

            macd_diff = macd - macd_signal
            macd_diff_prev = macd_prev - macd_signal_prev

            # Señales de crossover MACD
            if macd_diff_prev <= 0 and macd_diff > 0 and abs(macd_diff) > self.macd_threshold:
                direction = 'buy'
                confidence = 0.7
                reason = 'macd_bullish_crossover'
            elif macd_diff_prev >= 0 and macd_diff < 0 and abs(macd_diff) > self.macd_threshold:
                direction = 'sell'
                confidence = 0.7
                reason = 'macd_bearish_crossover'
            else:
                return signals  # No signal

            strength = min(1.0, abs(macd_diff) * 10)

            features = {
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_diff': macd_diff,
                'macd_prev': macd_prev,
                'macd_signal_prev': macd_signal_prev,
                'macd_threshold': self.macd_threshold,
                'reason': reason
            }

            signal = L1Signal(
                symbol=df.index.name or 'UNKNOWN',
                signal_type=L1SignalType.TECHNICAL_MACD,
                direction=direction,
                strength=strength,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                metadata={'model': 'TechnicalIndicatorsModel', 'indicator': 'macd', 'reason': reason}
            )
            signals.append(signal)

        except Exception as e:
            logger.error(f"Error calculating MACD signals: {e}")

        return signals

    def _calculate_bollinger_signals(self, df: pd.DataFrame) -> List[L1Signal]:
        """Calcular señales basadas en Bandas de Bollinger"""
        signals = []

        try:
            close = df['close'].iloc[-1]
            upper = df['bollinger_upper'].iloc[-1]
            lower = df['bollinger_lower'].iloc[-1]
            middle = df['bollinger_middle'].iloc[-1]

            # Calcular posición relativa en las bandas
            if upper > lower:
                position = (close - lower) / (upper - lower)
            else:
                position = 0.5

            # Señales de rebote en bandas
            if position <= 0.1:  # Precio cerca de banda inferior
                direction = 'buy'
                confidence = 0.65
                reason = 'bollinger_lower_rebound'
            elif position >= 0.9:  # Precio cerca de banda superior
                direction = 'sell'
                confidence = 0.65
                reason = 'bollinger_upper_rebound'
            else:
                return signals  # No signal

            # Calcular volatilidad como strength
            std = df['bollinger_std'].iloc[-1]
            strength = min(1.0, std / close * 10)  # Normalizar volatilidad

            features = {
                'close': close,
                'bollinger_upper': upper,
                'bollinger_lower': lower,
                'bollinger_middle': middle,
                'position_in_bands': position,
                'bollinger_std': std,
                'reason': reason
            }

            signal = L1Signal(
                symbol=df.index.name or 'UNKNOWN',
                signal_type=L1SignalType.TECHNICAL_BOLLINGER,
                direction=direction,
                strength=strength,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                metadata={'model': 'TechnicalIndicatorsModel', 'indicator': 'bollinger', 'reason': reason}
            )
            signals.append(signal)

        except Exception as e:
            logger.error(f"Error calculating Bollinger signals: {e}")

        return signals

class VolumeSignalsModel(BaseL1Model):
    """Modelo de señales de volumen - flujos de capital y liquidez"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.volume_period = self.config.get('volume_period', 20)
        self.volume_threshold = self.config.get('volume_threshold', 1.5)
        self.liquidity_threshold = self.config.get('liquidity_threshold', 0.7)

    def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[L1Signal]:
        signals = []

        for symbol, df in market_data.items():
            if not self._validate_market_data(df, self.volume_period):
                continue

            try:
                # Calcular indicadores técnicos (para volumen)
                indicators = calculate_technical_indicators({symbol: df})
                if symbol not in indicators or indicators[symbol].empty:
                    continue

                df_ind = indicators[symbol]

                # Señales de flujo de volumen
                volume_signals = self._calculate_volume_flow_signals(df_ind)
                signals.extend(volume_signals)

                # Señales de liquidez
                liquidity_signals = self._calculate_liquidity_signals(df_ind)
                signals.extend(liquidity_signals)

            except Exception as e:
                logger.error(f"Error generating volume signals for {symbol}: {e}")

        return signals

    def _calculate_volume_flow_signals(self, df: pd.DataFrame) -> List[L1Signal]:
        """Calcular señales basadas en flujos de volumen"""
        signals = []

        try:
            # Usar el z-score de volumen calculado
            vol_zscore = df['vol_zscore'].iloc[-1]
            vol_zscore_prev = df['vol_zscore'].iloc[-2] if len(df) > 1 else 0

            current_volume = df['volume'].iloc[-1]
            avg_volume = df['vol_mean_20'].iloc[-1]

            # Señales de volumen extremo
            if vol_zscore >= self.volume_threshold and vol_zscore_prev < self.volume_threshold:
                # Volumen alto - confirmar tendencia del precio
                close = df['close'].iloc[-1]
                prev_close = df['close'].iloc[-2] if len(df) > 1 else close

                if close > prev_close:
                    direction = 'buy'
                    reason = 'high_volume_upmove'
                else:
                    direction = 'sell'
                    reason = 'high_volume_downmove'

                confidence = min(0.8, 0.5 + vol_zscore * 0.2)
                strength = min(1.0, vol_zscore / 3)

            elif vol_zscore <= -self.volume_threshold and vol_zscore_prev > -self.volume_threshold:
                # Volumen bajo - señal de debilidad
                direction = 'hold'
                confidence = 0.6
                strength = 0.3
                reason = 'low_volume_weakness'
            else:
                return signals  # No signal

            features = {
                'vol_zscore': vol_zscore,
                'vol_zscore_prev': vol_zscore_prev,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_threshold': self.volume_threshold,
                'reason': reason
            }

            signal = L1Signal(
                symbol=df.index.name or 'UNKNOWN',
                signal_type=L1SignalType.VOLUME_FLOW,
                direction=direction,
                strength=strength,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                metadata={'model': 'VolumeSignalsModel', 'signal_type': 'flow', 'reason': reason}
            )
            signals.append(signal)

        except Exception as e:
            logger.error(f"Error calculating volume flow signals: {e}")

        return signals

    def _calculate_liquidity_signals(self, df: pd.DataFrame) -> List[L1Signal]:
        """Calcular señales basadas en liquidez"""
        signals = []

        try:
            # Medir liquidez basada en volatilidad del spread implícito
            # Usamos volatilidad de precios como proxy de liquidez
            close_std = df['close'].rolling(window=self.volume_period).std().iloc[-1]
            close_mean = df['close'].rolling(window=self.volume_period).mean().iloc[-1]

            if close_mean == 0:
                return signals

            # Coeficiente de variación como medida de liquidez
            cv = close_std / close_mean

            # Volumen relativo
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['vol_mean_20'].iloc[-1]

            if avg_volume == 0:
                volume_ratio = 0
            else:
                volume_ratio = current_volume / avg_volume

            # Combinar métricas de liquidez
            liquidity_score = (1 / (1 + cv)) * min(1.0, volume_ratio / 2)

            # Señales basadas en liquidez
            if liquidity_score >= self.liquidity_threshold:
                direction = 'buy'  # Alta liquidez favorece entradas
                confidence = 0.6
                reason = 'high_liquidity'
            elif liquidity_score <= 0.3:
                direction = 'hold'  # Baja liquidez - evitar operaciones
                confidence = 0.7
                reason = 'low_liquidity_risk'
            else:
                return signals  # Liquidez normal - no signal

            strength = liquidity_score

            features = {
                'liquidity_score': liquidity_score,
                'price_volatility': cv,
                'volume_ratio': volume_ratio,
                'close_std': close_std,
                'close_mean': close_mean,
                'liquidity_threshold': self.liquidity_threshold,
                'reason': reason
            }

            signal = L1Signal(
                symbol=df.index.name or 'UNKNOWN',
                signal_type=L1SignalType.VOLUME_LIQUIDITY,
                direction=direction,
                strength=strength,
                confidence=confidence,
                features=features,
                timestamp=datetime.now(),
                metadata={'model': 'VolumeSignalsModel', 'signal_type': 'liquidity', 'reason': reason}
            )
            signals.append(signal)

        except Exception as e:
            logger.error(f"Error calculating liquidity signals: {e}")

        return signals

# ============================================================================
# L1 MODELS - Operational Signals Layer
# ============================================================================

class L1Model:
    """Main L1 Model that combines all sub-models"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {
            'momentum': MomentumModel(self.config.get('momentum', {})),
            'technical': TechnicalIndicatorsModel(self.config.get('technical', {})),
            'volume': VolumeSignalsModel(self.config.get('volume', {}))
        }

    def predict(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate L1 signals and return metrics"""
        all_signals = []

        for model_name, model in self.models.items():
            try:
                signals = model.generate_signals(market_data)
                all_signals.extend(signals)
                logger.info(f"L1 {model_name} model generated {len(signals)} signals")
            except Exception as e:
                logger.error(f"Error in L1 {model_name} model: {e}")

        # Calcular métricas agregadas
        metrics = self._calculate_metrics(all_signals)

        return {
            'signals': all_signals,
            'metrics': metrics,
            'timestamp': datetime.now(),
            'model_count': len(self.models)
        }

    def _calculate_metrics(self, signals: List[L1Signal]) -> Dict[str, Any]:
        """Calculate aggregate metrics from signals"""
        if not signals:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_confidence': 0.0,
                'avg_strength': 0.0,
                'signal_types': {}
            }

        buy_count = sum(1 for s in signals if s.direction == 'buy')
        sell_count = sum(1 for s in signals if s.direction == 'sell')
        hold_count = sum(1 for s in signals if s.direction == 'hold')

        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        avg_strength = sum(s.strength for s in signals) / len(signals)

        signal_types = {}
        for s in signals:
            st = s.signal_type.value
            signal_types[st] = signal_types.get(st, 0) + 1

        return {
            'total_signals': len(signals),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'avg_confidence': avg_confidence,
            'avg_strength': avg_strength,
            'signal_types': signal_types
        }

# ============================================================================
# NUEVAS CLASES AGREGADAS (para compatibilidad con imports)
# ============================================================================

class SignalType(Enum):
    """Tipos de señales de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

    # Compatibilidad con nuestro sistema actual
    buy = "buy"
    sell = "sell"

class SignalSource(Enum):
    """Fuentes de las señales"""
    L2_TACTIC = "L2_TACTIC"
    L3_STRATEGY = "L3_STRATEGY"
    MANUAL = "MANUAL"
    RISK_MANAGER = "RISK_MANAGER"

class OrderStatus(Enum):
    """Estados de órdenes"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class ExecutionStatus(Enum):
    """Estados de ejecución para reportes"""
    EXECUTED = "EXECUTED"
    REJECTED_SAFETY = "REJECTED_SAFETY"
    REJECTED_AI = "REJECTED_AI"
    EXECUTION_ERROR = "EXECUTION_ERROR"

# ============================================================================
# CLASES EXISTENTES (mantenidas tal como están)
# ============================================================================

@dataclass
class Signal:
    """Señal de trading recibida de L2/L3"""
    signal_id: str
    strategy_id: str
    timestamp: float
    symbol: str
    side: str  # 'buy' or 'sell'
    qty: Optional[float] = None  # Will be calculated if not provided
    order_type: str = "market"  # market, limit
    price: Optional[float] = None  # para limit orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.5
    technical_indicators: Optional[Dict[str, float]] = None
    strength: float = 0.5
    features: Optional[Dict[str, Any]] = None
    signal_type: str = "tactical"

    def __post_init__(self):
        """Initialize default values and validate required fields"""
        if self.technical_indicators is None:
            self.technical_indicators = {}
        if self.features is None:
            self.features = {}

        # Add any features to technical_indicators for backward compatibility
        if self.features and isinstance(self.features, dict):
            # Copy all numeric features and required indicators
            required_indicators = ['rsi', 'macd', 'macd_signal', 'sma_20', 'sma_50',
                                'bollinger_upper', 'bollinger_lower', 'vol_zscore',
                                'close', 'signal_strength']

            for k, v in self.features.items():
                if k in required_indicators or isinstance(v, (int, float)):
                    self.technical_indicators[k] = float(v)

        # Add strength to technical indicators if not present
        if 'signal_strength' not in self.technical_indicators:
            self.technical_indicators['signal_strength'] = float(self.strength)

    # Métodos de compatibilidad con los enums
    def get_signal_type(self) -> SignalType:
        """Convertir side a SignalType"""
        if self.side.lower() == 'buy':
            return SignalType.BUY
        elif self.side.lower() == 'sell':
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def get_asset_from_symbol(self) -> str:
        """Extraer el asset base del símbolo (ej: BTCUSDT -> BTC)"""
        if self.symbol.endswith('USDT'):
            return self.symbol[:-4]
        elif self.symbol.endswith('BUSD'):
            return self.symbol[:-4]
        elif self.symbol.endswith('USD'):
            return self.symbol[:-3]
        else:
            # Fallback: tomar los primeros 3-4 caracteres
            return self.symbol[:3] if len(self.symbol) >= 6 else self.symbol[:4]

@dataclass
class OrderIntent:
    """Intención de orden después de validaciones"""
    signal_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    created_timestamp: float = None

    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = time.time()

@dataclass
class ExecutionResult:
    """Resultado de ejecución del exchange"""
    order_id: str
    filled_qty: float
    avg_price: float
    fees: float
    latency_ms: float
    status: str  # FILLED, PARTIAL, REJECTED

@dataclass
class ExecutionReport:
    """Reporte completo de ejecución para el bus"""
    signal_id: str
    status: str  # EXECUTED, REJECTED_SAFETY, REJECTED_AI, EXECUTION_ERROR
    timestamp: float
    reason: Optional[str] = None
    executed_qty: Optional[float] = None
    executed_price: Optional[float] = None
    fees: Optional[float] = None
    latency_ms: Optional[float] = None
    ai_confidence: Optional[float] = None
    ai_risk_score: Optional[float] = None
    ai_model_votes: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class RiskAlert:
    """Alerta de riesgo generada por el sistema"""
    alert_id: str
    level: str  # WARNING, CRITICAL
    message: str
    signal_id: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ValidationResult:
    """Resultado de validación de riesgo"""
    is_valid: bool
    reason: str = ""
    risk_score: float = 0.0
    warnings: Optional[List[str]] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

# ============================================================================
# FUNCIONES HELPER PARA COMPATIBILIDAD
# ============================================================================

def create_signal(
    signal_id: str,
    symbol: str,
    side: str,  # 'buy' o 'sell'
    qty: float,
    strategy_id: str = "L2_TACTIC",
    order_type: str = "market",
    price: Optional[float] = None,
    confidence: float = 0.5,
    strength: float = 0.5,
    timestamp: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    features: Optional[Dict[str, Any]] = None,
    technical_indicators: Optional[Dict[str, float]] = None,
    signal_type: str = "tactical"
) -> Signal:
    """
    Función helper para crear señales fácilmente con todos los campos necesarios
    """
    return Signal(
        signal_id=signal_id,
        strategy_id=strategy_id,
        timestamp=timestamp or time.time(),
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        price=price,
        confidence=confidence,
        strength=strength,
        stop_loss=stop_loss,
        take_profit=take_profit,
        features=features,
        technical_indicators=technical_indicators or {},
        signal_type=signal_type
    )

def create_signal_from_tactical(tactical_signal, qty: float = None) -> Signal:
    """
    Crea una señal de trading a partir de una señal táctica
    """
    signal_id = f"L2_{int(time.time() * 1000)}"

    # Get timestamp, handling both datetime and float
    if hasattr(tactical_signal, 'timestamp'):
        if hasattr(tactical_signal.timestamp, 'timestamp'):
            timestamp = tactical_signal.timestamp.timestamp()
        else:
            timestamp = float(tactical_signal.timestamp)
    else:
        timestamp = time.time()

    # Extract technical indicators from features
    technical_indicators = {}
    if hasattr(tactical_signal, 'features') and tactical_signal.features:
        for k, v in tactical_signal.features.items():
            if isinstance(v, (int, float)):
                technical_indicators[k] = float(v)

    # Add strength as a technical indicator
    if hasattr(tactical_signal, 'strength'):
        technical_indicators['signal_strength'] = float(tactical_signal.strength)

    return Signal(
        signal_id=signal_id,
        strategy_id='L2_TACTIC',
        timestamp=timestamp,
        symbol=tactical_signal.symbol,
        side=tactical_signal.side.lower(),
        qty=qty or 0.0,  # Will be calculated by OrderManager if 0
        order_type=getattr(tactical_signal, 'type', 'market'),
        confidence=getattr(tactical_signal, 'confidence', 0.5),
        technical_indicators=technical_indicators
    )

# ============================================================================
# EXPORTACIONES
# ============================================================================

__all__ = [
    # L1 Models
    'L1Model',
    'BaseL1Model',
    'MomentumModel',
    'TechnicalIndicatorsModel',
    'VolumeSignalsModel',
    'L1Signal',
    'L1SignalType',

    # Enums nuevos
    'SignalType',
    'SignalSource',
    'OrderStatus',
    'ExecutionStatus',

    # Clases existentes
    'Signal',
    'OrderIntent',
    'ExecutionResult',
    'ExecutionReport',
    'RiskAlert',
    'ValidationResult',

    # Helper functions
    'create_signal',
    'create_signal_from_tactical',
]
