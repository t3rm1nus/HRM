# signal_generator.py - Generador de señales para L2_tactic (adaptado para multiasset: BTC y ETH)

"""
Generador de señales para L2_tactic
===================================

Orquestador principal que combina:
- Modelo de IA como señal primaria
- Indicadores técnicos complementarios
- Reconocimiento de patrones
- Delegación de composición de señales a SignalComposer
Adaptado para manejar múltiples símbolos (BTC/USDT y ETH/USDT), asumiendo market_data es un dict con claves por símbolo.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import timedelta, timezone
import pandas as pd

from .config import L2Config
from .models import TacticalSignal, SignalDirection, SignalSource, L2State
from .ai_model_integration import AIModelWrapper
from .signal_composer import SignalComposer

logger = logging.getLogger(__name__)

def _make_utc(dt):
    """Convierte cualquier datetime a UTC aware"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# =====================================================
# Indicadores técnicos
# =====================================================
class TechnicalIndicators:
    """Calculadora de indicadores técnicos para señales complementarias"""

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        try:
            delta = prices.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window=window).mean()
            ma_down = down.rolling(window=window).mean()
            rs = ma_up / ma_down.replace(0, 1e-9)
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series()

    def calculate_macd(
        self, prices: pd.Series, short: int = 12, long: int = 26, signal: int = 9
    ):
        try:
            ema_short = prices.ewm(span=short, adjust=False).mean()
            ema_long = prices.ewm(span=long, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            hist = macd_line - signal_line
            return macd_line, signal_line, hist
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return pd.Series(), pd.Series(), pd.Series()

# =====================================================
# Reconocimiento de patrones
# =====================================================
class PatternRecognizer:
    """Reconocedor de patrones de velas y formaciones"""

    def detect_doji(
        self, open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold=0.1
    ) -> pd.Series:
        try:
            body = (close - open_prices).abs()
            rng = (high - low).replace(0, 1e-9)
            return (body / rng) < threshold
        except Exception as e:
            logger.error(f"Error detecting Doji: {e}")
            return pd.Series()

    def detect_hammer(
        self, open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold=2.0
    ) -> pd.Series:
        try:
            body = (close - open_prices).abs()
            lower_shadow = (open_prices.where(close > open_prices, close) - low)
            return (lower_shadow > threshold * body) & (close > open_prices)
        except Exception as e:
            logger.error(f"Error detecting Hammer: {e}")
            return pd.Series()

# =====================================================
# Signal Generator
# =====================================================
class SignalGenerator:
    """
    Generador principal de señales tácticas
    """
    def __init__(self, config: L2Config):
        self.config = config
        self.ai_model = AIModelWrapper(config.ai_model)
        self.technical = TechnicalIndicators()
        self.patterns = PatternRecognizer()
        self.state = L2State()
        self.composer = SignalComposer(config.__dict__)

        self.signal_performance = {
            symbol: {
                "ai_model": {"hits": 0, "total": 0, "recent_accuracy": 0.5},
                "technical": {"hits": 0, "total": 0, "recent_accuracy": 0.5},
                "patterns": {"hits": 0, "total": 0, "recent_accuracy": 0.5},
            } for symbol in config.signals.universe
        }

        logger.info("SignalGenerator initialized for multiasset")

    def generate_signals(
        self, market_data: Dict[str, pd.DataFrame], regime_context: Optional[Dict] = None
    ) -> List[TacticalSignal]:
        """Genera señales tácticas para todos los símbolos en el universo."""
        all_final_signals: List[TacticalSignal] = []

        for symbol in self.config.signals.universe:
            logger.info(f"Generating signals for {symbol}")

            try:
                data = market_data.get(symbol)
                if data is None:
                    logger.warning(f"No market data for {symbol}")
                    continue

                if "timestamp" in data.columns:
                    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

                if data.empty or len(data) < 50:
                    logger.warning(f"Insufficient data for {symbol}: {len(data)} rows")
                    continue

                self.state.cleanup_expired()

                all_signals: List[TacticalSignal] = []
                all_signals.extend(self._generate_ai_signals(data, symbol))
                all_signals.extend(self._generate_technical_signals(data, symbol))
                all_signals.extend(self._generate_pattern_signals(data, symbol))

                filtered_signals = self._apply_quality_filters(all_signals, data)
                final_signals = self.composer.compose(filtered_signals, regime_context)

                for signal in final_signals:
                    self.state.add_signal(signal)

                all_final_signals.extend(final_signals)

                logger.info(
                    f"Generated {len(final_signals)} final signals for {symbol} from {len(all_signals)} candidates"
                )

            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")

        return all_final_signals

    def _generate_ai_signals(
        self, market_data: pd.DataFrame, symbol: str
    ) -> List[TacticalSignal]:
        """Genera señales usando el modelo de IA"""
        logger.info(f"Generating AI signals for {symbol}")
        try:
            ai_signals = self.ai_model.predict(market_data, symbol)
            return ai_signals if ai_signals else []
        except Exception as e:
            logger.error(f"AI signal generation failed for {symbol}: {e}")
            return []

    def _generate_technical_signals(
        self, market_data: pd.DataFrame, symbol: str
    ) -> List[TacticalSignal]:
        """Genera señales usando indicadores técnicos"""
        logger.info(f"Generating technical signals for {symbol}")
        signals: List[TacticalSignal] = []
        try:
            current_time = pd.Timestamp.now(tz="UTC")
            if "close" not in market_data.columns:
                logger.warning(f"No 'close' column in market data for {symbol}")
                return signals

            prices = market_data["close"]
            if prices.empty or len(prices) < 14:
                logger.warning(f"Insufficient price data for {symbol}: {len(prices)} rows")
                return signals

            current_price = float(prices.iloc[-1])
            rsi = self.technical.calculate_rsi(prices)
            if not rsi.empty and len(rsi) >= 14:
                current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
                if current_rsi < 30:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            direction=SignalDirection.LONG,
                            strength=0.7,
                            confidence=0.7,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL,
                            metadata={"indicator": "RSI", "value": current_rsi, "condition": "oversold"},
                            expires_at=current_time + timedelta(
                                minutes=self.config.signals.signal_expiry_minutes
                            ),
                        )
                    )
                elif current_rsi > 70:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            direction=SignalDirection.SHORT,
                            strength=0.7,
                            confidence=0.7,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL,
                            metadata={"indicator": "RSI", "value": current_rsi, "condition": "overbought"},
                            expires_at=current_time + timedelta(
                                minutes=self.config.signals.signal_expiry_minutes
                            ),
                        )
                    )

            macd_line, signal_line, _ = self.technical.calculate_macd(prices)
            if not macd_line.empty and not signal_line.empty and len(macd_line) >= 26:
                if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            direction=SignalDirection.LONG,
                            strength=0.6,
                            confidence=0.6,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL,
                            metadata={"indicator": "MACD", "condition": "bullish_cross"},
                            expires_at=current_time + timedelta(
                                minutes=self.config.signals.signal_expiry_minutes
                            ),
                        )
                    )
                elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            direction=SignalDirection.SHORT,
                            strength=0.6,
                            confidence=0.6,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL,
                            metadata={"indicator": "MACD", "condition": "bearish_cross"},
                            expires_at=current_time + timedelta(
                                minutes=self.config.signals.signal_expiry_minutes
                            ),
                        )
                    )

            return signals
        except Exception as e:
            logger.error(f"Technical signal generation failed for {symbol}: {e}")
            return []

    def _generate_pattern_signals(
        self, market_data: pd.DataFrame, symbol: str
    ) -> List[TacticalSignal]:
        """Genera señales basadas en patrones de velas"""
        logger.info(f"Generating pattern signals for {symbol}")
        signals: List[TacticalSignal] = []
        try:
            current_time = pd.Timestamp.now(tz="UTC")
            required_cols = {"open", "high", "low", "close"}
            if not required_cols.issubset(market_data.columns):
                logger.warning(f"Missing required columns for {symbol}: {market_data.columns}")
                return signals

            open_prices = market_data["open"]
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]
            if close.empty or len(close) < 2:
                logger.warning(f"Insufficient price data for {symbol}: {len(close)} rows")
                return signals

            current_price = float(close.iloc[-1])
            doji = self.patterns.detect_doji(open_prices, high, low, close)
            if not doji.empty and doji.iloc[-1]:
                signals.append(
                    TacticalSignal(
                        symbol=symbol,
                        direction=SignalDirection.NEUTRAL,
                        strength=0.5,
                        confidence=0.5,
                        price=current_price,
                        timestamp=current_time,
                        source=SignalSource.PATTERN,
                        metadata={"pattern": "doji"},
                        expires_at=current_time + timedelta(
                            minutes=self.config.signals.signal_expiry_minutes
                        ),
                    )
                )

            hammer = self.patterns.detect_hammer(open_prices, high, low, close)
            if not hammer.empty and hammer.iloc[-1]:
                signals.append(
                    TacticalSignal(
                        symbol=symbol,
                        direction=SignalDirection.LONG,
                        strength=0.6,
                        confidence=0.6,
                        price=current_price,
                        timestamp=current_time,
                        source=SignalSource.PATTERN,
                        metadata={"pattern": "hammer"},
                        expires_at=current_time + timedelta(
                            minutes=self.config.signals.signal_expiry_minutes
                        ),
                    )
                )

            return signals
        except Exception as e:
            logger.error(f"Pattern signal generation failed for {symbol}: {e}")
            return []

    def _apply_quality_filters(
        self, signals: List[TacticalSignal], market_data: pd.DataFrame
    ) -> List[TacticalSignal]:
        """Filtra señales por fuerza mínima y elimina duplicados"""
        if not signals:
            return []

        filtered = [
            s for s in signals if s.strength >= self.config.signals.min_signal_strength
        ]

        unique = {}
        for s in filtered:
            key = (s.symbol, s.direction, s.source)
            if key not in unique or s.confidence > unique[key].confidence:
                unique[key] = s

        return list(unique.values())

    def update_signal_performance(
        self, symbol: str, signal_source: str, was_successful: bool
    ) -> None:
        if symbol in self.signal_performance and signal_source in self.signal_performance[symbol]:
            perf = self.signal_performance[symbol][signal_source]
            perf["total"] += 1
            if was_successful:
                perf["hits"] += 1
            if perf["total"] > 0:
                perf["recent_accuracy"] = perf["hits"] / perf["total"]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "ai_model": self.ai_model.get_model_info(),
            "signal_performance": self.signal_performance,
            "active_signals_count": len(self.state.active_signals),
            "config": {
                "min_signal_strength": self.config.signals.min_signal_strength,
                "ai_model_weight": self.config.signals.ai_model_weight,
                "technical_weight": self.config.signals.technical_weight,
                "pattern_weight": self.config.signals.pattern_weight,
                "universe": self.config.signals.universe,
            },
        }