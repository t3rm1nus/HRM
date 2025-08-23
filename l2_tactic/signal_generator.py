# signal_generator.py - Generador de señales para L2_tactic

"""
Generador de señales para L2_tactic
===================================

Orquestador principal que combina:
- Modelo de IA como señal primaria
- Indicadores técnicos complementarios
- Reconocimiento de patrones
- Delegación de composición de señales a SignalComposer
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import timedelta, timezone
import pandas as pd

from .config import L2Config
from .models import TacticalSignal, SignalDirection, SignalSource, L2State
from .ai_model_integration import AIModelWrapper
from .signal_composer import SignalComposer  # ✅ usamos el nuevo módulo

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
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(window=window).mean()
        ma_down = down.rolling(window=window).mean()
        rs = ma_up / ma_down.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def calculate_macd(
        self, prices: pd.Series, short: int = 12, long: int = 26, signal: int = 9
    ):
        ema_short = prices.ewm(span=short, adjust=False).mean()
        ema_long = prices.ewm(span=long, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist


# =====================================================
# Reconocimiento de patrones
# =====================================================
class PatternRecognizer:
    """Reconocedor de patrones de velas y formaciones"""

    def detect_doji(
        self, open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold=0.1
    ) -> pd.Series:
        body = (close - open_prices).abs()
        rng = (high - low).replace(0, 1e-9)
        return (body / rng) < threshold

    def detect_hammer(
        self, open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold=2.0
    ) -> pd.Series:
        body = (close - open_prices).abs()
        lower_shadow = (open_prices.where(close > open_prices, close) - low)
        return (lower_shadow > threshold * body) & (close > open_prices)


# =====================================================
# Signal Generator
# =====================================================
class SignalGenerator:
    """
    Generador principal de señales tácticas

    Combina múltiples fuentes de señales con pesos dinámicos:
    - Modelo de IA (señal primaria)
    - Indicadores técnicos (RSI, MACD, Bollinger)
    - Patrones de velas
    - Análisis de volumen
    """

    def __init__(self, config: L2Config):
        self.config = config
        self.ai_model = AIModelWrapper(config.ai_model)
        self.technical = TechnicalIndicators()
        self.patterns = PatternRecognizer()
        self.state = L2State()
        self.composer = SignalComposer(config.__dict__)  # ✅ nuevo

        # Métricas de performance por fuente
        self.signal_performance = {
            "ai_model": {"hits": 0, "total": 0, "recent_accuracy": 0.5},
            "technical": {"hits": 0, "total": 0, "recent_accuracy": 0.5},
            "patterns": {"hits": 0, "total": 0, "recent_accuracy": 0.5},
        }

        logger.info("SignalGenerator initialized")

    # =====================================================
    # Orquestador
    # =====================================================
    def generate_signals(
        self, market_data: pd.DataFrame, symbol: str, regime_context: Optional[Dict] = None
    ) -> List[TacticalSignal]:
        """Genera señales tácticas para un símbolo"""
        logger.info(f"Generating signals for {symbol}")

        try:
            if "timestamp" in market_data.columns:
                market_data["timestamp"] = pd.to_datetime(
                    market_data["timestamp"], utc=True
                )

            if market_data.empty or len(market_data) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(market_data)} rows")
                return []

            self.state.cleanup_expired()

            all_signals: List[TacticalSignal] = []
            all_signals.extend(self._generate_ai_signals(market_data, symbol))
            all_signals.extend(self._generate_technical_signals(market_data, symbol))
            all_signals.extend(self._generate_pattern_signals(market_data, symbol))

            filtered_signals = self._apply_quality_filters(all_signals, market_data)

            # ✅ Usamos SignalComposer
            final_signals = self.composer.compose(filtered_signals, regime_context)

            for signal in final_signals:
                self.state.add_signal(signal)

            logger.info(
                f"Generated {len(final_signals)} final signals for {symbol} from {len(all_signals)} candidates"
            )
            return final_signals

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return []

    # =====================================================
    # Generadores de señales
    # =====================================================
    def _generate_ai_signals(
        self, market_data: pd.DataFrame, symbol: str
    ) -> List[TacticalSignal]:
        """Genera señales usando el modelo de IA"""
        logger.info(f"Generating AI signals for {symbol}")
        try:
            features = market_data.tail(1).copy()
            ai_signals = self.ai_model.predict(features, symbol)

            if ai_signals is None:
                return []

            return list(ai_signals)
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
                return signals

            prices = market_data["close"]
            if prices.empty:
                return signals

            current_price = float(prices.iloc[-1])

            # RSI
            rsi = self.technical.calculate_rsi(prices)
            if not rsi.empty:
                current_rsi = rsi.iloc[-1]
                if current_rsi < 30:  # Sobreventa
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            direction=SignalDirection.LONG,
                            strength=min((30 - current_rsi) / 20, 1.0),
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
                elif current_rsi > 70:  # Sobrecompra
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            direction=SignalDirection.SHORT,
                            strength=min((current_rsi - 70) / 20, 1.0),
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

            # MACD
            macd_line, signal_line, _ = self.technical.calculate_macd(prices)
            if not macd_line.empty and not signal_line.empty:
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
                return signals

            open_prices = market_data["open"]
            high = market_data["high"]
            low = market_data["low"]
            close = market_data["close"]
            if close.empty:
                return signals

            current_price = float(close.iloc[-1])

            # Doji
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

            # Hammer
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

    # =====================================================
    # Filtros y métricas
    # =====================================================
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
        if signal_source in self.signal_performance:
            perf = self.signal_performance[signal_source]
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
            },
        }
