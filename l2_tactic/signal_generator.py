import logging
from typing import Dict, List, Optional, Any
from datetime import timedelta, timezone
import pandas as pd
from dataclasses import dataclass

from .config import L2Config
from .models import TacticalSignal, SignalDirection, SignalSource, L2State
from .ai_model_integration import AIModelWrapper
from .signal_composer import SignalComposer

logger = logging.getLogger("l2_tactic.signal_generator")

def _make_utc(dt):
    """Convierte cualquier datetime a UTC aware"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

class TechnicalIndicators:
    """Calculadora de indicadores técnicos para señales complementarias"""

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        try:
            if not pd.api.types.is_numeric_dtype(prices):
                raise ValueError("Prices must be numeric")
            delta = prices.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window=window).mean()
            ma_down = down.rolling(window=window).mean()
            rs = ma_up / ma_down.replace(0, 1e-9)
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}", exc_info=True)
            return pd.Series()

    def calculate_macd(
        self, prices: pd.Series, short: int = 12, long: int = 26, signal: int = 9
    ):
        try:
            if not pd.api.types.is_numeric_dtype(prices):
                raise ValueError("Prices must be numeric")
            ema_short = prices.ewm(span=short, adjust=False).mean()
            ema_long = prices.ewm(span=long, adjust=False).mean()
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            hist = macd_line - signal_line
            return macd_line, signal_line, hist
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}", exc_info=True)
            return pd.Series(), pd.Series(), pd.Series()

class PatternRecognizer:
    """Reconocedor de patrones de velas y formaciones"""

    def detect_doji(
        self, open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold=0.1
    ) -> pd.Series:
        try:
            for series in [open_prices, high, low, close]:
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(f"Series {series.name} must be numeric")
            body = (close - open_prices).abs()
            rng = (high - low).replace(0, 1e-9)
            return (body / rng) < threshold
        except Exception as e:
            logger.error(f"Error detecting Doji: {e}", exc_info=True)
            return pd.Series()

    def detect_hammer(
        self, open_prices: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, threshold=2.0
    ) -> pd.Series:
        try:
            for series in [open_prices, high, low, close]:
                if not pd.api.types.is_numeric_dtype(series):
                    raise ValueError(f"Series {series.name} must be numeric")
            body = (close - open_prices).abs()
            lower_shadow = (open_prices.where(close > open_prices, close) - low)
            return (lower_shadow > threshold * body) & (close > open_prices)
        except Exception as e:
            logger.error(f"Error detecting Hammer: {e}", exc_info=True)
            return pd.Series()

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

    def generate(self, state: Dict, symbol: str) -> List[Dict]:
        """
        Genera señales tácticas para un símbolo dado basado en el estado del mercado.
        Args:
            state: Diccionario con el estado actual del sistema.
            symbol: Símbolo para el cual generar señales (e.g., 'BTC/USDT').
        Returns:
            Lista de señales con formato {'symbol': str, 'direction': str, 'confidence': float, 'source': str}.
        """
        logger.info(f"Generating signals for {symbol}")
        signals = []

        # Verificar si el símbolo está en el mercado
        if symbol not in state["mercado"]:
            logger.error(f"Symbol {symbol} not found in market data")
            return signals

        # Obtener datos de mercado
        market_data = state["mercado"][symbol]
        if market_data.empty:
            logger.error(f"No market data available for {symbol}")
            return signals

        # Generar señales de IA
        logger.info(f"Generating AI signals for {symbol}")
        try:
            ai_signals = self.ai_model.predict(market_data, symbol)
            for signal in ai_signals:
                signals.append({
                    "symbol": signal.symbol,
                    "direction": signal.side,
                    "confidence": signal.confidence,
                    "source": signal.source
                })
            logger.info(f"Generated {len(ai_signals)} AI signals for {symbol}")
        except Exception as e:
            logger.error(f"Error generating AI signals for {symbol}: {e}", exc_info=True)

        # Generar señales técnicas
        logger.info(f"Generating technical signals for {symbol}")
        try:
            technical_signals = self._generate_technical_signals(market_data, symbol)
            for signal in technical_signals:
                signals.append({
                    "symbol": signal.symbol,
                    "direction": signal.side,
                    "confidence": signal.confidence,
                    "source": signal.source
                })
            logger.info(f"Generated {len(technical_signals)} technical signals for {symbol}")
        except Exception as e:
            logger.error(f"Technical signal generation failed for {symbol}: {e}", exc_info=True)

        # Generar señales de patrones
        logger.info(f"Generating pattern signals for {symbol}")
        try:
            pattern_signals = self._generate_pattern_signals(market_data, symbol)
            for signal in pattern_signals:
                signals.append({
                    "symbol": signal.symbol,
                    "direction": signal.side,
                    "confidence": signal.confidence,
                    "source": signal.source
                })
            logger.info(f"Generated {len(pattern_signals)} pattern signals for {symbol}")
        except Exception as e:
            logger.error(f"Pattern signal generation failed for {symbol}: {e}", exc_info=True)

        # Combinar señales usando SignalComposer
        try:
            final_signals = self.composer.compose(signals, market_data)
            logger.info(f"Generated {len(final_signals)} final signals for {symbol} from {len(signals)} candidates")
            return final_signals
        except Exception as e:
            logger.error(f"Error combining signals for {symbol}: {e}", exc_info=True)
            return []

    def _generate_technical_signals(
        self, market_data: pd.DataFrame, symbol: str
    ) -> List[TacticalSignal]:
        """Genera señales basadas en indicadores técnicos"""
        logger.info(f"Generating technical signals for {symbol}")
        signals: List[TacticalSignal] = []
        try:
            current_time = pd.Timestamp.now(tz="UTC")
            required_cols = {"close"}
            if not required_cols.issubset(market_data.columns):
                logger.warning(f"Missing required columns for {symbol}: {market_data.columns}")
                return signals

            close = market_data["close"]
            if close.empty or len(close) < 20:
                logger.warning(f"Insufficient price data for {symbol}: {len(close)} rows")
                return signals

            if not pd.api.types.is_numeric_dtype(close):
                logger.error(f"Close series for {symbol} is not numeric: {close.dtype}")
                return signals

            current_price = float(close.iloc[-1])
            # RSI
            rsi = self.technical.calculate_rsi(close)
            if not rsi.empty and len(rsi) > 0:
                rsi_value = rsi.iloc[-1]
                if rsi_value > 70:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            side=SignalDirection.SHORT.value,
                            strength=0.6,
                            confidence=0.6,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL.value,
                            features_used={"indicator": "rsi", "value": rsi_value},
                            horizon="1h",
                            reasoning="RSI overbought (>70)"
                        )
                    )
                elif rsi_value < 30:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            side=SignalDirection.LONG.value,
                            strength=0.6,
                            confidence=0.6,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL.value,
                            features_used={"indicator": "rsi", "value": rsi_value},
                            horizon="1h",
                            reasoning="RSI oversold (<30)"
                        )
                    )

            # MACD
            macd_line, signal_line, hist = self.technical.calculate_macd(close)
            if not macd_line.empty and len(macd_line) > 0:
                if macd_line.iloc[-1] > signal_line.iloc[-1] and hist.iloc[-1] > 0:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            side=SignalDirection.LONG.value,
                            strength=0.7,
                            confidence=0.7,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL.value,
                            features_used={"indicator": "macd", "hist": hist.iloc[-1]},
                            horizon="1h",
                            reasoning="MACD bullish crossover"
                        )
                    )
                elif macd_line.iloc[-1] < signal_line.iloc[-1] and hist.iloc[-1] < 0:
                    signals.append(
                        TacticalSignal(
                            symbol=symbol,
                            side=SignalDirection.SHORT.value,
                            strength=0.7,
                            confidence=0.7,
                            price=current_price,
                            timestamp=current_time,
                            source=SignalSource.TECHNICAL.value,
                            features_used={"indicator": "macd", "hist": hist.iloc[-1]},
                            horizon="1h",
                            reasoning="MACD bearish crossover"
                        )
                    )

            return signals
        except Exception as e:
            logger.error(f"Technical signal generation failed for {symbol}: {e}", exc_info=True)
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

            for series in [open_prices, high, low, close]:
                if not pd.api.types.is_numeric_dtype(series):
                    logger.error(f"Series {series.name} for {symbol} is not numeric: {series.dtype}")
                    return signals

            current_price = float(close.iloc[-1])
            doji = self.patterns.detect_doji(open_prices, high, low, close)
            if not doji.empty and doji.iloc[-1]:
                signals.append(
                    TacticalSignal(
                        symbol=symbol,
                        side=SignalDirection.NEUTRAL.value,
                        strength=0.5,
                        confidence=0.5,
                        price=current_price,
                        timestamp=current_time,
                        source=SignalSource.PATTERN.value,
                        features_used={"pattern": "doji"},
                        horizon="1h",
                        reasoning="Doji pattern detected"
                    )
                )

            hammer = self.patterns.detect_hammer(open_prices, high, low, close)
            if not hammer.empty and hammer.iloc[-1]:
                signals.append(
                    TacticalSignal(
                        symbol=symbol,
                        side=SignalDirection.LONG.value,
                        strength=0.6,
                        confidence=0.6,
                        price=current_price,
                        timestamp=current_time,
                        source=SignalSource.PATTERN.value,
                        features_used={"pattern": "hammer"},
                        horizon="1h",
                        reasoning="Hammer pattern detected"
                    )
                )

            return signals
        except Exception as e:
            logger.error(f"Pattern signal generation failed for {symbol}: {e}", exc_info=True)
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
            key = (s.symbol, s.side, s.source)
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