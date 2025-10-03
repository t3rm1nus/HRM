"""
Tight Range Handler - Production-grade TIGHT_RANGE processing for HRM PATH2

This module provides robust mean reversion signals specifically for TIGHT_RANGE regimes
in hybrid trading mode (PATH2), with comprehensive error handling and risk management.

Features:
- Pure numpy calculations (no pandas dependencies)
- Robust data validation with detailed error logging
- Production-grade technical indicators (RSI, Bollinger Bands, ATR)
- Dynamic risk management based on market volatility
- Confidence limits appropriate for PATH2 hybrid mode
"""

import numpy as np
from typing import Dict, Optional
from core.logging import logger


class PATH2TightRangeFix:
    """
    Production-grade handler for TIGHT_RANGE mean reversion in PATH2 hybrid mode.

    This class provides robust signal generation for tight range regimes with:
    - Comprehensive data validation and error handling
    - Pure numpy calculations for performance and reliability
    - Technical indicators: Bollinger Bands, RSI, ATR
    - Dynamic risk management with volatility-adjusted stops
    - Confidence limits suitable for hybrid mode (contra-allocation aware)
    """

    def __init__(self):
        """Initialize the tight range handler with production defaults"""
        self.min_data_points = 50  # Minimum data points for calculations
        self.bb_period = 20  # Bollinger Band period
        self.rsi_period = 14  # RSI calculation period
        self.atr_period = 14  # ATR for risk management
        self.max_confidence = 0.7  # PATH2 hybrid mode limit (below PATH3's 0.8+)

    def process_tight_range_signal(self, symbol: str, market_data, l3_confidence: float,
                                  l1_l2_signal: str = 'HOLD') -> Dict:
        """
        Generate mean reversion signals for TIGHT_RANGE regime in PATH2 mode.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            market_data: Market data (can be dict, pd.DataFrame, or raw arrays)
            l3_confidence: L3 regime analysis confidence (0.0-1.0)
            l1_l2_signal: L1/L2 combined signal for hybrid mode consideration

        Returns:
            Dict containing signal details with confidence, risk parameters, and metadata
        """
        try:
            # Step 1: Validate and extract data
            close_prices, highs, lows, volumes = self._validate_and_extract_data(market_data, symbol)

            if close_prices is None or len(close_prices) < self.min_data_points:
                return self._create_error_signal(symbol, "INSUFFICIENT_DATA",
                    f"Need minimum {self.min_data_points} data points, got {len(close_prices) if close_prices is not None else 0}")

            # Step 2: Calculate technical indicators
            bb_position, bb_width = self._calculate_bb_position(close_prices)
            rsi = self._calculate_rsi(close_prices)
            atr = self._calculate_atr(highs, lows, close_prices)

            # Step 3: Generate mean reversion signals based on indicators
            return self._generate_signal(symbol, close_prices[-1], bb_position, bb_width, rsi, atr,
                                       l3_confidence, l1_l2_signal)

        except Exception as e:
            logger.error(f"❌ Critical error in PATH2TightRangeFix for {symbol}: {str(e)}")
            return self._create_error_signal(symbol, "PROCESSING_ERROR", str(e))

    def _validate_and_extract_data(self, market_data, symbol: str):
        """
        Validate market data and extract numpy arrays for calculations.

        Supports multiple input formats for robustness.
        """
        try:
            # Handle different data formats
            if hasattr(market_data, 'values'):  # pandas Series/DataFrame
                close_prices = market_data['close'].values if 'close' in market_data.columns else None
                highs = market_data['high'].values if 'high' in market_data.columns else None
                lows = market_data['low'].values if 'low' in market_data.columns else None
                volumes = market_data['volume'].values if 'volume' in market_data.columns else None

            elif isinstance(market_data, dict):
                # Handle dictionary format
                close_prices = np.array(market_data.get('close', []))
                highs = np.array(market_data.get('high', []))
                lows = np.array(market_data.get('low', []))
                volumes = np.array(market_data.get('volume', []))

            else:
                # Assume it's already a numpy array
                close_prices = np.array(market_data)
                highs = None
                lows = None
                volumes = None

            # Validate data integrity
            if close_prices is None or len(close_prices) == 0:
                logger.error(f"❌ {symbol}: No close price data available")
                return None, None, None, None

            if highs is None or len(highs) != len(close_prices):
                logger.warning(f"⚠️ {symbol}: High prices not available or mismatched, using close prices as proxy")
                highs = close_prices.copy()

            if lows is None or len(lows) != len(close_prices):
                logger.warning(f"⚠️ {symbol}: Low prices not available or mismatched, using close prices as proxy")
                lows = close_prices.copy()

            logger.info(f"✅ {symbol}: Successfully validated {len(close_prices)} data points")
            return close_prices, highs, lows, volumes

        except Exception as e:
            logger.error(f"❌ {symbol}: Data validation error: {str(e)}")
            return None, None, None, None

    def _calculate_bb_position(self, prices: np.ndarray) -> tuple[float, float]:
        """Calculate Bollinger Band position and width."""
        try:
            sma = np.mean(prices[-self.bb_period:])
            std = np.std(prices[-self.bb_period:])
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            bb_width = (bb_upper - bb_lower) / sma  # Normalized width

            current_price = prices[-1]
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

            return max(0.0, min(1.0, bb_position)), bb_width

        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return 0.5, 0.1  # Neutral position

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate Relative Strength Index."""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-self.rsi_period:])
            avg_loss = np.mean(losses[-self.rsi_period:])

            if avg_loss == 0:
                return 50.0  # Neutral when no losses

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return max(0.0, min(100.0, rsi))

        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return 50.0  # Neutral RSI

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Calculate Average True Range for risk management."""
        try:
            high_low = highs - lows
            high_close = np.abs(highs - np.roll(closes, 1))[1:]  # Shifted to align
            low_close = np.abs(lows - np.roll(closes, 1))[1:]

            tr = np.maximum(high_low[1:], np.maximum(high_close, low_close))
            atr = np.mean(tr[-self.atr_period:]) if len(tr) >= self.atr_period else tr[-1]
            return max(atr, 1e-8)  # Avoid zero ATR

        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return abs(closes[-1] * 0.02)  # 2% default ATR

    def _generate_signal(self, symbol: str, current_price: float, bb_position: float,
                        bb_width: float, rsi: float, atr: float, l3_confidence: float,
                        l1_l2_signal: str) -> Dict:
        """
        Generate trading signal based on technical indicators.

        Mean reversion logic for TIGHT_RANGE:
        - BUY when oversold (BB < 0.25 and RSI < 35)
        - SELL when overbought (BB > 0.75 and RSI > 65)
        - HOLD when in balance (neutral zone)
        """
        try:
            # Enhanced mean reversion conditions with confidence adjustments
            if bb_position < 0.25 and rsi < 35:
                # Strong BUY signal - oversold in tight range
                base_confidence = min(self.max_confidence, l3_confidence * 0.8)

                # Additional confidence boost if L1/L2 agrees
                if l1_l2_signal.upper() in ['BUY', 'LONG']:
                    final_confidence = min(self.max_confidence, base_confidence * 1.2)
                else:
                    final_confidence = base_confidence

                stop_distance = max(atr * 1.5, current_price * 0.01)  # ATR-based or 1% minimum
                target_distance = max(atr * 2.5, current_price * 0.02)  # Higher target for MR

                return {
                    'action': 'BUY',
                    'confidence': final_confidence,
                    'reason': f'TIGHT_RANGE BUY: Oversold (BB:{bb_position:.2f}, RSI:{rsi:.1f})',
                    'stop_loss_pct': (stop_distance / current_price) * 100,
                    'take_profit_pct': (target_distance / current_price) * 100,
                    'entry_price': current_price,
                    'position_size_multiplier': 0.8,  # Conservative sizing in ranges
                    'indicators': {
                        'bb_position': bb_position,
                        'bb_width': bb_width,
                        'rsi': rsi,
                        'atr': atr,
                        'l3_confidence': l3_confidence
                    },
                    'signal_type': 'MEAN_REVERSION_BUY'
                }

            elif bb_position > 0.75 and rsi > 65:
                # Strong SELL signal - overbought in tight range
                base_confidence = min(self.max_confidence, l3_confidence * 0.8)

                # Additional confidence boost if L1/L2 agrees
                if l1_l2_signal.upper() in ['SELL', 'SHORT']:
                    final_confidence = min(self.max_confidence, base_confidence * 1.2)
                else:
                    final_confidence = base_confidence

                stop_distance = max(atr * 1.5, current_price * 0.01)
                target_distance = max(atr * 2.5, current_price * 0.02)

                return {
                    'action': 'SELL',
                    'confidence': final_confidence,
                    'reason': f'TIGHT_RANGE SELL: Overbought (BB:{bb_position:.2f}, RSI:{rsi:.1f})',
                    'stop_loss_pct': (stop_distance / current_price) * 100,
                    'take_profit_pct': (target_distance / current_price) * 100,
                    'entry_price': current_price,
                    'position_size_multiplier': 0.8,
                    'indicators': {
                        'bb_position': bb_position,
                        'bb_width': bb_width,
                        'rsi': rsi,
                        'atr': atr,
                        'l3_confidence': l3_confidence
                    },
                    'signal_type': 'MEAN_REVERSION_SELL'
                }

            elif bb_position < 0.4 and rsi < 45:
                # Moderate BUY signal
                confidence = min(0.55, l3_confidence * 0.6)
                stop_distance = max(atr * 2.0, current_price * 0.015)
                target_distance = max(atr * 3.0, current_price * 0.025)

                return {
                    'action': 'BUY',
                    'confidence': confidence,
                    'reason': f'TIGHT_RANGE MODERATE BUY: Lower zone (BB:{bb_position:.2f}, RSI:{rsi:.1f})',
                    'stop_loss_pct': (stop_distance / current_price) * 100,
                    'take_profit_pct': (target_distance / current_price) * 100,
                    'entry_price': current_price,
                    'position_size_multiplier': 0.6,
                    'indicators': {
                        'bb_position': bb_position,
                        'bb_width': bb_width,
                        'rsi': rsi,
                        'atr': atr,
                        'l3_confidence': l3_confidence
                    },
                    'signal_type': 'MODERATE_MR_BUY'
                }

            elif bb_position > 0.6 and rsi > 55:
                # Moderate SELL signal
                confidence = min(0.55, l3_confidence * 0.6)
                stop_distance = max(atr * 2.0, current_price * 0.015)
                target_distance = max(atr * 3.0, current_price * 0.025)

                return {
                    'action': 'SELL',
                    'confidence': confidence,
                    'reason': f'TIGHT_RANGE MODERATE SELL: Upper zone (BB:{bb_position:.2f}, RSI:{rsi:.1f})',
                    'stop_loss_pct': (stop_distance / current_price) * 100,
                    'take_profit_pct': (target_distance / current_price) * 100,
                    'entry_price': current_price,
                    'position_size_multiplier': 0.6,
                    'indicators': {
                        'bb_position': bb_position,
                        'bb_width': bb_width,
                        'rsi': rsi,
                        'atr': atr,
                        'l3_confidence': l3_confidence
                    },
                    'signal_type': 'MODERATE_MR_SELL'
                }

            else:
                # HOLD - price in balanced zone of tight range
                return {
                    'action': 'HOLD',
                    'confidence': 0.45,
                    'reason': f'TIGHT_RANGE HOLD: Balanced zone (BB:{bb_position:.2f}, RSI:{rsi:.1f})',
                    'stop_loss_pct': None,
                    'take_profit_pct': None,
                    'entry_price': current_price,
                    'position_size_multiplier': 0.0,
                    'indicators': {
                        'bb_position': bb_position,
                        'bb_width': bb_width,
                        'rsi': rsi,
                        'atr': atr,
                        'l3_confidence': l3_confidence
                    },
                    'signal_type': 'MR_HOLD'
                }

        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {str(e)}")
            return self._create_error_signal(symbol, "SIGNAL_GENERATION_ERROR", str(e))

    def _create_error_signal(self, symbol: str, error_type: str, error_details: str) -> Dict:
        """Create a safe error signal when processing fails."""
        logger.error(f"❌ {symbol} {error_type}: {error_details}")

        return {
            'action': 'HOLD',
            'confidence': 0.2,
            'reason': f'ERROR_{error_type}: {error_details[:100]}',  # Truncate long messages
            'stop_loss_pct': None,
            'take_profit_pct': None,
            'entry_price': None,
            'position_size_multiplier': 0.0,
            'indicators': {},
            'signal_type': 'ERROR_SIGNAL',
            'error_type': error_type,
            'error_details': error_details
        }
