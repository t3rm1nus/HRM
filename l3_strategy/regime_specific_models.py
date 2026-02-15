#!/usr/bin/env python3
"""
Regime-Specific L3 Models for HRM
Implements 3 specialized strategic models based on market regime:
- Bull Market Model: Aggressive growth strategies
- Bear Market Model: Defensive preservation strategies
- Range Market Model: Mean-reversion and volatility strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from core.logging import logger

@dataclass
class RegimeStrategy:
    """Strategy recommendations for specific market regime"""
    regime: str
    risk_appetite: float  # 0.0 to 1.0
    asset_allocation: Dict[str, float]
    position_sizing: Dict[str, Any]
    stop_loss_policy: Dict[str, Any]
    take_profit_policy: Dict[str, Any]
    rebalancing_frequency: str
    volatility_target: float
    correlation_limits: Dict[str, float]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BullMarketModel:
    """
    Bull Market L3 Model - Aggressive Growth Strategy
    Characteristics:
    - High risk appetite
    - Leveraged long positions
    - Momentum-based allocation
    - Loose stop losses
    - Aggressive take profits
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "BullMarketModel"
        self.min_trend_strength = self.config.get('min_trend_strength', 0.02)
        self.max_leverage = self.config.get('max_leverage', 2.0)
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'daily')

    def generate_strategy(self, market_data: Dict[str, pd.DataFrame],
                         regime_context: Dict[str, Any]) -> RegimeStrategy:
        """
        Generate bull market strategy based on market conditions
        """
        try:
            logger.info("🐂 Bull Market Model: Generating aggressive growth strategy")

            # Fixed bull market allocation as specified
            asset_allocation = {
                'BTC': 0.60,
                'ETH': 0.30,
                'USDT': 0.10
            }

            # Aggressive risk parameters for bull markets
            risk_appetite = 0.8

            # Position sizing based on volatility and momentum
            position_sizing = {
                'BTCUSDT': {
                    'max_position': 0.8,
                    'min_position': 0.3,
                    'vol_target': 0.15,
                    'momentum_multiplier': 1.5
                },
                'ETHUSDT': {
                    'max_position': 0.6,
                    'min_position': 0.2,
                    'vol_target': 0.18,
                    'momentum_multiplier': 1.3
                }
            }

            # Loose stop losses in bull markets
            stop_loss_policy = {
                'type': 'trailing',
                'initial_stop': 0.05,  # 5% initial stop
                'trailing_activation': 0.10,  # Activate after 10% gain
                'trailing_distance': 0.08,  # 8% trailing distance
                'max_drawdown': 0.15  # Allow 15% drawdown
            }

            # Aggressive take profits
            take_profit_policy = {
                'type': 'multiple_targets',
                'targets': [
                    {'price_level': 1.10, 'position_size': 0.25},  # Take 25% off at 10% gain
                    {'price_level': 1.25, 'position_size': 0.33},  # Take 33% off at 25% gain
                    {'price_level': 1.50, 'position_size': 1.00}   # Take all off at 50% gain
                ],
                'time_based_exit': '30d'  # Exit after 30 days regardless
            }

            strategy = RegimeStrategy(
                regime='bull',
                risk_appetite=risk_appetite,
                asset_allocation=asset_allocation,
                position_sizing=position_sizing,
                stop_loss_policy=stop_loss_policy,
                take_profit_policy=take_profit_policy,
                rebalancing_frequency=self.rebalance_frequency,
                volatility_target=0.20,  # Higher vol target in bull markets
                correlation_limits={
                    'max_correlation': 0.8,  # Allow higher correlation in bull markets
                    'min_diversification': 0.3
                },
                metadata={
                    'strategy_type': 'fixed_allocation',
                    'model_version': '1.0'
                }
            )

            logger.info(f"🐂 Bull Market Strategy: Risk={risk_appetite:.2f}, BTC={asset_allocation['BTC']:.2f}, ETH={asset_allocation['ETH']:.2f}")
            return strategy

        except Exception as e:
            logger.error(f"Bull Market Model error: {e}")
            return self._get_default_bull_strategy()

    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum score for asset"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.5

            prices = data['close'].tail(20)
            if len(prices) < 5:
                return 0.5

            # Short-term momentum (5 periods)
            short_momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]

            # Medium-term momentum (20 periods)
            if len(prices) >= 20:
                medium_momentum = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            else:
                medium_momentum = short_momentum

            # Combined momentum score (normalized)
            momentum_score = (short_momentum * 0.6 + medium_momentum * 0.4)
            return max(0, min(1, momentum_score + 0.5))  # Normalize to 0-1

        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.5

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for asset"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.05

            returns = data['close'].pct_change().dropna().tail(20)
            if len(returns) < 5:
                return 0.05

            return returns.std()
        except Exception:
            return 0.05

    def _get_default_bull_strategy(self) -> RegimeStrategy:
        """Return default bull market strategy"""
        return RegimeStrategy(
            regime='bull',
            risk_appetite=0.8,
            asset_allocation={'BTC': 0.6, 'ETH': 0.3, 'USDT': 0.1},
            position_sizing={},
            stop_loss_policy={},
            take_profit_policy={},
            rebalancing_frequency='daily',
            volatility_target=0.20,
            correlation_limits={'max_correlation': 0.8, 'min_diversification': 0.3}
        )

class BearMarketModel:
    """
    Bear Market L3 Model - Defensive Preservation Strategy
    Characteristics:
    - Low risk appetite
    - Heavy cash allocation
    - Short positions or put options
    - Tight stop losses
    - Conservative take profits
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "BearMarketModel"
        self.cash_allocation_min = self.config.get('cash_allocation_min', 0.6)
        self.max_short_exposure = self.config.get('max_short_exposure', 0.3)

    def generate_strategy(self, market_data: Dict[str, pd.DataFrame],
                         regime_context: Dict[str, Any]) -> RegimeStrategy:
        """
        Generate bear market strategy focused on capital preservation
        """
        try:
            logger.info("🐻 Bear Market Model: Generating defensive preservation strategy")

            # Fixed bear market allocation as specified
            asset_allocation = {
                'BTC': 0.00,
                'ETH': 0.00,
                'USDT': 1.00
            }

            # Ultra-conservative risk parameters
            risk_appetite = 0.2

            # Conservative position sizing
            position_sizing = {
                'BTCUSDT': {
                    'max_position': 0.15,
                    'min_position': 0.05,
                    'vol_target': 0.08,
                    'risk_multiplier': 0.5
                },
                'ETHUSDT': {
                    'max_position': 0.10,
                    'min_position': 0.03,
                    'vol_target': 0.10,
                    'risk_multiplier': 0.4
                }
            }

            # Tight stop losses in bear markets
            stop_loss_policy = {
                'type': 'fixed_tight',
                'initial_stop': 0.02,  # 2% initial stop
                'max_drawdown': 0.05,  # Max 5% drawdown
                'time_stop': '7d',     # Exit after 7 days regardless
                'volatility_adjusted': True
            }

            # Conservative take profits - quick profits in bear markets
            take_profit_policy = {
                'type': 'quick_profit',
                'targets': [
                    {'price_level': 1.03, 'position_size': 0.50},  # Take 50% off at 3% gain
                    {'price_level': 1.05, 'position_size': 1.00}   # Take all off at 5% gain
                ],
                'time_based_exit': '3d'  # Exit after 3 days regardless
            }

            strategy = RegimeStrategy(
                regime='bear',
                risk_appetite=risk_appetite,
                asset_allocation=asset_allocation,
                position_sizing=position_sizing,
                stop_loss_policy=stop_loss_policy,
                take_profit_policy=take_profit_policy,
                rebalancing_frequency='weekly',  # Less frequent rebalancing
                volatility_target=0.08,  # Lower vol target in bear markets
                correlation_limits={
                    'max_correlation': 0.5,  # Stricter correlation limits
                    'min_diversification': 0.7  # Higher diversification required
                },
                metadata={
                    'cash_allocation': 1.00,
                    'strategy_type': 'defensive_preservation',
                    'model_version': '1.0'
                }
            )

            logger.info(f"🐻 Bear Market Strategy: Risk={risk_appetite:.2f}, Cash={1.00:.2f}, BTC={0.00:.2f}")
            return strategy

        except Exception as e:
            logger.error(f"Bear Market Model error: {e}")
            return self._get_default_bear_strategy()

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate downside volatility (focus on losses)"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.10

            returns = data['close'].pct_change().dropna().tail(30)

            if len(returns) < 5:
                return 0.10

            # Focus on downside volatility (negative returns)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                return downside_returns.std()
            else:
                return returns.std() * 0.8  # Slightly lower if no downside moves

        except Exception:
            return 0.10

    def _get_default_bear_strategy(self) -> RegimeStrategy:
        """Return default bear market strategy"""
        return RegimeStrategy(
            regime='bear',
            risk_appetite=0.2,
            asset_allocation={'BTC': 0.00, 'ETH': 0.00, 'USDT': 1.00},
            position_sizing={},
            stop_loss_policy={},
            take_profit_policy={},
            rebalancing_frequency='weekly',
            volatility_target=0.08,
            correlation_limits={'max_correlation': 0.5, 'min_diversification': 0.7}
        )

class RangeMarketModel:
    """
    Range/Sideways Market L3 Model - CAUTIOUS Operations

    SOLUTION 1: Allow cautious operations in range regime instead of blocking all
    Characteristics:
    - Small position sizes with high caution
    - Mean-reversion with strict risk controls
    - Conservative volatility targeting
    - Frequent rebalancing to capture small moves
    - Portfolio-aware position limits
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "RangeMarketModel"

    def generate_strategy(self, market_data: Dict[str, pd.DataFrame],
                         regime_context: Dict[str, Any]) -> RegimeStrategy:
        """
        Generate range market strategy with fixed allocations
        """
        try:
            logger.info("📊 Range Market Model: Generating range regime strategy")

            # Fixed range market allocation as specified
            asset_allocation = {
                'BTC': 0.30,
                'ETH': 0.30,
                'USDT': 0.40
            }

            # Moderate risk appetite for range markets
            risk_appetite = 0.5

            # Position sizing with moderate limits
            position_sizing = {
                'BTCUSDT': {
                    'max_position': 0.40,
                    'min_position': 0.20,
                    'vol_target': 0.12,
                    'range_mode': True
                },
                'ETHUSDT': {
                    'max_position': 0.40,
                    'min_position': 0.20,
                    'vol_target': 0.12,
                    'range_mode': True
                }
            }

            # Stop losses for range trading
            stop_loss_policy = {
                'type': 'range_stops',
                'initial_stop': 0.03,  # 3% initial stop
                'max_drawdown': 0.06,  # Max 6% drawdown
                'time_stop': '3d',     # Exit after 3 days regardless
                'volatility_adjusted': True
            }

            # Take profits for range trading
            take_profit_policy = {
                'type': 'range_profits',
                'targets': [
                    {'price_level': 1.03, 'position_size': 0.50},  # Take 50% off at 3% gain
                    {'price_level': 1.06, 'position_size': 1.00}   # Take all off at 6% gain
                ],
                'time_based_exit': '2d',  # Exit after 2 days regardless
                'range_mode': True
            }

            strategy = RegimeStrategy(
                regime='range',
                risk_appetite=risk_appetite,
                asset_allocation=asset_allocation,
                position_sizing=position_sizing,
                stop_loss_policy=stop_loss_policy,
                take_profit_policy=take_profit_policy,
                rebalancing_frequency='daily',
                volatility_target=0.12,
                correlation_limits={
                    'max_correlation': 0.6,
                    'min_diversification': 0.5
                },
                metadata={
                    'strategy_type': 'fixed_allocation',
                    'model_version': '1.0'
                }
            )

            logger.info(f"📊 Range Strategy: Risk={risk_appetite:.2f}, BTC={asset_allocation['BTC']:.2f}, ETH={asset_allocation['ETH']:.2f}, Cash={asset_allocation['USDT']:.2f}")
            return strategy

        except Exception as e:
            logger.error(f"Range Market Model error: {e}")
            return self._get_default_range_strategy()

    def _calculate_signal_strength(self, bb_position: float, rsi: float, volatility: float) -> float:
        """Calculate comprehensive signal strength for position sizing"""
        try:
            # BB position strength (0-1, higher when near bands)
            bb_strength = 1 - abs(bb_position - 0.5) * 1.8  # Peaks at band edges

            # RSI strength (0-1, higher when oversold/overbought)
            if rsi < 30 or rsi > 70:
                rsi_strength = 1.0
            elif rsi < 40 or rsi > 60:
                rsi_strength = 0.7
            else:
                rsi_strength = 0.3

            # Volatility adjustment (prefer moderate volatility)
            if volatility < 0.03:
                vol_multiplier = 0.5  # Too low volatility = weaker signals
            elif volatility < 0.08:
                vol_multiplier = 1.0  # Optimal volatility
            else:
                vol_multiplier = 0.7  # High volatility = mixed signals

            signal_strength = (bb_strength * 0.5 + rsi_strength * 0.3) * vol_multiplier
            return min(1.0, max(0.0, signal_strength))

        except Exception:
            return 0.3  # Default moderate strength

    def _calculate_bb_position(self, data: pd.DataFrame) -> float:
        """Calculate position within Bollinger Bands (0=lower, 0.5=middle, 1=upper)"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.5

            prices = data['close'].tail(20)
            if len(prices) < 20:
                return 0.5

            # Calculate Bollinger Bands
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)

            current_price = prices.iloc[-1]
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]

            if upper > lower:
                position = (current_price - lower) / (upper - lower)
                return max(0, min(1, position))  # Clamp to 0-1
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Error calculating BB position: {e}")
            return 0.5

    def _calculate_rsi(self, data: pd.DataFrame) -> float:
        """Calculate RSI indicator"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 50.0

            prices = data['close'].tail(14)
            if len(prices) < 14:
                return 50.0

            deltas = prices.diff()
            gain = (deltas.where(deltas > 0, 0)).rolling(window=14).mean()
            loss = (-deltas.where(deltas < 0, 0)).rolling(window=14).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0

        except Exception:
            return 50.0

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for asset in range markets"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.08

            returns = data['close'].pct_change().dropna().tail(20)
            if len(returns) < 5:
                return 0.08

            return returns.std()
        except Exception:
            return 0.08

    def _get_default_range_strategy(self) -> RegimeStrategy:
        """Return default range market strategy"""
        return RegimeStrategy(
            regime='range',
            risk_appetite=0.5,
            asset_allocation={'BTC': 0.30, 'ETH': 0.30, 'USDT': 0.40},
            position_sizing={},
            stop_loss_policy={},
            take_profit_policy={},
            rebalancing_frequency='daily',
            volatility_target=0.12,
            correlation_limits={'max_correlation': 0.6, 'min_diversification': 0.5}
        )

class VolatileMarketModel:
    """
    Volatile Market L3 Model - Crisis Management Strategy
    Characteristics:
    - Moderate risk appetite with high volatility adjustments
    - Diversified allocation with volatility harvesting
    - Dynamic position sizing based on volatility spikes
    - Wide stop losses to avoid whipsaws
    - Quick profit taking in volatile conditions
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "VolatileMarketModel"
        self.volatility_threshold = self.config.get('volatility_threshold', 0.08)
        self.max_single_position = self.config.get('max_single_position', 0.25)

    def generate_strategy(self, market_data: Dict[str, pd.DataFrame],
                         regime_context: Dict[str, Any]) -> RegimeStrategy:
        """
        Generate volatile market strategy focused on crisis management and volatility harvesting
        """
        try:
            logger.info("🌪️ Volatile Market Model: Generating crisis management strategy")

            # Extract market metrics
            btc_data = market_data.get('BTCUSDT')
            eth_data = market_data.get('ETHUSDT')

            if btc_data is None or btc_data.empty:
                logger.warning("Volatile Market Model: Insufficient BTC data")
                return self._get_default_volatile_strategy()

            # Calculate volatility and crisis indicators
            btc_vol = self._calculate_volatility(btc_data)
            eth_vol = self._calculate_volatility(eth_data) if eth_data is not None else btc_vol

            avg_volatility = (btc_vol + eth_vol) / 2

            # Calculate drawdown risk
            btc_drawdown = self._calculate_drawdown(btc_data)
            eth_drawdown = self._calculate_drawdown(eth_data) if eth_data is not None else 0

            # Volatile market: Conservative but diversified approach
            # Allocate based on volatility - lower vol assets get higher weight
            total_vol = btc_vol + eth_vol
            if total_vol > 0:
                btc_weight = (1 - btc_vol / total_vol) / 2 + 0.25  # Base 25% + vol adjustment
                eth_weight = (1 - eth_vol / total_vol) / 2 + 0.25
            else:
                btc_weight = 0.4
                eth_weight = 0.3

            # Add alternative assets for diversification in volatile markets
            alt_weight = 0.3  # Gold, bonds, or other uncorrelated assets
            cash_weight = 1.0 - btc_weight - eth_weight - alt_weight

            # Ensure minimum cash holding for liquidity
            cash_weight = max(cash_weight, 0.15)

            asset_allocation = {
                'BTC': btc_weight,
                'ETH': eth_weight,
                'USDT': cash_weight
            }

            # Moderate risk appetite adjusted for volatility
            base_risk = 0.4
            vol_adjustment = max(0, (avg_volatility - self.volatility_threshold) * 5)  # Reduce risk with high vol
            risk_appetite = max(0.1, base_risk - vol_adjustment)

            # Conservative position sizing with volatility caps
            position_sizing = {
                'BTCUSDT': {
                    'max_position': min(self.max_single_position, 0.35 - btc_vol * 2),
                    'min_position': 0.05,
                    'vol_target': min(0.25, avg_volatility * 1.5),
                    'volatility_multiplier': 0.7  # Reduce size in high vol
                },
                'ETHUSDT': {
                    'max_position': min(self.max_single_position, 0.30 - eth_vol * 2),
                    'min_position': 0.04,
                    'vol_target': min(0.28, avg_volatility * 1.6),
                    'volatility_multiplier': 0.6
                }
            }

            # Wide stops to avoid whipsaws in volatile markets
            stop_loss_policy = {
                'type': 'volatility_adjusted_wide',
                'base_stop': 0.05,  # 5% base stop
                'volatility_addon': avg_volatility * 2,  # Add volatility-based buffer
                'max_stop': 0.15,  # Maximum 15% stop
                'time_stop': '10d',  # Exit after 10 days regardless
                'trailing_stop': True
            }

            # Quick profits in volatile conditions
            take_profit_policy = {
                'type': 'volatility_scaled',
                'base_target': 0.08,  # 8% base target
                'volatility_bonus': avg_volatility * 1.5,  # Higher vol allows bigger targets
                'max_target': 0.20,  # Maximum 20% target
                'scale_out': True,  # Scale out of positions
                'time_based_exit': '5d'  # Exit after 5 days
            }

            strategy = RegimeStrategy(
                regime='volatile',
                risk_appetite=risk_appetite,
                asset_allocation=asset_allocation,
                position_sizing=position_sizing,
                stop_loss_policy=stop_loss_policy,
                take_profit_policy=take_profit_policy,
                rebalancing_frequency='daily',  # Daily rebalancing for volatile markets
                volatility_target=min(0.35, avg_volatility * 1.2),  # Target slightly above current vol
                correlation_limits={
                    'max_correlation': 0.4,  # Very strict correlation limits
                    'min_diversification': 0.8,  # Maximum diversification required
                    'volatility_parity': True  # Balance volatility across assets
                },
                metadata={
                    'avg_volatility': avg_volatility,
                    'btc_volatility': btc_vol,
                    'eth_volatility': eth_vol,
                    'btc_drawdown': btc_drawdown,
                    'eth_drawdown': eth_drawdown,
                    'volatility_threshold': self.volatility_threshold,
                    'strategy_type': 'crisis_management_volatility_harvesting',
                    'model_version': '1.0'
                }
            )

            logger.info(f"🌪️ Volatile Market Strategy: Risk={risk_appetite:.2f}, BTC={btc_weight:.2f}, Cash={cash_weight:.2f}, Vol={avg_volatility:.3f}")
            return strategy

        except Exception as e:
            logger.error(f"Volatile Market Model error: {e}")
            return self._get_default_volatile_strategy()

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate realized volatility for crisis detection"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.10

            returns = data['close'].pct_change().dropna().tail(30)
            if len(returns) < 5:
                return 0.10

            # Use realized volatility (not annualized for crisis detection)
            return returns.std()
        except Exception:
            return 0.10

    def _calculate_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate current drawdown from recent high"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.0

            prices = data['close'].tail(30)
            if len(prices) < 5:
                return 0.0

            peak = prices.max()
            current = prices.iloc[-1]

            if peak > 0:
                return (peak - current) / peak
            return 0.0
        except Exception:
            return 0.0

    def _get_default_volatile_strategy(self) -> RegimeStrategy:
        """Return default volatile market strategy"""
        return RegimeStrategy(
            regime='volatile',
            risk_appetite=0.3,
            asset_allocation={'BTC': 0.30, 'ETH': 0.30, 'USDT': 0.40},
            position_sizing={},
            stop_loss_policy={},
            take_profit_policy={},
            rebalancing_frequency='daily',
            volatility_target=0.30,
            correlation_limits={'max_correlation': 0.4, 'min_diversification': 0.8}
        )

class CrisisMarketModel:
    """
    Crisis Market L3 Model - Emergency Preservation Strategy
    Characteristics:
    - Ultra-conservative risk appetite
    - Maximum cash allocation (90%+)
    - Emergency stop losses
    - Immediate profit taking
    - Circuit breaker logic
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "CrisisMarketModel"
        self.crisis_volatility_threshold = self.config.get('crisis_volatility_threshold', 0.15)
        self.crisis_drawdown_threshold = self.config.get('crisis_drawdown_threshold', 0.20)
        self.emergency_cash_allocation = self.config.get('emergency_cash_allocation', 0.95)

    def generate_strategy(self, market_data: Dict[str, pd.DataFrame],
                         regime_context: Dict[str, Any]) -> RegimeStrategy:
        """
        Generate crisis market strategy for emergency situations
        """
        try:
            logger.info("🚨 Crisis Market Model: Generating emergency preservation strategy")

            # Extract market metrics
            btc_data = market_data.get('BTCUSDT')
            eth_data = market_data.get('ETHUSDT')

            if btc_data is None or btc_data.empty:
                logger.warning("Crisis Market Model: Insufficient BTC data")
                return self._get_default_crisis_strategy()

            # Calculate crisis indicators
            btc_vol = self._calculate_volatility(btc_data)
            eth_vol = self._calculate_volatility(eth_data) if eth_data is not None else btc_vol

            btc_drawdown = self._calculate_drawdown(btc_data)
            eth_drawdown = self._calculate_drawdown(eth_data) if eth_data is not None else 0

            avg_volatility = (btc_vol + eth_vol) / 2
            max_drawdown = max(btc_drawdown, eth_drawdown)

            # Crisis detection
            is_crisis = (avg_volatility > self.crisis_volatility_threshold or
                        max_drawdown > self.crisis_drawdown_threshold)

            if is_crisis:
                logger.warning(f"🚨 CRISIS DETECTED: Vol={avg_volatility:.3f} > {self.crisis_volatility_threshold}, Drawdown={max_drawdown:.3f} > {self.crisis_drawdown_threshold}")

            # Emergency allocation: Maximum cash preservation
            cash_allocation = min(0.98, max(self.emergency_cash_allocation,
                                           0.90 + (avg_volatility * 2) + (max_drawdown * 3)))

            # Minimal emergency positions (if any)
            remaining_allocation = 1.0 - cash_allocation
            btc_allocation = remaining_allocation * 0.3  # Very small BTC position
            eth_allocation = remaining_allocation * 0.2  # Very small ETH position
            alt_allocation = remaining_allocation * 0.5   # Alternative safe assets

            asset_allocation = {
                'BTC': btc_allocation,
                'ETH': eth_allocation,
                'USDT': cash_allocation
            }

            # Ultra-conservative risk parameters
            risk_appetite = max(0.05, 0.1 - avg_volatility - max_drawdown)

            # Emergency position sizing - very small positions
            position_sizing = {
                'BTCUSDT': {
                    'max_position': 0.05,  # Maximum 5% position
                    'min_position': 0.01,  # Minimum 1% position
                    'vol_target': 0.05,    # Very low vol target
                    'emergency_mode': True
                },
                'ETHUSDT': {
                    'max_position': 0.03,  # Maximum 3% position
                    'min_position': 0.005, # Minimum 0.5% position
                    'vol_target': 0.06,    # Very low vol target
                    'emergency_mode': True
                }
            }

            # Emergency stop losses - very tight
            stop_loss_policy = {
                'type': 'emergency_circuit_breaker',
                'initial_stop': 0.01,  # 1% initial stop
                'max_drawdown': 0.02,  # Max 2% drawdown
                'time_stop': '1d',     # Exit after 1 day regardless
                'circuit_breaker': True,  # Enable circuit breaker logic
                'panic_sell_threshold': 0.05  # Sell everything if 5% loss
            }

            # Emergency take profits - take any profit immediately
            take_profit_policy = {
                'type': 'emergency_quick_profit',
                'targets': [
                    {'price_level': 1.005, 'position_size': 0.50},  # Take 50% off at 0.5% gain
                    {'price_level': 1.01, 'position_size': 1.00}    # Take all off at 1% gain
                ],
                'time_based_exit': '12h',  # Exit after 12 hours regardless
                'profit_lock': True  # Lock in any profits immediately
            }

            strategy = RegimeStrategy(
                regime='crisis',
                risk_appetite=risk_appetite,
                asset_allocation=asset_allocation,
                position_sizing=position_sizing,
                stop_loss_policy=stop_loss_policy,
                take_profit_policy=take_profit_policy,
                rebalancing_frequency='hourly',  # Very frequent rebalancing in crisis
                volatility_target=0.03,  # Very low vol target
                correlation_limits={
                    'max_correlation': 0.2,  # Extremely strict correlation limits
                    'min_diversification': 0.95,  # Maximum diversification required
                    'crisis_mode': True
                },
                metadata={
                    'avg_volatility': avg_volatility,
                    'btc_volatility': btc_vol,
                    'eth_volatility': eth_vol,
                    'btc_drawdown': btc_drawdown,
                    'eth_drawdown': eth_drawdown,
                    'max_drawdown': max_drawdown,
                    'crisis_detected': is_crisis,
                    'crisis_volatility_threshold': self.crisis_volatility_threshold,
                    'crisis_drawdown_threshold': self.crisis_drawdown_threshold,
                    'strategy_type': 'emergency_preservation_circuit_breaker',
                    'model_version': '1.0'
                }
            )

            logger.info(f"🚨 Crisis Market Strategy: Risk={risk_appetite:.3f}, Cash={cash_allocation:.3f}, Crisis={is_crisis}")
            return strategy

        except Exception as e:
            logger.error(f"Crisis Market Model error: {e}")
            return self._get_default_crisis_strategy()

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate extreme volatility for crisis detection"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.20

            returns = data['close'].pct_change().dropna().tail(10)  # Short-term crisis detection
            if len(returns) < 3:
                return 0.20

            # Use extreme volatility measure
            return returns.std() * 2  # Double the standard deviation for crisis sensitivity
        except Exception:
            return 0.20

    def _calculate_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate severe drawdown for crisis detection"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.30

            prices = data['close'].tail(20)  # Look at recent 20 periods
            if len(prices) < 5:
                return 0.30

            # Calculate maximum drawdown in recent period
            peak = prices.max()
            current = prices.iloc[-1]

            if peak > 0:
                return (peak - current) / peak
            return 0.30
        except Exception:
            return 0.30

    def _get_default_crisis_strategy(self) -> RegimeStrategy:
        """Return default crisis market strategy"""
        return RegimeStrategy(
            regime='crisis',
            risk_appetite=0.05,
            asset_allocation={'BTC': 0.00, 'ETH': 0.00, 'USDT': 1.00},
            position_sizing={},
            stop_loss_policy={},
            take_profit_policy={},
            rebalancing_frequency='hourly',
            volatility_target=0.03,
            correlation_limits={'max_correlation': 0.2, 'min_diversification': 0.95}
        )

class RegimeSpecificL3Processor:
    """
    Main processor for regime-specific L3 models
    Integrates bull, bear, range, volatile, and crisis market strategies
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {
            'bull': BullMarketModel(self.config.get('bull', {})),
            'bear': BearMarketModel(self.config.get('bear', {})),
            'range': RangeMarketModel(self.config.get('range', {})),
            'volatile': VolatileMarketModel(self.config.get('volatile', {})),
            'crisis': CrisisMarketModel(self.config.get('crisis', {}))
        }

    def generate_regime_strategy(self, market_data: Dict[str, pd.DataFrame],
                               regime_context: Dict[str, Any]) -> RegimeStrategy:
        """
        Generate strategy based on detected market regime
        """
        try:
            # Detect current regime
            regime = self._detect_regime(market_data, regime_context)

            logger.info(f"🎯 Regime-Specific L3: Detected regime '{regime}', generating strategy")

            # Get appropriate model
            model = self.models.get(regime)
            if model is None:
                logger.warning(f"No specific model for regime '{regime}', using range model")
                model = self.models['range']

            # Generate strategy
            strategy = model.generate_strategy(market_data, regime_context)

            # Add regime detection metadata
            strategy.metadata.update({
                'detected_regime': regime,
                'regime_confidence': regime_context.get('regime_confidence', 0.5),
                'detection_timestamp': datetime.now().isoformat()
            })

            logger.info(f"✅ Regime-Specific Strategy Generated: {regime} regime, risk_appetite={strategy.risk_appetite:.2f}")
            return strategy

        except Exception as e:
            logger.error(f"Error generating regime-specific strategy: {e}")
            # Return safe default strategy
            return RegimeStrategy(
                regime='neutral',
                risk_appetite=0.5,
                asset_allocation={'BTC': 0.40, 'ETH': 0.30, 'USDT': 0.30},
                position_sizing={},
                stop_loss_policy={},
                take_profit_policy={},
                rebalancing_frequency='weekly',
                volatility_target=0.10,
                correlation_limits={'max_correlation': 0.7, 'min_diversification': 0.4}
            )

    def _detect_regime(self, market_data: Dict[str, pd.DataFrame],
                      regime_context: Dict[str, Any]) -> str:
        """
        Detect current market regime using multiple signals including volatility and crisis detection
        Prioritizes regime from context if provided (from rule-based classifier in main.py)
        """
        try:
            # FIRST PRIORITY: Get regime from context if available (from rule-based classifier)
            if regime_context and 'regime' in regime_context:
                context_regime = regime_context['regime']
                logger.info(f"🎯 Using regime from context: {context_regime} (rule-based detection)")
                return context_regime

            # Enhanced regime detection using market data
            btc_data = market_data.get('BTCUSDT')
            eth_data = market_data.get('ETHUSDT')

            if btc_data is None or btc_data.empty:
                return 'range'

            # Calculate comprehensive market metrics
            prices = btc_data['close'].tail(30)
            if len(prices) < 10:
                return 'range'

            returns = prices.pct_change().dropna()
            trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
            volatility = returns.std()

            # Calculate additional crisis indicators
            drawdown = self._calculate_drawdown(btc_data)
            eth_drawdown = self._calculate_drawdown(eth_data) if eth_data is not None else 0
            max_drawdown = max(drawdown, eth_drawdown)

            # Calculate volatility for both assets
            btc_vol = volatility
            eth_vol = self._calculate_volatility(eth_data) if eth_data is not None else btc_vol
            avg_volatility = (btc_vol + eth_vol) / 2

            # 🛠️ AJUSTE CRÍTICO: Thresholds más altos para mercados crypto (menos conservadores)
            crisis_vol_threshold = 0.25  # Increased from 15% to 25% for crypto volatility
            crisis_dd_threshold = 0.30   # Increased from 20% to 30% for crypto drawdowns

            if avg_volatility > crisis_vol_threshold or max_drawdown > crisis_dd_threshold:
                logger.warning(f"🚨 CRISIS REGIME DETECTED: Vol={avg_volatility:.3f} > {crisis_vol_threshold}, DD={max_drawdown:.3f} > {crisis_dd_threshold}")
                return 'crisis'

            # VOLATILE DETECTION: High volatility but not crisis level (threshold más alto)
            volatile_vol_threshold = 0.12  # Increased from 8% to 12% for crypto markets
            if avg_volatility > volatile_vol_threshold:
                logger.info(f"🌪️ VOLATILE REGIME DETECTED: Vol={avg_volatility:.3f} > {volatile_vol_threshold}")
                return 'volatile'

            # Standard regime classification
            logger.info(f"📊 REGIME CLASSIFICATION: trend={trend:.3f}, volatility={volatility:.3f}, avg_vol={avg_volatility:.3f}, max_dd={max_drawdown:.3f}")

            if trend > 0.05 and volatility < 0.03:
                logger.info("🐂 BULL REGIME: Strong upward trend with low volatility")
                return 'bull'
            elif trend < -0.05:
                logger.info("🐻 BEAR REGIME: Strong downward trend")
                return 'bear'
            else:
                logger.info("📊 RANGE REGIME: Sideways market or mixed signals")
                return 'range'

        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'range'

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility for regime detection"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.05

            returns = data['close'].pct_change().dropna().tail(20)
            if len(returns) < 5:
                return 0.05

            return returns.std()
        except Exception:
            return 0.05

    def _calculate_drawdown(self, data: pd.DataFrame) -> float:
        """Calculate drawdown for crisis detection"""
        try:
            if data is None or data.empty or 'close' not in data.columns:
                return 0.0

            prices = data['close'].tail(20)
            if len(prices) < 5:
                return 0.0

            peak = prices.max()
            current = prices.iloc[-1]

            if peak > 0:
                return (peak - current) / peak
            return 0.0
        except Exception:
            return 0.0

    def get_model_health(self) -> Dict[str, Any]:
        """Check health of all regime models"""
        health = {}
        for regime, model in self.models.items():
            try:
                # Basic health check - model exists and has required methods
                has_generate = hasattr(model, 'generate_strategy')
                health[regime] = {
                    'status': 'healthy' if has_generate else 'error',
                    'model_name': model.name if hasattr(model, 'name') else 'unknown'
                }
            except Exception as e:
                health[regime] = {'status': 'error', 'error': str(e)}

        return {
            'overall_status': 'healthy' if all(h['status'] == 'healthy' for h in health.values()) else 'degraded',
            'models': health,
            'timestamp': datetime.now().isoformat()
        }
