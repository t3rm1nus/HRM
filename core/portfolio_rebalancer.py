"""
Portfolio Rebalancer

This module handles portfolio rebalancing logic to maintain target allocations,
including threshold-based rebalancing, calendar-based rebalancing, and trade execution.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from core.logging import logger
from l2_tactic.l2_utils import safe_float


class RebalanceTrigger(Enum):
    """Types of rebalance triggers"""
    THRESHOLD_BASED = "threshold_based"  # Rebalance when drift exceeds threshold
    CALENDAR_BASED = "calendar_based"   # Rebalance on fixed schedule
    MANUAL = "manual"                   # Manual rebalance trigger
    VOLATILITY_BASED = "volatility_based"  # Rebalance based on volatility changes
    CORRELATION_BASED = "correlation_based"  # Rebalance based on correlation changes


@dataclass
class RebalanceTrade:
    """Represents a trade needed for rebalancing"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    estimated_price: float
    estimated_value: float
    reason: str
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class RebalanceResult:
    """Result of a rebalancing operation"""
    success: bool
    trades_required: List[RebalanceTrade]
    total_value_change: float
    execution_status: str
    timestamp: str
    metadata: Dict[str, Any]


class PortfolioRebalancer:
    """
    Handles portfolio rebalancing to maintain target allocations
    """

    def __init__(self, weight_calculator, drift_threshold: float = 0.10,
                 transaction_costs: float = 0.001, min_trade_value: float = 10.0,
                 min_position_percentage: float = 0.10, rebalance_enabled: bool = True,
                 partial_rebalance_factor: float = 0.5):  # Nuevo parámetro para rebalance parcial
        """
        Initialize portfolio rebalancer

        Args:
            weight_calculator: WeightCalculator instance for target weights
            drift_threshold: Maximum allowed drift from target weights (10%)
            transaction_costs: Transaction cost as fraction (0.1%)
            min_trade_value: Minimum trade value in USD
            min_position_percentage: Minimum position size to maintain (10%)
            rebalance_enabled: Whether rebalancing is enabled (default: True)
            partial_rebalance_factor: Factor to scale down rebalance trades (0.0 to 1.0, 0.5 = 50% of full rebalance)
        """
        self.weight_calculator = weight_calculator
        self.drift_threshold = drift_threshold
        self.transaction_costs = transaction_costs
        self.min_trade_value = min_trade_value
        self.min_position_percentage = min_position_percentage  # Don't sell positions below this % of portfolio
        self.rebalance_enabled = rebalance_enabled
        self.partial_rebalance_factor = partial_rebalance_factor  # Factor para rebalance parcial

        # Rebalancing state
        self.last_rebalance = None
        self.target_weights = {}
        self.rebalance_history = []

        logger.info("🔄 Portfolio Rebalancer initialized")
        if not self.rebalance_enabled:
            logger.warning("⚠️ Portfolio rebalancing is DISABLED - only pure trades will be executed")
        if self.partial_rebalance_factor < 1.0:
            logger.info(f"📊 Partial rebalance enabled with factor: {self.partial_rebalance_factor:.1%}")

    def should_rebalance(self, current_weights: Dict[str, float],
                        trigger: RebalanceTrigger) -> Tuple[bool, str]:
        """
        Determine if portfolio rebalancing is needed

        Args:
            current_weights: Current portfolio weights
            trigger: Type of rebalance trigger to check

        Returns:
            Tuple of (should_rebalance, reason)
        """
        try:
            # Check if rebalancing is enabled
            if not self.rebalance_enabled:
                return False, "Rebalancing is disabled"
                
            if trigger == RebalanceTrigger.THRESHOLD_BASED:
                return self._check_threshold_rebalance(current_weights)
            elif trigger == RebalanceTrigger.CALENDAR_BASED:
                return self._check_calendar_rebalance()
            elif trigger == RebalanceTrigger.VOLATILITY_BASED:
                return self._check_volatility_rebalance(current_weights)
            elif trigger == RebalanceTrigger.CORRELATION_BASED:
                return self._check_correlation_rebalance(current_weights)
            elif trigger == RebalanceTrigger.MANUAL:
                return True, "Manual rebalance requested"
            else:
                return False, f"Unknown trigger type: {trigger}"

        except Exception as e:
            logger.error(f"❌ Error checking rebalance need: {e}")
            return False, f"Error: {str(e)}"

    def _check_threshold_rebalance(self, current_weights: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if drift from target weights exceeds threshold

        Args:
            current_weights: Current portfolio weights

        Returns:
            Tuple of (should_rebalance, reason)
        """
        try:
            if not self.target_weights:
                return False, "No target weights set"

            max_drift = 0.0
            max_drift_asset = None

            for symbol, target_weight in self.target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)
                drift = abs(current_weight - target_weight)

                if drift > max_drift:
                    max_drift = drift
                    max_drift_asset = symbol

            if max_drift > self.drift_threshold:
                return True, f"Max drift {max_drift:.1%} on {max_drift_asset} exceeds threshold {self.drift_threshold:.1%}"
            else:
                return False, f"Max drift {max_drift:.1%} within threshold {self.drift_threshold:.1%}"

        except Exception as e:
            logger.error(f"❌ Error checking threshold rebalance: {e}")
            return False, f"Error: {str(e)}"

    def _check_calendar_rebalance(self) -> Tuple[bool, str]:
        """
        Check if calendar-based rebalancing is due

        Returns:
            Tuple of (should_rebalance, reason)
        """
        try:
            if not self.last_rebalance:
                return True, "No previous rebalance recorded"

            # Check if it's been more than 30 days since last rebalance
            last_rebalance_time = pd.Timestamp(self.last_rebalance)
            days_since_rebalance = (pd.Timestamp.now() - last_rebalance_time).days

            if days_since_rebalance >= 30:
                return True, f"{days_since_rebalance} days since last rebalance (threshold: 30 days)"
            else:
                return False, f"Only {days_since_rebalance} days since last rebalance"

        except Exception as e:
            logger.error(f"❌ Error checking calendar rebalance: {e}")
            return False, f"Error: {str(e)}"

    def _check_volatility_rebalance(self, current_weights: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if volatility-based rebalancing is needed

        Args:
            current_weights: Current portfolio weights

        Returns:
            Tuple of (should_rebalance, reason)
        """
        try:
            # Get current portfolio volatility
            risk_metrics = self.weight_calculator.get_portfolio_risk_metrics(current_weights)
            current_vol = risk_metrics.get('volatility', 0.0)

            # Check if volatility has changed significantly (more than 20%)
            if hasattr(self, 'last_volatility') and self.last_volatility > 0:
                vol_change = abs(current_vol - self.last_volatility) / self.last_volatility

                if vol_change > 0.2:  # 20% change
                    self.last_volatility = current_vol
                    return True, f"Volatility changed {vol_change:.1%} from {self.last_volatility:.1%} to {current_vol:.1%}"
                else:
                    return False, f"Volatility change {vol_change:.1%} within threshold"
            else:
                # First time checking
                self.last_volatility = current_vol
                return False, "First volatility check - no baseline"

        except Exception as e:
            logger.error(f"❌ Error checking volatility rebalance: {e}")
            return False, f"Error: {str(e)}"

    def _check_correlation_rebalance(self, current_weights: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if correlation-based rebalancing is needed

        Args:
            current_weights: Current portfolio weights

        Returns:
            Tuple of (should_rebalance, reason)
        """
        try:
            # Check if correlations have changed significantly
            # This would require tracking historical correlations
            # For now, use a simple check based on correlation metrics

            if hasattr(self.weight_calculator, 'correlation_sizer'):
                correlation_report = self.weight_calculator.correlation_sizer.get_correlation_report(current_weights)
                avg_correlation = correlation_report.get('correlation_risk_metrics', {}).get('average_correlation', 0.5)

                # Check if average correlation has changed significantly
                if hasattr(self, 'last_avg_correlation'):
                    corr_change = abs(avg_correlation - self.last_avg_correlation)

                    if corr_change > 0.1:  # 10% change in average correlation
                        self.last_avg_correlation = avg_correlation
                        return True, f"Average correlation changed {corr_change:.1%} to {avg_correlation:.2f}"
                    else:
                        return False, f"Correlation change {corr_change:.1%} within threshold"
                else:
                    # First time checking
                    self.last_avg_correlation = avg_correlation
                    return False, "First correlation check - no baseline"
            else:
                return False, "No correlation sizer available"

        except Exception as e:
            logger.error(f"❌ Error checking correlation rebalance: {e}")
            return False, f"Error: {str(e)}"

    async def execute_rebalance(self, current_weights: Dict[str, float],
                              portfolio_value: float, market_data: Dict[str, Any],
                              trigger: RebalanceTrigger) -> RebalanceResult:
        """
        Execute portfolio rebalancing

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Current portfolio value in USD
            market_data: Current market data
            trigger: Rebalance trigger type

        Returns:
            RebalanceResult with execution details
        """
        try:
            logger.info(f"🔄 Executing portfolio rebalance (trigger: {trigger.value})")

            # Calculate required trades
            trades_required = self._calculate_rebalance_trades(
                current_weights, portfolio_value, market_data
            )

            # Filter out small trades
            significant_trades = [
                trade for trade in trades_required
                if trade.estimated_value >= self.min_trade_value
            ]

            if not significant_trades:
                result = RebalanceResult(
                    success=True,
                    trades_required=[],
                    total_value_change=0.0,
                    execution_status="No significant trades required",
                    timestamp=pd.Timestamp.now().isoformat(),
                    metadata={'reason': 'all_trades_below_minimum'}
                )
                logger.info("ℹ️ Rebalance completed - no significant trades required")
                return result

            # Calculate total value change
            total_value_change = sum(trade.estimated_value for trade in significant_trades)

            # Estimate transaction costs
            total_costs = total_value_change * self.transaction_costs

            # Check if rebalance is cost-effective
            if total_costs > portfolio_value * 0.001:  # Costs > 0.1% of portfolio
                logger.warning(f"⚠️ Rebalance costs (${total_costs:.2f}) may not be justified")

            # Update rebalance state
            self.last_rebalance = pd.Timestamp.now().isoformat()
            self.rebalance_history.append({
                'timestamp': self.last_rebalance,
                'trigger': trigger.value,
                'trades': len(significant_trades),
                'total_value_change': total_value_change,
                'portfolio_value': portfolio_value
            })

            result = RebalanceResult(
                success=True,
                trades_required=significant_trades,
                total_value_change=total_value_change,
                execution_status="Trades calculated successfully",
                timestamp=self.last_rebalance,
                metadata={
                    'transaction_costs': total_costs,
                    'portfolio_value': portfolio_value,
                    'trigger': trigger.value
                }
            )

            logger.info(f"✅ Rebalance executed: {len(significant_trades)} trades, ${total_value_change:.2f} value change")
            return result

        except Exception as e:
            logger.error(f"❌ Error executing rebalance: {e}")
            result = RebalanceResult(
                success=False,
                trades_required=[],
                total_value_change=0.0,
                execution_status=f"Error: {str(e)}",
                timestamp=pd.Timestamp.now().isoformat(),
                metadata={'error': str(e)}
            )
            return result

    def _calculate_rebalance_trades(self, current_weights: Dict[str, float],
                                  portfolio_value: float, market_data: Dict[str, Any],
                                  available_usdt: float = None, partial: bool = False) -> List[RebalanceTrade]:
        """
        Calculate the trades needed to rebalance to target weights
        with minimum position size protection and USDT balance constraints

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Current portfolio value
            market_data: Current market data
            available_usdt: Available USDT balance for paper mode
            partial: Whether to use partial rebalance factor

        Returns:
            List of RebalanceTrade objects
        """
        try:
            trades = []
            min_position_value = portfolio_value * self.min_position_percentage
            
            # If available USDT not provided, estimate from current portfolio
            if available_usdt is None:
                available_usdt = portfolio_value * (1 - sum(current_weights.values()))
            
            logger.debug(f"📊 Rebalance: Available USDT for buys: ${available_usdt:.2f}")

            # Calculate total buy value needed
            total_buy_value = 0.0
            buy_trades = []
            
            for symbol, target_weight in self.target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)
                weight_diff = target_weight - current_weight

                if abs(weight_diff) < 0.001:  # Less than 0.1% difference
                    continue

                # Calculate value to trade
                trade_value = weight_diff * portfolio_value

                # Apply partial rebalance factor if requested
                if partial and self.partial_rebalance_factor < 1.0:
                    trade_value *= self.partial_rebalance_factor
                    logger.debug(f"📊 Partial rebalance: Scaling trade value for {symbol} by {self.partial_rebalance_factor:.1%}")

                # Determine trade side
                if weight_diff > 0:
                    side = 'buy'
                else:
                    side = 'sell'

                # APPLY MINIMUM POSITION SIZE PROTECTION
                # Prevent selling positions that would fall below minimum size
                if side == 'sell':
                    current_value = current_weight * portfolio_value
                    new_value = current_value + trade_value  # trade_value is negative for sells

                    if new_value < min_position_value:
                        if current_value >= min_position_value:
                            # Adjust trade to maintain minimum position size
                            max_sell_value = current_value - min_position_value
                            if max_sell_value > 0.001 * portfolio_value:  # Only if meaningful
                                trade_value = -max_sell_value  # Negative because it's a sell
                                logger.info(f"🛡️ MIN POSITION SIZE: Adjusting {symbol} sell from ${-weight_diff * portfolio_value:.2f} to ${-trade_value:.2f} to maintain {self.min_position_percentage:.1%} minimum")
                            else:
                                logger.info(f"🛡️ MIN POSITION SIZE: Skipping {symbol} sell - would reduce position below {self.min_position_percentage:.1%} threshold")
                                continue  # Skip this trade entirely
                        else:
                            # Position already below minimum, don't sell
                            logger.info(f"🛡️ MIN POSITION SIZE: Skipping {symbol} sell - position already below {self.min_position_percentage:.1%} threshold")
                            continue

                # Get current price for the asset
                price = self._get_asset_price(symbol, market_data)
                if price <= 0:
                    logger.warning(f"⚠️ Could not get price for {symbol}, skipping")
                    continue

                # Calculate quantity to trade
                quantity = abs(trade_value) / price

                # Determine priority based on weight difference
                priority = 1 if abs(weight_diff) > 0.05 else (2 if abs(weight_diff) > 0.02 else 3)

                trade = RebalanceTrade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    estimated_price=price,
                    estimated_value=abs(trade_value),
                    reason=f"Rebalance to target weight {target_weight:.1%} (current: {current_weight:.1%})" + (" - Partial" if partial else ""),
                    priority=priority
                )

                if side == 'buy':
                    buy_trades.append(trade)
                    total_buy_value += trade.estimated_value
                else:
                    trades.append(trade)  # Sell trades can be executed immediately

            # Handle buy trades with USDT constraint
            if buy_trades:
                if total_buy_value > available_usdt:
                    logger.warning(f"⚠️ Rebalance: Total buy value ${total_buy_value:.2f} exceeds available USDT ${available_usdt:.2f}")
                    # Scale down buy trades proportionally
                    scale_factor = available_usdt / total_buy_value
                    logger.info(f"📊 Rebalance: Scaling buy trades by {scale_factor:.2%}")
                    for trade in buy_trades:
                        trade.quantity *= scale_factor
                        trade.estimated_value *= scale_factor
                        logger.debug(f"   {trade.symbol}: {trade.quantity:.6f} units (${trade.estimated_value:.2f})")
                else:
                    logger.info(f"✅ Rebalance: Total buy value ${total_buy_value:.2f} within available USDT ${available_usdt:.2f}")
                
                trades.extend(buy_trades)

            # Sort trades by priority (high priority first)
            trades.sort(key=lambda x: x.priority)

            logger.debug(f"📊 Calculated {len(trades)} rebalance trades (after position size and USDT constraints)")
            return trades

        except Exception as e:
            logger.error(f"❌ Error calculating rebalance trades: {e}")
            return []

    async def execute_partial_rebalance(self, current_weights: Dict[str, float],
                                     portfolio_value: float, market_data: Dict[str, Any],
                                     trigger: RebalanceTrigger,
                                     available_usdt: float = None) -> RebalanceResult:
        """
        Execute partial portfolio rebalancing using the configured partial rebalance factor

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Current portfolio value in USD
            market_data: Current market data
            trigger: Rebalance trigger type
            available_usdt: Available USDT balance for paper mode

        Returns:
            RebalanceResult with execution details
        """
        try:
            logger.info(f"🔄 Executing partial portfolio rebalance (trigger: {trigger.value}, factor: {self.partial_rebalance_factor:.1%})")

            # Calculate required trades with partial rebalance factor
            trades_required = self._calculate_rebalance_trades(
                current_weights, portfolio_value, market_data, available_usdt, partial=True
            )

            # Filter out small trades
            significant_trades = [
                trade for trade in trades_required
                if trade.estimated_value >= self.min_trade_value
            ]

            if not significant_trades:
                result = RebalanceResult(
                    success=True,
                    trades_required=[],
                    total_value_change=0.0,
                    execution_status="No significant partial rebalance trades required",
                    timestamp=pd.Timestamp.now().isoformat(),
                    metadata={'reason': 'all_trades_below_minimum', 'partial': True}
                )
                logger.info("ℹ️ Partial rebalance completed - no significant trades required")
                return result

            # Calculate total value change
            total_value_change = sum(trade.estimated_value for trade in significant_trades)

            # Estimate transaction costs
            total_costs = total_value_change * self.transaction_costs

            # Check if rebalance is cost-effective
            if total_costs > portfolio_value * 0.001:  # Costs > 0.1% of portfolio
                logger.warning(f"⚠️ Partial rebalance costs (${total_costs:.2f}) may not be justified")

            # Update rebalance state
            self.last_rebalance = pd.Timestamp.now().isoformat()
            self.rebalance_history.append({
                'timestamp': self.last_rebalance,
                'trigger': trigger.value,
                'trades': len(significant_trades),
                'total_value_change': total_value_change,
                'portfolio_value': portfolio_value,
                'partial': True
            })

            result = RebalanceResult(
                success=True,
                trades_required=significant_trades,
                total_value_change=total_value_change,
                execution_status="Partial rebalance trades calculated successfully",
                timestamp=self.last_rebalance,
                metadata={
                    'transaction_costs': total_costs,
                    'portfolio_value': portfolio_value,
                    'trigger': trigger.value,
                    'partial': True,
                    'partial_factor': self.partial_rebalance_factor
                }
            )

            logger.info(f"✅ Partial rebalance executed: {len(significant_trades)} trades, ${total_value_change:.2f} value change")
            return result

        except Exception as e:
            logger.error(f"❌ Error executing partial rebalance: {e}")
            result = RebalanceResult(
                success=False,
                trades_required=[],
                total_value_change=0.0,
                execution_status=f"Error: {str(e)}",
                timestamp=pd.Timestamp.now().isoformat(),
                metadata={'error': str(e), 'partial': True}
            )
            return result

    async def execute_rebalance(self, current_weights: Dict[str, float],
                              portfolio_value: float, market_data: Dict[str, Any],
                              trigger: RebalanceTrigger,
                              available_usdt: float = None,
                              partial: bool = False) -> RebalanceResult:
        """
        Execute portfolio rebalancing (full or partial)

        Args:
            current_weights: Current portfolio weights
            portfolio_value: Current portfolio value in USD
            market_data: Current market data
            trigger: Rebalance trigger type
            available_usdt: Available USDT balance for paper mode
            partial: Whether to use partial rebalance factor

        Returns:
            RebalanceResult with execution details
        """
        if partial:
            return await self.execute_partial_rebalance(
                current_weights, portfolio_value, market_data, trigger, available_usdt
            )

        try:
            logger.info(f"🔄 Executing full portfolio rebalance (trigger: {trigger.value})")

            # Calculate required trades
            trades_required = self._calculate_rebalance_trades(
                current_weights, portfolio_value, market_data, available_usdt, partial=False
            )

            # Filter out small trades
            significant_trades = [
                trade for trade in trades_required
                if trade.estimated_value >= self.min_trade_value
            ]

            if not significant_trades:
                result = RebalanceResult(
                    success=True,
                    trades_required=[],
                    total_value_change=0.0,
                    execution_status="No significant trades required",
                    timestamp=pd.Timestamp.now().isoformat(),
                    metadata={'reason': 'all_trades_below_minimum', 'partial': False}
                )
                logger.info("ℹ️ Rebalance completed - no significant trades required")
                return result

            # Calculate total value change
            total_value_change = sum(trade.estimated_value for trade in significant_trades)

            # Estimate transaction costs
            total_costs = total_value_change * self.transaction_costs

            # Check if rebalance is cost-effective
            if total_costs > portfolio_value * 0.001:  # Costs > 0.1% of portfolio
                logger.warning(f"⚠️ Rebalance costs (${total_costs:.2f}) may not be justified")

            # Update rebalance state
            self.last_rebalance = pd.Timestamp.now().isoformat()
            self.rebalance_history.append({
                'timestamp': self.last_rebalance,
                'trigger': trigger.value,
                'trades': len(significant_trades),
                'total_value_change': total_value_change,
                'portfolio_value': portfolio_value,
                'partial': False
            })

            result = RebalanceResult(
                success=True,
                trades_required=significant_trades,
                total_value_change=total_value_change,
                execution_status="Trades calculated successfully",
                timestamp=self.last_rebalance,
                metadata={
                    'transaction_costs': total_costs,
                    'portfolio_value': portfolio_value,
                    'trigger': trigger.value,
                    'partial': False
                }
            )

            logger.info(f"✅ Rebalance executed: {len(significant_trades)} trades, ${total_value_change:.2f} value change")
            return result

        except Exception as e:
            logger.error(f"❌ Error executing rebalance: {e}")
            result = RebalanceResult(
                success=False,
                trades_required=[],
                total_value_change=0.0,
                execution_status=f"Error: {str(e)}",
                timestamp=pd.Timestamp.now().isoformat(),
                metadata={'error': str(e), 'partial': False}
            )
            return result

    def _get_asset_price(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """
        Get current price for an asset from market data

        Args:
            symbol: Asset symbol
            market_data: Market data dictionary

        Returns:
            Asset price or 0 if not found
        """
        try:
            if symbol not in market_data:
                return 0.0

            symbol_data = market_data[symbol]

            # Handle different data formats
            if isinstance(symbol_data, dict) and 'close' in symbol_data:
                return safe_float(symbol_data['close'])
            elif isinstance(symbol_data, (pd.Series, pd.DataFrame)) and len(symbol_data) > 0:
                if isinstance(symbol_data, pd.DataFrame):
                    return safe_float(symbol_data['close'].iloc[-1])
                else:  # Series
                    return safe_float(symbol_data.iloc[-1])

            return 0.0

        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return 0.0

    def set_target_weights(self, target_weights: Dict[str, float]) -> bool:
        """
        Set target weights for rebalancing

        Args:
            target_weights: Dictionary of target weights

        Returns:
            Success status
        """
        try:
            # Validate weights sum to 1
            total_weight = sum(target_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                logger.warning(f"⚠️ Target weights sum to {total_weight:.4f}, normalizing to 1.0")
                # Normalize weights
                target_weights = {symbol: weight / total_weight for symbol, weight in target_weights.items()}

            self.target_weights = target_weights.copy()
            logger.info(f"🎯 Target weights set for {len(target_weights)} assets")
            return True

        except Exception as e:
            logger.error(f"❌ Error setting target weights: {e}")
            return False

    def get_rebalance_status(self) -> Dict[str, Any]:
        """
        Get current rebalancing status

        Returns:
            Status dictionary
        """
        try:
            status = {
                'last_rebalance': self.last_rebalance,
                'target_weights_set': bool(self.target_weights),
                'num_target_assets': len(self.target_weights),
                'drift_threshold': self.drift_threshold,
                'total_rebalances': len(self.rebalance_history),
                'rebalance_enabled': self.rebalance_enabled
            }

            # Calculate days since last rebalance
            if self.last_rebalance:
                last_rebalance_time = pd.Timestamp(self.last_rebalance)
                days_since = (pd.Timestamp.now() - last_rebalance_time).days
                status['days_since_last_rebalance'] = days_since

            return status

        except Exception as e:
            logger.error(f"❌ Error getting rebalance status: {e}")
            return {'error': str(e)}

    def get_rebalance_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get rebalancing history

        Args:
            limit: Maximum number of records to return

        Returns:
            List of rebalance records
        """
        try:
            return self.rebalance_history[-limit:] if limit > 0 else self.rebalance_history
        except Exception as e:
            logger.error(f"❌ Error getting rebalance history: {e}")
            return []

    def calculate_rebalance_impact(self, trades: List[RebalanceTrade],
                                 current_weights: Dict[str, float],
                                 portfolio_value: float) -> Dict[str, Any]:
        """
        Calculate the impact of rebalancing trades

        Args:
            trades: List of rebalance trades
            current_weights: Current portfolio weights
            portfolio_value: Current portfolio value

        Returns:
            Impact analysis dictionary
        """
        try:
            impact = {
                'current_portfolio_value': portfolio_value,
                'total_trade_value': 0.0,
                'expected_new_value': portfolio_value,
                'weight_changes': {},
                'transaction_costs': 0.0,
                'net_impact': 0.0
            }

            # Simulate weight changes
            new_weights = current_weights.copy()

            for trade in trades:
                symbol = trade.symbol
                trade_value = trade.estimated_value

                impact['total_trade_value'] += trade_value
                impact['transaction_costs'] += trade_value * self.transaction_costs

                # Calculate weight change
                weight_change = trade_value / portfolio_value
                if trade.side == 'buy':
                    new_weights[symbol] = new_weights.get(symbol, 0.0) + weight_change
                else:  # sell
                    new_weights[symbol] = new_weights.get(symbol, 0.0) - weight_change

                impact['weight_changes'][symbol] = {
                    'old_weight': current_weights.get(symbol, 0.0),
                    'new_weight': new_weights.get(symbol, 0.0),
                    'change': weight_change if trade.side == 'buy' else -weight_change
                }

            # Calculate net impact (after costs)
            impact['net_impact'] = impact['total_trade_value'] - impact['transaction_costs']

            # Calculate expected portfolio value after rebalance
            # This is approximate as it doesn't account for price changes during execution
            impact['expected_new_value'] = portfolio_value - impact['transaction_costs']

            logger.info(f"📊 Rebalance impact calculated: ${impact['total_trade_value']:.2f} trades, ${impact['transaction_costs']:.2f} costs")
            return impact

        except Exception as e:
            logger.error(f"❌ Error calculating rebalance impact: {e}")
            return {'error': str(e)}

    def optimize_rebalance_schedule(self, current_weights: Dict[str, float],
                                  market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize rebalancing schedule based on market conditions

        Args:
            current_weights: Current portfolio weights
            market_conditions: Current market conditions

        Returns:
            Optimization recommendations
        """
        try:
            recommendations = {
                'recommended_trigger': RebalanceTrigger.THRESHOLD_BASED.value,
                'recommended_threshold': self.drift_threshold,
                'next_rebalance_estimate': None,
                'reasoning': []
            }

            # Analyze market volatility
            volatility = market_conditions.get('volatility', 0.15)
            if volatility > 0.25:  # High volatility
                recommendations['recommended_threshold'] = min(self.drift_threshold * 1.5, 0.10)  # More tolerant
                recommendations['reasoning'].append("High market volatility - increasing drift threshold")
            elif volatility < 0.10:  # Low volatility
                recommendations['recommended_threshold'] = max(self.drift_threshold * 0.8, 0.02)  # Less tolerant
                recommendations['reasoning'].append("Low market volatility - decreasing drift threshold")

            # Analyze trading costs
            avg_spread = market_conditions.get('average_spread', 0.001)
            if avg_spread > 0.005:  # High spreads
                recommendations['recommended_trigger'] = RebalanceTrigger.CALENDAR_BASED.value
                recommendations['reasoning'].append("High trading costs - switching to calendar-based rebalancing")

            # Estimate next rebalance
            max_drift = self._calculate_max_drift(current_weights)
            if max_drift > 0:
                drift_rate = max_drift / 30  # Assume 30 days to reach current drift
                days_to_threshold = (recommendations['recommended_threshold'] - max_drift) / drift_rate
                if days_to_threshold > 0:
                    recommendations['next_rebalance_estimate'] = pd.Timestamp.now() + timedelta(days=days_to_threshold)

            return recommendations

        except Exception as e:
            logger.error(f"❌ Error optimizing rebalance schedule: {e}")
            return {'error': str(e)}

    def _calculate_max_drift(self, current_weights: Dict[str, float]) -> float:
        """
        Calculate maximum drift from target weights

        Args:
            current_weights: Current portfolio weights

        Returns:
            Maximum absolute drift
        """
        try:
            max_drift = 0.0

            for symbol, target_weight in self.target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)
                drift = abs(current_weight - target_weight)
                max_drift = max(max_drift, drift)

            return max_drift

        except Exception as e:
            logger.error(f"❌ Error calculating max drift: {e}")
            return 0.0


# Utility functions for rebalancing

def calculate_rebalance_efficiency(trades: List[RebalanceTrade],
                                 transaction_costs: float,
                                 portfolio_value: float) -> Dict[str, float]:
    """
    Calculate rebalancing efficiency metrics

    Args:
        trades: List of rebalance trades
        transaction_costs: Transaction cost rate
        portfolio_value: Portfolio value

    Returns:
        Efficiency metrics
    """
    try:
        if not trades:
            return {'efficiency': 1.0, 'cost_ratio': 0.0}

        total_trade_value = sum(trade.estimated_value for trade in trades)
        total_costs = total_trade_value * transaction_costs

        # Efficiency = (trade value) / (trade value + costs)
        efficiency = total_trade_value / (total_trade_value + total_costs) if total_trade_value > 0 else 1.0

        # Cost ratio = costs / portfolio value
        cost_ratio = total_costs / portfolio_value if portfolio_value > 0 else 0.0

        return {
            'efficiency': efficiency,
            'cost_ratio': cost_ratio,
            'total_trade_value': total_trade_value,
            'total_costs': total_costs
        }

    except Exception as e:
        logger.error(f"❌ Error calculating rebalance efficiency: {e}")
        return {'efficiency': 0.0, 'cost_ratio': 0.0}


def prioritize_rebalance_trades(trades: List[RebalanceTrade],
                              max_trades: int = 10) -> List[RebalanceTrade]:
    """
    Prioritize rebalance trades based on impact and urgency

    Args:
        trades: List of rebalance trades
        max_trades: Maximum number of trades to return

    Returns:
        Prioritized list of trades
    """
    try:
        # Sort by priority (ascending - 1 is highest), then by trade value (descending)
        prioritized = sorted(trades, key=lambda x: (x.priority, -x.estimated_value))

        return prioritized[:max_trades]

    except Exception as e:
        logger.error(f"❌ Error prioritizing rebalance trades: {e}")
        return trades[:max_trades] if trades else []


def validate_rebalance_trades(trades: List[RebalanceTrade],
                            portfolio_value: float,
                            min_trade_ratio: float = 0.001) -> Dict[str, Any]:
    """
    Validate rebalance trades for reasonableness

    Args:
        trades: List of rebalance trades
        portfolio_value: Portfolio value
        min_trade_ratio: Minimum trade size as ratio of portfolio

    Returns:
        Validation results
    """
    try:
        validation = {
            'valid': True,
            'issues': [],
            'total_trade_ratio': 0.0,
            'large_trades': [],
            'small_trades': []
        }

        total_trade_value = sum(trade.estimated_value for trade in trades)
        validation['total_trade_ratio'] = total_trade_value / portfolio_value if portfolio_value > 0 else 0.0

        min_trade_value = portfolio_value * min_trade_ratio

        for trade in trades:
            if trade.estimated_value < min_trade_value:
                validation['small_trades'].append(trade.symbol)
                validation['issues'].append(f"Trade for {trade.symbol} too small: ${trade.estimated_value:.2f}")

            if trade.estimated_value > portfolio_value * 0.1:  # More than 10% of portfolio
                validation['large_trades'].append(trade.symbol)
                validation['issues'].append(f"Trade for {trade.symbol} very large: ${trade.estimated_value:.2f}")

        if validation['issues']:
            validation['valid'] = False

        return validation

    except Exception as e:
        logger.error(f"❌ Error validating rebalance trades: {e}")
        return {'valid': False, 'issues': [str(e)]}
