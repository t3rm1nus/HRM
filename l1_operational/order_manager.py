# l1_operational/order_manager.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
import time


from core.logging import logger, log_trading_action
from core.config import HRM_PATH_MODE, PATH3_SIGNAL_SOURCE, MAX_CONTRA_ALLOCATION_PATH2
from .config import ConfigObject
from l2_tactic.models import TacticalSignal

class OrderManager:
    # Minimum order size in USD - REDUCED FOR BETTER SIGNAL EXECUTION
    MIN_ORDER_SIZE = 5.0  # $5 mínimo para órdenes válidas (reducido de $10)
    MIN_ORDER_USDT = 5.0  # Mínimo $5 USDT para órdenes

    def __init__(self, binance_client=None, market_data=None):
        """Initialize OrderManager."""
        self.config = ConfigObject
        self.binance_client = binance_client
        self.market_data = market_data or {}
        self.active_orders = {}
        self.execution_stats = {}
        self.risk_limits = ConfigObject.RISK_LIMITS
        self.portfolio_limits = ConfigObject.PORTFOLIO_LIMITS
        self.execution_config = ConfigObject.EXECUTION_CONFIG

        # Force testnet mode
        self.config.OPERATION_MODE = "TESTNET"
        self.execution_config["PAPER_MODE"] = True
        self.execution_config["USE_TESTNET"] = True

        # 🛡️ STOP-LOSS SIMULATION SYSTEM
        self.active_stop_losses = {}  # symbol -> list of stop orders
        self.stop_loss_monitor_active = True

        # 💰 PROFIT-TAKING ORDER MONITORING SYSTEM
        self.active_profit_orders = {}  # symbol -> list of profit orders
        self.profit_order_monitor_active = True

        # 🔄 PERSISTENT ORDER TRACKING - Track simulated orders across cycles
        self.simulated_orders_file = "portfolio_state_live.json"  # Use existing JSON file
        self._load_persistent_orders()

        # Max stop-loss % widened from 1.5% to 5% for range-bound markets
        self.MAX_STOP_LOSS_PCT = 0.050  # 5% instead of 1.5%

        # ⏰ TRADE COOLDOWN SYSTEM - Prevent overtrading same symbol
        self.trade_cooldowns = {}  # symbol -> last_trade_timestamp
        self.last_signal_type = {}  # symbol -> last_signal_type (buy/sell/initial_deployment)
        self.cooldown_periods = {
            "BTCUSDT": 15,  # 15 seconds for BTC (high volume, faster trading)
            "ETHUSDT": 15,  # 15 seconds for ETH (faster trading)
            "default": 15   # 15 seconds for other assets (improved responsiveness)
        }
        self.cooldown_monitor_active = True
        self.cooldown_blocked_count = 0  # Track cooldown blocks per cycle

        # 🎯 CONFIDENCE-BASED MULTIPLIER SYSTEM
        self.confidence_multipliers = {
            "enabled": True,
            "sizing_range": (0.5, 2.5),  # Min/Max sizing multiplier
            "stop_loss_range": (0.7, 1.3),  # Stop-loss tightness (lower = tighter stops)
            "risk_range": (0.8, 1.5),  # Risk adjustment (lower = more conservative)
            "priority_range": (0.5, 2.0),  # Execution priority
            "position_limit_range": (0.7, 1.8),  # Position size limits
        }

        # 💰 STAGGERED PROFIT-TAKING SYSTEM - REDISIGNED FOR PROPER RISK-REWARD
        # FIXED: Much wider profit targets to compensate for tighter stops
        self.profit_taking_config = {
            "enabled": True,
            "levels": [
                {"profit_pct": 0.03, "sell_pct": 0.25, "name": "quick_profit"},    # 3% profit, sell 25%
                {"profit_pct": 0.08, "sell_pct": 0.35, "name": "moderate_gain"},  # 8% profit, sell 35%
                {"profit_pct": 0.15, "sell_pct": 0.40, "name": "strong_gain"},    # 15% profit, sell 40%
            ],
            "regime_adjustments": {
                "RANGE": {"multiplier": 0.8, "description": "Slightly conservative for range markets"},
                "TREND": {"multiplier": 1.2, "description": "More aggressive for trending markets"},
                "VOLATILE": {"multiplier": 1.0, "description": "Standard for volatile markets"}
            },
            "confidence_adjustment": True,  # Adjust levels based on confidence
            "volatility_adjustment": True,  # Adjust profit targets based on volatility
            "convergence_adjustment": True,  # Adjust levels based on signal convergence
            "max_levels": 3,  # Maximum profit-taking levels
            "min_profit_threshold": 0.025,  # 2.5% minimum profit to enable profit-taking (was 1.5%)
        }

        # 🔄 CONVERGENCE-BASED PROFIT TAKING
        self.convergence_profit_taking = {
            "enabled": True,
            "convergence_multipliers": {
                "high": 1.4,    # 40% more aggressive profit-taking for high convergence
                "medium": 1.0,  # Baseline for medium convergence
                "low": 0.6      # 40% more conservative for low convergence
            },
            "convergence_levels": {
                "high": 0.8,    # Convergence score >= 0.8 is high
                "medium": 0.5,  # Convergence score >= 0.5 is medium
                "low": 0.0      # Below 0.5 is low
            },
            "early_exit_convergence": 0.9,  # Take all profits if convergence drops below this
            "profit_lock_convergence": 0.85, # Lock in profits if convergence reaches this level
        }

        # 🎯 PROFIT TARGET CALCULATOR
        self.profit_target_calculator = {
            "enabled": True,
            "risk_reward_ratios": [1.5, 2.0, 3.0, 4.0],  # RR ratios for different targets
            "confidence_multipliers": {
                "high": 1.3,    # 30% more aggressive for high confidence
                "medium": 1.0,  # Baseline for medium confidence
                "low": 0.7      # 30% more conservative for low confidence
            },
            "volatility_multipliers": {
                "low": 0.8,     # Tighter targets in low volatility
                "medium": 1.0,  # Baseline for medium volatility
                "high": 1.4     # Wider targets in high volatility
            },
            "time_based_adjustment": True,  # Adjust targets based on holding time
            "market_regime_adjustment": True,  # Adjust based on market conditions
        }

        logger.info(f"✅ OrderManager initialized - Mode: {ConfigObject.OPERATION_MODE}")
        logger.info(f"✅ Limits BTC: {ConfigObject.RISK_LIMITS['MAX_ORDER_SIZE_BTC']}, ETH: {ConfigObject.RISK_LIMITS['MAX_ORDER_SIZE_ETH']}")
        logger.info(f"🎯 Confidence multipliers: {self.confidence_multipliers['enabled']}")

    def validate_order(self, order: Dict[str, Any], path_mode: str = None) -> Dict[str, Any]:
        """
        Validate order against HRM path mode rules.

        In PATH3 (Full L3 Dominance), only allow orders from L3 trend-following signals.
        Blocks any order not originating from L3 trend-following in PATH3 mode.

        Args:
            order: Order dictionary to validate
            path_mode: Optional path mode override for testing

        Returns:
            Dict with 'valid': bool and 'reason': str
        """
        signal_source = order.get("signal_source", "")
        current_path_mode = path_mode if path_mode is not None else HRM_PATH_MODE

        # PATH3 VALIDATION: Only allow L3 trend-following orders
        if current_path_mode == "PATH3":
            if signal_source != PATH3_SIGNAL_SOURCE:
                logger.warning(f"🚫 PATH3 ORDER BLOCKED: {order.get('symbol', 'UNKNOWN')} {order.get('side', 'unknown')} from '{signal_source}' (only L3 trend-following allowed)")
                return {
                    "valid": False,
                    "reason": f"PATH3 mode blocks non-L3 orders. Source '{signal_source}' != '{PATH3_SIGNAL_SOURCE}'"
                }

            # Log allowed PATH3 orders
            logger.info(f"✅ PATH3 ORDER ALLOWED: {order.get('symbol', 'UNKNOWN')} {order.get('side', 'unknown')} from L3 trend-following")

        # For PATH1 and PATH2, all orders are allowed (no additional restrictions)
        elif current_path_mode in ["PATH1", "PATH2"]:
            logger.debug(f"✅ {current_path_mode} ORDER ALLOWED: {order.get('symbol', 'UNKNOWN')} {order.get('side', 'unknown')} from '{signal_source}'")

        else:
            logger.warning(f"⚠️ UNKNOWN HRM_PATH_MODE: {current_path_mode} - allowing order anyway")

        return {
            "valid": True,
            "reason": f"Order allowed in {current_path_mode} mode"
        }

    def calculate_confidence_multipliers(self, signal_confidence: float, signal_strength: float) -> Dict[str, float]:
        """
        Calculate comprehensive confidence-based multipliers for order generation.

        Args:
            signal_confidence: Signal confidence score (0.0 to 1.0)
            signal_strength: Signal strength score (0.0 to 1.0)

        Returns:
            Dictionary of multipliers for different aspects of order generation
        """
        if not self.confidence_multipliers["enabled"]:
            return {
                "sizing": 1.0,
                "stop_loss": 1.0,
                "risk": 1.0,
                "priority": 1.0,
                "position_limit": 1.0
            }

        # Ensure inputs are scalar floats
        signal_confidence = float(signal_confidence)
        signal_strength = float(signal_strength)

        # Combine confidence and strength for overall confidence score
        combined_confidence = (signal_confidence + signal_strength) / 2.0

        # Calculate multipliers using linear interpolation within defined ranges
        def interpolate_multiplier(confidence: float, min_val: float, max_val: float) -> float:
            """Interpolate multiplier based on confidence score"""
            return min_val + (max_val - min_val) * confidence

        multipliers = {
            "sizing": interpolate_multiplier(combined_confidence,
                                           self.confidence_multipliers["sizing_range"][0],
                                           self.confidence_multipliers["sizing_range"][1]),
            "stop_loss": interpolate_multiplier(combined_confidence,
                                             self.confidence_multipliers["stop_loss_range"][0],
                                             self.confidence_multipliers["stop_loss_range"][1]),
            "risk": interpolate_multiplier(combined_confidence,
                                         self.confidence_multipliers["risk_range"][0],
                                         self.confidence_multipliers["risk_range"][1]),
            "priority": interpolate_multiplier(combined_confidence,
                                            self.confidence_multipliers["priority_range"][0],
                                            self.confidence_multipliers["priority_range"][1]),
            "position_limit": interpolate_multiplier(combined_confidence,
                                                  self.confidence_multipliers["position_limit_range"][0],
                                                  self.confidence_multipliers["position_limit_range"][1])
        }

        logger.debug(f"🎯 CONFIDENCE MULTIPLIERS for conf={signal_confidence:.3f}, str={signal_strength:.3f}: {multipliers}")
        return multipliers

    def resolve_order_conflicts(self, order_params: Dict[str, Any], signal: TacticalSignal,
                               confidence_multipliers: Dict[str, float], max_position: float,
                               current_position: float, current_price: float) -> Dict[str, Any]:
        """
        Resolve conflicts between different order sizing and risk management parameters.

        Args:
            order_params: Current order parameters
            signal: The signal being processed
            confidence_multipliers: Confidence-based multipliers
            max_position: Maximum allowed position
            current_position: Current position size
            current_price: Current market price

        Returns:
            Resolved order parameters with conflicts resolved
        """
        resolved = order_params.copy()
        conflicts_resolved = []

        # 1. RESOLVE ORDER SIZING CONFLICTS
        # Check if combined multipliers would create an excessively large order
        sizing_multipliers = [
            float(getattr(signal, "strength", 0.5)) * 2.0,  # Strength multiplier
            max(0.7, 1.0 - order_params.get("volatility_used", 0.03) * 30),  # Volatility adjustment
            confidence_multipliers["sizing"]  # Confidence sizing
        ]

        combined_sizing_factor = 1.0
        for multiplier in sizing_multipliers:
            combined_sizing_factor *= multiplier

        # Safety check: prevent excessive sizing (max 5x base order)
        max_safe_sizing = 5.0
        if combined_sizing_factor > max_safe_sizing:
            original_factor = combined_sizing_factor
            combined_sizing_factor = max_safe_sizing
            resolved["sizing_adjusted"] = True
            resolved["original_sizing_factor"] = original_factor
            resolved["adjusted_sizing_factor"] = combined_sizing_factor
            conflicts_resolved.append(f"Sizing factor reduced from {original_factor:.2f}x to {max_safe_sizing:.1f}x (safety limit)")

        # 2. RESOLVE POSITION LIMIT CONFLICTS
        # Check if confidence-adjusted position limit conflicts with base limits
        base_max_position = max_position
        confidence_adjusted_limit = max_position * confidence_multipliers["position_limit"]

        # Ensure confidence adjustment doesn't create unsafe positions
        max_reasonable_position = base_max_position * 2.0  # Max 2x base limit even for high confidence
        if confidence_adjusted_limit > max_reasonable_position:
            original_limit = confidence_adjusted_limit
            confidence_adjusted_limit = max_reasonable_position
            resolved["position_limit_adjusted"] = True
            resolved["original_position_limit"] = original_limit
            resolved["adjusted_position_limit"] = confidence_adjusted_limit
            conflicts_resolved.append(f"Position limit capped at {max_reasonable_position:.4f} (was {original_limit:.4f})")

        # 3. RESOLVE STOP-LOSS CONFLICTS
        # Ensure stop-loss remains within safe ranges after confidence adjustment
        stop_loss_pct = resolved.get("stop_loss_pct", 0.02)
        confidence_adjusted_sl = stop_loss_pct * confidence_multipliers["stop_loss"]

        # Safety bounds for stop-loss
        min_safe_sl = 0.005  # 0.5% minimum
        max_safe_sl = 0.10   # 10% maximum

        if confidence_adjusted_sl < min_safe_sl:
            original_sl = confidence_adjusted_sl
            confidence_adjusted_sl = min_safe_sl
            resolved["stop_loss_adjusted"] = True
            resolved["original_stop_loss_pct"] = original_sl
            resolved["adjusted_stop_loss_pct"] = confidence_adjusted_sl
            conflicts_resolved.append(f"Stop-loss increased from {original_sl:.3f} to {min_safe_sl:.3f} (minimum safety)")

        elif confidence_adjusted_sl > max_safe_sl:
            original_sl = confidence_adjusted_sl
            confidence_adjusted_sl = max_safe_sl
            resolved["stop_loss_adjusted"] = True
            resolved["original_stop_loss_pct"] = original_sl
            resolved["adjusted_stop_loss_pct"] = confidence_adjusted_sl
            conflicts_resolved.append(f"Stop-loss reduced from {original_sl:.3f} to {max_safe_sl:.3f} (maximum safety)")

        # 4. RESOLVE CAPITAL ALLOCATION CONFLICTS
        # Ensure order size doesn't exceed available capital after all adjustments
        order_value = resolved.get("order_value", 0)
        max_order_cap = resolved.get("max_order_cap", 1000.0)  # Default $1000

        if order_value > max_order_cap:
            original_value = order_value
            order_value = max_order_cap
            resolved["order_value"] = order_value
            resolved["quantity"] = order_value / current_price
            resolved["capital_adjusted"] = True
            resolved["original_order_value"] = original_value
            conflicts_resolved.append(f"Order size reduced from ${original_value:.2f} to ${order_value:.2f} (capital limit)")

        # 5. FINAL POSITION SIZE VALIDATION
        # Ensure final position won't exceed adjusted limits
        final_quantity = resolved.get("quantity", 0)
        final_position = current_position + final_quantity

        if final_position > confidence_adjusted_limit:
            original_quantity = final_quantity
            max_additional = confidence_adjusted_limit - current_position
            final_quantity = max(0, max_additional)
            resolved["quantity"] = final_quantity
            resolved["order_value"] = abs(final_quantity) * current_price
            resolved["position_limit_enforced"] = True
            resolved["original_quantity"] = original_quantity
            conflicts_resolved.append(f"Position size reduced to respect limit: {confidence_adjusted_limit:.4f} (was {final_position:.4f})")

        # Log resolved conflicts
        if conflicts_resolved:
            logger.warning(f"⚖️ CONFLICT RESOLUTION for {signal.symbol} {signal.side}:")
            for conflict in conflicts_resolved:
                logger.warning(f"   • {conflict}")
            resolved["conflicts_resolved"] = conflicts_resolved
        else:
            logger.debug(f"✅ No conflicts detected for {signal.symbol} {signal.side}")

        return resolved

    def generate_staggered_profit_taking(self, signal: TacticalSignal, quantity: float,
                                        current_price: float, confidence_multipliers: Dict[str, float],
                                        volatility_forecast: float, convergence_score: float = 0.5,
                                        market_regime: str = "TREND") -> List[Dict[str, Any]]:
        """
        Generate staggered profit-taking orders for long positions.

        Args:
            signal: The buy signal
            quantity: Position size to apply profit-taking to
            current_price: Current market price
            confidence_multipliers: Confidence-based multipliers
            volatility_forecast: Market volatility forecast

        Returns:
            List of profit-taking orders
        """
        if not self.profit_taking_config["enabled"] or signal.side != "buy":
            return []

        profit_taking_orders = []
        remaining_quantity = abs(quantity)

        # Adjust profit levels based on confidence, volatility, and convergence
        adjusted_levels = self._adjust_profit_levels(confidence_multipliers, volatility_forecast, convergence_score)

        logger.info(f"💰 STAGGERED PROFIT-TAKING for {signal.symbol}: {len(adjusted_levels)} levels")

        for level in adjusted_levels:
            if remaining_quantity <= 0:
                break

            # Calculate profit target price
            profit_target = current_price * (1 + level["profit_pct"])

            # Calculate quantity to sell at this level
            sell_quantity = min(remaining_quantity * level["sell_pct"], remaining_quantity)

            if sell_quantity * current_price >= self.MIN_ORDER_SIZE:  # Only create if meets minimum
                profit_order = {
                    "symbol": signal.symbol,
                    "side": "SELL",  # Sell to take profits
                    "type": "TAKE_PROFIT",
                    "quantity": sell_quantity,
                    "profit_target": profit_target,
                    "profit_pct": level["profit_pct"],
                    "price": current_price,
                    "timestamp": datetime.utcnow().isoformat(),
                    "signal_strength": getattr(signal, "strength", 0.5),
                    "signal_source": f"profit_taking_{level['name']}",
                    "parent_order": f"{signal.symbol}_buy_{datetime.utcnow().isoformat()}",
                    "status": "pending",
                    "level_name": level["name"],
                    "order_type": "PROFIT_TAKING",  # Tag as profit-taking order
                    "execution_type": "LIMIT"  # Execute as limit order at target price
                }

                profit_taking_orders.append(profit_order)
                remaining_quantity -= sell_quantity

                logger.info(f"💰 PROFIT-TAKING LEVEL: {signal.symbol} sell {sell_quantity:.4f} @ ${profit_target:.2f} ({level['profit_pct']*100:.1f}% profit) [{level['name']}]")

        if profit_taking_orders:
            logger.info(f"💰 TOTAL PROFIT-TAKING ORDERS: {len(profit_taking_orders)} levels for {signal.symbol}")

        return profit_taking_orders

    def _adjust_profit_levels(self, confidence_multipliers: Dict[str, float],
                            volatility_forecast: float, convergence_score: float = 0.5,
                            market_regime: str = "TREND") -> List[Dict[str, Any]]:
        """
        Adjust profit-taking levels based on confidence, market volatility, signal convergence, and market regime.

        Args:
            confidence_multipliers: Confidence-based multipliers
            volatility_forecast: Market volatility forecast
            convergence_score: Signal convergence score (0.0 to 1.0)
            market_regime: Current market regime ('RANGE', 'TREND', 'VOLATILE')

        Returns:
            Adjusted profit-taking levels
        """
        adjusted_levels = []

        # Determine convergence category
        if convergence_score >= self.convergence_profit_taking["convergence_levels"]["high"]:
            convergence_key = "high"
        elif convergence_score >= self.convergence_profit_taking["convergence_levels"]["medium"]:
            convergence_key = "medium"
        else:
            convergence_key = "low"

        convergence_multiplier = self.convergence_profit_taking["convergence_multipliers"][convergence_key]

        # Determine regime-based multiplier - CONSERVATIVE FOR RANGE MARKETS
        regime_key = market_regime.upper() if market_regime.upper() in self.profit_taking_config["regime_adjustments"] else "TREND"
        regime_multiplier = self.profit_taking_config["regime_adjustments"][regime_key]["multiplier"]
        regime_name = self.profit_taking_config["regime_adjustments"][regime_key]["description"]

        logger.debug(f"🔄 PROFIT ADJUSTMENTS: convergence={convergence_score:.3f} ({convergence_key}, {convergence_multiplier:.2f}x), regime={market_regime} ({regime_multiplier:.2f}x - {regime_name})")

        for level in self.profit_taking_config["levels"]:
            adjusted_level = level.copy()

            # Adjust profit percentage based on confidence (higher confidence = more aggressive targets)
            confidence_adjustment = confidence_multipliers.get("risk", 1.0)
            adjusted_level["profit_pct"] *= confidence_adjustment

            # Adjust based on volatility (higher volatility = wider profit targets)
            vol_adjustment = max(0.8, min(1.5, 1 + volatility_forecast * 5))
            adjusted_level["profit_pct"] *= vol_adjustment

            # Adjust based on convergence (higher convergence = more aggressive profit-taking)
            adjusted_level["profit_pct"] *= convergence_multiplier

            # 🎯 CRITICAL: Apply regime-based adjustments (conservative for range markets)
            adjusted_level["profit_pct"] *= regime_multiplier
            adjusted_level["sell_pct"] *= regime_multiplier  # Also adjust sell percentages

            # Log final adjustments for RANGE markets
            if market_regime.upper() == "RANGE":
                logger.info(f"📊 RANGE MARKET ADJUSTMENT: {level['name']} profit_target {level['profit_pct']*100:.1f}% → {adjusted_level['profit_pct']*100:.1f}% (multiplier: {regime_multiplier:.2f})")

            # Ensure profit target meets minimum threshold
            if adjusted_level["profit_pct"] >= self.profit_taking_config["min_profit_threshold"]:
                adjusted_levels.append(adjusted_level)

        # Limit to maximum levels
        return adjusted_levels[:self.profit_taking_config["max_levels"]]

    def check_convergence_profit_actions(self, symbol: str, current_convergence: float,
                                       position_quantity: float, entry_price: float,
                                       current_price: float) -> List[Dict[str, Any]]:
        """
        Check if convergence changes require profit-taking actions.

        Args:
            symbol: Trading symbol
            current_convergence: Current convergence score
            position_quantity: Current position size
            entry_price: Position entry price
            current_price: Current market price

        Returns:
            List of profit-taking actions to execute
        """
        if not self.convergence_profit_taking["enabled"]:
            return []

        actions = []

        # Check for early exit condition (convergence dropped too low)
        if current_convergence < self.convergence_profit_taking["early_exit_convergence"]:
            logger.warning(f"🔄 CONVERGENCE EARLY EXIT: {symbol} convergence {current_convergence:.3f} < threshold {self.convergence_profit_taking['early_exit_convergence']:.3f}")

            # Create order to exit entire position
            exit_order = {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": position_quantity,
                "price": current_price,
                "timestamp": datetime.utcnow().isoformat(),
                "signal_source": "convergence_early_exit",
                "reason": f"Convergence dropped below {self.convergence_profit_taking['early_exit_convergence']:.3f}",
                "convergence_score": current_convergence,
                "status": "pending"
            }
            actions.append(exit_order)

        # Check for profit locking condition (convergence reached high level)
        elif current_convergence >= self.convergence_profit_taking["profit_lock_convergence"]:
            logger.info(f"🔄 CONVERGENCE PROFIT LOCK: {symbol} convergence {current_convergence:.3f} >= threshold {self.convergence_profit_taking['profit_lock_convergence']:.3f}")

            # Calculate current profit
            if entry_price > 0:
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                if profit_pct > 1.0:  # Only lock profits if we have meaningful gains
                    # Lock in 50% of current profits
                    lock_quantity = position_quantity * 0.5

                    if lock_quantity * current_price >= self.MIN_ORDER_SIZE:
                        lock_order = {
                            "symbol": symbol,
                            "side": "SELL",
                            "type": "MARKET",
                            "quantity": lock_quantity,
                            "price": current_price,
                            "timestamp": datetime.utcnow().isoformat(),
                            "signal_source": "convergence_profit_lock",
                            "reason": f"Convergence reached {current_convergence:.3f}, locking 50% profits",
                            "convergence_score": current_convergence,
                            "current_profit_pct": profit_pct,
                            "status": "pending"
                        }
                        actions.append(lock_order)

        if actions:
            logger.info(f"🔄 CONVERGENCE ACTIONS for {symbol}: {len(actions)} orders generated")

        return actions

    def calculate_profit_targets(self, entry_price: float, stop_loss_price: float,
                               confidence_level: float, volatility_forecast: float,
                               symbol: str = "GENERIC") -> Dict[str, Any]:
        """
        Calculate optimal profit targets using risk-reward ratios and market conditions.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop-loss price for risk calculation
            confidence_level: Signal confidence (0.0 to 1.0)
            volatility_forecast: Market volatility forecast
            symbol: Trading symbol for context

        Returns:
            Dictionary with profit targets and analysis
        """
        if not self.profit_target_calculator["enabled"]:
            return {"targets": [], "analysis": "Calculator disabled"}

        # Calculate risk amount (distance to stop-loss)
        if entry_price > stop_loss_price:  # Long position
            risk_amount = entry_price - stop_loss_price
            risk_pct = (risk_amount / entry_price) * 100
        else:  # Short position
            risk_amount = stop_loss_price - entry_price
            risk_pct = (risk_amount / entry_price) * 100

        # Determine confidence category
        if confidence_level >= 0.8:
            confidence_key = "high"
        elif confidence_level >= 0.5:
            confidence_key = "medium"
        else:
            confidence_key = "low"

        # Determine volatility category
        if volatility_forecast >= 0.05:
            volatility_key = "high"
        elif volatility_forecast >= 0.02:
            volatility_key = "medium"
        else:
            volatility_key = "low"

        # Get adjustment multipliers
        confidence_multiplier = self.profit_target_calculator["confidence_multipliers"][confidence_key]
        volatility_multiplier = self.profit_target_calculator["volatility_multipliers"][volatility_key]

        # Calculate profit targets using risk-reward ratios
        targets = []
        for rr_ratio in self.profit_target_calculator["risk_reward_ratios"]:
            # Base reward amount
            reward_amount = risk_amount * rr_ratio

            # Apply confidence and volatility adjustments
            adjusted_reward = reward_amount * confidence_multiplier * volatility_multiplier

            # Calculate target price
            if entry_price > stop_loss_price:  # Long position
                target_price = entry_price + adjusted_reward
                profit_pct = (adjusted_reward / entry_price) * 100
            else:  # Short position
                target_price = entry_price - adjusted_reward
                profit_pct = (adjusted_reward / entry_price) * 100

            # Calculate risk-reward ratio for this target
            actual_rr = adjusted_reward / risk_amount

            targets.append({
                "target_price": target_price,
                "profit_amount": adjusted_reward,
                "profit_pct": profit_pct,
                "risk_reward_ratio": actual_rr,
                "confidence_level": confidence_level,
                "confidence_category": confidence_key,
                "volatility_level": volatility_forecast,
                "volatility_category": volatility_key,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "risk_amount": risk_amount,
                "risk_pct": risk_pct
            })

        # Analysis and recommendations
        analysis = self._analyze_profit_targets(targets, symbol)

        return {
            "targets": targets,
            "analysis": analysis,
            "summary": {
                "total_targets": len(targets),
                "best_rr_ratio": max(t["risk_reward_ratio"] for t in targets),
                "avg_profit_pct": sum(t["profit_pct"] for t in targets) / len(targets),
                "confidence_category": confidence_key,
                "volatility_category": volatility_key
            }
        }

    def _analyze_profit_targets(self, targets: List[Dict[str, Any]], symbol: str) -> str:
        """
        Analyze profit targets and provide trading recommendations.

        Args:
            targets: List of calculated profit targets
            symbol: Trading symbol

        Returns:
            Analysis string with recommendations
        """
        if not targets:
            return "No profit targets calculated"

        # Find best target based on risk-reward
        best_target = max(targets, key=lambda x: x["risk_reward_ratio"])

        # Analyze target distribution
        high_rr_targets = [t for t in targets if t["risk_reward_ratio"] >= 3.0]
        moderate_rr_targets = [t for t in targets if 2.0 <= t["risk_reward_ratio"] < 3.0]
        low_rr_targets = [t for t in targets if t["risk_reward_ratio"] < 2.0]

        analysis_parts = []

        # Overall assessment
        if best_target["risk_reward_ratio"] >= 3.0:
            analysis_parts.append(f"🎯 EXCELLENT setup for {symbol}: Best RR ratio {best_target['risk_reward_ratio']:.1f}:1")
        elif best_target["risk_reward_ratio"] >= 2.0:
            analysis_parts.append(f"✅ GOOD setup for {symbol}: Best RR ratio {best_target['risk_reward_ratio']:.1f}:1")
        else:
            analysis_parts.append(f"⚠️ POOR setup for {symbol}: Best RR ratio {best_target['risk_reward_ratio']:.1f}:1")

        # Target distribution analysis
        if high_rr_targets:
            analysis_parts.append(f"   • {len(high_rr_targets)} high-quality targets (RR ≥ 3.0)")
        if moderate_rr_targets:
            analysis_parts.append(f"   • {len(moderate_rr_targets)} moderate targets (RR 2.0-3.0)")
        if low_rr_targets:
            analysis_parts.append(f"   • {len(low_rr_targets)} low-quality targets (RR < 2.0)")

        # Confidence and volatility context
        confidence_cat = best_target["confidence_category"]
        volatility_cat = best_target["volatility_category"]

        if confidence_cat == "high" and volatility_cat == "low":
            analysis_parts.append("   • HIGH CONFIDENCE + LOW VOLATILITY = IDEAL conditions")
        elif confidence_cat == "low" or volatility_cat == "high":
            analysis_parts.append("   • CONSIDER reducing position size due to uncertainty")

        # Profit potential assessment
        avg_profit_pct = sum(t["profit_pct"] for t in targets) / len(targets)
        if avg_profit_pct >= 10.0:
            analysis_parts.append(f"   • STRONG profit potential: {avg_profit_pct:.1f}% average target")
        elif avg_profit_pct >= 5.0:
            analysis_parts.append(f"   • MODERATE profit potential: {avg_profit_pct:.1f}% average target")
        else:
            analysis_parts.append(f"   • LIMITED profit potential: {avg_profit_pct:.1f}% average target")

        return "\n".join(analysis_parts)

    def get_optimal_profit_target(self, entry_price: float, stop_loss_price: float,
                                confidence_level: float, volatility_forecast: float,
                                symbol: str = "GENERIC") -> Dict[str, Any]:
        """
        Get the single optimal profit target based on all factors.

        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop-loss price for risk calculation
            confidence_level: Signal confidence (0.0 to 1.0)
            volatility_forecast: Market volatility forecast
            symbol: Trading symbol

        Returns:
            Optimal profit target with full analysis
        """
        all_targets = self.calculate_profit_targets(
            entry_price, stop_loss_price, confidence_level,
            volatility_forecast, symbol
        )

        if not all_targets["targets"]:
            return {"error": "No targets calculated"}

        # Select optimal target based on risk-reward ratio
        optimal_target = max(all_targets["targets"], key=lambda x: x["risk_reward_ratio"])

        # Add optimization reasoning
        optimal_target["optimization_reason"] = "Selected for highest risk-reward ratio"
        optimal_target["all_targets_count"] = len(all_targets["targets"])
        optimal_target["analysis"] = all_targets["analysis"]

        return optimal_target

    async def generate_orders(self, state: Dict[str, Any], valid_signals: List[TacticalSignal]) -> List[Dict[str, Any]]:
        """Generate orders from tactical signals."""
        try:
            orders = []
            portfolio = state.get("portfolio", {})

            # Debug logging
            market_data_dict = state.get("market_data")
            logger.info(f"🐛 DEBUG OrderManager - market_data type: {type(market_data_dict)}")
            logger.info(f"🐛 DEBUG OrderManager - market_data keys: {list(market_data_dict.keys()) if isinstance(market_data_dict, dict) else 'N/A'}")

            for signal in valid_signals:
                try:
                    # ⏰ CHECK TRADE COOLDOWN BEFORE PROCESSING SIGNAL - PASAR TIPO DE SEÑAL
                    signal_type = signal.side  # buy, sell, hold
                    if self.should_apply_cooldown(signal.symbol, signal_type,
                                                self.last_signal_type.get(signal.symbol),
                                                self.trade_cooldowns.get(signal.symbol)):
                        logger.warning(f"⏰ SIGNAL REJECTED: {signal.symbol} in trade cooldown, skipping signal generation")
                        self.cooldown_blocked_count += 1  # Track cooldown blocks
                        continue

                    # Ensure market_data is a dict
                    if not isinstance(market_data_dict, dict):
                        logger.error(f"❌ Invalid market_data type: {type(market_data_dict)}")
                        continue

                    market_data = market_data_dict.get(signal.symbol)
                    logger.info(f"🐛 DEBUG OrderManager - {signal.symbol} market_data type: {type(market_data)}")

                    if market_data is None:
                        logger.warning(f"⚠️ No market data for {signal.symbol} - key not found")
                        continue

                    if isinstance(market_data, pd.DataFrame):
                        if market_data.empty:
                            logger.warning(f"⚠️ Empty DataFrame for {signal.symbol}")
                            continue
                        # CRITICAL FIX: Extract scalar value properly from DataFrame/Series
                        close_series = market_data["close"]
                        if hasattr(close_series, 'iloc'):
                            current_price = float(close_series.iloc[-1])
                        else:
                            current_price = float(close_series)
                        logger.info(f"🐛 DEBUG OrderManager - {signal.symbol} current_price from DataFrame: {current_price}")
                    elif isinstance(market_data, dict):
                        # Handle dict format
                        if 'close' in market_data:
                            close_value = market_data['close']
                            # Handle list/dict formats
                            if isinstance(close_value, list):
                                current_price = float(close_value[-1]) if close_value else 50000.0
                            else:
                                current_price = float(close_value)
                            logger.info(f"🐛 DEBUG OrderManager - {signal.symbol} current_price from dict: {current_price}")
                        else:
                            logger.warning(f"⚠️ No 'close' key in market data dict for {signal.symbol}")
                            continue
                    else:
                        logger.warning(f"⚠️ Unsupported market data format for {signal.symbol}: {type(market_data)}")
                        continue
                    max_position = 1200.0 / current_price  # Coordinate with rotator's $1200 limit
                    current_position = portfolio.get(signal.symbol, {}).get("position", 0.0)

                    # ✅ FIXED: Dynamic threshold adjustment based on market conditions
                    # Get market volatility and risk context
                    l3_context = state.get("l3_output", {})

                    # Handle volatility_forecast - can be nested dict or direct value
                    vol_forecast_value = l3_context.get("volatility_forecast", {})
                    if isinstance(vol_forecast_value, dict):
                        volatility_forecast = float(vol_forecast_value.get(signal.symbol, 0.03) or 0.03)
                    else:
                        # Handle direct numeric value or string that should be numeric
                        try:
                            volatility_forecast = float(vol_forecast_value) if vol_forecast_value is not None else 0.03
                        except (ValueError, TypeError):
                            volatility_forecast = 0.03  # Default fallback

                    # Handle risk_appetite - can be string category or numeric
                    risk_appetite_value = l3_context.get("risk_appetite", 0.5)
                    try:
                        risk_appetite = float(risk_appetite_value) if risk_appetite_value is not None else 0.5
                    except (ValueError, TypeError):
                        # Map string categories to numeric values
                        if isinstance(risk_appetite_value, str):
                            risk_mapping = {
                                "low": 0.3,
                                "moderate": 0.5,
                                "medium": 0.5,
                                "high": 0.7,
                                "conservative": 0.3,
                                "aggressive": 0.8
                            }
                            risk_appetite = risk_mapping.get(risk_appetite_value.lower(), 0.5)
                        else:
                            risk_appetite = 0.5  # Default fallback

                    # MICRO-POSITIONS FIX: Higher minimums to prevent insignificant positions
                    # This prevents orders that cost less than $10 (too small to execute)
                    base_min_order = 10.0  # Minimum $10 order value to be worth executing

                    # Adjust minimum based on volatility (higher vol = slightly higher min to avoid slippage)
                    vol_multiplier = max(0.3, min(1.5, volatility_forecast * 30))  # 0.3x to 1.5x based on vol %
                    dynamic_min_order = base_min_order * vol_multiplier

                    # Adjust based on risk appetite (higher risk = smaller orders)
                    risk_multiplier = 1.5 - risk_appetite * 0.5  # 1.0 for high risk, 1.5 for low risk
                    dynamic_min_order *= risk_multiplier

                    # Ensure minimum doesn't go below MIN_ORDER_SIZE, max $25
                    dynamic_min_order = max(self.MIN_ORDER_SIZE, min(25.0, dynamic_min_order))

                    logger.info(f"📊 Dynamic thresholds for {signal.symbol}: min_order=${dynamic_min_order:.2f}, vol={volatility_forecast:.4f}, risk={risk_appetite:.2f}")

                    # 🎯 CALCULATE CONFIDENCE MULTIPLIERS
                    signal_confidence = float(getattr(signal, "confidence", 0.5))
                    signal_strength = float(getattr(signal, "strength", 0.5))
                    confidence_multipliers = self.calculate_confidence_multipliers(signal_confidence, signal_strength)

                    logger.info(f"🎯 CONFIDENCE MULTIPLIERS for {signal.symbol}: sizing={confidence_multipliers['sizing']:.2f}x, stop_loss={confidence_multipliers['stop_loss']:.2f}x, risk={confidence_multipliers['risk']:.2f}x")

                    # ✅ FIXED: Proper buy/sell/hold logic with dynamic thresholds
                    if signal.side == "buy":
                        # Buy logic
                        usdt_balance = portfolio.get("USDT", {}).get("free", 0.0)
                        if usdt_balance < dynamic_min_order:
                            logger.warning(f"⚠️ Insufficient USDT balance: {usdt_balance:.2f} < {dynamic_min_order:.2f}")
                            continue

                        # COORDINATE WITH POSITION ROTATOR LIMITS - PRIORITY 3 FIX
                        # Calculate TOTAL position including planned buy to respect $1200 limit
                        base_order_pct = 0.25  # Base 25% of balance (increased for better capital utilization)
                        strength_multiplier = getattr(signal, "strength", 0.5) * 2.0  # 0.5 to 2.0x
                        vol_adjustment = max(0.7, 1.0 - volatility_forecast * 30)  # Reduce size in high vol (less aggressive reduction)
                        confidence_sizing = confidence_multipliers["sizing"]  # Additional confidence-based sizing

                        order_pct = base_order_pct * strength_multiplier * vol_adjustment * confidence_sizing
                        order_value = min(usdt_balance * order_pct, 1000.0 * confidence_multipliers["position_limit"])  # Cap adjusted by confidence
                        quantity = order_value / current_price

                        # CRITICAL FIX: COORDINATE WITH POSITION ROTATOR - Calculate TOTAL position including planned buy
                        total_after_buy = current_position + quantity
                        max_position_rotator_limit = 1200.0 / current_price  # $1200 position limit used by rotator

                        # If total after buy would exceed rotator limit, adjust buy size DOWN
                        if total_after_buy > max_position_rotator_limit:
                            logger.warning(f"🔄 POSITION COORDINATION: Planned buy would exceed rotator limit ${1200.0:.0f}")
                            logger.warning(f"   Current position: {current_position:.4f}, Planned buy: {quantity:.4f}, Total: {total_after_buy:.4f} > {max_position_rotator_limit:.4f}")
                            # Adjust quantity to fit within limit, leaving small buffer
                            adjusted_quantity = max(0.0, max_position_rotator_limit - current_position - 0.001)  # Leave tiny buffer
                            logger.info(f"   Adjusted buy size: {quantity:.4f} → {adjusted_quantity:.4f} to respect $1200 limit")
                            quantity = adjusted_quantity
                            order_value = quantity * current_price

                        # Apply confidence-based position limit multiplier (additional safety layer)
                        adjusted_max_position = max_position * confidence_multipliers["position_limit"]
                        if current_position + quantity > adjusted_max_position:
                            quantity = max(0.0, adjusted_max_position - current_position)
                            order_value = quantity * current_price

                    elif signal.side == "sell":
                        # Sell logic - CRITICAL: Allow selling even with small positions for signal execution
                        if current_position <= 0:
                            logger.warning(f"⚠️ No position to sell for {signal.symbol}")
                            continue

                        # Dynamic sell sizing based on signal strength and confidence multipliers - HIGH CONFIDENCE AGGRESSIVE SIZING
                        strength_multiplier = getattr(signal, "strength", 0.5) * 2.0
                        base_sell_pct = min(0.7, 0.20 * strength_multiplier)  # 10% to 70% of position (base calculation)
                        confidence_sell_multiplier = confidence_multipliers["sizing"]  # Additional confidence-based sizing
                        sell_pct = min(0.9, base_sell_pct * confidence_sell_multiplier)  # Apply confidence multiplier, max 90%
                        quantity = -current_position * sell_pct  # Negative for sell

                        logger.info(f"📈 SELL EXECUTION: {signal.symbol} selling {sell_pct:.1%} of position ({abs(quantity):.4f} units) [confidence: {confidence_sell_multiplier:.2f}x]")

                    else:  # hold
                        # Do nothing for hold signals
                        logger.info(f"📊 Hold signal for {signal.symbol} - no action taken")
                        continue

                    # Check against dynamic minimum order size WITH MICRO-POSITION FIX
                    quantity = float(quantity)  # Ensure scalar value
                    current_price = float(current_price)  # Ensure scalar value
                    order_value_usdt = abs(quantity) * current_price
                    validation_result = self.validate_order_size(signal.symbol, quantity, current_price, portfolio)

                    # 💰 MICRO-POSITION FIX: Handle modified quantity for 100% sells
                    if validation_result.get("modified_quantity") is not None and validation_result["conversion_type"] == "100_sell_micro_position":
                        logger.info(f"💰 MICRO-POSITION FIX: Updating {signal.symbol} sell quantity from {quantity:.4f} to {validation_result['modified_quantity']:.4f} (100% position sell)")
                        quantity = validation_result["modified_quantity"]
                        order_value_usdt = validation_result["order_value"]  # Use the full position value

                    if validation_result["valid"]:
                        # 🎯 PATH MODE VALIDATION: Check HRM_PATH_MODE rules before proceeding
                        path_order = {
                            "symbol": signal.symbol,
                            "side": signal.side,
                            "signal_source": getattr(signal, "source", "unknown"),
                            "quantity": quantity,
                            "price": current_price
                        }

                        path_validation = self.validate_order(path_order)
                        if not path_validation["valid"]:
                            logger.warning(f"🚫 PATH MODE BLOCKED: {signal.symbol} {signal.side} - {path_validation['reason']}")
                            continue  # Skip this signal, don't generate order

                        logger.info(f"✅ PATH MODE VALIDATION PASSED: {signal.symbol} {signal.side} in {HRM_PATH_MODE}")

                        # Prepare order parameters for conflict resolution
                        order_params = {
                            "quantity": quantity,
                            "order_value": order_value_usdt,
                            "max_order_cap": 1000.0,  # Default $1000 cap
                            "volatility_used": volatility_forecast,
                            "stop_loss_pct": max(0.008, min(0.015, volatility_forecast * 10)),  # FIXED: Tighter base stop-loss (0.8-1.5% range)
                        }

                        # 🎯 RESOLVE CONFLICTS BETWEEN DIFFERENT SIZING AND RISK PARAMETERS
                        resolved_params = self.resolve_order_conflicts(
                            order_params, signal, confidence_multipliers,
                            max_position, current_position, current_price
                        )

                        # Update quantity and order value with resolved parameters
                        quantity = resolved_params["quantity"]
                        order_value_usdt = resolved_params["order_value"]

                        # Create main market order with resolved parameters
                        order = {
                            "symbol": signal.symbol,
                            "side": signal.side,
                            "type": "MARKET",
                            "quantity": quantity,
                            "price": current_price,
                            "timestamp": datetime.utcnow().isoformat(),
                            "signal_strength": signal_strength,
                            "signal_confidence": signal_confidence,
                            "signal_source": getattr(signal, "source", "tactical"),
                            "dynamic_min_order": dynamic_min_order,
                            "volatility_used": volatility_forecast,
                            "risk_appetite_used": risk_appetite,
                            "confidence_multipliers": confidence_multipliers,
                            "conflicts_resolved": resolved_params.get("conflicts_resolved", []),
                            "l3_regime": l3_context.get("regime", "unknown"),
                            "status": "pending",
                            "order_type": "ENTRY",  # Tag as entry order
                            "execution_type": "MARKET"  # Execute as market order immediately
                        }

                        # CRÍTICO: Agregar STOP-LOSS order - SIEMPRE generar para todas las órdenes buy/sell
                        stop_loss = getattr(signal, "stop_loss", None)

                        # Si no hay stop-loss en la señal, calcular uno automático
                        if not stop_loss or stop_loss <= 0:
                            # Use resolved stop-loss percentage from conflict resolution
                            stop_loss_pct = resolved_params.get("adjusted_stop_loss_pct",
                                                               resolved_params.get("stop_loss_pct",
                                                                                  max(0.015, min(0.05, volatility_forecast * 10))))

                            if signal.side == "buy":
                                stop_loss = current_price * (1 - stop_loss_pct)
                            else:  # sell
                                stop_loss = current_price * (1 + stop_loss_pct)
                            logger.info(f"🛡️ AUTO STOP-LOSS: {signal.symbol} {signal.side} @ ${stop_loss:.2f} ({stop_loss_pct*100:.1f}% from entry) [resolved]")

                        # 🛡️ CRITICAL VALIDATION: Stop-loss calculations and positioning
                        stop_loss_valid, validation_details = self._validate_stop_loss_calculation(
                            signal.side, current_price, stop_loss, signal.symbol
                        )

                        if stop_loss_valid:
                            # Create appropriate stop-loss order based on signal direction
                            if signal.side == "buy":
                                # BUY signals: Stop-loss below current price, triggers SELL to exit long position
                                sl_order = {
                                    "symbol": signal.symbol,
                                    "side": "SELL",  # Stop-loss sells to exit long position
                                    "type": "STOP_LOSS",
                                    "quantity": abs(quantity),  # Always positive
                                    "stop_price": stop_loss,
                                    "price": current_price,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "signal_strength": getattr(signal, "strength", 0.5),
                                    "signal_source": "stop_loss_protection",
                                    "parent_order": f"{signal.symbol}_{signal.side}_{datetime.utcnow().isoformat()}",
                                    "status": "pending",
                                    "stop_loss_validation": validation_details
                                }
                                orders.append(sl_order)
                                logger.info(f"🛡️ STOP-LOSS VALIDADO: {signal.symbol} BUY→SELL {abs(quantity):.4f} @ stop={stop_loss:.2f} (below {current_price:.2f}) | Distance: {validation_details['distance_pct']:.2f}%")

                            elif signal.side == "sell":
                                # SELL signals: Stop-loss above current price, triggers BUY to cover short position
                                sl_order = {
                                    "symbol": signal.symbol,
                                    "side": "BUY",  # Stop-loss buys to cover short position
                                    "type": "STOP_LOSS",
                                    "quantity": abs(quantity),  # Always positive
                                    "stop_price": stop_loss,
                                    "price": current_price,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "signal_strength": getattr(signal, "strength", 0.5),
                                    "signal_source": "stop_loss_protection",
                                    "parent_order": f"{signal.symbol}_{signal.side}_{datetime.utcnow().isoformat()}",
                                    "status": "pending",
                                    "stop_loss_validation": validation_details
                                }
                                orders.append(sl_order)
                                logger.info(f"🛡️ STOP-LOSS VALIDADO: {signal.symbol} SELL→BUY {abs(quantity):.4f} @ stop={stop_loss:.2f} (above {current_price:.2f}) | Distance: {validation_details['distance_pct']:.2f}%")
                        else:
                            logger.error(f"🚨 STOP-LOSS REJECTED for {signal.symbol} {signal.side}: {validation_details['reason']}")
                            logger.error(f"   Current Price: {current_price:.8f}, Stop Loss: {stop_loss:.8f}, Side: {signal.side}")
                            # Continue without stop-loss - main order still executes but without protection
                            logger.warning(f"⚠️ MAIN ORDER EXECUTING WITHOUT STOP-LOSS PROTECTION for {signal.symbol}")

                        # ✅ FIXED: CORRECT ORDER EXECUTION
                        # 1. FIRST: Add main market order
                        orders.append(order)

                        # 2. SECOND: Generate profit-taking orders (after main order, so position exists)
                        if signal.side == "buy":
                            # Get convergence score for profit-taking adjustments
                            convergence_score = getattr(signal, "convergence", 0.5)
                            # Get market regime from L3 context for regime-specific profit-taking
                            market_regime = l3_context.get("regime", "TREND")  # Default to TREND if not available
                            profit_taking_orders = self.generate_staggered_profit_taking(
                                signal, quantity, current_price, confidence_multipliers, volatility_forecast, convergence_score, market_regime
                            )
                            orders.extend(profit_taking_orders)

                        # 3. THIRD: Add stop-loss orders (after main order and profit-taking, ensuring position exists)
                        logger.info(f"✅ Order generated: {signal.symbol} {signal.side} {quantity:.4f} (${order_value_usdt:.2f}) [min: ${dynamic_min_order:.2f}]")
                    else:
                        logger.warning(f"⚠️ Order too small: {signal.symbol} {signal.side} {quantity:.4f} (${order_value_usdt:.2f}) < ${dynamic_min_order:.2f} minimum")

                except Exception as e:
                    logger.error(f"❌ Error processing signal {signal}: {e}")
                    continue

            return orders

        except Exception as e:
            logger.error(f"❌ Error generating orders: {e}")
            return []

    async def execute_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute orders by properly categorizing them:
        - ENTRY orders: Execute as market orders immediately
        - PROFIT_TAKING orders: Place as limit orders at target prices (don't execute now)
        - STOP_LOSS orders: Place for monitoring (don't execute now)
        """
        if not orders:
            return []

        processed_orders = []

        # 🔄 CATEGORIZE ORDERS BY TYPE (CRITICAL FIX FOR PROFIT-TAKING)
        entry_orders = [o for o in orders if o.get('order_type') == 'ENTRY' or (o.get('type') == 'MARKET' and not o.get('order_type'))]
        profit_orders = [o for o in orders if o.get('order_type') == 'PROFIT_TAKING' or o.get('type') == 'TAKE_PROFIT']
        stop_orders = [o for o in orders if o.get('type') == 'STOP_LOSS']

        logger.info(f"🔄 ORDER CATEGORIZATION: {len(entry_orders)} ENTRY, {len(profit_orders)} PROFIT_TAKING, {len(stop_orders)} STOP_LOSS")

        # 1️⃣ EXECUTE ENTRY ORDERS AS MARKET ORDERS (IMMEDIATE EXECUTION)
        for order in entry_orders:
            try:
                if self.execution_config["PAPER_MODE"]:
                    # Market orders se ejecutan inmediatamente
                    order["status"] = "filled"
                    order["filled_price"] = float(order["price"])  # Ensure scalar
                    order["filled_quantity"] = float(order["quantity"])  # Ensure scalar
                    # Calculate commission consistently with portfolio manager
                    order_value = abs(order["filled_price"] * order["filled_quantity"])
                    order["commission"] = order_value * 0.001  # 0.1% fee
                    logger.info(f"✅ MARKET ejecutado: {order['symbol']} {order['side']} {order['quantity']:.4f} @ {order['price']:.2f}")

                    # ⏰ UPDATE TRADE COOLDOWN - Apply cooldown after successful trade
                    self.update_trade_cooldown(order["symbol"], order["side"])

                    # 🎯 LOG STANDARDIZED TRADING ACTION
                    log_trading_action(
                        symbol=order["symbol"],
                        strategy="L2_SIGNAL",
                        regime=order.get("l3_regime", "unknown").upper(),
                        action=order["side"].upper(),
                        confidence=order.get("signal_confidence", 0.5),
                        reason=f"L2 signal execution: {order.get('signal_source', 'unknown')} strength={order.get('signal_strength', 0.5):.2f}"
                    )
                else:
                    # Production market orders (not implemented yet)
                    raise NotImplementedError(f"Real market orders not implemented yet. Order: {order}")

                processed_orders.append(order)

            except Exception as e:
                logger.error(f"❌ Error executing entry order {order}: {e}")
                order["status"] = "rejected"
                order["error"] = str(e)
                processed_orders.append(order)

        # 2️⃣ PLACE PROFIT-TAKING ORDERS AS LIMIT ORDERS (DON'T EXECUTE NOW)
        for order in profit_orders:
            try:
                target_price = order.get('profit_target', order.get('target_price', 0))
                if target_price <= 0:
                    logger.error(f"❌ Invalid target price for profit-taking order: {target_price}")
                    order["status"] = "rejected"
                    order["error"] = "Invalid target price"
                    processed_orders.append(order)
                    continue

                if self.execution_config["PAPER_MODE"]:
                    # PLACE LIMIT ORDER AT TARGET PRICE (don't execute immediately)
                    order["status"] = "placed"
                    order["type"] = "LIMIT"
                    order["price"] = target_price  # Set price to target (this is where limit order sits)
                    order["time_in_force"] = "GTC"
                    order["order_id"] = f"pt_{order['symbol']}_{order['side']}_{target_price:.6f}"
                    logger.info(f"💰 PROFIT LIMIT PLACED: {order['symbol']} SELL {order['quantity']:.4f} @ ${target_price:.2f} ({order.get('profit_pct', 0)*100:.1f}% profit) - WAITING FOR TARGET")

                    # 🚨 CRITICAL: Add profit-taking orders to monitoring system
                    self.add_simulated_profit_order(order["symbol"], order)
                else:
                    # PLACE REAL LIMIT ORDER FOR PROFIT-TAKING
                    if not self.binance_client:
                        logger.error("❌ No Binance client available for profit-taking orders")
                        order["status"] = "rejected"
                        order["error"] = "No Binance client"
                    else:
                        try:
                            # Place limit order at profit target
                            tp_order = await self.binance_client.place_limit_order(
                                symbol=order["symbol"],
                                side="SELL",  # Profit-taking always sells
                                quantity=order["quantity"],
                                price=target_price,
                                stop_price=None,  # No stop for profit-taking
                                order_type="LIMIT"
                            )
                            order["status"] = "placed"
                            order["order_id"] = tp_order.get("id", f"pt_{order['symbol']}")
                            order["exchange_order"] = tp_order
                            logger.info(f"💰 PROFIT LIMIT PLACED: {order['symbol']} SELL {order['quantity']:.4f} @ limit=${target_price:.2f} (Real order ID: {order['order_id']})")

                        except Exception as tp_error:
                            logger.error(f"❌ Error placing profit-taking limit order: {tp_error}")
                            order["status"] = "rejected"
                            order["error"] = str(tp_error)

                processed_orders.append(order)

            except Exception as e:
                logger.error(f"❌ Error placing profit-taking order {order}: {e}")
                order["status"] = "rejected"
                order["error"] = str(e)
                processed_orders.append(order)

        # 3️⃣ PLACE STOP-LOSS ORDERS FOR MONITORING (DON'T EXECUTE NOW)
        for order in stop_orders:
            try:
                if self.execution_config["PAPER_MODE"]:
                    # PLACE STOP-LOSS FOR MONITORING (don't execute immediately)
                    order["status"] = "placed"
                    order["order_id"] = f"sl_{order['symbol']}_{order['side']}_{order['stop_price']:.6f}"
                    logger.info(f"🛡️ STOP-LOSS PLACED: {order['symbol']} {order['side']} {order['quantity']:.4f} @ stop={order['stop_price']:.2f} - MONITORING FOR TRIGGER")

                    # 🛡️ REGISTER FOR MONITORING (this is the correct place)
                    self.add_simulated_stop_loss(order["symbol"], order)
                else:
                    # PLACE REAL STOP-LOSS ORDER
                    # CRÍTICO: Ejecutar STOP_LOSS orders reales en modo producción
                    if not self.binance_client:
                        logger.error("❌ No Binance client available for stop-loss orders")
                        order["status"] = "rejected"
                        order["error"] = "No Binance client"
                    else:
                        try:
                            # Colocar orden STOP_LOSS real en Binance
                            sl_order = await self.binance_client.place_stop_loss_order(
                                symbol=order["symbol"],
                                side=order["side"],
                                quantity=order["quantity"],
                                stop_price=order["stop_price"],
                                limit_price=order.get("price")  # Usar precio de mercado como limit
                            )
                            order["status"] = "placed"
                            order["order_id"] = sl_order.get("id", f"sl_{order['symbol']}")
                            order["exchange_order"] = sl_order
                            logger.info(f"🛡️ STOP-LOSS REAL colocado: {order['symbol']} {order['side']} {order['quantity']:.4f} @ stop={order['stop_price']:.2f} (ID: {order['order_id']})")
                        except Exception as sl_error:
                            logger.error(f"❌ Error colocando stop-loss real: {sl_error}")
                            order["status"] = "rejected"
                            order["error"] = str(sl_error)



                processed_orders.append(order)

            except Exception as e:
                logger.error(f"❌ Error executing order {order}: {e}")
                order["status"] = "rejected"
                order["error"] = str(e)
                processed_orders.append(order)

        return processed_orders

    def cleanup_stale_orders(self, portfolio_positions: Dict[str, float]) -> Dict[str, int]:
        """
        Clean up stale stop-loss orders for positions that no longer exist.

        Args:
            portfolio_positions: Dictionary mapping symbol to current position size

        Returns:
            Dictionary with cleanup statistics
        """
        logger.info(f"🧹 CLEANUP STALE ORDERS | Checking {len(self.active_stop_losses)} symbols for stale orders")

        cleaned_stops = 0
        cleaned_profits = 0

        # Check stop-loss orders
        symbols_to_remove_sl = []
        for symbol, stop_orders in list(self.active_stop_losses.items()):
            current_position = portfolio_positions.get(symbol, 0.0)

            # If position is effectively zero, remove all stop orders for this symbol
            if abs(current_position) < 0.0001:
                logger.info(f"🧹 CLEANUP STALE: Removing {len(stop_orders)} stop-loss orders for {symbol} (position: {current_position:.6f})")
                cleaned_stops += len(stop_orders)
                symbols_to_remove_sl.append(symbol)
            else:
                logger.debug(f"✅ STALE CHECK: {symbol} has valid position {current_position:.6f}, keeping {len(stop_orders)} stop orders")

        # Remove stale stop-loss orders
        for symbol in symbols_to_remove_sl:
            del self.active_stop_losses[symbol]

        # Check profit-taking orders
        symbols_to_remove_pt = []
        for symbol, profit_orders in list(self.active_profit_orders.items()):
            current_position = portfolio_positions.get(symbol, 0.0)

            # If position is effectively zero, remove all profit orders for this symbol
            if abs(current_position) < 0.0001:
                logger.info(f"🧹 CLEANUP STALE: Removing {len(profit_orders)} profit-taking orders for {symbol} (position: {current_position:.6f})")
                cleaned_profits += len(profit_orders)
                symbols_to_remove_pt.append(symbol)

        # Remove stale profit-taking orders
        for symbol in symbols_to_remove_pt:
            del self.active_profit_orders[symbol]

        # Save cleaned state
        self._save_persistent_orders()

        stats = {
            "stop_loss_orders_cleaned": cleaned_stops,
            "profit_orders_cleaned": cleaned_profits,
            "symbols_cleaned": len(symbols_to_remove_sl) + len(symbols_to_remove_pt)
        }

        logger.info(f"🧹 STALE ORDERS CLEANED | Stop-loss: {cleaned_stops}, Profit-taking: {cleaned_profits}, Symbols: {len(symbols_to_remove_sl) + len(symbols_to_remove_pt)}")
        return stats

    # 🛡️ STOP-LOSS SIMULATION SYSTEM WITH VALIDATION
    async def monitor_and_execute_stop_losses_with_validation(self, current_market_data: Dict[str, Any],
                                                            portfolio_positions: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Monitor active stop-loss orders and execute when triggered, with portfolio validation."""
        logger.info(f"🛡️ STOP-LOSS MONITOR | Estado: activo={self.stop_loss_monitor_active}, órdenes_activas={len(self.active_stop_losses)}")

        if not self.stop_loss_monitor_active or not self.active_stop_losses:
            logger.debug("🛡️ STOP-LOSS MONITOR | No hay órdenes activas para monitorear")
            return []

        executed_stops = []

        for symbol, stop_orders in list(self.active_stop_losses.items()):
            if not stop_orders:
                continue

            logger.debug(f"🛡️ STOP-LOSS MONITOR | Monitoreando {len(stop_orders)} órdenes para {symbol}")

            # NEW: Validate portfolio position before processing orders
            if portfolio_positions is not None:
                current_position = portfolio_positions.get(symbol, 0.0)
                if abs(current_position) < 0.0001:
                    logger.warning(f"🚨 STALE STOP-LOSS DETECTED: {symbol} has {len(stop_orders)} stop orders but no position ({current_position:.6f}) - clearing")
                    # Clear all stop orders for this symbol
                    del self.active_stop_losses[symbol]
                    self._save_persistent_orders()
                    continue

            # Get current price for this symbol
            market_data = current_market_data.get(symbol)
            if market_data is None:
                logger.warning(f"🛡️ STOP-LOSS MONITOR | No hay datos de mercado para {symbol}")
                continue

            if isinstance(market_data, pd.DataFrame):
                current_price = float(market_data["close"].iloc[-1])
            elif isinstance(market_data, dict) and 'close' in market_data:
                current_price = float(market_data['close'])
            else:
                logger.warning(f"🛡️ STOP-LOSS MONITOR | Formato de datos inválido para {symbol}")
                continue

            logger.debug(f"🛡️ STOP-LOSS MONITOR | {symbol} precio actual: ${current_price:.2f}")

            # Check each stop-loss order
            remaining_orders = []
            for stop_order in stop_orders:
                stop_price = stop_order["stop_price"]
                side = stop_order["side"]

                logger.debug(f"🛡️ STOP-LOSS MONITOR | Verificando orden: {symbol} {side} stop=${stop_price:.2f} vs precio=${current_price:.2f}")

                # Check if stop-loss should trigger
                triggered = False
                if side == "SELL" and current_price <= stop_price:
                    # Long position stop-loss triggered (price fell below stop)
                    triggered = True
                    logger.warning(f"🚨 STOP-LOSS TRIGGERED: {symbol} LONG position stopped at {current_price:.2f} (stop: {stop_price:.2f})")
                elif side == "BUY" and current_price >= stop_price:
                    # Short position stop-loss triggered (price rose above stop)
                    triggered = True
                    logger.warning(f"🚨 STOP-LOSS TRIGGERED: {symbol} SHORT position stopped at {current_price:.2f} (stop: {stop_price:.2f})")

                if triggered:
                    # Execute the stop-loss order
                    stop_order["status"] = "filled"
                    stop_order["filled_price"] = current_price
                    stop_order["filled_quantity"] = stop_order["quantity"]
                    stop_order["triggered_at"] = datetime.utcnow().isoformat()
                    stop_order["trigger_price"] = current_price

                    # Calculate commission
                    order_value = abs(current_price * stop_order["quantity"])
                    stop_order["commission"] = order_value * 0.001  # 0.1% fee

                    executed_stops.append(stop_order)

                    logger.info(f"🛡️ STOP-LOSS EXECUTADO: {symbol} {side} {stop_order['quantity']:.4f} @ {current_price:.2f} (Pérdida protegida)")

                else:
                    # Keep order active
                    remaining_orders.append(stop_order)
                    logger.debug(f"🛡️ STOP-LOSS MONITOR | Orden {symbol} {side} permanece activa (stop=${stop_price:.2f})")

            # Update active orders for this symbol
            if remaining_orders:
                self.active_stop_losses[symbol] = remaining_orders
                logger.debug(f"🛡️ STOP-LOSS MONITOR | {len(remaining_orders)} órdenes activas restantes para {symbol}")
            else:
                del self.active_stop_losses[symbol]
                logger.info(f"🛡️ STOP-LOSS MONITOR | Todas las órdenes ejecutadas para {symbol}")

        logger.info(f"🛡️ STOP-LOSS MONITOR | Ciclo completado: {len(executed_stops)} órdenes ejecutadas")
        return executed_stops

    # Keep the old method for backward compatibility (but route to new one)
    async def monitor_and_execute_stop_losses(self, current_market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Legacy method - routes to new validated version without portfolio validation."""
        return await self.monitor_and_execute_stop_losses_with_validation(current_market_data, None)

    def _load_persistent_orders(self):
        """Load persistent orders from JSON file."""
        try:
            import json
            import os

            if os.path.exists(self.simulated_orders_file):
                with open(self.simulated_orders_file, 'r') as f:
                    data = json.load(f)

                # Load stop-loss orders
                if 'active_stop_losses' in data:
                    for symbol, orders in data['active_stop_losses'].items():
                        if symbol not in self.active_stop_losses:
                            self.active_stop_losses[symbol] = []
                        self.active_stop_losses[symbol].extend(orders)
                    logger.info(f"📁 Loaded {sum(len(v) for v in data.get('active_stop_losses', {}).values())} persistent stop-loss orders")

                # Load profit orders
                if 'active_profit_orders' in data:
                    for symbol, orders in data['active_profit_orders'].items():
                        if symbol not in self.active_profit_orders:
                            self.active_profit_orders[symbol] = []
                        self.active_profit_orders[symbol].extend(orders)
                    logger.info(f"📁 Loaded {sum(len(v) for v in data.get('active_profit_orders', {}).values())} persistent profit orders")

        except Exception as e:
            logger.error(f"❌ Error loading persistent orders: {e}")

    def _save_persistent_orders(self):
        """Save persistent orders to JSON file."""
        try:
            import json

            data = {
                'timestamp': datetime.utcnow().isoformat(),
                'active_stop_losses': self.active_stop_losses,
                'active_profit_orders': self.active_profit_orders
            }

            with open(self.simulated_orders_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            # Log summary without spamming
            sl_count = sum(len(v) for v in self.active_stop_losses.values())
            pt_count = sum(len(v) for v in self.active_profit_orders.values())
            if sl_count > 0 or pt_count > 0:
                logger.debug(f"💾 Saved {sl_count} stop-loss and {pt_count} profit orders to persistent storage")

        except Exception as e:
            logger.error(f"❌ Error saving persistent orders: {e}")

    def add_simulated_stop_loss(self, symbol: str, stop_order: Dict[str, Any]):
        """Add a stop-loss order to the monitoring system."""
        if symbol not in self.active_stop_losses:
            self.active_stop_losses[symbol] = []

        self.active_stop_losses[symbol].append(stop_order)

        # Save to persistent storage
        self._save_persistent_orders()

        logger.info(f"🛡️ STOP-LOSS registrado para monitoreo: {symbol} {stop_order['side']} @ {stop_order['stop_price']:.2f}")

    def get_active_stop_losses(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all currently active stop-loss orders."""
        return self.active_stop_losses.copy()

    def cancel_stop_loss(self, symbol: str, order_id: str) -> bool:
        """Cancel a specific stop-loss order."""
        if symbol not in self.active_stop_losses:
            return False

        for i, order in enumerate(self.active_stop_losses[symbol]):
            if order.get("order_id") == order_id:
                del self.active_stop_losses[symbol][i]
                logger.info(f"🛡️ STOP-LOSS cancelado: {symbol} {order_id}")
                return True

        return False

    # 💰 PROFIT-TAKING ORDER MONITORING SYSTEM
    async def monitor_and_execute_profit_orders(self, current_market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Monitor active profit-taking orders and execute when targets are reached."""
        logger.info(f"💰 PROFIT MONITOR | Estado: activo={self.profit_order_monitor_active}, órdenes_activas={sum(len(v) for v in self.active_profit_orders.values())}")

        if not self.profit_order_monitor_active or not self.active_profit_orders:
            logger.debug("💰 PROFIT MONITOR | No hay órdenes activas para monitorear")
            return []

        executed_profits = []

        for symbol, profit_orders in list(self.active_profit_orders.items()):
            if not profit_orders:
                continue

            logger.debug(f"💰 PROFIT MONITOR | Monitoreando {len(profit_orders)} órdenes para {symbol}")

            # Get current price for this symbol
            market_data = current_market_data.get(symbol)
            if market_data is None:
                logger.warning(f"💰 PROFIT MONITOR | No hay datos de mercado para {symbol}")
                continue

            if isinstance(market_data, pd.DataFrame):
                current_price = float(market_data["close"].iloc[-1])
            elif isinstance(market_data, dict) and 'close' in market_data:
                current_price = float(market_data['close'])
            else:
                logger.warning(f"💰 PROFIT MONITOR | Formato de datos inválido para {symbol}")
                continue

            logger.debug(f"💰 PROFIT MONITOR | {symbol} precio actual: ${current_price:.2f}")

            # Check each profit-taking order
            remaining_orders = []
            for profit_order in profit_orders:
                target_price = profit_order.get("price", 0)  # This should be the limit target price
                side = profit_order.get("side", "")

                logger.debug(f"💰 PROFIT MONITOR | Verificando orden: {symbol} {side} target=${target_price:.2f} vs precio=${current_price:.2f}")

                # Check if profit-taking should trigger (price meets or exceeds target for sell orders)
                triggered = False
                if side == "SELL" and current_price >= target_price:
                    # Profit-taking triggered (price reached target)
                    triggered = True
                    logger.info(f"💰 PROFIT-TAKING TRIGGERED: {symbol} target ${target_price:.2f} reached at {current_price:.2f}")

                if triggered:
                    # Execute the profit-taking order
                    profit_order["status"] = "filled"
                    profit_order["filled_price"] = target_price  # Fill at target price for limit orders
                    profit_order["filled_quantity"] = profit_order["quantity"]
                    profit_order["triggered_at"] = datetime.utcnow().isoformat()
                    profit_order["trigger_price"] = current_price

                    # Calculate commission
                    order_value = abs(target_price * profit_order["quantity"])
                    profit_order["commission"] = order_value * 0.001  # 0.1% fee

                    executed_profits.append(profit_order)

                    logger.info(f"💰 PROFIT-TAKING EXECUTADO: {symbol} {side} {profit_order['quantity']:.4f} @ {target_price:.2f} (Ganancias tomadas)")

                else:
                    # Keep order active
                    remaining_orders.append(profit_order)
                    logger.debug(f"💰 PROFIT MONITOR | Orden {symbol} {side} permanece activa (target=${target_price:.2f})")

            # Update active orders for this symbol
            if remaining_orders:
                self.active_profit_orders[symbol] = remaining_orders
                logger.debug(f"💰 PROFIT MONITOR | {len(remaining_orders)} órdenes activas restantes para {symbol}")
            else:
                del self.active_profit_orders[symbol]
                logger.info(f"💰 PROFIT MONITOR | Todas las órdenes ejecutadas para {symbol}")

        logger.info(f"💰 PROFIT MONITOR | Ciclo completado: {len(executed_profits)} órdenes ejecutadas")
        return executed_profits

    def add_simulated_profit_order(self, symbol: str, profit_order: Dict[str, Any]):
        """Add a profit-taking order to the monitoring system."""
        if symbol not in self.active_profit_orders:
            self.active_profit_orders[symbol] = []

        self.active_profit_orders[symbol].append(profit_order)

        # Save to persistent storage
        self._save_persistent_orders()

        logger.info(f"💰 PROFIT ORDER registrado para monitoreo: {symbol} {profit_order.get('side')} @ target={profit_order.get('price'):.2f}")

    # ⏰ TRADE COOLDOWN SYSTEM METHODS
    def should_apply_cooldown(self, symbol: str, signal_type: str, last_signal_type: str = None, last_trade_time: float = None) -> bool:
        """
        Cooldown solo para órdenes repetidas del mismo tipo.
        NO cooldown para deployment inicial o órdenes opuestas.

        Args:
            symbol: Trading symbol
            signal_type: Current signal type (buy/sell/initial_deployment)
            last_signal_type: Last signal type executed
            last_trade_time: Last trade timestamp

        Returns:
            True if cooldown should be applied
        """
        if not self.cooldown_monitor_active:
            return False  # Cooldown disabled

        # Nunca cooldown para deployment inicial
        if signal_type == "initial_deployment":
            return False

        # Si no hay último trade, no hay cooldown
        if last_trade_time is None or last_signal_type is None:
            return False

        current_time = time.time()
        cooldown_period = self.cooldown_periods.get(symbol, self.cooldown_periods["default"])

        time_elapsed = current_time - last_trade_time
        if time_elapsed < cooldown_period:
            # Solo cooldown si es MISMO tipo de orden en <30s
            if signal_type == last_signal_type:
                remaining_time = cooldown_period - time_elapsed
                logger.info(f"⏰ TRADE COOLDOWN: {symbol} {signal_type} blocked - same type order {time_elapsed:.0f}s ago (wait {remaining_time:.0f}s)")
                return True  # Cooldown para órdenes repetidas del mismo tipo
            else:
                logger.info(f"✅ TRADE COOLDOWN: {symbol} {signal_type} allowed after {signal_type} - opposite to last {last_signal_type}")
                return False  # Permitir órdenes opuestas
        else:
            # Cooldown expirado
            return False

    def check_trade_cooldown(self, symbol: str, signal_type: str = "unknown") -> bool:
        """
        Legacy method - redirects to new should_apply_cooldown method.
        """
        last_trade_time = self.trade_cooldowns.get(symbol)
        last_signal_type = self.last_signal_type.get(symbol) if hasattr(self, 'last_signal_type') else None
        return self.should_apply_cooldown(symbol, signal_type, last_signal_type, last_trade_time)

    def update_trade_cooldown(self, symbol: str, signal_type: str = "unknown"):
        """
        Update the last trade timestamp and signal type for a symbol to start cooldown.

        Args:
            symbol: Symbol that was just traded
            signal_type: Type of signal executed (buy/sell)
        """
        if self.cooldown_monitor_active:
            self.trade_cooldowns[symbol] = time.time()
            self.last_signal_type[symbol] = signal_type
            cooldown_period = self.cooldown_periods.get(symbol, self.cooldown_periods["default"])
            logger.info(f"⏰ TRADE COOLDOWN: {symbol} {signal_type} executed, cooldown {cooldown_period}s active")

    def get_cooldown_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all active cooldowns.

        Returns:
            Dictionary with cooldown information for each symbol
        """
        status = {}
        current_time = time.time()

        for symbol, last_trade_time in self.trade_cooldowns.items():
            cooldown_period = self.cooldown_periods.get(symbol, self.cooldown_periods["default"])
            time_elapsed = current_time - last_trade_time
            remaining_time = max(0, cooldown_period - time_elapsed)

            status[symbol] = {
                "last_trade": last_trade_time,
                "cooldown_period": cooldown_period,
                "time_elapsed": time_elapsed,
                "remaining_time": remaining_time,
                "active": remaining_time > 0
            }

        return status

    def clear_cooldown(self, symbol: str) -> bool:
        """
        Manually clear cooldown for a symbol.

        Args:
            symbol: Symbol to clear cooldown for

        Returns:
            True if cooldown was cleared, False if no cooldown existed
        """
        if symbol in self.trade_cooldowns:
            del self.trade_cooldowns[symbol]
            logger.info(f"⏰ TRADE COOLDOWN: {symbol} cooldown manually cleared")
            return True
        return False

    def validate_order_size(self, symbol, quantity, current_price, portfolio=None):
        """Valida que la orden cumpla con los requisitos mínimos y liquidez del mercado"""

        order_value = abs(quantity) * current_price
        min_order = self.MIN_ORDER_SIZE  # Usar la constante de clase actualizada

        # 💰 MICRO-POSITION FIX: For sell orders with insufficient value, sell 100% of position
        if order_value < min_order and quantity < 0:  # Sell order too small
            current_position = portfolio.get(symbol, {}).get("position", 0.0) if portfolio else 0.0
            position_value = current_position * current_price

            logger.warning(f"🛑 Sell order too small: {symbol} value ${order_value:.2f} < ${min_order} minimum")

            # Check if we have a meaningful position to sell fully
            if current_position > 0 and position_value >= 1.0:  # At least $1 position value
                # Force 100% sell of the position
                full_sell_quantity = -current_position
                full_sell_value = current_position * current_price

                logger.info(f"💰 MICRO-POSITION FIX: Converting to 100% sell for {symbol} - ${full_sell_value:.2f} position")

                # Return modified order that's valid
                return {
                    "valid": True,
                    "reason": f"Micro-position converted to 100% sell: ${full_sell_value:.2f}",
                    "order_value": full_sell_value,
                    "required_capital": 0.0,
                    "available_capital": current_position,
                    "modified_quantity": full_sell_quantity,  # Return the new quantity
                    "original_quantity": quantity,
                    "conversion_type": "100_sell_micro_position"
                }
            else:
                return {
                    "valid": False,
                    "reason": f"Micro-position too small to sell fully: position_value=${position_value:.2f} < $1.00 or no position",
                    "order_value": order_value,
                    "required_capital": min_order,
                    "available_capital": current_position
                }

        if order_value < min_order:
            logger.warning(f"🛑 Buy order rejected: {symbol} value ${order_value:.2f} < ${min_order} minimum")
            return {
                "valid": False,
                "reason": f"Order value ${order_value:.2f} below minimum ${min_order:.2f}",
                "order_value": order_value,
                "required_capital": min_order,
                "available_capital": 0.0
            }

        # 🏊 LIQUIDITY MANAGEMENT: Check market liquidity
        liquidity_check = self._check_market_liquidity(symbol, quantity, current_price)
        if not liquidity_check["sufficient"]:
            logger.warning(f"🛑 Insufficient liquidity: {symbol} order ${order_value:.2f} > ${liquidity_check['max_order_value']:.2f} available")
            return {
                "valid": False,
                "reason": f"Insufficient market liquidity: order ${order_value:.2f} exceeds ${liquidity_check['max_order_value']:.2f} (avg volume: ${liquidity_check['avg_volume']:.2f})",
                "order_value": order_value,
                "required_capital": order_value,
                "available_capital": liquidity_check['max_order_value']
            }

        # Verificar que tenemos suficiente capital para buy orders
        if quantity > 0:  # Buy order
            required_usdt = order_value * 1.002  # Incluir fees
            available_usdt = portfolio.get('USDT', {}).get('free', 0.0) if portfolio else 0.0
            if required_usdt > available_usdt:
                logger.warning(f"🛑 Insufficient capital: {symbol} requires ${required_usdt:.2f}, available ${available_usdt:.2f}")
                return {
                    "valid": False,
                    "reason": f"Insufficient capital: ${available_usdt:.2f} < ${required_usdt:.2f} required",
                    "order_value": order_value,
                    "required_capital": required_usdt,
                    "available_capital": available_usdt
                }
        else:  # Sell order
            # Check if we have sufficient position to sell
            current_position = portfolio.get(symbol, {}).get("position", 0.0) if portfolio else 0.0
            if current_position <= 0:
                return {
                    "valid": False,
                    "reason": f"No position to sell for {symbol}",
                    "order_value": order_value,
                    "required_capital": 0.0,
                    "available_capital": current_position
                }
            elif abs(quantity) > current_position:
                return {
                    "valid": False,
                    "reason": f"Insufficient position: {current_position:.6f} < {abs(quantity):.6f}",
                    "order_value": order_value,
                    "required_capital": abs(quantity),
                    "available_capital": current_position
                }

        return {
            "valid": True,
            "reason": "Order size, capital, and liquidity requirements met",
            "order_value": order_value,
            "required_capital": order_value * 1.002 if quantity > 0 else 0.0,
            "available_capital": portfolio.get('USDT', {}).get('free', 0.0) if quantity > 0 else portfolio.get(symbol, {}).get("position", 0.0) if portfolio else 0.0
        }

    def _check_market_liquidity(self, symbol: str, quantity: float, current_price: float) -> Dict[str, Any]:
        """
        Check if the market has sufficient liquidity for the order size.
        Uses recent volume data to estimate available liquidity.
        """
        try:
            # Get market data for this symbol
            market_data = self.market_data.get(symbol)
            if market_data is None:
                # Fallback: assume sufficient liquidity if no data
                return {
                    "sufficient": True,
                    "max_order_value": float('inf'),
                    "avg_volume": 0.0,
                    "reason": "no_market_data"
                }

            # Extract volume data from recent periods
            if isinstance(market_data, pd.DataFrame):
                if market_data.empty or 'volume' not in market_data.columns:
                    return {
                        "sufficient": True,
                        "max_order_value": float('inf'),
                        "avg_volume": 0.0,
                        "reason": "insufficient_data"
                    }

                # Use last 20 periods for volume calculation
                recent_volumes = market_data['volume'].tail(20).astype(float)
                avg_volume = recent_volumes.mean()
                max_volume = recent_volumes.max()

            elif isinstance(market_data, dict):
                # Handle dict format (fallback)
                volume_data = market_data.get('volume', [])
                if not volume_data or len(volume_data) < 5:
                    return {
                        "sufficient": True,
                        "max_order_value": float('inf'),
                        "avg_volume": 0.0,
                        "reason": "insufficient_data"
                    }

                recent_volumes = pd.Series(volume_data[-20:]).astype(float)
                avg_volume = recent_volumes.mean()
                max_volume = recent_volumes.max()
            else:
                return {
                    "sufficient": True,
                    "max_order_value": float('inf'),
                    "avg_volume": 0.0,
                    "reason": "invalid_data_format"
                }

            # Calculate maximum safe order size
            # Conservative approach: max order = 5% of average volume to avoid slippage
            max_order_volume_pct = 0.05  # 5% of average volume
            max_order_volume = avg_volume * max_order_volume_pct
            max_order_value = max_order_volume * current_price

            # For very liquid markets (high volume), allow larger orders
            if avg_volume > 1000000:  # $1M+ average volume
                max_order_volume_pct = 0.10  # 10% for highly liquid markets
                max_order_value = avg_volume * max_order_volume_pct * current_price

            order_value = abs(quantity) * current_price
            sufficient = order_value <= max_order_value

            logger.debug(f"🏊 Liquidity check for {symbol}: order=${order_value:.2f}, max_allowed=${max_order_value:.2f}, avg_volume=${avg_volume:.2f}")

            return {
                "sufficient": sufficient,
                "max_order_value": max_order_value,
                "avg_volume": avg_volume,
                "max_volume": max_volume,
                "order_volume_pct": (order_value / current_price) / avg_volume if avg_volume > 0 else 0.0
            }

        except Exception as e:
            logger.error(f"❌ Error checking market liquidity for {symbol}: {e}")
            # Fallback: assume sufficient liquidity on error
            return {
                "sufficient": True,
                "max_order_value": float('inf'),
                "avg_volume": 0.0,
                "reason": f"error: {str(e)}"
            }

    def validate_order_timing(self, order: Dict[str, Any], recent_orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate order timing to prevent immediate sell execution.

        Don't sell what you just bought at same price.

        Args:
            order: The order to validate
            recent_orders: List of recent orders within last 5 seconds

        Returns:
            Modified order or original if validation passes
        """
        try:
            now = datetime.utcnow()

            if order.get("side") == "SELL":
                # Check for recent buy orders with same symbol
                recent_buys = [
                    o for o in recent_orders
                    if (o.get("symbol") == order.get("symbol") and
                        o.get("side") == "BUY" and
                        o.get("timestamp") and
                        (now - datetime.fromisoformat(o["timestamp"])).seconds < 5)
                ]

                if recent_buys:
                    # This is a profit-taking order - don't market execute
                    if abs(order.get("price", 0) - recent_buys[0].get("price", 0)) < 0.001:
                        logger.warning(f"⏰ IMMEDIATE SELL PREVENTED: {order['symbol']} sell @ {order.get('price', 0):.2f} blocked (just bought at same price)")
                        order["execution_type"] = "LIMIT"
                        order["price"] = order.get("target_price", order.get("price", 0))  # Use target price not current
                        order["validated_timing"] = True
                        order["timing_reason"] = "Immediate sell prevented - converted to limit order"

            return order
        except Exception as e:
            logger.error(f"❌ Error validating order timing for {order.get('symbol', 'unknown')}: {e}")
            return order

    def _validate_stop_loss_calculation(self, signal_side: str, current_price: float,
                                       stop_loss: float, symbol: str) -> tuple[bool, Dict[str, Any]]:
        """
        Comprehensive validation of stop-loss calculations to ensure proper positioning.

        Args:
            signal_side: 'buy' or 'sell'
            current_price: Current market price
            stop_loss: Calculated stop-loss price
            symbol: Trading symbol

        Returns:
            Tuple of (is_valid, validation_details)
        """
        try:
            validation_details = {
                "signal_side": signal_side,
                "current_price": current_price,
                "stop_loss": stop_loss,
                "symbol": symbol,
                "distance_pct": 0.0,
                "is_valid": False,
                "reason": "validation_pending"
            }

            # Basic input validation
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                validation_details.update({
                    "is_valid": False,
                    "reason": f"Invalid current price: {current_price}"
                })
                return False, validation_details

            if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
                validation_details.update({
                    "is_valid": False,
                    "reason": f"Invalid stop-loss price: {stop_loss}"
                })
                return False, validation_details

            if signal_side.lower() not in ['buy', 'sell']:
                validation_details.update({
                    "is_valid": False,
                    "reason": f"Invalid signal side: {signal_side}"
                })
                return False, validation_details

            # Calculate distance and percentage
            if signal_side.lower() == 'buy':
                if stop_loss >= current_price:
                    validation_details.update({
                        "is_valid": False,
                        "reason": f"BUY stop-loss ({stop_loss:.8f}) must be BELOW current price ({current_price:.8f})"
                    })
                    return False, validation_details
                distance = current_price - stop_loss
                distance_pct = (distance / current_price) * 100
                expected_position = "below"
                actual_position = "below"
            else:  # sell
                if stop_loss <= current_price:
                    validation_details.update({
                        "is_valid": False,
                        "reason": f"SELL stop-loss ({stop_loss:.8f}) must be ABOVE current price ({current_price:.8f})"
                    })
                    return False, validation_details
                distance = stop_loss - current_price
                distance_pct = (distance / current_price) * 100
                expected_position = "above"
                actual_position = "above"

            validation_details["distance_pct"] = distance_pct

            # 🔥 FIXED: WIDER STOP-LOSSES increased to 5% max for range-bound markets
            MIN_STOP_DISTANCE_PCT = 0.5   # 0.5% minimum (much tighter)
            MAX_STOP_DISTANCE_PCT = 5.0   # 5.0% maximum (wider stops for range markets)

            if distance_pct < MIN_STOP_DISTANCE_PCT:
                validation_details.update({
                    "is_valid": False,
                    "reason": f"Stop-loss distance ({distance_pct:.2f}%) below minimum {MIN_STOP_DISTANCE_PCT}% (too tight, will cause overtrading)"
                })
                return False, validation_details

            if distance_pct > MAX_STOP_DISTANCE_PCT:
                validation_details.update({
                    "is_valid": False,
                    "reason": f"Stop-loss distance ({distance_pct:.2f}%) above maximum {MAX_STOP_DISTANCE_PCT}% (too wide, poor risk-reward)"
                })
                return False, validation_details

            # Validate price precision (crypto requires 8 decimal places)
            stop_loss_str = f"{stop_loss:.10f}"
            current_price_str = f"{current_price:.10f}"

            # Check for floating point precision issues
            if len(stop_loss_str.split('.')[-1].rstrip('0')) > 8:
                validation_details.update({
                    "is_valid": False,
                    "reason": f"Stop-loss price has too many decimal places: {stop_loss}"
                })
                return False, validation_details

            # Additional validation: ensure stop-loss is reasonable for the asset
            if symbol.endswith('USDT'):
                # For crypto assets, ensure stop-loss is within reasonable bounds
                if distance_pct < 0.5:  # Too tight (< 0.5%)
                    validation_details.update({
                        "is_valid": False,
                        "reason": f"Stop-loss too tight for crypto asset ({distance_pct:.2f}%)"
                    })
                    return False, validation_details

            # All validations passed
            validation_details.update({
                "is_valid": True,
                "reason": f"Valid {signal_side.upper()} stop-loss {distance_pct:.2f}% {expected_position} current price",
                "distance": distance,
                "min_distance_pct": MIN_STOP_DISTANCE_PCT,
                "max_distance_pct": MAX_STOP_DISTANCE_PCT
            })

            logger.debug(f"✅ STOP-LOSS VALIDATION PASSED for {symbol} {signal_side}: {distance_pct:.2f}% distance")
            return True, validation_details

        except Exception as e:
            logger.error(f"❌ Error validating stop-loss calculation for {symbol}: {e}")
            validation_details.update({
                "is_valid": False,
                "reason": f"Validation error: {str(e)}"
            })
            return False, validation_details

    def process_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and separate orders based on type to prevent immediate execution issues.

        Entry orders execute immediately as market orders.
        Profit-taking orders are placed as limit orders at target prices.

        Args:
            orders: List of orders to process

        Returns:
            Processed orders with execution modifications
        """
        try:
            # Separate orders by type
            entry_orders = [o for o in orders if o.get("order_type") == "ENTRY"]
            profit_orders = [o for o in orders if o.get("order_type") == "PROFIT_TAKING"]
            stop_loss_orders = [o for o in orders if o.get("type") == "STOP_LOSS"]

            processed_orders = []

            logger.info(f"📊 PROCESSING ORDERS: {len(entry_orders)} entry, {len(profit_orders)} profit-taking, {len(stop_loss_orders)} stop-loss")

            # 1. ENTRY ORDERS: Execute immediately as market orders
            for order in entry_orders:
                logger.info(f"📈 ENTRY ORDER: {order['symbol']} {order['side']} {order['quantity']:.4f} @ market")
                # Keep as market order for immediate execution
                processed_orders.append(order)

            # 2. PROFIT-TAKING ORDERS: Convert to limit orders at target prices
            for order in profit_orders:
                target_price = order.get("profit_target", order.get("target_price", order.get("price", 0)))
                if target_price > 0:
                    # Modify order to be limit order at target price instead of market
                    order["type"] = "LIMIT"
                    order["price"] = target_price
                    order["time_in_force"] = "GTC"
                    order["processed_type"] = "PROFIT_LIMIT"  # Track this was converted
                    logger.info(f"💰 PROFIT LIMIT: {order['symbol']} SELL {order['quantity']:.4f} @ ${target_price:.2f} (target price)")
                else:
                    logger.warning(f"⚠️ PROFIT ORDER MISSING TARGET: {order['symbol']} - keeping as market")
                processed_orders.append(order)

            # 3. STOP-LOSS ORDERS: Keep as-is for monitoring
            processed_orders.extend(stop_loss_orders)

            logger.info(f"✅ ORDER PROCESSING COMPLETE: {len(processed_orders)} orders ready for execution")
            return processed_orders

        except Exception as e:
            logger.error(f"❌ Error processing orders: {e}")
            return orders  # Return original orders on error

    def place_market_order(self, symbol: str, side: str, quantity: float, price: float,
                          state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place market order with validation and USDT balance check.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Current market price
            state: Current system state

        Returns:
            Dict with order result and validation details
        """
        try:
            # Get portfolio for validation
            portfolio = state.get("portfolio", {})
            usdt_balance = portfolio.get("USDT", {}).get("free", 0.0)

            logger.info(f"🔄 VALIDATING MARKET ORDER: {symbol} {side} {quantity:.4f} @ ${price:.2f}")

            # 1. Validate USDT balance for buy orders
            if side.lower() == "buy":
                order_value = quantity * price
                required_usdt = order_value * 1.002  # Include 0.2% fee buffer

                if required_usdt > usdt_balance:
                    logger.error(f"🚫 INSUFFICIENT USDT BALANCE: Required ${required_usdt:.2f}, Available ${usdt_balance:.2f}")
                    return {
                        "success": False,
                        "error": "INSUFFICIENT_USDT_BALANCE",
                        "required": required_usdt,
                        "available": usdt_balance,
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity
                    }

                logger.info(f"✅ USDT BALANCE VALIDATION PASSED: Required ${required_usdt:.2f}, Available ${usdt_balance:.2f}")

            # 2. Validate position for sell orders
            elif side.lower() == "sell":
                current_position = portfolio.get(symbol, {}).get("position", 0.0)
                if current_position <= 0:
                    logger.error(f"🚫 NO POSITION TO SELL: {symbol} position is {current_position:.6f}")
                    return {
                        "success": False,
                        "error": "NO_POSITION_TO_SELL",
                        "current_position": current_position,
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity
                    }

                if abs(quantity) > current_position:
                    logger.error(f"🚫 INSUFFICIENT POSITION: Requested {abs(quantity):.6f}, Available {current_position:.6f}")
                    return {
                        "success": False,
                        "error": "INSUFFICIENT_POSITION",
                        "requested": abs(quantity),
                        "available": current_position,
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity
                    }

                logger.info(f"✅ POSITION VALIDATION PASSED: Requested {abs(quantity):.6f}, Available {current_position:.6f}")

            # 3. Validate minimum order size
            order_value = abs(quantity) * price
            min_order_value = self.MIN_ORDER_USDT  # Use the class constant

            if order_value < min_order_value:
                logger.error(f"🚫 ORDER TOO SMALL: Value ${order_value:.2f} < minimum ${min_order_value:.2f}")
                return {
                    "success": False,
                    "error": "ORDER_TOO_SMALL",
                    "order_value": order_value,
                    "min_required": min_order_value,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity
                }

            # 4. Create market order structure
            order = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.utcnow().isoformat(),
                "signal_source": "market_order",
                "status": "pending",
                "order_type": "ENTRY",
                "execution_type": "MARKET",
                "validated_usdt_balance": side.lower() == "buy",
                "validated_position": side.lower() == "sell",
                "min_order_check": True
            }

            logger.info(f"✅ MARKET ORDER VALIDATION PASSED: {symbol} {side} {quantity:.4f} @ ${price:.2f} (${order_value:.2f})")

            return {
                "success": True,
                "order": order,
                "validation_details": {
                    "usdt_balance_check": side.lower() == "buy",
                    "position_check": side.lower() == "sell",
                    "min_order_check": True,
                    "order_value": order_value,
                    "required_usdt": required_usdt if side.lower() == "buy" else 0.0
                },
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price
            }

        except Exception as e:
            logger.error(f"❌ Error in place_market_order: {e}")
            return {
                "success": False,
                "error": "VALIDATION_ERROR",
                "exception": str(e),
                "symbol": symbol,
                "side": side,
                "quantity": quantity
            }

    # Properties for external access
    @property
    def logger(self):
        """Expose logger for external access."""
        return logger

    @property
    def min_order_usdt(self):
        """Expose min_order_usdt for external access."""
        return self.MIN_ORDER_USDT
