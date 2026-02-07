"""
Position Manager - Handles position sizing and order calculations
"""

from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from core.logging import logger


class PositionManager:
    """
    Manages position sizing calculations and risk-adjusted order sizes
    """

    def __init__(self, state_manager, portfolio_manager, config: Dict):
        """
        Initialize PositionManager

        Args:
            state_manager: System state manager
            portfolio_manager: Portfolio management interface
            config: System configuration
        """
        self.state = state_manager
        self.portfolio = portfolio_manager
        self.config = config

        logger.info("PositionManager initialized")

    def calculate_order_size(self, symbol: str, action: str, signal_confidence: float,
                           current_price: float, position_qty: float) -> float:
        """
        Calculate order size based on signal parameters and risk management

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            action: 'buy' or 'sell'
            signal_confidence: Signal confidence (0.0-1.0)
            current_price: Current market price
            position_qty: Current position quantity

        Returns:
            Order size in base currency units
        """
        try:
            # 💰 CRITICAL FIX: Mode-dependent USDT balance check
            if action.lower() == 'buy':
                # Check if we're in paper/simulated mode
                paper_mode = False
                if hasattr(self.portfolio, 'mode') and self.portfolio.mode == "simulated":
                    paper_mode = True
                elif self.config.get("PAPER_MODE", False) or self.config.get("OPERATION_MODE", "").upper() == "PAPER":
                    paper_mode = True
                
                try:
                    if paper_mode:
                        # Get available USDT balance from simulated client
                        if hasattr(self.portfolio, 'client') and hasattr(self.portfolio.client, 'get_balances'):
                            available_usdt = self.portfolio.client.get_balances().get('USDT', 0.0)
                        elif hasattr(self.portfolio, 'get_balance'):
                            available_usdt = self.portfolio.get_balance('USDT')
                        else:
                            logger.warning("Paper mode: No direct balance access, using fallback")
                            available_usdt = 1000.0
                        logger.debug(f"📊 PAPER MODE: Using portfolio USDT balance: ${available_usdt:.2f}")
                    else:
                        # Real mode - use exchange client
                        available_usdt = self.portfolio.get_available_balance("USDT")
                        logger.debug(f"📊 REAL MODE: Using exchange USDT balance: ${available_usdt:.2f}")
                        
                    # Fallback if available_usdt is 0 or None (missing exposure data)
                    if available_usdt <= 0:
                        logger.warning(f"Available USDT is {available_usdt:.2f}, using fallback value for {symbol}")
                        available_usdt = 1000.0  # Fallback to $1000 available balance
                        
                except Exception as e:
                    logger.warning(f"Failed to get available USDT balance: {e}, using fallback value")
                    available_usdt = 1000.0  # Fallback value if exposure data is missing

                # Base allocation (5-10% of available USDT for paper mode, 5-15% for real mode)
                if paper_mode:
                    base_allocation_pct = 0.075  # Fixed 7.5% allocation for paper mode (middle of 5-10% range)
                else:
                    base_allocation_pct = 0.05 + (signal_confidence * 0.10)  # 5% to 15% for real mode
                
                order_size_usdt = available_usdt * base_allocation_pct

                # Apply confidence multiplier
                if paper_mode:
                    confidence_multiplier = 1.0  # Fixed multiplier for paper mode
                else:
                    confidence_multiplier = 1.0 + (signal_confidence - 0.5) * 1.0  # 0.5 to 1.5 for real mode
                    
                order_size_usdt *= confidence_multiplier

                # Convert to base currency units
                order_size = order_size_usdt / current_price

                logger.info(f"BUY order size calculated: {symbol} ${order_size_usdt:.2f} ({order_size:.6f} units) @ confidence {signal_confidence:.2f}")

            elif action.lower() == 'sell':
                # For sells, size based on current position
                if position_qty <= 0:
                    logger.warning(f"SELL order requested but no position: {symbol}")
                    return 0.0

                # Base sell percentage (75-100% based on confidence) - MORE AGGRESSIVE
                base_sell_pct = 0.75 + (signal_confidence * 0.25)  # 75% to 100%
                order_size = position_qty * base_sell_pct

                logger.info(f"SELL order size calculated: {symbol} {order_size:.6f} units ({base_sell_pct*100:.0f}%) @ confidence {signal_confidence:.2f}")

            else:
                logger.error(f"Unknown action: {action}")
                return 0.0

            # Apply minimum order size check - REDUCED TO $2 TO ALLOW SMALLER ORDERS
            min_order_value = 2.0  # $2 minimum (matches MIN_ORDER_USDT in order_manager)
            if order_size * current_price < min_order_value:
                logger.warning(f"Order size too small: ${order_size * current_price:.2f} < ${min_order_value:.2f}")
                # For paper mode, we don't want to return 0 for small orders - allow them
                if paper_mode:
                    logger.warning("Paper mode: Allowing small order despite minimum size requirement")
                else:
                    return 0.0

            return order_size

        except Exception as e:
            logger.error(f"Error calculating order size for {symbol}: {e}")
            # In paper mode, return a fallback order size if calculation fails
            if self.config.get("PAPER_MODE", False) or self.config.get("OPERATION_MODE", "").upper() == "PAPER":
                fallback_notional = 75.0  # $75 fallback order size
                fallback_qty = fallback_notional / current_price
                logger.warning(f"Using fallback order size for {symbol}: {fallback_qty:.6f} units")
                return fallback_qty
            return 0.0
