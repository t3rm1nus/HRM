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
                # Check if we're in simulated mode
                if hasattr(self.portfolio, 'mode') and self.portfolio.mode == "simulated":
                    available_usdt = self.portfolio.get_balance('USDT')
                    logger.debug(f"📊 SIMULATED MODE: Using portfolio USDT balance: ${available_usdt:.2f}")
                else:
                    # Real mode - use exchange client
                    available_usdt = self.portfolio.get_available_balance("USDT")
                    logger.debug(f"📊 REAL MODE: Using exchange USDT balance: ${available_usdt:.2f}")

                # Base allocation (1-5% of available USDT based on confidence)
                base_allocation_pct = 0.01 + (signal_confidence * 0.04)  # 1% to 5%
                order_size_usdt = available_usdt * base_allocation_pct

                # Apply confidence multiplier
                confidence_multiplier = 1.0 + (signal_confidence - 0.5) * 0.5  # 0.75 to 1.25
                order_size_usdt *= confidence_multiplier

                # Convert to base currency units
                order_size = order_size_usdt / current_price

                logger.info(f"BUY order size calculated: {symbol} ${order_size_usdt:.2f} (${order_size:.6f} units) @ confidence {signal_confidence:.2f}")

            elif action.lower() == 'sell':
                # For sells, size based on current position
                if position_qty <= 0:
                    logger.warning(f"SELL order requested but no position: {symbol}")
                    return 0.0

                # Base sell percentage (50-100% based on confidence)
                base_sell_pct = 0.5 + (signal_confidence * 0.5)  # 50% to 100%
                order_size = position_qty * base_sell_pct

                logger.info(f"SELL order size calculated: {symbol} {order_size:.6f} units ({base_sell_pct*100:.0f}%) @ confidence {signal_confidence:.2f}")

            else:
                logger.error(f"Unknown action: {action}")
                return 0.0

            # Apply minimum order size check
            min_order_value = 5.0  # $5 minimum
            if order_size * current_price < min_order_value:
                logger.warning(f"Order size too small: ${order_size * current_price:.2f} < ${min_order_value:.2f}")
                return 0.0

            return order_size

        except Exception as e:
            logger.error(f"Error calculating order size for {symbol}: {e}")
            return 0.0
