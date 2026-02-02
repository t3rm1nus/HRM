from typing import Dict, Any
from core.logging import logger


class OrderValidators:
    """
    Validates trading orders according to HRM rules and market conditions.
    """

    def __init__(self, config):
        """
        Initialize OrderValidators with configuration.

        Args:
            config: Configuration object (can be dict or ConfigObject)
        """
        self.config = config

        # Handle both dict and ConfigObject instances
        if hasattr(config, 'RISK_LIMITS'):
            # ConfigObject instance - extract values from its attributes
            self.min_order_value = config.RISK_LIMITS.get('MIN_ORDER_SIZE_USDT', 5.0)  # Minimum $5 order
            self.max_position_pct = config.PORTFOLIO_LIMITS.get('MAX_PORTFOLIO_EXPOSURE_BTC', 0.30)  # Maximum 30% of portfolio
        else:
            # Fallback for dict-like objects
            self.min_order_value = getattr(config, 'get', lambda key, default: default)('MIN_ORDER_VALUE', 5.0)
            self.max_position_pct = getattr(config, 'get', lambda key, default: default)('MAX_POSITION_PCT', 0.30)

        logger.info(f"✅ OrderValidators initialized - Min order: ${self.min_order_value}, Max position: {self.max_position_pct*100:.1f}%")

    def validate_order(self, symbol: str, action: str, quantity: float,
                      current_price: float, portfolio_value: float, position_qty: float) -> Dict[str, Any]:
        """
        Validate a trading order according to HRM trading rules.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            action: Order action ('buy' or 'sell')
            quantity: Order quantity in base asset
            current_price: Current market price
            portfolio_value: Total portfolio value in USDT
            position_qty: Current position quantity

        Returns:
            dict: {'valid': bool, 'reason': str}
        """
        try:
            logger.debug(f"🔍 Validating {action.upper()} order: {symbol} qty={quantity:.6f} @ ${current_price:.2f}")

            # Calculate order value
            order_value_usdt = abs(quantity) * current_price

            # 1. Minimum order value check
            if order_value_usdt < self.min_order_value:
                return {
                    'valid': False,
                    'reason': f"Order value ${order_value_usdt:.2f} below minimum ${self.min_order_value:.2f}"
                }

            # 2. Action validation
            if action.lower() not in ['buy', 'sell']:
                return {
                    'valid': False,
                    'reason': f"Invalid action '{action}'. Must be 'buy' or 'sell'"
                }

            # 3. Position validation for sell orders
            if action.lower() == 'sell':
                if position_qty <= 0:
                    return {
                        'valid': False,
                        'reason': f"No position to sell for {symbol} (current position: {position_qty:.6f})"
                    }
                if abs(quantity) > position_qty:
                    return {
                        'valid': False,
                        'reason': f"Insufficient position: {position_qty:.6f} < {abs(quantity):.6f}"
                    }

            # 4. Position size limit validation for buy orders
            if action.lower() == 'buy':
                new_position_value = (position_qty + quantity) * current_price
                max_allowed_value = portfolio_value * self.max_position_pct

                if new_position_value > max_allowed_value:
                    return {
                        'valid': False,
                        'reason': f"Position size ${new_position_value:.2f} exceeds maximum ${max_allowed_value:.2f} ({self.max_position_pct*100:.1f}% of portfolio)"
                    }

            # 5. Price validation
            if current_price <= 0:
                return {
                    'valid': False,
                    'reason': f"Invalid price: ${current_price:.2f}"
                }

            # 6. Quantity validation
            if quantity == 0:
                return {
                    'valid': False,
                    'reason': "Order quantity cannot be zero"
                }

            logger.info(f"✅ Order validation passed: {symbol} {action.upper()} {quantity:.6f} @ ${current_price:.2f} (${order_value_usdt:.2f})")
            return {
                'valid': True,
                'reason': f"Order meets all validation criteria (${order_value_usdt:.2f})"
            }

        except Exception as e:
            logger.error(f"❌ Error validating order for {symbol}: {e}")
            return {
                'valid': False,
                'reason': f"Validation error: {str(e)}"
            }
