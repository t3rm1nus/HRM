from typing import Dict, Any, Optional
from datetime import datetime
from core.logging import logger
from .simulated_exchange_client import SimulatedExchangeClient


class OrderExecutors:
    """
    Executes trading orders through simulated or real trading interfaces.
    """

    def __init__(self, state_manager, portfolio_manager, config: Dict[str, Any], simulated_client=None):
        """
        Initialize OrderExecutors with required managers and configuration.

        Args:
            state_manager: State management object
            portfolio_manager: Portfolio management object
            config: Configuration object
            simulated_client: Pre-initialized SimulatedExchangeClient instance (DI)
        """
        self.state_manager = state_manager
        self.portfolio_manager = portfolio_manager
        self.config = config

        # Execution settings - Clear distinction between paper mode and testnet
        self.paper_mode = config.get('PAPER_MODE', True)
        self.use_testnet = config.get('USE_TESTNET', False) and not self.paper_mode

        # Initialize simulated exchange client for paper trading (DI)
        if self.paper_mode:
            logger.info("🧪 Paper trading with simulated execution")
            logger.info("📊 Using REAL Binance market data (public endpoints)")
            
            if simulated_client is None:
                logger.critical("🚨 FATAL: Paper mode requires a pre-initialized SimulatedExchangeClient")
                raise RuntimeError("Paper mode requires a pre-initialized SimulatedExchangeClient")
                
            self.simulated_client = simulated_client
            logger.info(f"✅ SimulatedExchangeClient initialized with balances: {simulated_client.get_balances()}")
        else:
            self.simulated_client = None

        logger.info(f"✅ OrderExecutors initialized (paper_mode: {self.paper_mode}, testnet: {self.use_testnet})")

    def execute_order(self, symbol: str, action: str, quantity: float,
                     current_price: float, stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a trading order with optional stop-loss and take-profit orders.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            action: Order action ('buy' or 'sell')
            quantity: Order quantity in base asset
            current_price: Current market price
            stop_loss: Optional stop-loss price
            take_profit: Optional take-profit price

        Returns:
            dict: Order execution result with status and details
        """
        try:
            logger.info(f"🔄 Executing {action.upper()} order: {symbol} qty={quantity:.6f} @ ${current_price:.2f}")

            # Validate inputs
            if action.lower() not in ['buy', 'sell']:
                return {
                    'status': 'rejected',
                    'symbol': symbol,
                    'action': action,
                    'reason': f"Invalid action '{action}'. Must be 'buy' or 'sell'",
                    'timestamp': datetime.now().isoformat()
                }

            if current_price <= 0:
                return {
                    'status': 'rejected',
                    'symbol': symbol,
                    'action': action,
                    'reason': f"Invalid price: ${current_price:.2f}",
                    'timestamp': datetime.now().isoformat()
                }

            if quantity == 0:
                return {
                    'status': 'rejected',
                    'symbol': symbol,
                    'action': action,
                    'reason': "Order quantity cannot be zero",
                    'timestamp': datetime.now().isoformat()
                }

            # Calculate order value
            order_value_usdt = abs(quantity) * current_price

            # In paper mode (simulation), use SimulatedExchangeClient
            if self.paper_mode:
                try:
                    # Execute order through simulated client
                    try:
                        trade_result = self.simulated_client.execute_order(
                            symbol=symbol,
                            side=action.upper(),
                            qty=quantity,
                            market_price=current_price
                        )
                    except Exception as client_error:
                        logger.error(f"❌ Simulated client error: {client_error}")
                        return {
                            'status': 'failed',
                            'symbol': symbol,
                            'action': action,
                            'reason': f"Simulated client error: {str(client_error)}",
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Get updated balances
                    current_balances = self.simulated_client.get_balances()
                    
                    execution_result = {
                        'status': 'executed',
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'market_price': current_price,
                        'execution_price': trade_result['execution_price'],
                        'order_value': trade_result['cost'],
                        'commission': trade_result['fee'],
                        'slippage_cost': trade_result['slippage_cost'],
                        'filled_quantity': quantity,
                        'timestamp': datetime.now().isoformat(),
                        'execution_type': 'SIMULATED_TRADE',
                        'order_id': trade_result['trade_id'],
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'balances_after': current_balances
                    }

                    # Update portfolio via portfolio manager if available
                    if self.portfolio_manager and hasattr(self.portfolio_manager, 'update_position'):
                        try:
                            self.portfolio_manager.update_position(
                                symbol=symbol,
                                action=action,
                                quantity=quantity,
                                price=trade_result['execution_price'],
                                commission=trade_result['fee']
                            )
                            logger.debug(f"📊 Portfolio updated for {symbol} {action} {quantity:.6f}")
                        except Exception as e:
                            logger.warning(f"⚠️ Portfolio update failed for {symbol}: {e}")

                    logger.info(f"✅ SIMULATED TRADE EXECUTED: {symbol} {action.upper()} {quantity:.6f} @ ${trade_result['execution_price']:.2f} (slippage: {self.simulated_client.slippage*100:.2f}%, fee: {self.simulated_client.fee*100:.2f}%)")
                    logger.info(f"   Balances: {current_balances}")

                except Exception as e:
                    logger.error(f"❌ Simulated order execution failed: {e}")
                    return {
                        'status': 'failed',
                        'symbol': symbol,
                        'action': action,
                        'reason': f"Simulation error: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    }

            else:
                # Real trading execution would go here
                logger.warning("🚨 REAL TRADING NOT IMPLEMENTED: Falling back to paper mode")
                return {
                    'status': 'rejected',
                    'symbol': symbol,
                    'action': action,
                    'reason': "Real trading execution not implemented",
                    'timestamp': datetime.now().isoformat()
                }

            return execution_result

        except Exception as e:
            logger.error(f"❌ Error executing order for {symbol}: {e}")
            return {
                'status': 'failed',
                'symbol': symbol,
                'action': action,
                'reason': f"Execution error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
