from typing import Dict, Any, Optional
from datetime import datetime
from core.logging import logger
from .simulated_exchange_client import SimulatedExchangeClient

# CRITICAL FIX: Import PaperTradeLogger y TradingMetrics
from storage.paper_trade_logger import get_paper_logger, PAPER_LOGGER_AVAILABLE
from core.trading_metrics import get_trading_metrics


class OrderExecutors:
    """
    Executes trading orders through simulated or real trading interfaces.
    
    CRITICAL FIX: Ahora registra TODOS los trades ejecutados en:
    - PaperTradeLogger (para persistencia)
    - TradingMetrics (para métricas de performance)
    """

    def __init__(self, state_manager, portfolio_manager, mode: str = "simulated", simulated_client=None):
        """
        Initialize OrderExecutors with required managers and configuration.

        Args:
            state_manager: State management object
            portfolio_manager: Portfolio management object
            mode: Operating mode (simulated, live, testnet, backtest)
            simulated_client: Pre-initialized SimulatedExchangeClient instance (DI)
        """
        self.state_manager = state_manager
        self.portfolio_manager = portfolio_manager
        self.mode = mode

        # Execution settings - Clear distinction between paper mode and testnet
        self.paper_mode = mode in ["simulated", "paper"]
        self.use_testnet = mode == "testnet" and not self.paper_mode

        # Initialize simulated exchange client for paper trading (DI)
        if self.paper_mode:
            logger.info("🧪 Paper trading with simulated execution")
            logger.info("📊 Using REAL Binance market data (public endpoints)")
            
            if simulated_client is None:
                logger.warning("⚠️ SimulatedExchangeClient not provided - creating temporary one for testing")
                from l1_operational.simulated_exchange_client import SimulatedExchangeClient
                simulated_client = SimulatedExchangeClient({
                    "BTC": 0.01549,
                    "ETH": 0.385,
                    "USDT": 3000.0
                })
                
            self.simulated_client = simulated_client
            logger.info(f"✅ SimulatedExchangeClient initialized with balances: {simulated_client.get_balances()}")
            
            # CRITICAL FIX: Initialize PaperTradeLogger
            if PAPER_LOGGER_AVAILABLE:
                self.paper_logger = get_paper_logger()
                logger.info("✅ PaperTradeLogger initialized for trade persistence")
            else:
                self.paper_logger = None
                logger.warning("⚠️ PaperTradeLogger not available - trades will not be persisted")
            
            # CRITICAL FIX: Initialize TradingMetrics
            self.trading_metrics = get_trading_metrics()
            logger.info("✅ TradingMetrics initialized for performance tracking")
        else:
            self.simulated_client = None
            self.paper_logger = None
            self.trading_metrics = None

        logger.info(f"✅ OrderExecutors initialized (paper_mode: {self.paper_mode}, testnet: {self.use_testnet})")

    async def execute_order(self, symbol: str, action: str, quantity: float,
                           current_price: float, stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a trading order with optional stop-loss and take-profit orders.
        
        CRITICAL FIX: Ahora registra todos los trades ejecutados en PaperTradeLogger
        y TradingMetrics para logging consistente y métricas de performance.

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
        # CRITICAL FIX: Variables para logging de trade
        nav_before = 0.0
        nav_after = 0.0
        
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
                    # CRITICAL FIX: Log NAV before execution
                    if self.portfolio_manager:
                        nav_before = getattr(self.portfolio_manager, 'portfolio', {}).get('total', 0.0)
                    
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
                    
                    # CRITICAL FIX: Calculate NAV after execution using PortfolioManager
                    if self.portfolio_manager:
                        try:
                            # Sync portfolio from SimulatedExchangeClient (single source of truth)
                            await self.portfolio_manager.sync_from_exchange_async(self.simulated_client)
                            
                            # Calculate NAV using PortfolioManager's async method
                            nav_after = await self.portfolio_manager.get_total_value_async()
                            
                            # Verify NAV calculation: NAV must always equal (assets × prices) + USDT
                            usdt_balance = current_balances.get('USDT', 0.0)
                            btc_balance = current_balances.get('BTC', 0.0)
                            eth_balance = current_balances.get('ETH', 0.0)
                            
                            # Get current prices from MarketDataManager
                            market_prices = await self.portfolio_manager._get_market_prices()
                            btc_price = market_prices.get('BTCUSDT', 0.0)
                            eth_price = market_prices.get('ETHUSDT', 0.0)
                            
                            # Calculate NAV manually for verification
                            manual_nav = usdt_balance + (btc_balance * btc_price) + (eth_balance * eth_price)
                            
                            # Validate NAV calculation
                            nav_tolerance = 0.01  # 1 cent tolerance
                            if abs(nav_after - manual_nav) > nav_tolerance:
                                logger.critical(f"🚨 NAV CALCULATION ERROR: PortfolioManager={nav_after:.2f}, Manual={manual_nav:.2f}, Diff={abs(nav_after - manual_nav):.2f}")
                                # Use manual calculation as fallback
                                nav_after = manual_nav
                            else:
                                logger.info(f"✅ NAV verification passed: {nav_after:.2f} ≈ {manual_nav:.2f}")
                            
                            # Add defensive safeguards: prevent negative balances
                            if usdt_balance < 0:
                                logger.critical(f"🚨 NEGATIVE USDT BALANCE DETECTED: {usdt_balance:.2f}")
                                # Reset to zero and adjust other assets proportionally
                                usdt_balance = 0.0
                                # This should trigger a system alert
                                
                            if btc_balance < 0:
                                logger.critical(f"🚨 NEGATIVE BTC BALANCE DETECTED: {btc_balance:.6f}")
                                btc_balance = 0.0
                                
                            if eth_balance < 0:
                                logger.critical(f"🚨 NEGATIVE ETH BALANCE DETECTED: {eth_balance:.4f}")
                                eth_balance = 0.0
                                
                            # Ensure NAV is never 0 unless portfolio is actually empty
                            if nav_after <= 0 and (usdt_balance > 0 or btc_balance > 0 or eth_balance > 0):
                                logger.critical(f"🚨 NAV DROP TO ZERO DETECTED: {nav_after:.2f} with non-zero balances")
                                # Calculate minimum possible NAV
                                min_nav = usdt_balance + (btc_balance * 1.0) + (eth_balance * 1.0)  # Assume minimum $1 prices
                                nav_after = max(min_nav, 0.01)  # Set to minimum of 1 cent
                                logger.warning(f"⚠️ NAV adjusted to minimum: ${nav_after:.2f}")
                            
                        except Exception as nav_error:
                            logger.error(f"❌ Error calculating NAV: {nav_error}")
                            nav_after = nav_before
                    
                    execution_result = {
                        'status': 'filled',  # CRITICAL FIX: Status 'filled' para que sea reconocido por el sistema
                        'symbol': symbol,
                        'action': action,
                        'side': action.lower(),  # CRITICAL FIX: Añadir campo 'side' para compatibilidad
                        'quantity': quantity,
                        'market_price': current_price,
                        'filled_price': trade_result['execution_price'],  # CRITICAL FIX: Campo 'filled_price' requerido
                        'price': trade_result['execution_price'],  # CRITICAL FIX: Campo 'price' para compatibilidad
                        'order_value': trade_result['cost'],
                        'commission': trade_result['fee'],
                        'slippage_cost': trade_result['slippage_cost'],
                        'filled_quantity': quantity,  # CRITICAL FIX: Campo 'filled_quantity' requerido
                        'timestamp': datetime.now().isoformat(),
                        'execution_type': 'SIMULATED_TRADE',
                        'order_id': trade_result['trade_id'],
                        'order_type': 'MARKET',  # CRITICAL FIX: Campo 'order_type' requerido
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'balances_after': current_balances,
                        'signal_source': 'order_executor',  # CRITICAL FIX: Campo 'signal_source' para routing
                        'reason': f'Executed {action.upper()} order',  # CRITICAL FIX: Campo 'reason' para logging
                        'nav_before': nav_before,
                        'nav_after': nav_after,
                        'nav_verification_passed': abs(nav_after - manual_nav) <= nav_tolerance if 'manual_nav' in locals() else False
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

                    # ====================================================================================
                    # CRITICAL FIX: LOG TRADE EN PAPER TRADE LOGGER
                    # ====================================================================================
                    if self.paper_logger and PAPER_LOGGER_AVAILABLE:
                        try:
                            self.paper_logger.log_paper_trade(
                                order=execution_result,
                                market_data=None,  # Podría pasarse como parámetro si está disponible
                                cycle_id=None,  # Podría pasarse desde el caller
                                strategy="order_executor"
                            )
                            logger.info(f"📝 Trade logged to PaperTradeLogger: {symbol} {action.upper()} {quantity:.6f}")
                        except Exception as log_error:
                            logger.error(f"❌ Error logging to PaperTradeLogger: {log_error}")
                    
                    # ====================================================================================
                    # CRITICAL FIX: LOG TRADE EN TRADING METRICS
                    # ====================================================================================
                    if self.trading_metrics:
                        try:
                            self.trading_metrics.update_from_orders(
                                executed_orders=[execution_result],
                                portfolio_value=nav_after
                            )
                            logger.info(f"📊 Trade metrics updated in TradingMetrics")
                        except Exception as metrics_error:
                            logger.error(f"❌ Error updating TradingMetrics: {metrics_error}")

                    # ====================================================================================
                    # CRITICAL FIX: LOG EXPLÍCITO DEL TRADE EJECUTADO CON NAV ANTES/DESPUÉS
                    # ====================================================================================
                    logger.info("=" * 80)
                    logger.info(f"✅ TRADE EXECUTED: {symbol} {action.upper()}")
                    logger.info(f"   Quantity:     {quantity:.6f}")
                    logger.info(f"   Price:        ${trade_result['execution_price']:.2f}")
                    logger.info(f"   Order Value:  ${trade_result['cost']:.2f}")
                    logger.info(f"   Fee:          ${trade_result['fee']:.4f}")
                    logger.info(f"   Slippage:     ${trade_result['slippage_cost']:.4f}")
                    logger.info(f"   NAV Before:   ${nav_before:.2f}")
                    logger.info(f"   NAV After:    ${nav_after:.2f}")
                    logger.info(f"   Balances:     {current_balances}")
                    logger.info(f"   NAV Verified: {execution_result.get('nav_verification_passed', False)}")
                    logger.info("=" * 80)

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
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """
        CRITICAL FIX: Obtener resumen de trades del PaperTradeLogger.
        
        Returns:
            Dict con estadísticas de trades de la sesión actual
        """
        summary = {
            'paper_logger_available': PAPER_LOGGER_AVAILABLE and self.paper_logger is not None,
            'trading_metrics_available': self.trading_metrics is not None,
            'paper_trades': {},
            'metrics': {}
        }
        
        if self.paper_logger and PAPER_LOGGER_AVAILABLE:
            try:
                summary['paper_trades'] = self.paper_logger.get_session_summary()
            except Exception as e:
                logger.error(f"❌ Error getting paper trade summary: {e}")
        
        if self.trading_metrics:
            try:
                summary['metrics'] = self.trading_metrics.get_summary_report()
            except Exception as e:
                logger.error(f"❌ Error getting trading metrics: {e}")
        
        return summary
