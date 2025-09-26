# l1_operational/order_manager.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from core.logging import logger
from .config import ConfigObject
from l2_tactic.models import TacticalSignal

class OrderManager:
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

        logger.info(f"✅ OrderManager initialized - Mode: {ConfigObject.OPERATION_MODE}")
        logger.info(f"✅ Limits BTC: {ConfigObject.RISK_LIMITS['MAX_ORDER_SIZE_BTC']}, ETH: {ConfigObject.RISK_LIMITS['MAX_ORDER_SIZE_ETH']}")

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
                        current_price = float(market_data["close"].iloc[-1])
                        logger.info(f"🐛 DEBUG OrderManager - {signal.symbol} current_price from DataFrame: {current_price}")
                    elif isinstance(market_data, dict):
                        # Handle dict format
                        if 'close' in market_data:
                            current_price = float(market_data['close'])
                            logger.info(f"🐛 DEBUG OrderManager - {signal.symbol} current_price from dict: {current_price}")
                        else:
                            logger.warning(f"⚠️ No 'close' key in market data dict for {signal.symbol}")
                            continue
                    else:
                        logger.warning(f"⚠️ Unsupported market data format for {signal.symbol}: {type(market_data)}")
                        continue
                    max_position = self.risk_limits.get(f"MAX_ORDER_SIZE_{signal.symbol[:3]}", 0.0)
                    current_position = portfolio.get(signal.symbol, {}).get("position", 0.0)

                    # ✅ FIXED: Dynamic threshold adjustment based on market conditions
                    # Get market volatility and risk context
                    l3_context = state.get("l3_output", {})
                    volatility_forecast = l3_context.get("volatility_forecast", {}).get(signal.symbol, 0.03)
                    risk_appetite = l3_context.get("risk_appetite", 0.5)

                    # CRITICAL FIX: Much lower minimum for BTCUSDT to allow signals to execute
                    if signal.symbol == "BTCUSDT":
                        base_min_order = 0.5  # Allow BTC orders as low as $0.50 to execute signals
                    else:
                        base_min_order = 1.0  # Other assets minimum $1.0

                    # Adjust minimum based on volatility (higher vol = slightly higher min to avoid slippage)
                    vol_multiplier = max(0.3, min(1.5, volatility_forecast * 30))  # 0.3x to 1.5x based on vol %
                    dynamic_min_order = base_min_order * vol_multiplier

                    # Adjust based on risk appetite (higher risk = smaller orders)
                    risk_multiplier = 1.5 - risk_appetite * 0.5  # 1.0 for high risk, 1.5 for low risk
                    dynamic_min_order *= risk_multiplier

                    # Ensure minimum doesn't go below $0.50 for BTC or $1.00 for others, max $25
                    if signal.symbol == "BTCUSDT":
                        dynamic_min_order = max(0.5, min(25.0, dynamic_min_order))
                    else:
                        dynamic_min_order = max(1.0, min(25.0, dynamic_min_order))

                    logger.info(f"📊 Dynamic thresholds for {signal.symbol}: min_order=${dynamic_min_order:.2f}, vol={volatility_forecast:.4f}, risk={risk_appetite:.2f}")

                    # ✅ FIXED: Proper buy/sell/hold logic with dynamic thresholds
                    if signal.side == "buy":
                        # Buy logic
                        usdt_balance = portfolio.get("USDT", {}).get("free", 0.0)
                        if usdt_balance < dynamic_min_order:
                            logger.warning(f"⚠️ Insufficient USDT balance: {usdt_balance:.2f} < {dynamic_min_order:.2f}")
                            continue

                        # Dynamic order sizing based on signal strength and market conditions
                        base_order_pct = 0.05  # Base 5% of balance
                        strength_multiplier = getattr(signal, "strength", 0.5) * 2.0  # 0.5 to 2.0x
                        vol_adjustment = max(0.5, 1.0 - volatility_forecast * 50)  # Reduce size in high vol

                        order_pct = base_order_pct * strength_multiplier * vol_adjustment
                        order_value = min(usdt_balance * order_pct, 200.0)  # Cap at $200
                        quantity = order_value / current_price

                        if current_position + quantity > max_position:
                            quantity = max(0.0, max_position - current_position)

                    elif signal.side == "sell":
                        # Sell logic - only if we have a position
                        if current_position <= 0:
                            logger.warning(f"⚠️ No position to sell for {signal.symbol}")
                            continue

                        # Dynamic sell sizing based on signal strength
                        strength_multiplier = getattr(signal, "strength", 0.5) * 2.0
                        sell_pct = min(0.3, 0.1 * strength_multiplier)  # 3% to 30% of position
                        quantity = -current_position * sell_pct  # Negative for sell

                    else:  # hold
                        # Do nothing for hold signals
                        logger.info(f"📊 Hold signal for {signal.symbol} - no action taken")
                        continue

                    # Check against dynamic minimum order size
                    order_value_usdt = abs(quantity) * current_price
                    if order_value_usdt >= dynamic_min_order:
                        # Create main market order
                        order = {
                            "symbol": signal.symbol,
                            "side": signal.side,
                            "type": "MARKET",
                            "quantity": quantity,
                            "price": current_price,
                            "timestamp": datetime.utcnow().isoformat(),
                            "signal_strength": getattr(signal, "strength", 0.5),
                            "signal_source": getattr(signal, "source", "tactical"),
                            "dynamic_min_order": dynamic_min_order,
                            "volatility_used": volatility_forecast,
                            "risk_appetite_used": risk_appetite,
                            "status": "pending"
                        }

                        # CRÍTICO: Agregar STOP-LOSS order si está disponible en la señal
                        stop_loss = getattr(signal, "stop_loss", None)
                        if stop_loss and stop_loss > 0:
                            # Validar stop-loss según dirección
                            if signal.side == "buy" and stop_loss < current_price:
                                sl_order = {
                                    "symbol": signal.symbol,
                                    "side": "SELL",  # Stop-loss siempre vende
                                    "type": "STOP_LOSS",
                                    "quantity": abs(quantity),  # Siempre positivo
                                    "stop_price": stop_loss,
                                    "price": current_price,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "signal_strength": getattr(signal, "strength", 0.5),
                                    "signal_source": "stop_loss_protection",
                                    "parent_order": f"{signal.symbol}_{signal.side}_{datetime.utcnow().isoformat()}",
                                    "status": "pending"
                                }
                                orders.append(sl_order)
                                logger.info(f"🛡️ STOP-LOSS generado: {signal.symbol} SELL {abs(quantity):.4f} @ stop={stop_loss:.2f}")
                            elif signal.side == "sell" and stop_loss > current_price:
                                sl_order = {
                                    "symbol": signal.symbol,
                                    "side": "BUY",  # Stop-loss para short vende para cubrir
                                    "type": "STOP_LOSS",
                                    "quantity": abs(quantity),  # Siempre positivo
                                    "stop_price": stop_loss,
                                    "price": current_price,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "signal_strength": getattr(signal, "strength", 0.5),
                                    "signal_source": "stop_loss_protection",
                                    "parent_order": f"{signal.symbol}_{signal.side}_{datetime.utcnow().isoformat()}",
                                    "status": "pending"
                                }
                                orders.append(sl_order)
                                logger.info(f"🛡️ STOP-LOSS generado: {signal.symbol} BUY {abs(quantity):.4f} @ stop={stop_loss:.2f}")

                        orders.append(order)
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
        """Execute a list of orders, including STOP_LOSS orders."""
        if not orders:
            return []

        processed_orders = []

        for order in orders:
            try:
                order_type = order.get("type", "MARKET")

                if self.execution_config["PAPER_MODE"]:
                    # Simular ejecución en paper mode
                    if order_type == "STOP_LOSS":
                        # Stop-loss orders se simulan como pendientes hasta activación
                        order["status"] = "placed"  # No "filled" hasta que se active
                        order["order_id"] = f"sl_{order['symbol']}_{order['side']}_{order['stop_price']}"
                        logger.info(f"🛡️ STOP-LOSS simulado: {order['symbol']} {order['side']} {order['quantity']:.4f} @ stop={order['stop_price']:.2f}")
                    else:
                        # Market orders se ejecutan inmediatamente
                        order["status"] = "filled"
                        order["filled_price"] = order["price"]
                        order["filled_quantity"] = order["quantity"]
                        # Calculate commission consistently with portfolio manager
                        order_value = abs(order["price"] * order["quantity"])
                        order["commission"] = order_value * 0.001  # 0.1% fee
                        logger.info(f"✅ MARKET ejecutado: {order['symbol']} {order['side']} {order['quantity']:.4f} @ {order['price']:.2f}")

                elif order_type == "STOP_LOSS":
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

                else:
                    # Market orders en modo real (no implementado aún)
                    raise NotImplementedError(f"Real market orders not implemented yet. Order: {order}")

                processed_orders.append(order)

            except Exception as e:
                logger.error(f"❌ Error executing order {order}: {e}")
                order["status"] = "rejected"
                order["error"] = str(e)
                processed_orders.append(order)

        return processed_orders
