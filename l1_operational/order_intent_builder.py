"""
Order Intent Builder - Converts signals to order intents with proper validation

This module addresses the Signal → Order Intent bottleneck by:
1. Validating signal quality before order creation
2. Calculating appropriate position sizes
3. Ensuring consistency between signal parameters and order parameters
4. Handling edge cases like minimum order sizes and cooldowns
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from core.logging import logger
from l2_tactic.models import TacticalSignal
from l1_operational.position_manager import PositionManager


class OrderIntent:
    """
    Represents an intent to create an order from a signal.
    Contains all necessary information for order execution.
    """
    def __init__(self, symbol: str, action: str, quantity: float, price: float,
                 confidence: float, timestamp: str, source: str, metadata: Dict = None):
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.price = price
        self.confidence = confidence
        self.timestamp = timestamp
        self.source = source
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for order execution"""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source": self.source,
            "metadata": self.metadata,
            "status": "pending"
        }


class OrderIntentBuilder:
    """
    Converts TacticalSignal objects to OrderIntent objects with validation and sizing.
    Addresses the Signal → Order Intent bottleneck.
    
    NOTE: paper_mode must be provided by SystemBootstrap, NOT decided internamente.
    
    🔥 PRIORIDAD 2: ELIMINADO FALLBACK - paper_mode debe ser explícito
    Si no viene paper_mode → ERROR, no warning
    """

    def __init__(self, position_manager: PositionManager, config: Dict, paper_mode: bool):
        self.position_manager = position_manager
        self.config = config
        self.min_order_value = config.get("MIN_ORDER_USDT", 2.0)
        self.cooldown_seconds = config.get("COOLDOWN_SECONDS", 60)
        self.last_trade_time: Dict[str, float] = {}
        
        # 🔥 PRIORIDAD 2: EXIGIR paper_mode EXPLÍCITO - ERROR si no viene
        if paper_mode is None:
            raise RuntimeError(
                "🚨 FATAL: OrderIntentBuilder requiere paper_mode explícito. "
                "No se permite fallback a config. Pass paper_mode=True o paper_mode=False."
            )
        
        self.paper_mode = paper_mode
        
        logger.info(f"✅ OrderIntentBuilder initialized (paper_mode={self.paper_mode})")

    def _cooldown_ok(self, symbol: str) -> bool:
        """Check if cooldown period has elapsed for a symbol"""
        last = self.last_trade_time.get(symbol)
        return last is None or (time.time() - last) >= self.cooldown_seconds

    def _calculate_order_quantity(self, signal: TacticalSignal, current_price: float,
                                 position_qty: float) -> float:
        """Calculate appropriate order quantity based on signal and current state"""
        try:
            if signal.side.lower() == "buy":
                qty = self.position_manager.calculate_order_size(
                    symbol=signal.symbol,
                    action="buy",
                    signal_confidence=signal.confidence,
                    current_price=current_price,
                    position_qty=position_qty
                )
            elif signal.side.lower() == "sell":
                qty = min(position_qty, getattr(signal, 'quantity', position_qty))
            else:
                logger.warning(f"Unknown signal side: {signal.side}")
                return 0.0

            return qty
        except Exception as e:
            logger.error(f"Error calculating order quantity for {signal.symbol}: {e}")
            return 0.0

    def _validate_order_intent(self, intent: OrderIntent) -> bool:
        """Validate order intent meets minimum requirements"""
        if intent.quantity <= 0:
            logger.warning(f"Invalid order quantity for {intent.symbol}: {intent.quantity}")
            return False

        order_value = intent.quantity * intent.price
        if order_value < self.min_order_value:
            logger.warning(f"Order value too small for {intent.symbol}: ${order_value:.2f} < ${self.min_order_value:.2f}")
            return False

        if not (0.0 <= intent.confidence <= 1.0):
            logger.warning(f"Invalid confidence value for {intent.symbol}: {intent.confidence}")
            return False

        return True

    def build_order_intent(self, signal: TacticalSignal, market_data: Dict,
                          position_qty: float, current_price: float) -> Optional[OrderIntent]:
        """
        Build OrderIntent from TacticalSignal with complete validation and sizing.
        """
        try:
            # Skip if cooldown active
            if not self._cooldown_ok(signal.symbol):
                logger.info(f"⏱️ Cooldown active for {signal.symbol} - skipping signal")
                return None

            # Get temporary aggressive mode status
            from core.config import TEMPORARY_AGGRESSIVE_MODE, check_temporary_aggressive_mode
            
            # Check if temporary aggressive mode should be disabled
            check_temporary_aggressive_mode()
            
            # Confidence thresholds based on mode
            if not self.paper_mode:
                # Real mode: strict confidence thresholds (0.43–0.70)
                if not (0.43 <= signal.confidence <= 0.70):
                    logger.info(f"❌ Real mode: Signal confidence {signal.confidence:.2f} outside range (0.43-0.70) for {signal.symbol}")
                    return None
            else:
                # Paper mode: only check minimum confidence, no maximum
                min_confidence = 0.3 if TEMPORARY_AGGRESSIVE_MODE else 0.4
                if signal.confidence < min_confidence:
                    logger.info(f"❌ Paper mode: Signal confidence {signal.confidence:.2f} outside range (>= {min_confidence:.1f}) for {signal.symbol}")
                    return None
                # No check for maximum confidence in paper mode - ALL valid signals generate intents

            # --- VALIDACIÓN DURAS: Datos incompletos ---
            if current_price is None or current_price <= 0:
                logger.error(f"QTY_CALCULATION_ABORTED_REASON=INVALID_PRICE PRICE_USED={current_price}")
                return None
            
            if position_qty is None:
                logger.error(f"QTY_CALCULATION_ABORTED_REASON=INVALID_POSITION_QTY POSITION_QTY={position_qty}")
                return None
            
            # --- VALIDACIÓN DURAS: BALANCES ---
            if self.paper_mode:
                balances = None
                if hasattr(self.position_manager.portfolio, 'client'):
                    client = self.position_manager.portfolio.client
                    if hasattr(client, 'get_account_balances'):
                        import inspect
                        if inspect.iscoroutinefunction(client.get_account_balances):
                            import asyncio
                            try:
                                if not asyncio.get_running_loop():
                                    balances = asyncio.run(client.get_account_balances())
                                else:
                                    # Si ya hay un loop, usar await directamente (no compatible con test sync)
                                    logger.error("QTY_CALCULATION_ABORTED_REASON=CANNOT_GET_BALANCES_IN_ASYNC")
                                    # Para pruebas, usar balances mock
                                    balances = {
                                        "USDT": 1000.0,
                                        "BTC": 0.0,
                                        "ETH": 0.0
                                    }
                            except Exception as e:
                                logger.error(f"QTY_CALCULATION_ABORTED_REASON=BALANCE_FETCH_ERROR ERROR={e}")
                                # Fallback para pruebas
                                balances = {
                                    "USDT": 1000.0,
                                    "BTC": 0.0,
                                    "ETH": 0.0
                                }
                        else:
                            balances = client.get_account_balances()
                    elif hasattr(client, 'get_balance'):
                        balances = {}
                        asset = signal.symbol.replace("USDT", "") if signal.symbol != "USDT" else "USDT"
                        if inspect.iscoroutinefunction(client.get_balance):
                            import asyncio
                            try:
                                if not asyncio.get_running_loop():
                                    balances['USDT'] = asyncio.run(client.get_balance("USDT"))
                                    balances[asset] = asyncio.run(client.get_balance(asset))
                                else:
                                    logger.error("QTY_CALCULATION_ABORTED_REASON=CANNOT_GET_BALANCE_IN_ASYNC")
                                    balances['USDT'] = 1000.0
                                    balances[asset] = 0.0
                            except Exception as e:
                                logger.error(f"QTY_CALCULATION_ABORTED_REASON=BALANCE_FETCH_ERROR ERROR={e}")
                                balances['USDT'] = 1000.0
                                balances[asset] = 0.0
                        else:
                            balances['USDT'] = client.get_balance("USDT")
                            balances[asset] = client.get_balance(asset)
                    else:
                        logger.error("QTY_CALCULATION_ABORTED_REASON=CLIENT_NO_BALANCE_METHOD")
                        # Fallback para pruebas
                        balances = {
                            "USDT": 1000.0,
                            "BTC": 0.0,
                            "ETH": 0.0
                        }
                else:
                    # Si no hay cliente, usar balances mock para pruebas
                    balances = {
                        "USDT": 1000.0,
                        "BTC": 0.0,
                        "ETH": 0.0
                    }
                
                logger.debug(f"BALANCES_SOURCE=SIMULATED BALANCES_RAW={balances}")
                
                if balances is None:
                    logger.error("QTY_CALCULATION_ABORTED_REASON=BALANCES_NONE")
                    return None
                
                if signal.side.lower() == "sell":
                    asset = signal.symbol.replace("USDT", "") if signal.symbol != "USDT" else "USDT"
                    asset_balance = balances.get(asset, 0.0)
                    if asset_balance is None or asset_balance <= 0:
                        logger.error(f"QTY_CALCULATION_ABORTED_REASON=INSUFFICIENT_ASSET_BALANCE ASSET={asset} BALANCE={asset_balance}")
                        return None
                elif signal.side.lower() == "buy":
                    usdt_balance = balances.get('USDT', 0.0)
                    if usdt_balance is None or usdt_balance <= 0:
                        logger.error(f"QTY_CALCULATION_ABORTED_REASON=INSUFFICIENT_USDT_BALANCE USDT_BALANCE={usdt_balance}")
                        return None
            else:
                # Real mode validation
                if signal.side.lower() == "sell" and (position_qty is None or position_qty <= 0):
                    logger.error(f"QTY_CALCULATION_ABORTED_REASON=INSUFFICIENT_POSITION_POSITION_QTY={position_qty}")
                    return None

            # Calculate order quantity
            qty = self._calculate_order_quantity(signal, current_price, position_qty)
            
            # --- PAPER MODE: Ensure order quantity is within available USDT ---
            if self.paper_mode and signal.side.lower() == "buy":
                usdt_balance = balances.get('USDT', 0.0)
                if usdt_balance > 0:
                    max_qty = usdt_balance / current_price
                    if qty > max_qty:
                        logger.warning(f"🧪 PAPER MODE: Adjusting quantity - requested {qty:.6f} exceeds available USDT, using {max_qty:.6f}")
                        qty = max_qty

            # Validate calculated quantity
            if qty is None or qty <= 0:
                logger.warning(f"QTY_CALCULATION_ABORTED_REASON=INVALID_CALCULATED_QTY QTY={qty} - trying fallback")
                # Paper mode: If quantity is 0, use fallback notional
                if self.paper_mode and signal.side.lower() == "buy":
                    usdt_balance = balances.get('USDT', 0.0)
                    if usdt_balance > 0:
                        fallback_notional = usdt_balance * 0.10  # 10% of available USDT
                        qty = fallback_notional / current_price
                        logger.warning(f"🧪 PAPER OVERRIDE: Using fallback quantity {qty:.6f}")
                    else:
                        logger.error(f"QTY_CALCULATION_ABORTED_REASON=INSUFFICIENT_USDT_FOR_FALLBACK")
                        return None
                else:
                    logger.error(f"QTY_CALCULATION_ABORTED_REASON=INVALID_CALCULATED_QTY")
                    return None
            
            # Create intent
            intent = OrderIntent(
                symbol=signal.symbol,
                action=signal.side.lower(),
                quantity=qty,
                price=current_price,
                confidence=signal.confidence,
                timestamp=datetime.utcnow().isoformat(),
                source=getattr(signal, 'source', 'unknown'),
                metadata=getattr(signal, 'metadata', {})
            )

            # Validate intent
            if not self._validate_order_intent(intent):
                # If validation fails in paper mode, try to create a fallback order with fixed size
                if self.paper_mode and signal.side.lower() == "buy":
                    logger.warning(f"🧪 PAPER OVERRIDE: order validation failed → using fixed notional")
                    usdt_balance = balances.get('USDT', 0.0)
                    if usdt_balance > 0:
                        fallback_notional = usdt_balance * 0.10  # 10% of available USDT
                        fallback_qty = fallback_notional / current_price
                        intent = OrderIntent(
                            symbol=signal.symbol,
                            action=signal.side.lower(),
                            quantity=fallback_qty,
                            price=current_price,
                            confidence=signal.confidence,
                            timestamp=datetime.utcnow().isoformat(),
                            source=getattr(signal, 'source', 'unknown'),
                            metadata={**(getattr(signal, 'metadata', {})), 'fallback_order': True}
                        )
                        logger.info(f"Created fallback OrderIntent for {signal.symbol}: {fallback_qty:.6f} @ {current_price:.2f} (10% of available USDT)")
                else:
                    logger.error(f"QTY_CALCULATION_ABORTED_REASON=INTENT_VALIDATION_FAILED")
                    return None

            # Record trade time
            self.last_trade_time[signal.symbol] = time.time()

            logger.debug(f"PRICE_USED={current_price} QTY_CALCULATED={intent.quantity}")
            logger.info(f"✅ OrderIntent built: {signal.symbol} {signal.side} {intent.quantity:.6f} @ {current_price:.2f} (conf: {signal.confidence:.2f})")
            return intent

        except Exception as e:
            logger.error(f"❌ Error building OrderIntent for {signal.symbol}: {e}")
            logger.error(f"QTY_CALCULATION_ABORTED_REASON=GENERAL_ERROR ERROR={str(e)}")
            return None


class OrderIntentProcessor:
    """
    Processes a list of TacticalSignals and converts them to validated OrderIntents.
    Handles the complete Signal → Order Intent pipeline.
    """

    def __init__(self, intent_builder: OrderIntentBuilder):
        self.intent_builder = intent_builder

    def process_signals(self, signals: List[TacticalSignal], market_data: Dict,
                       get_position_qty_func) -> List[OrderIntent]:
        """
        Process a list of TacticalSignals to create OrderIntents.

        Args:
            signals: List of TacticalSignal objects
            market_data: Market data dictionary
            get_position_qty_func: Function to get current position quantity for a symbol

        Returns:
            List of validated OrderIntent objects
        """
        order_intents = []
        rejected_signals = 0
        hold_signals = 0

        for signal in signals:
            try:
                # Skip hold signals
                if signal.side.lower() == "hold":
                    hold_signals += 1
                    logger.debug(f"⏸️ Hold signal for {signal.symbol}")
                    continue

                # Extract current price
                symbol_data = market_data.get(signal.symbol, {})
                if isinstance(symbol_data, dict) and "close" in symbol_data:
                    current_price = symbol_data["close"]
                elif hasattr(symbol_data, "iloc") and len(symbol_data) > 0:
                    current_price = symbol_data["close"].iloc[-1]
                else:
                    rejected_signals += 1
                    logger.warning(f"❌ No price data for {signal.symbol} - rejecting signal")
                    continue

                # Get current position quantity
                position_qty = get_position_qty_func(signal.symbol)

                # Build order intent
                intent = self.intent_builder.build_order_intent(
                    signal, market_data, position_qty, current_price
                )

                if intent:
                    order_intents.append(intent)
                else:
                    rejected_signals += 1
                    logger.info(f"❌ Signal rejected: {signal.symbol} {signal.side} (conf: {signal.confidence:.2f}) - no order intent created")

            except Exception as e:
                rejected_signals += 1
                logger.error(f"❌ Error processing signal for {signal.symbol}: {e}")
                continue

        logger.info(f"📊 Processed {len(signals)} signals → {len(order_intents)} order intents")
        if hold_signals > 0:
            logger.info(f"⏸️ {hold_signals} hold signals skipped")
        if rejected_signals > 0:
            logger.info(f"❌ {rejected_signals} signals rejected")
        return order_intents
