# Fix path for imports - MUST be first
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Order Intent Builder - Converts signals to order intents with proper validation

This module addresses the Signal → Order Intent bottleneck by:
1. Validating signal quality before order creation
2. Calculating appropriate position sizes
3. Ensuring consistency between signal parameters and order parameters
4. Handling edge cases like minimum order sizes and cooldowns

CRITICAL FIXES:
- Capital allocation: Computes available capital ONCE per cycle
- Proportional allocation: Distributes capital proportionally among BUY intents
- Defensive safeguards: Prevents negative balances and over-allocation
- Logging consistency: Logs NAV before/after and balances after execution
- ASYNC FIX: Fully async - NO event loop nesting (no asyncio.run, no run_until_complete)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import asyncio

from core.logging import logger
from l2_tactic.models import TacticalSignal
from l1_operational.position_manager import PositionManager
from utils.position_size_cli_helper import PositionSizeCLIHelper, PositionSizeResult


class OrderIntent:
    """
    Represents an intent to create an order from a signal.
    Contains all necessary information for order execution.
    """
    def __init__(self, symbol: str, action: str, quantity: float, price: float,
                 confidence: float, timestamp: str, source: str, metadata: Dict = None,
                 reason: str = None):
        self.symbol = symbol
        self.action = action
        self.quantity = quantity
        self.price = price
        self.confidence = confidence
        self.timestamp = timestamp
        self.source = source
        self.metadata = metadata or {}
        self.reason = reason or "strategy_signal"  # Default reason for backward compatibility

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
            "reason": self.reason,
            "status": "pending"
        }


class OrderIntentBuilder:
    """
    Converts TacticalSignal objects to OrderIntent objects with validation and sizing.
    Addresses the Signal → Order Intent bottleneck.
    
    NOTE: paper_mode must be provided by SystemBootstrap, NOT decided internally.
    
    CRITICAL: paper_mode must be explicit. If not provided → ERROR.
    
    CRITICAL FIX: Capital allocation is computed ONCE per cycle and distributed
    proportionally among BUY intents to prevent over-allocation.
    """

    def __init__(self, position_manager: PositionManager, config: Dict, paper_mode: bool):
        self.position_manager = position_manager
        self.config = config
        self.min_order_value = config.get("MIN_ORDER_USDT", 2.0)
        self.cooldown_seconds = config.get("COOLDOWN_SECONDS", 36)
        self.last_trade_time: Dict[str, float] = {}
        
        # REQUIRE explicit paper_mode - ERROR if not provided
        if paper_mode is None:
            raise RuntimeError(
                "FATAL: OrderIntentBuilder requires explicit paper_mode. "
                "Fallback to config is not allowed. Pass paper_mode=True or paper_mode=False."
            )
        
        self.paper_mode = paper_mode
        
        # Track capital allocation state per cycle
        self._cycle_usdt_snapshot: Optional[float] = None
        self._cycle_allocated_usdt: float = 0.0
        self._cycle_intent_count: int = 0
        
        logger.info(f"OrderIntentBuilder initialized (paper_mode={self.paper_mode})")

    def reset_cycle_allocation(self):
        """
        Reset cycle allocation state. MUST be called at the start of each cycle.
        """
        self._cycle_usdt_snapshot = None
        self._cycle_allocated_usdt = 0.0
        self._cycle_intent_count = 0
        logger.debug("Cycle allocation state reset")

    async def get_cycle_usdt_balance(self) -> float:
        """
        Get USDT balance for the current cycle.
        Computes it ONCE per cycle and caches it.
        
        Returns:
            float: Available USDT balance for this cycle
        """
        if self._cycle_usdt_snapshot is None:
            portfolio_manager = self.position_manager.portfolio
            self._cycle_usdt_snapshot = await portfolio_manager.get_asset_balance_async("USDT")
            logger.info(f"USDT balance snapshot for cycle: ${self._cycle_usdt_snapshot:.2f}")
        
        return self._cycle_usdt_snapshot

    def _cooldown_ok(self, symbol: str, signal_confidence: float = None, l3_regime: str = None) -> bool:
        """Check if cooldown period has elapsed for a symbol with adaptive cooldown"""
        last = self.last_trade_time.get(symbol)
        
        if last is None:
            return True
            
        cooldown_seconds = self._get_adaptive_cooldown(signal_confidence, l3_regime)
        return (time.time() - last) >= cooldown_seconds
    
    def _get_adaptive_cooldown(self, signal_confidence: float, l3_regime: str) -> float:
        """
        Get adaptive cooldown based on L3 regime and signal confidence.
        
        - TRENDING with confidence >0.6: 2 cycles (24s)
        - TRENDING with confidence <0.6: 3 cycles (36s)
        - RANGE: 4 cycles (48s)
        - Confidence <0.4: 5 cycles (60s)
        """
        if signal_confidence is None and l3_regime is None:
            return self.cooldown_seconds
        
        if signal_confidence is not None and signal_confidence < 0.4:
            return 60
        
        if l3_regime is not None:
            l3_regime = l3_regime.lower()
            
            if "trend" in l3_regime or "trending" in l3_regime:
                if signal_confidence is not None and signal_confidence > 0.6:
                    return 24
                else:
                    return 36
            elif "range" in l3_regime:
                return 48
        
        return self.cooldown_seconds

    async def _calculate_order_quantity(self, signal: TacticalSignal, current_price: float,
                                        position_qty: float, available_usdt: Optional[float] = None) -> float:
        """
        Calculate appropriate order quantity using PositionSizeCLIHelper.
        
        ASYNC REQUIREMENTS:
        - This method MUST be async
        - PositionSizeCLIHelper.calculate_position_size() MUST be awaited directly
        - NO asyncio.run(), NO loop.run_until_complete(), NO run_coroutine_threadsafe()
        """
        try:
            helper = PositionSizeCLIHelper(
                self.position_manager.portfolio,
                min_order_value=self.min_order_value
            )
            
            allocation_pct = getattr(signal, 'confidence', 0.1)
            
            # For BUY: use available_usdt as custom balance
            # For SELL: use position_qty as custom balance
            custom_balance = available_usdt if signal.side.lower() == "buy" else position_qty
            
            # CRITICAL: Await the async method directly - no event loop manipulation
            result = await helper.calculate_position_size(
                symbol=signal.symbol,
                side=signal.side.lower(),
                current_price=current_price,
                allocation_pct=allocation_pct,
                min_order_value=self.min_order_value,
                use_balance_cache=True,
                custom_balance=custom_balance if custom_balance and custom_balance > 0 else None
            )
            
            if result.is_valid:
                logger.debug(f"[HELPER] Calculated qty for {signal.symbol}: {result.qty:.8f}")
                return result.qty
            else:
                logger.warning(f"[HELPER] Invalid qty for {signal.symbol}: {result.rejection_reason}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in helper calculation: {e}")
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

    async def build_order_intent(self, signal: TacticalSignal, market_data: Dict,
                                  position_qty: float, current_price: float, l3_regime: str = None,
                                  remaining_usdt: Optional[float] = None) -> Optional[OrderIntent]:
        """
        Build OrderIntent from TacticalSignal with complete validation and sizing.
        
        Args:
            signal: TacticalSignal to convert
            market_data: Market data dictionary
            position_qty: Current position quantity
            current_price: Current market price
            l3_regime: L3 regime (trending, range, etc.)
            remaining_usdt: Optional remaining USDT available for this intent
        """
        try:
            # Skip if cooldown active
            if not self._cooldown_ok(signal.symbol, signal.confidence, l3_regime):
                logger.info(f"Cooldown active for {signal.symbol} - skipping signal (regime: {l3_regime or 'unknown'}, conf: {signal.confidence:.2f})")
                return None

            # Get temporary aggressive mode status
            from core.config import TEMPORARY_AGGRESSIVE_MODE, check_temporary_aggressive_mode
            check_temporary_aggressive_mode()
            
            # Confidence thresholds based on mode
            if not self.paper_mode:
                if not (0.43 <= signal.confidence <= 0.70):
                    logger.info(f"Real mode: Signal confidence {signal.confidence:.2f} outside range (0.43-0.70) for {signal.symbol}")
                    return None
            else:
                min_confidence = 0.3 if TEMPORARY_AGGRESSIVE_MODE else 0.4
                if signal.confidence < min_confidence:
                    logger.info(f"Paper mode: Signal confidence {signal.confidence:.2f} outside range (>= {min_confidence:.1f}) for {signal.symbol}")
                    return None

            # --- VALIDATION: Data completeness ---
            if current_price is None or current_price <= 0:
                logger.error(f"QTY_CALCULATION_ABORTED_REASON=INVALID_PRICE PRICE_USED={current_price}")
                return None
            
            if position_qty is None:
                logger.error(f"QTY_CALCULATION_ABORTED_REASON=INVALID_POSITION_QTY POSITION_QTY={position_qty}")
                return None
            
            # --- VALIDATION: Balances (ASYNC) ---
            portfolio_manager = self.position_manager.portfolio
            asset = signal.symbol.replace("USDT", "") if signal.symbol != "USDT" else "USDT"
            
            if signal.side.lower() == "sell":
                # For SELL: verify we have the asset
                asset_balance = await portfolio_manager.get_asset_balance_async(asset)
                
                if asset_balance is None or asset_balance <= 0:
                    logger.debug(f"SELL ignored: no {asset} (balance={asset_balance})")
                    return None
                
                logger.info(f"[BALANCE_CHECK] SELL {asset}: Available={asset_balance:.6f}")
                
            elif signal.side.lower() == "buy":
                # For BUY: verify we have USDT
                if remaining_usdt is not None:
                    usdt_balance = remaining_usdt
                    logger.info(f"[BALANCE_CHECK] BUY {asset}: Using remaining USDT=${usdt_balance:.2f}")
                else:
                    usdt_balance = await portfolio_manager.get_asset_balance_async("USDT")
                    logger.info(f"[BALANCE_CHECK] BUY {asset}: USDT Available=${usdt_balance:.2f}")
                
                if usdt_balance is None or usdt_balance <= 0:
                    logger.error(f"QTY_CALCULATION_ABORTED_REASON=INSUFFICIENT_USDT_BALANCE USDT_BALANCE={usdt_balance}")
                    return None
            
            # Get all balances for position sizing
            balances = await portfolio_manager.get_balances_async()
            logger.debug(f"[BALANCE_ACCESS] ASYNC | All balances for sizing | Values: {balances}")

            # Calculate order quantity
            usdt_balance_for_sizing = remaining_usdt if remaining_usdt is not None else balances.get('USDT', 0.0) if signal.side.lower() == "buy" else None
            
            # CRITICAL: Await the async method
            qty = await self._calculate_order_quantity(signal, current_price, position_qty, usdt_balance_for_sizing)
            
            # Paper mode: ensure order quantity is within available USDT
            if self.paper_mode and signal.side.lower() == "buy":
                effective_usdt = remaining_usdt if remaining_usdt is not None else balances.get('USDT', 0.0)
                if effective_usdt > 0:
                    max_qty = effective_usdt / current_price
                    if qty > max_qty:
                        logger.warning(f"PAPER MODE: Adjusting quantity - requested {qty:.6f} exceeds available USDT, using {max_qty:.6f}")
                        qty = max_qty

            # SAFE VALIDATION: Check calculated quantity
            if qty is None or qty <= 0:
                # Log detailed debug info for troubleshooting
                logger.warning(
                    f"QTY_VALIDATION_FAILED: "
                    f"signal_side={signal.side}, "
                    f"current_price={current_price}, "
                    f"position_qty={position_qty}, "
                    f"available_usdt={usdt_balance_for_sizing}, "
                    f"min_order_value={self.min_order_value}, "
                    f"calculated_qty={qty}"
                )
                
                # Paper mode fallback
                if self.paper_mode and signal.side.lower() == "buy":
                    fallback_usdt = remaining_usdt if remaining_usdt is not None else balances.get('USDT', 0.0)
                    if fallback_usdt > 0:
                        fallback_notional = fallback_usdt * 0.10
                        qty = fallback_notional / current_price
                        logger.warning(f"PAPER OVERRIDE: Using fallback quantity {qty:.6f}")
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
                if self.paper_mode and signal.side.lower() == "buy":
                    logger.warning(f"PAPER OVERRIDE: order validation failed - using fixed notional")
                    fallback_usdt = remaining_usdt if remaining_usdt is not None else balances.get('USDT', 0.0)
                    if fallback_usdt > 0:
                        fallback_notional = fallback_usdt * 0.10
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
                        logger.info(f"Created fallback OrderIntent for {signal.symbol}: {fallback_qty:.6f} @ {current_price:.2f}")
                else:
                    logger.error(f"QTY_CALCULATION_ABORTED_REASON=INTENT_VALIDATION_FAILED")
                    return None

            # Record trade time
            self.last_trade_time[signal.symbol] = time.time()

            logger.debug(f"PRICE_USED={current_price} QTY_CALCULATED={intent.quantity}")
            logger.info(f"OrderIntent built: {signal.symbol} {signal.side} {intent.quantity:.6f} @ {current_price:.2f} (conf: {signal.confidence:.2f})")
            return intent

        except Exception as e:
            logger.error(f"Error building OrderIntent for {signal.symbol}: {e}")
            logger.error(f"QTY_CALCULATION_ABORTED_REASON=GENERAL_ERROR ERROR={str(e)}")
            return None


class OrderIntentProcessor:
    """
    Processes a list of TacticalSignals and converts them to validated OrderIntents.
    Handles the complete Signal → Order Intent pipeline.
    """

    def __init__(self, intent_builder: OrderIntentBuilder):
        self.intent_builder = intent_builder

    async def process_signals(self, signals: List[TacticalSignal], market_data: Dict,
                             get_position_qty_func, l3_regime: str = None) -> List[OrderIntent]:
        """
        Process a list of TacticalSignals to create OrderIntents.
        
        Args:
            signals: List of TacticalSignal objects
            market_data: Market data dictionary
            get_position_qty_func: Function to get current position quantity for a symbol
            l3_regime: Régimen L3 (trending, range, etc.)

        Returns:
            List of validated OrderIntent objects
        """
        order_intents = []
        rejected_signals = 0
        hold_signals = 0

        # Reset cycle allocation state at start
        self.intent_builder.reset_cycle_allocation()

        # Get USDT balance ONCE per cycle
        portfolio_manager = self.intent_builder.position_manager.portfolio
        usdt_available = await portfolio_manager.get_asset_balance_async("USDT")
        logger.info(f"USDT balance snapshot for cycle: ${usdt_available:.2f}")

        # First pass: collect all BUY signals
        buy_signals = []
        other_signals = []
        
        for signal in signals:
            if signal.side.lower() == "hold":
                hold_signals += 1
                logger.debug(f"Hold signal for {signal.symbol}")
                continue
            elif signal.side.lower() == "buy":
                buy_signals.append(signal)
            else:
                other_signals.append(signal)
        
        # Calculate proportional allocation for BUY signals
        num_buy_signals = len(buy_signals)
        proportional_allocation = usdt_available / num_buy_signals if num_buy_signals > 0 else 0.0
        
        if num_buy_signals > 0:
            logger.info(f"Capital allocation: ${usdt_available:.2f} USDT across {num_buy_signals} BUY signals = ${proportional_allocation:.2f} each")

        # Process BUY signals with proportional allocation
        remaining_usdt = usdt_available
        buy_intents_created = 0
        
        for signal in buy_signals:
            try:
                # Extract current price
                symbol_data = market_data.get(signal.symbol, {})
                if isinstance(symbol_data, dict) and "close" in symbol_data:
                    current_price = symbol_data["close"]
                elif hasattr(symbol_data, "iloc") and len(symbol_data) > 0:
                    current_price = symbol_data["close"].iloc[-1]
                else:
                    rejected_signals += 1
                    logger.warning(f"No price data for {signal.symbol} - rejecting signal")
                    continue

                # Get current position quantity
                if asyncio.iscoroutinefunction(get_position_qty_func):
                    position_qty = await get_position_qty_func(signal.symbol)
                else:
                    position_qty = get_position_qty_func(signal.symbol)
                    logger.warning(f"Using sync function for position_qty: {signal.symbol}")

                # Build order intent
                intent = await self.intent_builder.build_order_intent(
                    signal, market_data, position_qty, current_price, l3_regime,
                    remaining_usdt=proportional_allocation
                )

                if intent:
                    order_intents.append(intent)
                    buy_intents_created += 1
                    allocated_for_this_intent = intent.quantity * intent.price
                    remaining_usdt -= allocated_for_this_intent
                    logger.info(f"Allocated ${allocated_for_this_intent:.2f} for {signal.symbol}, remaining: ${remaining_usdt:.2f}")
                else:
                    rejected_signals += 1
                    logger.info(f"Signal rejected: {signal.symbol} {signal.side} (conf: {signal.confidence:.2f}) - no order intent created")

            except Exception as e:
                rejected_signals += 1
                logger.error(f"Error processing signal for {signal.symbol}: {e}")
                continue

        # Process SELL and other signals
        for signal in other_signals:
            try:
                # Extract current price
                symbol_data = market_data.get(signal.symbol, {})
                if isinstance(symbol_data, dict) and "close" in symbol_data:
                    current_price = symbol_data["close"]
                elif hasattr(symbol_data, "iloc") and len(symbol_data) > 0:
                    current_price = symbol_data["close"].iloc[-1]
                else:
                    rejected_signals += 1
                    logger.warning(f"No price data for {signal.symbol} - rejecting signal")
                    continue

                # Get current position quantity
                if asyncio.iscoroutinefunction(get_position_qty_func):
                    position_qty = await get_position_qty_func(signal.symbol)
                else:
                    position_qty = get_position_qty_func(signal.symbol)

                # Build order intent
                intent = await self.intent_builder.build_order_intent(
                    signal, market_data, position_qty, current_price, l3_regime
                )

                if intent:
                    order_intents.append(intent)
                else:
                    rejected_signals += 1
                    logger.info(f"Signal rejected: {signal.symbol} {signal.side} (conf: {signal.confidence:.2f})")

            except Exception as e:
                rejected_signals += 1
                logger.error(f"Error processing signal for {signal.symbol}: {e}")
                continue

        # Validate total capital allocation
        buy_intents = [intent for intent in order_intents if intent.action == "buy"]
        if buy_intents:
            total_intended_value = sum(intent.quantity * intent.price for intent in buy_intents)
            
            if total_intended_value > usdt_available * 1.05:
                scale_factor = usdt_available / total_intended_value
                logger.warning(f"CAPITAL OVER-ALLOCATION: Adjusting buy intents by factor {scale_factor:.2f}")
                logger.warning(f"   Total intended: ${total_intended_value:.2f}, Available: ${usdt_available:.2f}")
                
                for intent in buy_intents:
                    original_qty = intent.quantity
                    intent.quantity *= scale_factor
                    logger.debug(f"   {intent.symbol}: {original_qty:.6f} -> {intent.quantity:.6f}")
                
                adjusted_total = sum(intent.quantity * intent.price for intent in buy_intents)
                logger.info(f"Capital allocation adjusted: ${adjusted_total:.2f} <= ${usdt_available:.2f}")
            else:
                logger.info(f"Capital allocation OK: ${total_intended_value:.2f} <= ${usdt_available:.2f}")
            
            final_total = sum(intent.quantity * intent.price for intent in buy_intents)
            if final_total > usdt_available:
                logger.critical(f"OVER-ALLOCATION AFTER ADJUSTMENT: ${final_total:.2f} > ${usdt_available:.2f}")
                force_scale = (usdt_available * 0.95) / final_total
                for intent in buy_intents:
                    intent.quantity *= force_scale
                final_total = sum(intent.quantity * intent.price for intent in buy_intents)
                logger.warning(f"Forced scale down applied. Final allocation: ${final_total:.2f}")

        logger.info(f"Processed {len(signals)} signals -> {len(order_intents)} order intents ({buy_intents_created} BUY)")
        if hold_signals > 0:
            logger.info(f"{hold_signals} hold signals skipped")
        if rejected_signals > 0:
            logger.info(f"{rejected_signals} signals rejected")
        return order_intents
