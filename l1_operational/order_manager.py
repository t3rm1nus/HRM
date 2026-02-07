from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import time

from core.logging import logger
from l2_tactic.models import TacticalSignal
from .position_manager import PositionManager
from l1_operational.order_validators import OrderValidators
from l1_operational.order_executors import OrderExecutors
from .order_intent_builder import OrderIntentBuilder, OrderIntentProcessor, OrderIntent

# ========================================================================================
# CONSTANTES
# ========================================================================================
DUST_THRESHOLD_BTC = 0.00005
DUST_THRESHOLD_ETH = 0.005
MIN_ORDER_USDT = 2.0


class OrderManager:
    """
    OrderManager CORREGIDO
    - Contrato único de orden
    - Campo `action` obligatorio SIEMPRE
    - Sin métodos duplicados
    - SELL y BUY coherentes
    - Integra Order Intent Builder para resolver bottleneck Signal → Order Intent
    """

    def __init__(self, state_manager, portfolio_manager, config: Dict, simulated_client=None):
        self.state = state_manager
        self.portfolio = portfolio_manager
        self.config = config

        self.position_manager = PositionManager(
            state_manager=state_manager,
            portfolio_manager=portfolio_manager,
            config=config
        )

        self.validators = OrderValidators(config)
        self.executors = OrderExecutors(state_manager, portfolio_manager, config, simulated_client)

        # Initialize Order Intent Builder
        self.intent_builder = OrderIntentBuilder(self.position_manager, config)
        self.intent_processor = OrderIntentProcessor(self.intent_builder)

        self.last_trade_time: Dict[str, float] = {}
        self.cooldown_seconds = config.get("COOLDOWN_SECONDS", 60)

        logger.info("✅ OrderManager inicializado (FIXED) con Order Intent Builder")

    # ======================================================================
    # CORE ENTRY - USING ORDER INTENT BUILDER
    # ======================================================================

    async def generate_orders(self, state: Dict, signals: List[TacticalSignal]) -> List[Dict]:
        # Process signals to order intents (Signal → Order Intent step)
        order_intents = self.intent_processor.process_signals(
            signals,
            state.get("market_data", {}),
            self._get_effective_position
        )

        # Convert order intents to executable orders
        orders = []
        for intent in order_intents:
            order = self._intent_to_order(intent)
            if order:
                orders.append(order)

        logger.info(f"📊 Order generation complete: {len(signals)} signals → {len(order_intents)} intents → {len(orders)} orders")
        return orders

    # ======================================================================
    # SIGNAL HANDLER (legacy, kept for compatibility)
    # ======================================================================

    def handle_signal(self, signal: TacticalSignal, market_data: Dict) -> Dict[str, Any]:
        symbol = signal.symbol
        action = signal.side.lower()  # buy / sell / hold

        current_price = self._extract_current_price(market_data, symbol)
        if current_price <= 0:
            return self._reject(symbol, action, "invalid_price")

        if action == "hold":
            return self._hold(symbol)

        if not self._cooldown_ok(symbol):
            return self._reject(symbol, action, "cooldown_active")

        position_qty = self._get_effective_position(symbol)

        # ================= BUY =================
        if action == "buy":
            if position_qty > 0:
                return self._hold(symbol)

            qty = self.position_manager.calculate_order_size(
                symbol=symbol,
                action="buy",
                signal_confidence=signal.confidence,
                current_price=current_price,
                position_qty=0.0
            )

            return self._build_order(symbol, "buy", qty, current_price)

        # ================= SELL =================
        if action == "sell":
            if position_qty <= self._dust(symbol):
                return self._hold(symbol)

            qty = min(position_qty, signal.quantity or position_qty)
            return self._build_order(symbol, "sell", qty, current_price)

        return self._hold(symbol)

    # ======================================================================
    # ORDER FROM INTENT
    # ======================================================================

    def _intent_to_order(self, intent: OrderIntent) -> Optional[Dict[str, Any]]:
        """Convert OrderIntent to executable order with validation"""
        try:
            # Build order from intent
            order = {
                "status": "accepted",
                "symbol": intent.symbol,
                "action": intent.action,
                "quantity": float(intent.quantity),
                "price": float(intent.price),
                "value_usdt": float(intent.quantity * intent.price),
                "timestamp": intent.timestamp,
                "mode": "paper",
                "confidence": intent.confidence,
                "source": intent.source,
                "metadata": intent.metadata
            }

            # Validate order
            validation = self.validators.validate_and_normalize_order(order)
            if validation["validation"]["status"] != "valid":
                logger.warning(f"❌ Order validation failed for {intent.symbol}: {validation['validation']['reason']}")
                return None

            logger.info(f"🎯 Order created from intent: {intent.symbol} {intent.action} {intent.quantity:.6f} @ {intent.price:.2f}")
            return validation["order"]

        except Exception as e:
            logger.error(f"❌ Error converting intent to order: {e}")
            return None

    # ======================================================================
    # ORDER BUILDER (CRITICAL FIX)
    # ======================================================================

    def _build_order(self, symbol: str, action: str, qty: float, price: float) -> Dict[str, Any]:
        value = qty * price
        if qty <= 0 or value < MIN_ORDER_USDT:
            return self._reject(symbol, action, "order_too_small")

        order = {
            "status": "accepted",
            "symbol": symbol,
            "action": action,          # ✅ SIEMPRE PRESENTE
            "quantity": float(qty),    # ✅ SELL NO USA NEGATIVOS
            "price": float(price),
            "value_usdt": float(value),
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "paper"
        }

        validation = self.validators.validate_and_normalize_order(order)
        if validation["validation"]["status"] != "valid":
            return self._reject(symbol, action, "validation_failed")

        self.last_trade_time[symbol] = time.time()
        return validation["order"]

    # ======================================================================
    # EXECUTION
    # ======================================================================

    async def execute_orders(self, orders: List[Dict]) -> List[Dict]:
        results = []
        for o in orders:
            try:
                result = self.executors.execute_order(
                    symbol=o["symbol"],
                    action=o["action"],
                    quantity=o["quantity"],
                    current_price=o["price"]
                )
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Execution error {o['symbol']}: {e}")
                o["status"] = "failed"
                o["error"] = str(e)
                results.append(o)
        return results

    # ======================================================================
    # HELPERS
    # ======================================================================

    def _cooldown_ok(self, symbol: str) -> bool:
        last = self.last_trade_time.get(symbol)
        return last is None or (time.time() - last) >= self.cooldown_seconds

    def _dust(self, symbol: str) -> float:
        return DUST_THRESHOLD_BTC if "BTC" in symbol else DUST_THRESHOLD_ETH

    def _get_effective_position(self, symbol: str) -> float:
        asset = symbol.replace("USDT", "")
        try:
            return float(self.portfolio.get_balance(asset))
        except Exception:
            return 0.0

    def _extract_current_price(self, market_data: Dict, symbol: str) -> float:
        try:
            if symbol not in market_data:
                return 0.0
            data = market_data[symbol]
            if isinstance(data, pd.DataFrame):
                return float(data["close"].iloc[-1])
            if isinstance(data, dict) and "close" in data:
                return float(data["close"])
        except Exception:
            pass
        return 0.0

    # ======================================================================
    # RESPONSES
    # ======================================================================

    def _hold(self, symbol: str) -> Dict[str, Any]:
        return {
            "status": "hold",
            "symbol": symbol,
            "action": "hold",
            "timestamp": datetime.utcnow().isoformat()
        }

    def _reject(self, symbol: str, action: str, reason: str) -> Dict[str, Any]:
        return {
            "status": "rejected",
            "symbol": symbol,
            "action": action,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
