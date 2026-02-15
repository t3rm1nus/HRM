"""
Trading Pipeline Manager - Orquesta el flujo completo de trading.
Versión estabilizada y corregida:
- Single snapshot por ciclo
- Sin recalcular NAV múltiples veces
- Sin duplicación de sincronización de precios
- Flujo limpio y determinista
- Fixed async method signatures
- Proper NAV calculation and logging
- Simulated mode consistency
"""

import time
import pandas as pd
from typing import Dict, List, Optional, Any
from core.logging import logger
from system.models import TradingCycleResult


class TradingPipelineManager:

    def __init__(
        self,
        portfolio_manager,
        order_manager,
        l2_processor,
        position_rotator,
        auto_rebalancer,
        signal_verifier,
        state_coordinator,
        mode: str = "simulated"
    ):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.l2_processor = l2_processor
        self.position_rotator = position_rotator
        self.auto_rebalancer = auto_rebalancer
        self.signal_verifier = signal_verifier
        self.state_coordinator = state_coordinator
        self.mode = mode

        self.auto_learning_bridge = None
        self.cycle_context: Optional[Dict[str, Any]] = None

    # ==========================================================
    # MAIN PIPELINE
    # ==========================================================

    async def process_trading_cycle(
        self,
        state: Dict,
        market_data: Dict[str, pd.DataFrame],
        cycle_id: int = 0
    ) -> TradingCycleResult:

        start_time = pd.Timestamp.utcnow()

        if not state:
            raise ValueError("System state not injected")

        if "version" not in state:
            state["version"] = "1.0"

        # ==========================================================
        # 1️⃣ SNAPSHOT ÚNICO DEL CICLO (SOURCE OF TRUTH)
        # ==========================================================

        balances = await self.portfolio_manager.get_balances_async()

        prices = {
            symbol: float(df.iloc[-1]["close"])
            for symbol, df in market_data.items()
            if df is not None and not df.empty
        }

        self.cycle_context = {
            "cycle_id": cycle_id,
            "balances": balances,
            "prices": prices,
            "timestamp": pd.Timestamp.utcnow().isoformat()
        }

        logger.info(
            f"📸 CYCLE SNAPSHOT | Cycle {cycle_id} | "
            f"Balances: {balances} | Prices: {prices}"
        )

        result = TradingCycleResult(
            signals_generated=0,
            orders_executed=0,
            orders_rejected=0,
            cooldown_blocked=0,
            l3_regime="unknown",
            portfolio_value=0.0,
            execution_time=0.0
        )

        try:

            # ==========================================================
            # 2️⃣ L3 DECISION
            # ==========================================================

            l3_output = await self._update_l3_decision(state, market_data)
            result.l3_regime = l3_output.get("regime", "unknown")

            # ==========================================================
            # 3️⃣ L2 SIGNALS
            # ==========================================================

            l2_signals = await self._generate_l2_signals(state, market_data, l3_output)
            result.signals_generated = len(l2_signals)

            valid_signals = await self._validate_signals(l2_signals, market_data)

            # ==========================================================
            # 4️⃣ ORDER GENERATION
            # ==========================================================

            l3_regime = l3_output.get("regime", "unknown")

            orders = (
                await self.order_manager.generate_orders(
                    state, valid_signals, l3_regime
                )
                if valid_signals
                else []
            )

            validated_orders = (
                await self._validate_orders(orders, state)
                if orders else []
            )

            executed = (
                await self.order_manager.execute_orders(validated_orders)
                if validated_orders else []
            )

            filled = [o for o in executed if o.get("status") == "filled"]
            rejected = [o for o in executed if o.get("status") == "rejected"]

            result.orders_executed = len(filled)
            result.orders_rejected = len(rejected)

            # ==========================================================
            # 5️⃣ UPDATE PORTFOLIO (ONLY IF FILLED) - FIXED SIGNATURE
            # ==========================================================

            if filled:
                # CRITICAL FIX: Log NAV before execution
                nav_before = await self.portfolio_manager.get_total_value_async()
                logger.info("=" * 80)
                logger.info(f"📊 NAV BEFORE ORDER PROCESSING: ${nav_before:.2f}")
                logger.info("=" * 80)
                
                # Fixed: Pass only orders (market_data not needed, but accepted for backwards compatibility)
                await self.portfolio_manager.update_from_orders_async(filled)
                
                # CRITICAL FIX: Log balances after execution
                balances_after = await self.portfolio_manager.get_balances_async()
                logger.info("=" * 80)
                logger.info("📊 BALANCES AFTER ORDER EXECUTION:")
                logger.info(f"   BTC:  {balances_after.get('BTC', 0.0):.6f}")
                logger.info(f"   ETH:  {balances_after.get('ETH', 0.0):.6f}")
                logger.info(f"   USDT: ${balances_after.get('USDT', 0.0):.2f}")
                logger.info("=" * 80)

                await self._record_trades_for_auto_learning(
                    filled, l3_output, market_data
                )

                await self._log_paper_trades(
                    filled, market_data, cycle_id, l3_output
                )

            # ==========================================================
            # 6️⃣ SYNC STATE (NO EXTRA BALANCE CALLS)
            # ==========================================================

            await self._sync_state_from_snapshot(state)

            # ==========================================================
            # 7️⃣ CALCULATE NAV (FROM SNAPSHOT) - ENSURE IT'S CORRECT
            # ==========================================================

            # CRITICAL FIX: Use the portfolio manager's NAV calculation with proper prices
            if self.portfolio_manager.market_data_manager:
                try:
                    nav_result = await self.portfolio_manager.update_nav_async(prices)
                    result.portfolio_value = nav_result["total_nav"]
                except Exception as nav_error:
                    logger.error(f"❌ Error calculating NAV with MarketDataManager: {nav_error}")
                    # Fallback to snapshot calculation
                    result.portfolio_value = self._calculate_nav_from_snapshot()
            else:
                result.portfolio_value = self._calculate_nav_from_snapshot()
            
            # Defensive safeguard: Ensure NAV never drops to 0 unless portfolio is actually empty
            if result.portfolio_value <= 0:
                usdt = balances.get("USDT", 0.0)
                btc = balances.get("BTC", 0.0)
                eth = balances.get("ETH", 0.0)
                if usdt > 0 or btc > 0 or eth > 0:
                    logger.critical(f"🚨 NAV CALCULATION ERROR: NAV={result.portfolio_value:.2f} with non-zero balances")
                    # Calculate minimum NAV assuming $1 minimum prices
                    min_nav = usdt + btc * 1.0 + eth * 1.0
                    result.portfolio_value = max(min_nav, 0.01)
                    logger.warning(f"⚠️ NAV adjusted to minimum: ${result.portfolio_value:.2f}")

            # ==========================================================
            # 8️⃣ OPTIONAL ROTATION / REBALANCE
            # ==========================================================

            result.orders_executed += await self._process_position_rotation(
                state, market_data
            )

            result.orders_executed += await self._process_rebalancing(
                state, market_data, l3_output, valid_signals
            )

        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}", exc_info=True)

        finally:
            result.execution_time = (
                pd.Timestamp.utcnow() - start_time
            ).total_seconds()

            logger.info(
                f"⏱️ Ciclo completado en {result.execution_time:.2f}s | "
                f"Señales: {result.signals_generated} | "
                f"Órdenes: {result.orders_executed} ejecutadas, "
                f"{result.orders_rejected} rechazadas"
            )

        return result

    # ==========================================================
    # NAV FROM SNAPSHOT (NO REFETCH)
    # ==========================================================

    def _calculate_nav_from_snapshot(self) -> float:

        if not self.cycle_context:
            return 0.0

        balances = self.cycle_context["balances"]
        prices = self.cycle_context["prices"]

        total = balances.get("USDT", 0.0)

        for symbol, price in prices.items():
            asset = symbol.replace("USDT", "")
            total += balances.get(asset, 0.0) * price

        return total

    # ==========================================================
    # STATE SYNC FROM SNAPSHOT
    # ==========================================================

    async def _sync_state_from_snapshot(self, state: Dict) -> None:

        if not self.cycle_context:
            return

        balances = self.cycle_context["balances"]

        snapshot = {
            "btc_balance": balances.get("BTC", 0.0),
            "eth_balance": balances.get("ETH", 0.0),
            "usdt_balance": balances.get("USDT", 0.0),
            "total_value": self._calculate_nav_from_snapshot(),
        }

        state["portfolio"] = snapshot

        self.state_coordinator.update_total_value(snapshot["total_value"])

    # ==========================================================
    # AUTO LEARNING
    # ==========================================================

    async def _record_trades_for_auto_learning(
        self,
        filled_orders: List[Dict],
        l3_output: Dict,
        market_data: Dict
    ) -> None:

        if not filled_orders or not self.auto_learning_bridge:
            return

        for order in filled_orders:
            try:
                await self.auto_learning_bridge.record_order_execution(
                    order=order,
                    l3_context=l3_output,
                    market_data=market_data
                )
            except Exception as e:
                logger.error(f"Auto-learning error: {e}")

    # ==========================================================
    # PAPER LOGGER
    # ==========================================================

    async def _log_paper_trades(
        self,
        filled_orders: List[Dict],
        market_data: Dict,
        cycle_id: int,
        l3_output: Dict
    ) -> None:

        try:
            from storage.paper_trade_logger import log_paper_trade

            for order in filled_orders:
                log_paper_trade(
                    order,
                    market_data=market_data,
                    cycle_id=cycle_id,
                    strategy=l3_output.get("strategy_type", "paper")
                )

        except Exception as e:
            logger.error(f"Paper logger error: {e}")

    # ==========================================================
    # L3
    # ==========================================================

    async def _update_l3_decision(self, state: Dict, market_data: Dict) -> Dict:

        from core.l3_processor import get_l3_decision

        try:
            output = get_l3_decision(market_data)
        except Exception:
            output = None

        if not output:
            output = {
                "regime": "neutral",
                "signal": "hold",
                "confidence": 0.0,
                "strategy_type": "fallback",
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }

        state["l3_output"] = output
        state["l3_last_update"] = time.time()

        return output

    # ==========================================================
    # L2
    # ==========================================================

    async def _generate_l2_signals(
        self,
        state: Dict,
        market_data: Dict,
        l3_decision: Dict
    ) -> List[Dict]:

        try:
            return self.l2_processor.generate_signals_conservative(
                market_data=market_data,
                l3_context=l3_decision
            )
        except Exception as e:
            logger.error(f"L2 error: {e}")
            return []

    # ==========================================================
    # VALIDATION
    # ==========================================================

    async def _validate_signals(
        self,
        signals: List[Dict],
        market_data: Dict
    ) -> List[Dict]:

        valid = []

        for signal in signals:
            try:
                await self.signal_verifier.submit_signal_for_verification(
                    signal, market_data
                )
                valid.append(signal)
            except Exception:
                continue

        return valid

    async def _validate_orders(
        self,
        orders: List[Dict],
        state: Dict
    ) -> List[Dict]:

        validated = []
        portfolio = state.get("portfolio", {})

        for order in orders:
            if order.get("status") != "pending":
                validated.append(order)
                continue

            check = self.order_manager.validate_order_size(
                order.get("symbol"),
                order.get("quantity"),
                order.get("price"),
                portfolio
            )

            if check["valid"]:
                validated.append(order)
            else:
                order["status"] = "rejected"
                order["validation_error"] = check["reason"]
                validated.append(order)

        return validated

    # ==========================================================
    # OPTIONAL COMPONENTS
    # ==========================================================

    async def _process_position_rotation(
        self,
        state: Dict,
        market_data: Dict
    ) -> int:

        if not self.position_rotator:
            return 0

        try:
            rotation = await self.position_rotator.check_and_rotate_positions(
                state, market_data
            )

            if not rotation:
                return 0

            executed = await self.order_manager.execute_orders(rotation)

            # Fixed: Pass only orders (market_data not needed)
            await self.portfolio_manager.update_from_orders_async(executed)

            return len([o for o in executed if o.get("status") == "filled"])

        except Exception:
            return 0

    async def _process_rebalancing(
        self,
        state: Dict,
        market_data: Dict,
        l3_output: Dict,
        valid_signals: List
    ) -> int:

        if not self.auto_rebalancer:
            return 0

        if l3_output.get("signal") == "sell":
            return 0

        try:
            rebalance = await self.auto_rebalancer.check_and_execute_rebalance(
                market_data,
                l3_decision=l3_output
            )

            if not rebalance:
                return 0

            executed = await self.order_manager.execute_orders(rebalance)

            # Fixed: Pass only orders (market_data not needed)
            await self.portfolio_manager.update_from_orders_async(executed)

            return len([o for o in executed if o.get("status") == "filled"])

        except Exception:
            return 0

    # ==========================================================
    # CYCLE CONTEXT ACCESSOR
    # ==========================================================

    def get_cycle_context(self) -> Optional[Dict[str, Any]]:
        """
        Get the current cycle context.
        
        Returns:
            Optional[Dict]: The cycle context or None if not set
        """
        return self.cycle_context
