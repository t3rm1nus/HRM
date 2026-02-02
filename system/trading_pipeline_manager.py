# system/trading_pipeline_manager.py
"""
Trading Pipeline Manager - Orquesta el flujo completo de trading.
Paper-safe, sin llamadas a APIs inexistentes.
"""

import time
import pandas as pd
from typing import Dict, List
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
        config: Dict
    ):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.l2_processor = l2_processor
        self.position_rotator = position_rotator
        self.auto_rebalancer = auto_rebalancer
        self.signal_verifier = signal_verifier
        self.state_coordinator = state_coordinator
        self.config = config

    # ==================================================================
    # MAIN PIPELINE
    # ==================================================================

    async def process_trading_cycle(
        self,
        state: Dict,
        market_data: Dict[str, pd.DataFrame]
    ) -> TradingCycleResult:

        start_time = pd.Timestamp.utcnow()

        if not state or "version" not in state:
            raise RuntimeError("System state not injected")

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
            # PASO 1 – L3
            l3_output = await self._update_l3_decision(state, market_data)
            result.l3_regime = l3_output.get("regime", "unknown")

            # PASO 2 – L2
            l2_signals = await self._generate_l2_signals(state, market_data, l3_output)
            result.signals_generated = len(l2_signals)

            # PASO 3 – Verificación
            valid_signals = await self._validate_signals(l2_signals, market_data)

            # PASO 4 – Generar órdenes
            orders = await self.order_manager.generate_orders(state, valid_signals)

            # PASO 5 – Validar órdenes
            validated_orders = await self._validate_orders(orders, state)

            # PASO 6 – Ejecutar
            executed = await self.order_manager.execute_orders(validated_orders)

            filled = [o for o in executed if o.get("status") == "filled"]
            rejected = [o for o in executed if o.get("status") == "rejected"]

            result.orders_executed += len(filled)
            result.orders_rejected += len(rejected)

            # PASO 7 – ACTUALIZAR PORTFOLIO (CRÍTICO)
            if filled:
                await self.portfolio_manager.update_from_orders_async(filled, market_data)

            # PASO 8 – Refrescar STATE desde portfolio real
            self._sync_state_from_portfolio(state)

            result.portfolio_value = self.portfolio_manager.get_total_value(market_data)

            # PASO 9 – Rotación (opcional)
            if self.position_rotator:
                rotation = await self.position_rotator.check_and_rotate_positions(state, market_data)
                if rotation:
                    executed_rot = await self.order_manager.execute_orders(rotation)
                    await self.portfolio_manager.update_from_orders_async(executed_rot, market_data)
                    result.orders_executed += len(executed_rot)

            # PASO 10 – Rebalance (opcional)
            if self.auto_rebalancer:
                rebalance = await self.auto_rebalancer.check_and_execute_rebalance(
                    market_data, l3_decision=l3_output
                )
                if rebalance:
                    executed_reb = await self.order_manager.execute_orders(rebalance)
                    await self.portfolio_manager.update_from_orders_async(executed_reb, market_data)
                    result.orders_executed += len(executed_reb)

        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}", exc_info=True)

        finally:
            result.execution_time = (pd.Timestamp.utcnow() - start_time).total_seconds()
            logger.info(f"⏱️ Ciclo completado en {result.execution_time:.2f}s")

        return result

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _sync_state_from_portfolio(self, state: Dict):
        """
        Fuente de verdad ÚNICA: PortfolioManager
        """
        portfolio = self.portfolio_manager.get_portfolio_state()

        snapshot = {
            "btc_balance": portfolio.get("BTC", 0.0),
            "eth_balance": portfolio.get("ETH", 0.0),
            "usdt_balance": portfolio.get("USDT", 0.0),
            "total_value": portfolio.get("total", 0.0),
        }

        state["portfolio"] = snapshot
        self.state_coordinator.update_state({"portfolio": snapshot})

    async def _update_l3_decision(self, state: Dict, market_data: Dict) -> Dict:
        from core.l3_processor import get_l3_decision

        output = get_l3_decision(market_data) or {
            "regime": "neutral",
            "signal": "hold",
            "confidence": 0.5,
            "strategy_type": "fallback",
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }

        state["l3_output"] = output
        state["l3_last_update"] = time.time()
        return output

    async def _generate_l2_signals(self, state, market_data, l3_decision) -> List:
        context = {
            "regime": l3_decision.get("regime"),
            "signal": l3_decision.get("signal"),
            "confidence": l3_decision.get("confidence"),
            "allow_l2": l3_decision.get("allow_l2_signals", True),
            "l3_output": l3_decision,
        }

        return self.l2_processor.generate_signals_conservative(
            market_data=market_data,
            l3_context=context
        )

    async def _validate_signals(self, signals: List, market_data: Dict) -> List:
        valid = []
        for s in signals:
            try:
                await self.signal_verifier.submit_signal_for_verification(s, market_data)
                valid.append(s)
            except Exception:
                continue
        return valid

    async def _validate_orders(self, orders: List, state: Dict) -> List:
        validated = []
        portfolio = state.get("portfolio", {})

        for o in orders:
            if o.get("status") != "pending":
                validated.append(o)
                continue

            check = self.order_manager.validate_order_size(
                o.get("symbol"),
                o.get("quantity"),
                o.get("price"),
                portfolio
            )

            if check["valid"]:
                validated.append(o)
            else:
                o["status"] = "rejected"
                o["validation_error"] = check["reason"]
                validated.append(o)

        return validated
