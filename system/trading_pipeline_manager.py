
"""
Trading Pipeline Manager.

Orquesta el ciclo de trading y delega toda ejecución de órdenes
en ExecutionGateway.

La ejecución de esta fase está limitada a PAPER/SIMULATED.
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd

from core.logging import logger
from system.models import TradingCycleResult


class TradingPipelineManager:
    """Coordina señales, validaciones, ejecución y sincronización."""

    def __init__(
        self,
        portfolio_manager,
        order_manager,
        l2_processor,
        position_rotator,
        auto_rebalancer,
        signal_verifier,
        state_coordinator,
        mode: str = "simulated",
        execution_gateway=None,
    ):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.l2_processor = l2_processor
        self.position_rotator = position_rotator
        self.auto_rebalancer = auto_rebalancer
        self.signal_verifier = signal_verifier
        self.state_coordinator = state_coordinator
        self.execution_gateway = execution_gateway

        self.mode = str(mode).strip().lower()

        if self.mode not in {"paper", "simulated"}:
            raise ValueError(
                "TradingPipelineManager solo admite PAPER/SIMULATED"
            )

        self.auto_learning_bridge = None
        self.cycle_context: Optional[Dict[str, Any]] = None

        self._bootstrap_completed_at: Optional[int] = None
        self._bootstrap_grace_cycles = 3
        self._bootstrap_grace_seconds = 30

    async def process_trading_cycle(
        self,
        state: Dict,
        market_data: Dict[str, pd.DataFrame],
        cycle_id: int = 0,
    ) -> TradingCycleResult:
        """Procesa un ciclo completo de trading."""

        start_time = pd.Timestamp.utcnow()

        if not state:
            raise ValueError("System state not injected")

        state.setdefault("version", "1.0")

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
            "timestamp": pd.Timestamp.utcnow().isoformat(),
        }

        logger.info(
            "CYCLE SNAPSHOT | Cycle %s | Balances: %s | Prices: %s",
            cycle_id,
            balances,
            prices,
        )

        result = TradingCycleResult(
            signals_generated=0,
            orders_executed=0,
            orders_rejected=0,
            cooldown_blocked=0,
            l3_regime="unknown",
            portfolio_value=0.0,
            execution_time=0.0,
        )

        try:
            l3_output = await self._update_l3_decision(
                state,
                market_data,
            )
            result.l3_regime = l3_output.get("regime", "unknown")

            l2_signals = await self._generate_l2_signals(
                state,
                market_data,
                l3_output,
            )
            result.signals_generated = len(l2_signals)

            valid_signals = await self._validate_signals(
                l2_signals,
                market_data,
            )

            l3_regime = l3_output.get("regime", "unknown")

            orders = (
                await self.order_manager.generate_orders(
                    state,
                    valid_signals,
                    l3_regime,
                )
                if valid_signals
                else []
            )

            validated_orders = (
                await self._validate_orders(orders, state)
                if orders
                else []
            )

            executed = await self._ejecutar_ordenes_por_gateway(
                validated_orders
            )

            filled = [
                order
                for order in executed
                if order.get("status") == "filled"
            ]

            rejected = [
                order
                for order in executed
                if order.get("status") == "rejected"
            ]

            result.orders_executed = len(filled)
            result.orders_rejected = len(rejected)

            if filled:
                await self._actualizar_contexto_cartera()

                await self._record_trades_for_auto_learning(
                    filled,
                    l3_output,
                    market_data,
                )

                await self._log_paper_trades(
                    filled,
                    market_data,
                    cycle_id,
                    l3_output,
                )

            rotation_filled = await self._process_position_rotation(
                state,
                market_data,
            )
            result.orders_executed += rotation_filled

            rebalancing_filled = await self._process_rebalancing(
                state,
                market_data,
                l3_output,
                valid_signals,
            )
            result.orders_executed += rebalancing_filled

            if rotation_filled > 0 or rebalancing_filled > 0:
                await self._actualizar_contexto_cartera()

            await self._sync_state_from_snapshot(state)

            if self.portfolio_manager.market_data_manager:
                try:
                    nav_result = (
                        await self.portfolio_manager.update_nav_async(
                            prices
                        )
                    )
                    result.portfolio_value = nav_result["total_nav"]

                except Exception as nav_error:
                    logger.error(
                        "Error calculando NAV: %s",
                        nav_error,
                        exc_info=True,
                    )
                    result.portfolio_value = (
                        self._calculate_nav_from_snapshot()
                    )
            else:
                result.portfolio_value = (
                    self._calculate_nav_from_snapshot()
                )

            if result.portfolio_value <= 0:
                balances_actuales = self.cycle_context["balances"]

                usdt = balances_actuales.get("USDT", 0.0)
                btc = balances_actuales.get("BTC", 0.0)
                eth = balances_actuales.get("ETH", 0.0)

                if usdt > 0 or btc > 0 or eth > 0:
                    logger.critical(
                        "NAV inválido con balances no nulos: %s",
                        result.portfolio_value,
                    )

        except Exception as error:
            logger.error(
                "Error en ciclo de trading: %s",
                error,
                exc_info=True,
            )

        finally:
            result.execution_time = (
                pd.Timestamp.utcnow() - start_time
            ).total_seconds()

            logger.info(
                "Ciclo completado en %.2fs | Señales: %s | "
                "Órdenes: %s ejecutadas, %s rechazadas",
                result.execution_time,
                result.signals_generated,
                result.orders_executed,
                result.orders_rejected,
            )

        return result

    async def _ejecutar_ordenes_por_gateway(
        self,
        ordenes: List[Dict],
    ) -> List[Dict]:
        """
        Ejecuta órdenes exclusivamente mediante ExecutionGateway.

        No existe fallback a OrderManager: si el Gateway no está
        disponible, las órdenes se rechazan.
        """

        if not ordenes:
            return []

        if self.execution_gateway is None:
            logger.critical(
                "ExecutionGateway no inicializado. "
                "Se bloquea la ejecución de %s órdenes.",
                len(ordenes),
            )

            rechazadas = []

            for orden in ordenes:
                resultado = dict(orden)
                resultado["status"] = "rejected"
                resultado["validation_error"] = (
                    "ExecutionGateway no está inicializado"
                )
                rechazadas.append(resultado)

            return rechazadas

        try:
            resultados = await self.execution_gateway.execute_orders(
                ordenes
            )

            if not isinstance(resultados, list):
                raise TypeError(
                    "ExecutionGateway debe devolver una lista"
                )

            return resultados

        except Exception as error:
            logger.exception(
                "ExecutionGateway ha fallado"
            )

            fallidas = []

            for orden in ordenes:
                resultado = dict(orden)
                resultado["status"] = "failed"
                resultado["execution_error"] = str(error)
                fallidas.append(resultado)

            return fallidas

    async def _actualizar_contexto_cartera(self) -> None:
        """
        Actualiza el snapshot desde la cartera ya sincronizada.

        No aplica operaciones a PortfolioManager. La sincronización
        corresponde a OrderExecutors.
        """

        if self.cycle_context is None:
            return

        balances = await self.portfolio_manager.get_balances_async()

        self.cycle_context["balances"] = balances
        self.cycle_context["timestamp"] = (
            pd.Timestamp.utcnow().isoformat()
        )

        logger.info(
            "Snapshot de cartera actualizado: %s",
            balances,
        )

    def _calculate_nav_from_snapshot(self) -> float:
        """Calcula el NAV utilizando el snapshot del ciclo."""

        if not self.cycle_context:
            return 0.0

        balances = self.cycle_context["balances"]
        prices = self.cycle_context["prices"]

        total = float(balances.get("USDT", 0.0))

        for symbol, price in prices.items():
            asset = symbol.replace("USDT", "")
            total += float(balances.get(asset, 0.0)) * float(price)

        return total

    async def _sync_state_from_snapshot(self, state: Dict) -> None:
        """Sincroniza el estado general con el snapshot actual."""

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

        self.state_coordinator.update_total_value(
            snapshot["total_value"]
        )

    async def _record_trades_for_auto_learning(
        self,
        filled_orders: List[Dict],
        l3_output: Dict,
        market_data: Dict,
    ) -> None:
        """Registra operaciones ejecutadas para autoaprendizaje."""

        if not filled_orders or not self.auto_learning_bridge:
            return

        for order in filled_orders:
            try:
                await self.auto_learning_bridge.record_order_execution(
                    order=order,
                    l3_context=l3_output,
                    market_data=market_data,
                )
            except Exception as error:
                logger.error(
                    "Error en auto-learning: %s",
                    error,
                    exc_info=True,
                )

    async def _log_paper_trades(
        self,
        filled_orders: List[Dict],
        market_data: Dict,
        cycle_id: int,
        l3_output: Dict,
    ) -> None:
        """
        Mantiene compatibilidad con el registro PAPER.

        OrderExecutors registra las operaciones ejecutadas. Este
        método solo registra las que no estén marcadas previamente.
        """

        try:
            from storage.paper_trade_logger import log_paper_trade

            for order in filled_orders:
                if order.get("_already_logged"):
                    continue

                log_paper_trade(
                    order,
                    market_data=market_data,
                    cycle_id=cycle_id,
                    strategy=l3_output.get(
                        "strategy_type",
                        "paper",
                    ),
                )

                logger.warning(
                    "Trade sin marca de registro previo: %s",
                    order.get("order_id"),
                )

        except Exception:
            logger.exception("Error registrando trades PAPER")

    async def _update_l3_decision(
        self,
        state: Dict,
        market_data: Dict,
    ) -> Dict:
        """Obtiene la decisión estratégica L3."""

        from core.l3_processor import get_l3_decision

        try:
            output = get_l3_decision(market_data)
        except Exception:
            logger.exception("Error obteniendo decisión L3")
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

    async def _generate_l2_signals(
        self,
        state: Dict,
        market_data: Dict,
        l3_decision: Dict,
    ) -> List[Dict]:
        """Genera señales L2 y aplica el periodo de gracia."""

        try:
            signals = self.l2_processor.generate_signals_conservative(
                market_data=market_data,
                l3_context=l3_decision,
            )

            if not self._is_bootstrap_grace_period_active(state):
                return signals

            filtered_signals = []

            for signal in signals:
                try:
                    signal_type = getattr(
                        signal,
                        "side",
                        "",
                    ).lower()

                    symbol = getattr(
                        signal,
                        "symbol",
                        "UNKNOWN",
                    )

                    if signal_type in {"sell", "buy"}:
                        logger.info(
                            "Periodo de gracia: señal %s bloqueada para %s",
                            signal_type.upper(),
                            symbol,
                        )
                        continue

                    filtered_signals.append(signal)

                except AttributeError:
                    logger.exception(
                        "Señal incompatible durante periodo de gracia"
                    )

            return filtered_signals

        except Exception:
            logger.exception("Error generando señales L2")
            return []

    def mark_bootstrap_completed(
        self,
        current_cycle: int = 0,
    ) -> None:
        """Activa el periodo de gracia tras el bootstrap."""

        self._bootstrap_completed_at = current_cycle

        logger.info(
            "Bootstrap completado en ciclo %s",
            current_cycle,
        )

    def _is_bootstrap_grace_period_active(
        self,
        state: Dict,
    ) -> bool:
        """Comprueba si sigue activo el periodo de gracia."""

        if self._bootstrap_completed_at is None:
            return False

        current_cycle = state.get("cycle_id", 0)

        cycles_since_bootstrap = (
            current_cycle - self._bootstrap_completed_at
        )

        return cycles_since_bootstrap < self._bootstrap_grace_cycles

    async def _validate_signals(
        self,
        signals: List[Dict],
        market_data: Dict,
    ) -> List[Dict]:
        """Envía señales al verificador."""

        valid = []

        for signal in signals:
            try:
                await self.signal_verifier.submit_signal_for_verification(
                    signal,
                    market_data,
                )
                valid.append(signal)

            except Exception:
                logger.exception("Señal rechazada por el verificador")

        return valid

    async def _validate_orders(
        self,
        orders: List[Dict],
        state: Dict,
    ) -> List[Dict]:
        """Aplica la validación previa existente del OrderManager."""

        validated = []
        portfolio = state.get("portfolio", {})

        for order in orders:
            if order.get("status") != "pending":
                validated.append(order)
                continue

            try:
                check = self.order_manager.validate_order_size(
                    order.get("symbol"),
                    order.get("quantity"),
                    order.get("price"),
                    portfolio,
                )

                if check.get("valid"):
                    validated.append(order)
                else:
                    rejected = dict(order)
                    rejected["status"] = "rejected"
                    rejected["validation_error"] = check.get(
                        "reason",
                        "Orden no válida",
                    )
                    validated.append(rejected)

            except Exception as error:
                rejected = dict(order)
                rejected["status"] = "rejected"
                rejected["validation_error"] = str(error)
                validated.append(rejected)

        return validated

    async def _process_position_rotation(
        self,
        state: Dict,
        market_data: Dict,
    ) -> int:
        """Procesa rotaciones mediante ExecutionGateway."""

        if not self.position_rotator:
            return 0

        try:
            rotation = (
                await self.position_rotator.check_and_rotate_positions(
                    state,
                    market_data,
                )
            )

            if not rotation:
                return 0

            executed = await self._ejecutar_ordenes_por_gateway(
                rotation
            )

            filled = [
                order
                for order in executed
                if order.get("status") == "filled"
            ]

            if filled:
                await self._actualizar_contexto_cartera()

            return len(filled)

        except Exception:
            logger.exception("Error procesando rotación")
            return 0

    async def _process_rebalancing(
        self,
        state: Dict,
        market_data: Dict,
        l3_output: Dict,
        valid_signals: List,
    ) -> int:
        """Procesa rebalanceos mediante ExecutionGateway."""

        if not self.auto_rebalancer:
            return 0

        if l3_output.get("signal") == "sell":
            return 0

        if self.is_in_grace_period:
            logger.info(
                "AutoRebalancer: rebalanceo bloqueado durante periodo de gracia"
            )
            return 0

        try:
            rebalance = (
                await self.auto_rebalancer.check_and_execute_rebalance(
                    market_data,
                    l3_decision=l3_output,
                )
            )

            if not rebalance:
                return 0

            executed = await self._ejecutar_ordenes_por_gateway(
                rebalance
            )

            filled = [
                order
                for order in executed
                if order.get("status") == "filled"
            ]

            if filled:
                await self._actualizar_contexto_cartera()

            return len(filled)

        except Exception:
            logger.exception("Error procesando rebalanceo")
            return 0

    def get_cycle_context(self) -> Optional[Dict[str, Any]]:
        """Devuelve el contexto del ciclo actual."""

        return self.cycle_context

    @property
    def is_in_grace_period(self) -> bool:
        """Indica si sigue activo el periodo de gracia."""

        if self._bootstrap_completed_at is None:
            return False

        current_cycle = 0

        if self.cycle_context:
            current_cycle = self.cycle_context.get(
                "cycle_id",
                0,
            )

        cycles_since_bootstrap = (
            current_cycle - self._bootstrap_completed_at
        )

        return cycles_since_bootstrap < self._bootstrap_grace_cycles
