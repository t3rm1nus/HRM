from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.logging import logger
from l1_operational.simulated_exchange_client import (
    SimulatedExchangeClient,
)

try:
    from storage.paper_trade_logger import (
        get_paper_logger,
        PAPER_LOGGER_AVAILABLE,
    )
except ImportError:
    get_paper_logger = None
    PAPER_LOGGER_AVAILABLE = False

try:
    from core.trading_metrics import get_trading_metrics
except ImportError:
    get_trading_metrics = None


class OrderExecutors:
    """
    Ejecuta órdenes de trading en los modos soportados.

    PAPER:
        Ejecuta operaciones exclusivamente contra el exchange simulado.

    LIVE y TESTNET:
        No ejecuta operaciones desde esta clase. Se necesita un
        ejecutor real integrado y explícitamente habilitado.

    No crea capital ficticio ni cambia de modo automáticamente.
    """

    MODOS_PAPER = {"paper", "simulated"}
    MODOS_REALES = {"live", "testnet", "backtest"}

    def __init__(
        self,
        state_manager,
        portfolio_manager,
        mode: str = "simulated",
        simulated_client=None,
    ):
        self.state_manager = state_manager
        self.portfolio_manager = portfolio_manager
        self.mode = self._normalizar_modo(mode)

        self.paper_mode = self.mode in self.MODOS_PAPER
        self.use_testnet = self.mode == "testnet"

        self.simulated_client = simulated_client
        self.paper_logger = None
        self.trading_metrics = None

        if self.paper_mode:
            self._inicializar_simulador()
            self._inicializar_registro()
            self._inicializar_metricas()

        logger.info(
            "OrderExecutors inicializado: modo=%s, paper=%s",
            self.mode,
            self.paper_mode,
        )

    @staticmethod
    def _normalizar_modo(mode: str) -> str:
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("El modo de ejecución es obligatorio")

        modo = mode.strip().lower()

        modos_validos = {
            "paper",
            "simulated",
            "live",
            "testnet",
            "backtest",
        }

        if modo not in modos_validos:
            raise ValueError(f"Modo de ejecución no válido: {modo}")

        return modo

    def _inicializar_simulador(self) -> None:
        if self.simulated_client is None:
            self.simulated_client = (
                SimulatedExchangeClient._instance
            )

        if self.simulated_client is None:
            raise RuntimeError(
                "No existe un SimulatedExchangeClient inicializado. "
                "Debe crearse desde bootstrap con el capital "
                "configurado. No se permite crear capital ficticio."
            )

        if not hasattr(self.simulated_client, "execute_order"):
            raise TypeError(
                "El cliente simulado no implementa execute_order()"
            )

        logger.info(
            "Exchange PAPER conectado. Balances: %s",
            self.simulated_client.get_balances(),
        )

    def _inicializar_registro(self) -> None:
        if not PAPER_LOGGER_AVAILABLE or get_paper_logger is None:
            logger.warning(
                "PaperTradeLogger no disponible"
            )
            return

        try:
            self.paper_logger = get_paper_logger()
        except Exception:
            logger.exception(
                "No se pudo inicializar PaperTradeLogger"
            )

    def _inicializar_metricas(self) -> None:
        if get_trading_metrics is None:
            return

        try:
            self.trading_metrics = get_trading_metrics()
        except Exception:
            logger.exception(
                "No se pudieron inicializar TradingMetrics"
            )

    @staticmethod
    def _validar_orden(
        symbol: str,
        action: str,
        quantity: float,
        current_price: float,
    ) -> Optional[str]:
        if not isinstance(symbol, str) or not symbol.strip():
            return "Símbolo inválido"

        if not isinstance(action, str):
            return "Acción inválida"

        if action.strip().lower() not in {"buy", "sell"}:
            return "La acción debe ser buy o sell"

        try:
            cantidad = float(quantity)
            precio = float(current_price)
        except (TypeError, ValueError):
            return "Cantidad y precio deben ser numéricos"

        import math

        if not math.isfinite(cantidad) or cantidad <= 0:
            return "La cantidad debe ser positiva y finita"

        if not math.isfinite(precio) or precio <= 0:
            return "El precio debe ser positivo y finito"

        return None

    async def _obtener_nav(self) -> Optional[float]:
        if self.portfolio_manager is None:
            return None

        try:
            return float(
                await self.portfolio_manager.get_total_value_async()
            )
        except Exception:
            logger.exception(
                "No se pudo calcular el valor del portfolio"
            )
            return None

    async def _sincronizar_portfolio(self) -> None:
        if self.portfolio_manager is None:
            return

        metodo = getattr(
            self.portfolio_manager,
            "sync_from_exchange_async",
            None,
        )

        if metodo is None:
            return

        await metodo(self.simulated_client)

    async def _registrar_trade(
        self,
        resultado: Dict[str, Any],
        nav_after: Optional[float],
    ) -> None:
        if self.paper_logger is not None:
            try:
                self.paper_logger.log_paper_trade(
                    order=resultado,
                    market_data=None,
                    cycle_id=None,
                    strategy="order_executor",
                )
                resultado["_already_logged"] = True
            except Exception:
                logger.exception(
                    "No se pudo persistir el trade PAPER"
                )

        if self.trading_metrics is not None:
            try:
                self.trading_metrics.update_from_orders(
                    executed_orders=[resultado],
                    portfolio_value=nav_after or 0.0,
                )
            except Exception:
                logger.exception(
                    "No se pudieron actualizar las métricas"
                )

    async def execute_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        current_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta una orden MARKET simulada.

        Las órdenes pendientes y las protecciones stop-loss /
        take-profit necesitan un gestor específico. Recibir esos
        parámetros no significa que se hayan colocado.
        """
        error = self._validar_orden(
            symbol,
            action,
            quantity,
            current_price,
        )

        if error:
            return self._crear_resultado(
                "rejected",
                symbol,
                action,
                error,
            )

        if not self.paper_mode:
            return self._crear_resultado(
                "rejected",
                symbol,
                action,
                (
                    f"El modo {self.mode} no está habilitado en "
                    "OrderExecutors. No se cambiará automáticamente "
                    "a PAPER ni se enviará una orden real."
                ),
            )

        if stop_loss is not None or take_profit is not None:
            return self._crear_resultado(
                "rejected",
                symbol,
                action,
                (
                    "No se aceptan stop_loss ni take_profit en esta "
                    "ejecución MARKET: todavía no existe un gestor "
                    "de órdenes de protección integrado."
                ),
            )

        nav_before = await self._obtener_nav()

        try:
            await self._sincronizar_portfolio()

            trade = self.simulated_client.execute_order(
                symbol=symbol.strip().upper(),
                side=action.strip().upper(),
                qty=float(quantity),
                market_price=float(current_price),
            )

            balances = self.simulated_client.get_balances()

            await self._sincronizar_portfolio()

            nav_after = await self._obtener_nav()

            resultado = {
                "status": "filled",
                "symbol": symbol.strip().upper(),
                "action": action.strip().lower(),
                "side": action.strip().lower(),
                "quantity": float(quantity),
                "filled_quantity": float(quantity),
                "market_price": float(current_price),
                "filled_price": trade["execution_price"],
                "fill_price": trade["execution_price"],
                "price": trade["execution_price"],
                "order_value": trade["cost"],
                "commission": trade["fee"],
                "slippage_cost": trade["slippage_cost"],
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "execution_type": "SIMULATED_TRADE",
                "order_id": trade["order_id"],
                "order_type": "MARKET",
                "balances_after": balances,
                "signal_source": "order_executor",
                "reason": "Orden simulada ejecutada",
                "nav_before": nav_before,
                "nav_after": nav_after,
                "simulated": True,
            }

            await self._registrar_trade(
                resultado,
                nav_after,
            )

            logger.info(
                "PAPER ejecutado: %s %s cantidad=%s precio=%s",
                action.upper(),
                symbol,
                quantity,
                trade["execution_price"],
            )

            return resultado

        except Exception as error:
            logger.exception(
                "Error ejecutando orden PAPER: %s",
                symbol,
            )

            return self._crear_resultado(
                "failed",
                symbol,
                action,
                str(error),
            )

    @staticmethod
    def _crear_resultado(
        status: str,
        symbol: str,
        action: str,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "symbol": symbol,
            "action": action,
            "side": action,
            "reason": reason,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

    def get_trade_summary(self) -> Dict[str, Any]:
        resumen = {
            "paper_logger_available": (
                PAPER_LOGGER_AVAILABLE
                and self.paper_logger is not None
            ),
            "trading_metrics_available": (
                self.trading_metrics is not None
            ),
            "paper_trades": {},
            "metrics": {},
        }

        if self.paper_logger is not None:
            try:
                resumen["paper_trades"] = (
                    self.paper_logger.get_session_summary()
                )
            except Exception:
                logger.exception(
                    "Error consultando historial PAPER"
                )

        if self.trading_metrics is not None:
            try:
                resumen["metrics"] = (
                    self.trading_metrics.get_summary_report()
                )
            except Exception:
                logger.exception(
                    "Error consultando métricas"
                )

        return resumen