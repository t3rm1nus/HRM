
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway central de ejecución de HRM.

Centraliza la validación, el control de riesgo y la delegación
de órdenes al OrderManager existente.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from core.logging import logger
from core.portfolio_manager import PortfolioManager
from core.trading_metrics import get_trading_metrics
from l1_operational.order_manager import OrderManager
from l1_operational.risk_guard import RiskGuard


class ExecutionGateway:
    """Punto único de entrada para ejecutar órdenes."""

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager
    ):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager

        # Reutilizar los componentes ya inicializados por OrderManager.
        self.validator = order_manager.validators
        self.risk_guard = RiskGuard(portfolio_manager)
        self.trading_metrics = get_trading_metrics()

        self.execution_stats = {
            "total_orders": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "rejected_orders": 0,
            "risk_blocks": 0,
        }

    async def execute_orders(
        self,
        orders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Valida, filtra y delega las órdenes al motor existente."""

        if not orders:
            return []

        self.execution_stats["total_orders"] += len(orders)

        logger.info(
            "Gateway: procesando %s órdenes",
            len(orders)
        )

        ordenes_validadas = self._validar_ordenes(orders)
        ordenes_autorizadas = await self._validar_riesgo(
            ordenes_validadas
        )

        if not ordenes_autorizadas:
            return self._obtener_ordenes_rechazadas(
                ordenes_validadas
            )

        resultados = await self._ejecutar_ordenes(
            ordenes_autorizadas
        )

        self._actualizar_estadisticas(resultados)

        await self._registrar_resultados(resultados)

        return self._combinar_resultados(
            ordenes_validadas,
            resultados
        )

    def _validar_ordenes(
        self,
        ordenes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normaliza y valida las órdenes antes de evaluar su riesgo."""

        ordenes_validas = []

        for orden_original in ordenes:
            orden = dict(orden_original)

            try:
                resultado = self.validator.validate_and_normalize_order(
                    orden
                )

                reporte = resultado["validation"]

                if reporte["status"] != "valid":
                    orden["status"] = "rejected"
                    orden["validation_error"] = "; ".join(
                        reporte.get("errors", [])
                    )

                    self.execution_stats["rejected_orders"] += 1
                    ordenes_validas.append(orden)
                    continue

                orden_normalizada = resultado["order"]

                # Mantener los campos adicionales del contrato original.
                orden_normalizada.update({
                    clave: valor
                    for clave, valor in orden.items()
                    if clave not in orden_normalizada
                })

                orden_normalizada["side"] = (
                    orden_normalizada["action"]
                )

                orden_normalizada["action"] = (
                    orden_normalizada["action"].lower()
                )

                ordenes_validas.append(orden_normalizada)

            except Exception as error:
                orden["status"] = "rejected"
                orden["validation_error"] = str(error)

                self.execution_stats["rejected_orders"] += 1
                ordenes_validas.append(orden)

                logger.exception(
                    "Error validando una orden"
                )

        return ordenes_validas

    async def _validar_riesgo(
        self,
        ordenes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Comprueba el riesgo individual de cada orden."""

        ordenes_autorizadas = []

        for orden in ordenes:
            if orden.get("status") == "rejected":
                continue

            try:
                resultado = await self.risk_guard.check_order_risk(
                    orden
                )

                if not resultado.get("allowed", False):
                    orden["status"] = "rejected"
                    orden["risk_error"] = resultado.get(
                        "reason",
                        "Orden bloqueada por riesgo"
                    )

                    self.execution_stats["risk_blocks"] += 1
                    self.execution_stats["rejected_orders"] += 1

                    logger.warning(
                        "Orden bloqueada por riesgo: %s",
                        orden["risk_error"]
                    )
                    continue

                ordenes_autorizadas.append(orden)

            except Exception as error:
                # Ante un fallo del control de riesgo, bloquear.
                orden["status"] = "rejected"
                orden["risk_error"] = str(error)

                self.execution_stats["risk_blocks"] += 1
                self.execution_stats["rejected_orders"] += 1

                logger.exception(
                    "Error durante la validación de riesgo"
                )

        return ordenes_autorizadas

    async def _ejecutar_ordenes(
        self,
        ordenes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Delega la ejecución al OrderManager real."""

        try:
            return await self.order_manager.execute_orders(
                ordenes
            )

        except Exception:
            logger.exception(
                "Error ejecutando órdenes en OrderManager"
            )

            resultados = []

            for orden in ordenes:
                resultado = dict(orden)
                resultado["status"] = "failed"
                resultado["execution_error"] = (
                    "Error interno del motor de ejecución"
                )
                resultados.append(resultado)

            return resultados

    def _actualizar_estadisticas(
        self,
        resultados: List[Dict[str, Any]]
    ) -> None:
        """Actualiza los contadores de ejecución."""

        for resultado in resultados:
            estado = str(
                resultado.get("status", "")
            ).lower()

            if estado == "filled":
                self.execution_stats[
                    "successful_executions"
                ] += 1

            elif estado in ("failed", "error"):
                self.execution_stats[
                    "failed_executions"
                ] += 1

            elif estado == "rejected":
                self.execution_stats[
                    "rejected_orders"
                ] += 1

    async def _registrar_resultados(
        self,
        resultados: List[Dict[str, Any]]
    ) -> None:
        """Registra los resultados sin duplicar actualizaciones del portfolio."""

        ejecutadas = [
            resultado
            for resultado in resultados
            if resultado.get("status") == "filled"
        ]

        if not ejecutadas:
            return

        try:
            valor_cartera = (
                await self.portfolio_manager.get_total_value_async()
            )

            self.trading_metrics.update_from_orders(
                executed_orders=ejecutadas,
                portfolio_value=valor_cartera
            )

        except Exception:
            logger.exception(
                "No se pudieron actualizar las métricas de trading"
            )

    def _obtener_ordenes_rechazadas(
        self,
        ordenes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Devuelve las órdenes rechazadas durante los controles."""

        return [
            orden
            for orden in ordenes
            if orden.get("status") == "rejected"
        ]

    def _combinar_resultados(
        self,
        ordenes_originales: List[Dict[str, Any]],
        resultados: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Conserva tanto las órdenes ejecutadas como las rechazadas."""

        rechazadas = [
            orden
            for orden in ordenes_originales
            if orden.get("status") == "rejected"
        ]

        return resultados + rechazadas

    async def execute_signal(
        self,
        signal: Any,
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convierte una señal táctica en una orden ejecutable."""

        try:
            ordenes = await self._convertir_senal_a_orden(
                signal,
                market_data
            )

            return await self.execute_orders(ordenes)

        except Exception:
            logger.exception(
                "Error ejecutando señal táctica"
            )
            return []

    async def _convertir_senal_a_orden(
        self,
        signal: Any,
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Construye una orden a partir de una señal."""

        symbol = getattr(signal, "symbol", "")
        side = str(
            getattr(signal, "side", "hold")
        ).lower()

        confidence = float(
            getattr(signal, "confidence", 0.5)
        )

        if side not in ("buy", "sell"):
            return []

        precio = self._obtener_precio_actual(
            symbol,
            market_data
        )

        if precio is None or precio <= 0:
            logger.warning(
                "No hay precio válido para %s. Orden cancelada.",
                symbol
            )
            return []

        cantidad = await self._calcular_cantidad(
            symbol,
            confidence,
            precio
        )

        if cantidad <= 0:
            return []

        return [{
            "symbol": symbol,
            "action": side,
            "side": side,
            "type": "MARKET",
            "quantity": cantidad,
            "price": precio,
            "confidence": confidence,
            "signal_source": "tactical",
            "status": "pending",
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }]

    def _obtener_precio_actual(
        self,
        symbol: str,
        market_data: Dict[str, Any]
    ) -> Optional[float]:
        """Obtiene el último precio disponible en los datos recibidos."""

        datos = market_data.get(symbol)

        try:
            if isinstance(datos, dict):
                precio = datos.get("close")

            elif isinstance(datos, pd.DataFrame):
                if datos.empty or "close" not in datos.columns:
                    return None

                precio = datos["close"].iloc[-1]

            else:
                return None

            precio = float(precio)

            if precio <= 0:
                return None

            return precio

        except (TypeError, ValueError, KeyError, IndexError):
            return None

    async def _calcular_cantidad(
        self,
        symbol: str,
        confidence: float,
        precio: float
    ) -> float:
        """Calcula la cantidad respetando el capital disponible."""

        try:
            valor_cartera = float(
                await self.portfolio_manager.get_total_value_async()
            )

            if valor_cartera <= 0:
                return 0.0

            confianza = max(
                0.0,
                min(confidence, 1.0)
            )

            asignacion = valor_cartera * 0.10 * confianza

            cantidad = asignacion / precio

            # No forzar mínimos que puedan superar el capital disponible.
            return max(0.0, cantidad)

        except Exception:
            logger.exception(
                "No se pudo calcular la cantidad para %s",
                symbol
            )
            return 0.0

    def get_execution_stats(self) -> Dict[str, int]:
        """Devuelve una copia de las estadísticas."""

        return self.execution_stats.copy()

    async def cleanup_stale_orders(
        self,
        current_positions: Dict[str, float]
    ) -> Dict[str, Any]:
        """Delega la limpieza al gestor de órdenes."""

        try:
            resultado = self.order_manager.cleanup_stale_orders(
                current_positions
            )

            if hasattr(resultado, "__await__"):
                resultado = await resultado

            return resultado or {}

        except Exception:
            logger.exception(
                "Error limpiando órdenes obsoletas"
            )
            return {}


async def create_execution_gateway(
    portfolio_manager: PortfolioManager,
    order_manager: OrderManager
) -> ExecutionGateway:
    """Construye el gateway de ejecución."""

    gateway = ExecutionGateway(
        portfolio_manager,
        order_manager
    )

    logger.info(
        "Execution Gateway inicializado"
    )

    return gateway
