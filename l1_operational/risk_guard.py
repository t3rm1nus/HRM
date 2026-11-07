
# -*- coding: utf-8 -*-
"""
Control de riesgo de HRM.

Las comprobaciones de órdenes utilizan balances reales del PortfolioManager.
No se utilizan balances, precios ni capital ficticios para autorizar órdenes.
"""

import math
import time
from typing import Any, Dict, Optional

from core.logging import logger

from l1_operational.config import (
    RISK_LIMITS,
    PORTFOLIO_LIMITS
)

from .models import (
    Signal,
    ValidationResult
)


class RiskGuard:
    """Aplica los límites de riesgo antes de ejecutar operaciones."""

    def __init__(self, portfolio_manager=None):
        self.portfolio_manager = portfolio_manager

        self.daily_pnl = 0.0
        self.daily_start_balance: Optional[float] = None

        self.current_positions: Dict[str, float] = {}
        self.account_balance: Optional[float] = None

        self.daily_start_timestamp = time.time()

    async def check_order_risk(
        self,
        order: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprueba si una orden puede ejecutarse con el capital disponible."""

        if self.portfolio_manager is None:
            return self._rechazar(
                "PortfolioManager no está disponible"
            )

        try:
            symbol = str(
                order.get("symbol", "")
            ).upper()

            action = str(
                order.get(
                    "action",
                    order.get("side", "")
                )
            ).upper()

            cantidad = self._convertir_numero(
                order.get("quantity")
            )

            precio = self._convertir_numero(
                order.get("price")
            )

            if not symbol or not symbol.endswith("USDT"):
                return self._rechazar(
                    "Símbolo no soportado o inválido"
                )

            if action not in ("BUY", "SELL"):
                return self._rechazar(
                    "Acción inválida"
                )

            if cantidad is None or cantidad <= 0:
                return self._rechazar(
                    "Cantidad inválida"
                )

            if precio is None or precio <= 0:
                return self._rechazar(
                    "Precio de mercado no válido"
                )

            valor_orden = cantidad * precio

            error_tamano = self._validar_tamano_orden(
                symbol,
                cantidad,
                valor_orden
            )

            if error_tamano:
                return self._rechazar(error_tamano)

            balances = await (
                self.portfolio_manager.get_balances_async()
            )

            if not isinstance(balances, dict) or not balances:
                return self._rechazar(
                    "No se pudieron obtener los balances de la cartera"
                )

            balances = {
                str(activo).upper(): self._convertir_numero(
                    saldo
                )
                for activo, saldo in balances.items()
            }

            if any(
                saldo is None or saldo < 0
                for saldo in balances.values()
            ):
                return self._rechazar(
                    "Los balances contienen valores inválidos"
                )

            balance_usdt = balances.get("USDT", 0.0)

            if action == "BUY":
                return await self._validar_compra(
                    symbol,
                    cantidad,
                    precio,
                    valor_orden,
                    balance_usdt,
                    balances
                )

            return await self._validar_venta(
                symbol,
                cantidad,
                balances
            )

        except Exception as error:
            logger.exception(
                "Error durante la comprobación de riesgo"
            )

            return self._rechazar(
                f"No se pudo completar el control de riesgo: {error}"
            )

    async def _validar_compra(
        self,
        symbol: str,
        cantidad: float,
        precio: float,
        valor_orden: float,
        balance_usdt: float,
        balances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Valida fondos, reserva y exposición antes de comprar."""

        coste_estimado = valor_orden * 1.002

        if balance_usdt < coste_estimado:
            return self._rechazar(
                "Saldo USDT insuficiente para cubrir la compra y comisiones"
            )

        valor_cartera = await self._obtener_valor_cartera()

        if valor_cartera is None or valor_cartera <= 0:
            return self._rechazar(
                "No se puede determinar el valor actual de la cartera"
            )

        reserva_minima = (
            valor_cartera
            * PORTFOLIO_LIMITS["MIN_USDT_RESERVE"]
        )

        if balance_usdt - coste_estimado < reserva_minima:
            return self._rechazar(
                "La compra vulneraría la reserva mínima de USDT"
            )

        activo = self._obtener_activo_base(symbol)

        limite_exposicion = self._obtener_limite_exposicion(
            activo
        )

        if limite_exposicion is not None:
            posicion_actual = balances.get(activo, 0.0)

            exposicion_actual = posicion_actual * precio
            exposicion_nueva = (
                exposicion_actual + valor_orden
            )

            maximo_permitido = (
                valor_cartera * limite_exposicion
            )

            if exposicion_nueva > maximo_permitido:
                return self._rechazar(
                    "La compra superaría el límite de exposición "
                    f"de {activo}"
                )

        maximo_posicion = PORTFOLIO_LIMITS.get(
            "MAX_POSITION_SIZE_USDT"
        )

        if (
            maximo_posicion is not None
            and valor_orden > maximo_posicion
        ):
            return self._rechazar(
                "La orden supera el tamaño máximo permitido por posición"
            )

        return self._permitir(
            "Compra autorizada"
        )

    async def _validar_venta(
        self,
        symbol: str,
        cantidad: float,
        balances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Impide vender más unidades de las que existen."""

        activo = self._obtener_activo_base(symbol)
        balance_activo = balances.get(activo, 0.0)

        if cantidad > balance_activo:
            return self._rechazar(
                f"Balance insuficiente de {activo}: "
                f"disponible {balance_activo}, solicitado {cantidad}"
            )

        return self._permitir(
            "Venta autorizada"
        )

    async def _obtener_valor_cartera(self) -> Optional[float]:
        """Obtiene una valoración actual de la cartera."""

        try:
            valor = await (
                self.portfolio_manager.get_total_value_async()
            )

            valor = self._convertir_numero(valor)

            if valor is None or valor <= 0:
                return None

            return valor

        except Exception:
            logger.exception(
                "No se pudo obtener el valor de la cartera"
            )
            return None

    def _validar_tamano_orden(
        self,
        symbol: str,
        cantidad: float,
        valor_orden: float
    ) -> Optional[str]:
        """Comprueba los límites configurados por operación."""

        minimo = RISK_LIMITS.get(
            "MIN_ORDER_SIZE_USDT",
            0
        )

        maximo_valor = RISK_LIMITS.get(
            "MAX_ORDER_SIZE_USDT"
        )

        if valor_orden < minimo:
            return (
                f"Orden inferior al mínimo de {minimo} USDT"
            )

        if (
            maximo_valor is not None
            and valor_orden > maximo_valor
        ):
            return (
                f"Orden superior al máximo de {maximo_valor} USDT"
            )

        activo = self._obtener_activo_base(symbol)

        maximo_cantidad = RISK_LIMITS.get(
            f"MAX_ORDER_SIZE_{activo}"
        )

        if (
            maximo_cantidad is not None
            and cantidad > maximo_cantidad
        ):
            return (
                f"Cantidad superior al límite configurado "
                f"para {activo}"
            )

        return None

    def _obtener_limite_exposicion(
        self,
        activo: str
    ) -> Optional[float]:
        """Obtiene el porcentaje máximo de exposición del activo."""

        clave = f"MAX_PORTFOLIO_EXPOSURE_{activo}"

        limite = PORTFOLIO_LIMITS.get(clave)

        if limite is None:
            return None

        return float(limite)

    @staticmethod
    def _obtener_activo_base(symbol: str) -> str:
        """Extrae el activo base de un par terminado en USDT."""

        if symbol.endswith("USDT"):
            return symbol[:-4]

        return symbol

    @staticmethod
    def _convertir_numero(
        valor: Any
    ) -> Optional[float]:
        """Convierte un valor a float y rechaza datos no finitos."""

        try:
            numero = float(valor)

            if not math.isfinite(numero):
                return None

            return numero

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _permitir(
        motivo: str
    ) -> Dict[str, Any]:
        """Construye una respuesta de autorización."""

        return {
            "allowed": True,
            "reason": motivo
        }

    @staticmethod
    def _rechazar(
        motivo: str
    ) -> Dict[str, Any]:
        """Construye una respuesta de rechazo."""

        return {
            "allowed": False,
            "reason": motivo
        }

    # ================================================================
    # API LEGACY: compatibilidad con señales L1
    # ================================================================

    def validate_signal(
        self,
        signal: Signal
    ) -> ValidationResult:
        """Valida los parámetros de una señal sin inventar balances."""

        logger.info(
            "Validando señal %s",
            getattr(signal, "signal_id", "unknown")
        )

        validaciones = [
            self._validate_basic_params,
            self._validate_stop_loss,
            self._validate_order_size
        ]

        for validar in validaciones:
            try:
                resultado = validar(signal)

                if not resultado.is_valid:
                    logger.warning(
                        "Señal rechazada: %s",
                        resultado.reason
                    )
                    return resultado

            except Exception as error:
                logger.exception(
                    "Error validando señal"
                )

                return ValidationResult(
                    False,
                    f"Error de validación: {error}"
                )

        # Las comprobaciones de balance y exposición requieren
        # el control asíncrono check_order_risk().
        return ValidationResult(
            True,
            "Parámetros de señal válidos. "
            "Pendiente de comprobación de cartera antes de ejecutar."
        )

    def _validate_basic_params(
        self,
        signal: Signal
    ) -> ValidationResult:
        """Comprueba los parámetros básicos de la señal."""

        if not getattr(signal, "signal_id", None):
            return ValidationResult(
                False,
                "Falta signal_id"
            )

        if not getattr(signal, "symbol", None):
            return ValidationResult(
                False,
                "Falta symbol"
            )

        side = str(
            getattr(signal, "side", "")
        ).lower()

        if side not in ("buy", "sell"):
            return ValidationResult(
                False,
                f"Acción inválida: {side}"
            )

        cantidad = self._convertir_numero(
            getattr(signal, "qty", None)
        )

        if cantidad is None or cantidad <= 0:
            return ValidationResult(
                False,
                "Cantidad inválida"
            )

        tipo = str(
            getattr(signal, "order_type", "market")
        ).lower()

        if tipo not in ("market", "limit"):
            return ValidationResult(
                False,
                "Tipo de orden inválido"
            )

        if tipo == "limit":
            precio = self._convertir_numero(
                getattr(signal, "price", None)
            )

            if precio is None or precio <= 0:
                return ValidationResult(
                    False,
                    "Una orden LIMIT necesita un precio válido"
                )

        return ValidationResult(
            True,
            "Parámetros básicos válidos"
        )

    def _validate_stop_loss(
        self,
        signal: Signal
    ) -> ValidationResult:
        """Valida el stop loss cuando la señal lo incluye."""

        stop_loss = getattr(
            signal,
            "stop_loss",
            None
        )

        if stop_loss is None:
            return ValidationResult(
                True,
                "Sin stop loss explícito"
            )

        precio = self._convertir_numero(
            getattr(signal, "price", None)
        )

        stop_loss = self._convertir_numero(
            stop_loss
        )

        if (
            precio is None
            or precio <= 0
            or stop_loss is None
            or stop_loss <= 0
        ):
            return ValidationResult(
                False,
                "Precio o stop loss inválido"
            )

        side = str(
            getattr(signal, "side", "")
        ).lower()

        if side == "buy" and stop_loss >= precio:
            return ValidationResult(
                False,
                "El stop loss de una compra debe estar por debajo del precio"
            )

        if side == "sell" and stop_loss <= precio:
            return ValidationResult(
                False,
                "El stop loss de una venta debe estar por encima del precio"
            )

        return ValidationResult(
            True,
            "Stop loss válido"
        )

    def _validate_order_size(
        self,
        signal: Signal
    ) -> ValidationResult:
        """Valida los límites de cantidad de la señal."""

        symbol = str(
            getattr(signal, "symbol", "")
        ).upper()

        activo = self._obtener_activo_base(symbol)

        cantidad = self._convertir_numero(
            getattr(signal, "qty", None)
        )

        if cantidad is None or cantidad <= 0:
            return ValidationResult(
                False,
                "Cantidad inválida"
            )

        limite = RISK_LIMITS.get(
            f"MAX_ORDER_SIZE_{activo}"
        )

        if limite is not None and cantidad > limite:
            return ValidationResult(
                False,
                f"Cantidad superior al máximo permitido para {activo}"
            )

        precio = self._convertir_numero(
            getattr(signal, "price", None)
        )

        if precio is None or precio <= 0:
            return ValidationResult(
                False,
                "No se dispone de un precio válido para valorar la señal"
            )

        valor = cantidad * precio

        minimo = RISK_LIMITS.get(
            "MIN_ORDER_SIZE_USDT",
            0
        )

        maximo = RISK_LIMITS.get(
            "MAX_ORDER_SIZE_USDT"
        )

        if valor < minimo:
            return ValidationResult(
                False,
                f"Orden inferior al mínimo de {minimo} USDT"
            )

        if maximo is not None and valor > maximo:
            return ValidationResult(
                False,
                f"Orden superior al máximo de {maximo} USDT"
            )

        return ValidationResult(
            True,
            "Tamaño de orden válido"
        )

    def _validate_account_balance(
        self,
        signal: Signal
    ) -> ValidationResult:
        """No autoriza operaciones con balances simulados o inventados."""

        return ValidationResult(
            False,
            "La comprobación de balance requiere check_order_risk() asíncrono"
        )

    def _validate_position_limits(
        self,
        signal: Signal
    ) -> ValidationResult:
        """No calcula exposición usando precios ficticios."""

        return ValidationResult(
            False,
            "La comprobación de exposición requiere check_order_risk() asíncrono"
        )

    def _validate_daily_drawdown(
        self,
        signal: Signal
    ) -> ValidationResult:
        """Comprueba drawdown solo si existe una referencia inicial real."""

        if (
            self.daily_start_balance is None
            or self.daily_start_balance <= 0
        ):
            return ValidationResult(
                True,
                "Drawdown no evaluado: falta balance inicial real",
                warnings=[
                    "Configura el balance inicial diario para activar este control"
                ]
            )

        valor_actual = (
            self.daily_start_balance + self.daily_pnl
        )

        drawdown = (
            self.daily_start_balance - valor_actual
        ) / self.daily_start_balance

        limite = PORTFOLIO_LIMITS.get(
            "MAX_DAILY_DRAWDOWN",
            1.0
        )

        if drawdown >= limite:
            return ValidationResult(
                False,
                f"Límite de drawdown diario alcanzado: {drawdown:.2%}"
            )

        advertencias = []

        if drawdown >= limite * 0.8:
            advertencias.append(
                f"Drawdown próximo al límite: {drawdown:.2%}"
            )

        return ValidationResult(
            True,
            "Drawdown dentro del límite",
            warnings=advertencias
        )

    def update_position(
        self,
        symbol: str,
        qty_change: float
    ) -> None:
        """Actualiza el registro auxiliar de posiciones."""

        activo = self._obtener_activo_base(
            symbol.upper()
        )

        cantidad = self._convertir_numero(
            qty_change
        )

        if cantidad is None:
            raise ValueError(
                "El cambio de posición debe ser numérico"
            )

        self.current_positions[activo] = (
            self.current_positions.get(activo, 0.0)
            + cantidad
        )

        logger.info(
            "Posición actualizada: %s = %s",
            activo,
            self.current_positions[activo]
        )

    def update_daily_pnl(
        self,
        pnl_change: float
    ) -> None:
        """Actualiza el PnL diario comunicado por el sistema."""

        pnl = self._convertir_numero(
            pnl_change
        )

        if pnl is None:
            raise ValueError(
                "El cambio de PnL debe ser numérico"
            )

        self.daily_pnl += pnl

        logger.info(
            "PnL diario actualizado: %s",
            self.daily_pnl
        )

    def reset_daily_metrics(
        self,
        account_balance: Optional[float] = None
    ) -> None:
        """Reinicia métricas y acepta el balance real del nuevo día."""

        balance = self._convertir_numero(
            account_balance
        )

        if balance is not None and balance > 0:
            self.account_balance = balance
            self.daily_start_balance = balance

        self.daily_pnl = 0.0
        self.daily_start_timestamp = time.time()

        logger.info(
            "Métricas diarias reiniciadas"
        )
