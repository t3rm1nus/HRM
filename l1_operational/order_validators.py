import math
from typing import Any, Dict, Optional

from core.logging import logger
from l1_operational.config import ConfigObject


class OrderValidators:
    """
    Valida y normaliza órdenes antes de su ejecución.

    Comprueba:
    - Campos obligatorios.
    - Tipos y valores numéricos.
    - Símbolos y acciones.
    - Tipo de orden.
    - Límites de tamaño y valor.
    """

    CAMPOS_OBLIGATORIOS = (
        "action",
        "symbol",
        "quantity",
        "price",
        "type",
        "side",
    )

    TIPOS_ORDEN = {"MARKET", "LIMIT"}
    ACCIONES = {"BUY", "SELL"}

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or ConfigObject()
        self.logger = logger

    @staticmethod
    def _convertir_numero(
        valor: Any,
        nombre: str,
    ) -> Optional[float]:
        if isinstance(valor, bool):
            return None

        try:
            numero = float(valor)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(numero):
            return None

        return numero

    @staticmethod
    def _normalizar_texto(valor: Any) -> Optional[str]:
        if not isinstance(valor, str):
            return None

        texto = valor.strip()

        if not texto:
            return None

        return texto

    def _obtener_limite(
        self,
        clave: str,
        predeterminado: float,
    ) -> float:
        if isinstance(self.config, dict):
            valor = self.config.get(clave, predeterminado)
        else:
            valor = getattr(self.config, clave, predeterminado)

        numero = self._convertir_numero(valor, clave)

        if numero is None or numero < 0:
            return predeterminado

        return numero

    def validate_order_fields(
        self,
        order: Dict[str, Any],
    ) -> Optional[str]:
        if not isinstance(order, dict):
            return "La orden debe ser un diccionario"

        faltantes = [
            campo
            for campo in self.CAMPOS_OBLIGATORIOS
            if campo not in order
        ]

        if faltantes:
            return (
                "Campos obligatorios faltantes: "
                + ", ".join(faltantes)
            )

        action = self._normalizar_texto(order["action"])
        side = self._normalizar_texto(order["side"])
        symbol = self._normalizar_texto(order["symbol"])
        order_type = self._normalizar_texto(order["type"])

        if action is None or action.upper() not in self.ACCIONES:
            return "Campo action inválido"

        if side is None or side.upper() not in self.ACCIONES:
            return "Campo side inválido"

        if action.upper() != side.upper():
            return "Los campos action y side no coinciden"

        if symbol is None:
            return "Campo symbol inválido"

        if order_type is None or order_type.upper() not in self.TIPOS_ORDEN:
            return "Campo type inválido"

        quantity = self._convertir_numero(
            order["quantity"],
            "quantity",
        )

        if quantity is None or quantity <= 0:
            return "Campo quantity debe ser positivo y finito"

        price = self._convertir_numero(
            order["price"],
            "price",
        )

        if price is None or price <= 0:
            return "Campo price debe ser positivo y finito"

        return None

    def validate_order_values(
        self,
        order: Dict[str, Any],
    ) -> Optional[str]:
        if not isinstance(order, dict):
            return "La orden no es válida"

        cantidad = self._convertir_numero(
            order.get("quantity"),
            "quantity",
        )
        precio = self._convertir_numero(
            order.get("price"),
            "price",
        )

        if cantidad is None or precio is None:
            return "Cantidad o precio no válidos"

        if cantidad <= 0 or precio <= 0:
            return "Cantidad y precio deben ser positivos"

        valor_orden = cantidad * precio

        if not math.isfinite(valor_orden):
            return "El valor de la orden no es finito"

        minimo = self._obtener_limite(
            "MIN_ORDER_SIZE_USDT",
            2.0,
        )

        if valor_orden < minimo:
            return (
                f"Valor de orden ({valor_orden:.2f} USDT) "
                f"inferior al mínimo ({minimo:.2f} USDT)"
            )

        simbolo = order.get("symbol", "").strip().upper()

        if not simbolo.endswith("USDT"):
            return "Solo se admiten pares cotizados en USDT"

        activo = simbolo[:-4]

        if not activo:
            return "Activo base no válido"

        limite = self._obtener_limite(
            f"MAX_ORDER_SIZE_{activo}",
            0.0,
        )

        if limite > 0 and cantidad > limite:
            return (
                f"Cantidad ({cantidad}) excede el máximo "
                f"permitido para {activo} ({limite})"
            )

        maximo_valor = self._obtener_limite(
            "MAX_ORDER_SIZE_USDT",
            1500.0,
        )

        if valor_orden > maximo_valor:
            return (
                f"Valor de orden ({valor_orden:.2f} USDT) "
                f"supera el máximo ({maximo_valor:.2f} USDT)"
            )

        return None

    def validate_order_completeness(
        self,
        order: Dict[str, Any],
    ) -> Dict[str, Any]:
        reporte = {
            "status": "valid",
            "valid": True,
            "symbol": (
                order.get("symbol", "unknown")
                if isinstance(order, dict)
                else "unknown"
            ),
            "action": (
                order.get("action", "unknown")
                if isinstance(order, dict)
                else "unknown"
            ),
            "errors": [],
        }

        error_campos = self.validate_order_fields(order)

        if error_campos:
            reporte["status"] = "invalid"
            reporte["valid"] = False
            reporte["errors"].append(error_campos)
            return reporte

        error_valores = self.validate_order_values(order)

        if error_valores:
            reporte["status"] = "invalid"
            reporte["valid"] = False
            reporte["errors"].append(error_valores)

        return reporte

    def normalize_order(
        self,
        order: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(order, dict):
            raise TypeError("La orden debe ser un diccionario")

        normalizada = dict(order)

        action = order.get("action", order.get("side", ""))
        side = order.get("side", action)

        if isinstance(action, str):
            normalizada["action"] = action.strip().upper()

        if isinstance(side, str):
            normalizada["side"] = side.strip().upper()

        symbol = order.get("symbol")

        if isinstance(symbol, str):
            normalizada["symbol"] = symbol.strip().upper()

        order_type = order.get(
            "type",
            order.get("order_type", "MARKET"),
        )

        if isinstance(order_type, str):
            normalizada["type"] = order_type.strip().upper()

        for campo in ("quantity", "price"):
            if campo in order:
                numero = self._convertir_numero(
                    order[campo],
                    campo,
                )

                if numero is not None:
                    normalizada[campo] = numero

        normalizada.setdefault(
            "reason",
            "strategy_signal",
        )

        return normalizada

    def validate_and_normalize_order(
        self,
        order: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            normalizada = self.normalize_order(order)
            validacion = self.validate_order_completeness(
                normalizada
            )

            return {
                "order": normalizada,
                "validation": validacion,
            }

        except (TypeError, ValueError) as error:
            return {
                "order": order if isinstance(order, dict) else {},
                "validation": {
                    "status": "invalid",
                    "valid": False,
                    "errors": [str(error)],
                },
            }


# Compatibilidad con ExecutionGateway y módulos antiguos.
OrderValidator = OrderValidators