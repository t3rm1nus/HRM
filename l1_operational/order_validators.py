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

    Admite órdenes procedentes de módulos antiguos siempre
    que puedan normalizarse sin inventar datos financieros.
    """

    CAMPOS_OBLIGATORIOS = (
        "action",
        "symbol",
        "quantity",
        "price",
        "type",
        "side",
    )

    TIPOS_ORDEN = {
        "MARKET",
        "LIMIT",
    }

    ACCIONES = {
        "BUY",
        "SELL",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = (
            config if config is not None else ConfigObject()
        )
        self.logger = logger

    @staticmethod
    def _convertir_numero(
        valor: Any,
        nombre: str,
    ) -> Optional[float]:
        """
        Convierte un valor a float y rechaza valores no finitos.
        """
        if isinstance(valor, bool):
            return None

        try:
            numero = float(valor)

        except (TypeError, ValueError, OverflowError):
            return None

        if not math.isfinite(numero):
            return None

        return numero

    @staticmethod
    def _normalizar_texto(
        valor: Any,
    ) -> Optional[str]:
        """
        Elimina espacios y rechaza textos vacíos.
        """
        if not isinstance(valor, str):
            return None

        texto = valor.strip()

        if not texto:
            return None

        return texto

    def _obtener_configuracion(
        self,
        clave: str,
        predeterminado: Any = None,
    ) -> Any:
        """
        Obtiene una opción tanto de configuraciones planas
        como de configuraciones anidadas.
        """
        if isinstance(self.config, dict):
            if clave in self.config:
                return self.config[clave]

            limites = self.config.get(
                "RISK_LIMITS",
                self.config.get("risk_limits", {}),
            )

            if isinstance(limites, dict):
                if clave in limites:
                    return limites[clave]

            return predeterminado

        valor = getattr(
            self.config,
            clave,
            None,
        )

        if valor is not None:
            return valor

        limites = getattr(
            self.config,
            "RISK_LIMITS",
            getattr(
                self.config,
                "risk_limits",
                {},
            ),
        )

        if isinstance(limites, dict):
            return limites.get(
                clave,
                predeterminado,
            )

        return predeterminado

    def _obtener_limite(
        self,
        clave: str,
        predeterminado: float,
    ) -> float:
        """
        Obtiene un límite numérico de configuración.

        Si el valor configurado es inválido, utiliza el
        valor predeterminado seguro.
        """
        valor = self._obtener_configuracion(
            clave,
            predeterminado,
        )

        numero = self._convertir_numero(
            valor,
            clave,
        )

        if numero is None or numero < 0:
            self.logger.warning(
                "Límite de configuración inválido: %s. "
                "Se utiliza el valor predeterminado.",
                clave,
            )

            return predeterminado

        return numero

    def validate_order_fields(
        self,
        order: Dict[str, Any],
    ) -> Optional[str]:
        """
        Comprueba la presencia y validez estructural
        de los campos obligatorios.
        """
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

        action = self._normalizar_texto(
            order["action"]
        )

        side = self._normalizar_texto(
            order["side"]
        )

        symbol = self._normalizar_texto(
            order["symbol"]
        )

        order_type = self._normalizar_texto(
            order["type"]
        )

        if action is None or action.upper() not in self.ACCIONES:
            return "Campo action inválido"

        if side is None or side.upper() not in self.ACCIONES:
            return "Campo side inválido"

        if action.upper() != side.upper():
            return (
                "Los campos action y side no coinciden"
            )

        if symbol is None:
            return "Campo symbol inválido"

        if order_type is None:
            return "Campo type inválido"

        if order_type.upper() not in self.TIPOS_ORDEN:
            return "Tipo de orden no permitido"

        quantity = self._convertir_numero(
            order["quantity"],
            "quantity",
        )

        if quantity is None or quantity <= 0:
            return (
                "Campo quantity debe ser positivo y finito"
            )

        price = self._convertir_numero(
            order["price"],
            "price",
        )

        if price is None or price <= 0:
            return (
                "Campo price debe ser positivo y finito"
            )

        return None

    def validate_order_values(
        self,
        order: Dict[str, Any],
    ) -> Optional[str]:
        """
        Comprueba el valor económico y los límites
        configurados para una orden.
        """
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
            return (
                "Cantidad o precio no válidos"
            )

        if cantidad <= 0 or precio <= 0:
            return (
                "Cantidad y precio deben ser positivos"
            )

        valor_orden = cantidad * precio

        if not math.isfinite(valor_orden):
            return (
                "El valor de la orden no es finito"
            )

        minimo = self._obtener_limite(
            "MIN_ORDER_SIZE_USDT",
            2.0,
        )

        if valor_orden < minimo:
            return (
                f"Valor de orden ({valor_orden:.2f} USDT) "
                f"inferior al mínimo ({minimo:.2f} USDT)"
            )

        simbolo = self._normalizar_texto(
            order.get("symbol")
        )

        if simbolo is None:
            return "Símbolo no válido"

        simbolo = simbolo.upper()

        if not simbolo.endswith("USDT"):
            return (
                "Solo se admiten pares cotizados en USDT"
            )

        activo = simbolo[:-4]

        if not activo:
            return "Activo base no válido"

        if not activo.isalnum():
            return "El símbolo contiene caracteres no permitidos"

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
        """
        Ejecuta las validaciones estructurales y económicas.
        """
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

        error_campos = self.validate_order_fields(
            order
        )

        if error_campos:
            reporte["status"] = "invalid"
            reporte["valid"] = False
            reporte["errors"].append(
                error_campos
            )

            return reporte

        error_valores = self.validate_order_values(
            order
        )

        if error_valores:
            reporte["status"] = "invalid"
            reporte["valid"] = False
            reporte["errors"].append(
                error_valores
            )

        return reporte

    def normalize_order(
        self,
        order: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normaliza una orden sin modificar el objeto original.

        No inventa cantidad, precio ni símbolo.
        """
        if not isinstance(order, dict):
            raise TypeError(
                "La orden debe ser un diccionario"
            )

        normalizada = dict(order)

        action = order.get(
            "action",
            order.get("side", ""),
        )

        side = order.get(
            "side",
            action,
        )

        if isinstance(action, str):
            normalizada["action"] = (
                action.strip().upper()
            )

        if isinstance(side, str):
            normalizada["side"] = (
                side.strip().upper()
            )

        symbol = order.get("symbol")

        if isinstance(symbol, str):
            normalizada["symbol"] = (
                symbol.strip().upper()
            )

        order_type = order.get(
            "type",
            order.get(
                "order_type",
                "MARKET",
            ),
        )

        if isinstance(order_type, str):
            normalizada["type"] = (
                order_type.strip().upper()
            )

        for campo in (
            "quantity",
            "price",
        ):
            if campo not in order:
                continue

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
        """
        Normaliza y valida una orden.

        Mantiene el contrato esperado por ExecutionGateway:
        {
            "order": orden_normalizada,
            "validation": reporte
        }
        """
        try:
            normalizada = self.normalize_order(
                order
            )

            validacion = (
                self.validate_order_completeness(
                    normalizada
                )
            )

            return {
                "order": normalizada,
                "validation": validacion,
            }

        except (TypeError, ValueError) as error:
            return {
                "order": (
                    order
                    if isinstance(order, dict)
                    else {}
                ),
                "validation": {
                    "status": "invalid",
                    "valid": False,
                    "errors": [
                        str(error)
                    ],
                },
            }


# Compatibilidad con ExecutionGateway y módulos antiguos.
OrderValidator = OrderValidators