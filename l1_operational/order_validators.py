from typing import Dict, Any, Optional, List
from core.logging import logger
from l1_operational.config import ConfigObject

class OrderValidators:
    """
    Clase centralizada para validar órdenes antes de su ejecución.
    Asegura que todas las órdenes tengan los campos obligatorios y sean válidas.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.required_fields = ['action', 'symbol', 'quantity', 'price', 'type', 'side']
        self.logger = logger

    def validate_order_fields(self, order: Dict[str, Any]) -> Optional[str]:
        """
        Valida que la orden tenga todos los campos obligatorios.

        Args:
            order: Diccionario con los datos de la orden

        Returns:
            None si la orden es válida, o un mensaje de error si falta algún campo
        """
        missing_fields = [field for field in self.required_fields if field not in order]

        if missing_fields:
            return f"Campos obligatorios faltantes: {', '.join(missing_fields)}"

        # Validar que los campos tengan valores válidos
        if not order.get('action') or order['action'].upper() not in ['BUY', 'SELL']:
            return "Campo 'action' inválido o faltante"

        if not order.get('symbol') or not isinstance(order['symbol'], str):
            return "Campo 'symbol' inválido o faltante"

        if not isinstance(order.get('quantity'), (int, float)) or order['quantity'] <= 0:
            return "Campo 'quantity' inválido o faltante"

        if not isinstance(order.get('price'), (int, float)) or order['price'] <= 0:
            return "Campo 'price' inválido o faltante"

        if not order.get('type') or order['type'].upper() not in ['MARKET', 'LIMIT']:
            return "Campo 'type' inválido o faltante"

        if not order.get('side') or order['side'].upper() not in ['BUY', 'SELL']:
            return "Campo 'side' inválido o faltante"

        return None

    def validate_order_values(self, order: Dict[str, Any]) -> Optional[str]:
        """
        Valida que los valores de la orden estén dentro de los límites permitidos.

        Args:
            order: Diccionario con los datos de la orden

        Returns:
            None si los valores son válidos, o un mensaje de error si hay problemas
        """
        # Validar tamaño mínimo de orden
        order_value = order['quantity'] * order['price']
        min_order_size = self.config.get('MIN_ORDER_SIZE_USDT', 5.0)

        if order_value < min_order_size:
            return f"Valor de orden (${order_value:.2f}) menor al mínimo permitido (${min_order_size:.2f})"

        # Validar límites de riesgo por activo
        asset = order['symbol'].replace('USDT', '')
        max_order_size = self.config.get(f"MAX_ORDER_SIZE_{asset}", 0.05 if asset == 'BTC' else 0.5)

        if order['quantity'] > max_order_size:
            return f"Cantidad ({order['quantity']}) excede el límite máximo para {asset}"

        return None

    def validate_order_completeness(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida completamente una orden y devuelve un reporte detallado.

        Args:
            order: Diccionario con los datos de la orden

        Returns:
            Diccionario con el resultado de la validación
        """
        validation_report = {
            'status': 'valid',
            'symbol': order.get('symbol', 'unknown'),
            'action': order.get('action', 'unknown'),
            'errors': []
        }

        # Validar campos obligatorios
        field_error = self.validate_order_fields(order)
        if field_error:
            validation_report['status'] = 'invalid'
            validation_report['errors'].append(field_error)
            return validation_report

        # Validar valores
        value_error = self.validate_order_values(order)
        if value_error:
            validation_report['status'] = 'invalid'
            validation_report['errors'].append(value_error)

        return validation_report

    def normalize_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza una orden para asegurar consistencia en el formato.

        Args:
            order: Diccionario con los datos de la orden

        Returns:
            Orden normalizada con todos los campos en el formato correcto
        """
        normalized_order = {}

        # Normalizar action
        if 'action' in order:
            normalized_order['action'] = order['action'].upper()
        else:
            normalized_order['action'] = order.get('side', 'UNKNOWN').upper()

        # Normalizar symbol
        normalized_order['symbol'] = order.get('symbol', 'UNKNOWN')

        # Normalizar quantity
        normalized_order['quantity'] = float(order.get('quantity', 0))

        # Normalizar price
        normalized_order['price'] = float(order.get('price', 0))

        # Normalizar type
        if 'type' in order:
            normalized_order['type'] = order['type'].upper()
        else:
            normalized_order['type'] = 'MARKET'

        # Normalizar side (fallback a action si no existe)
        if 'side' in order:
            normalized_order['side'] = order['side'].upper()
        else:
            normalized_order['side'] = normalized_order['action']

        # Agregar campos adicionales si existen
        for key in ['timestamp', 'signal_source', 'reason', 'status', 'order_type', 'execution_type']:
            if key in order:
                normalized_order[key] = order[key]

        return normalized_order

    def validate_and_normalize_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida y normaliza una orden en un solo paso.

        Args:
            order: Diccionario con los datos de la orden

        Returns:
            Diccionario con el resultado de la validación y la orden normalizada
        """
        normalized_order = self.normalize_order(order)
        validation_report = self.validate_order_completeness(normalized_order)

        return {
            'order': normalized_order,
            'validation': validation_report
        }