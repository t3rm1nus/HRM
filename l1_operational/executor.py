from loguru import logger
from l1_operational.binance_client import exchange

def execute_order(symbol: str, side: str, amount: float, price: float = None, order_type: str = "market"):
    """
    Ejecuta una orden en Binance (via ccxt).
    Retorna un dict con estado de la orden.
    L1 solo ejecuta órdenes pre-validadas, sin tomar decisiones de trading.
    """
    try:
        if order_type == "market":
            order = exchange.create_market_order(symbol, side, amount)
        elif order_type == "limit":
            if price is None:
                raise ValueError("Las órdenes limit requieren un precio")
            order = exchange.create_limit_order(symbol, side, amount, price)
        else:
            raise ValueError(f"Tipo de orden no soportado: {order_type}")

        logger.info(f"Orden ejecutada: {order}")
        return {"status": "success", "order": order}

    except Exception as e:
        logger.error(f"Error ejecutando orden: {e}")
        return {"status": "error", "message": str(e)}
