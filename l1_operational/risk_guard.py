from l1_operational.data_feed import get_balance
from l1_operational.config import RISK_LIMITS, PORTFOLIO_LIMITS
from loguru import logger

def validate_order(symbol: str, side: str, amount: float, price: float = None) -> bool:
    """
    Valida que la orden cumpla con todos los límites de riesgo.
    Retorna True si la orden es segura para ejecutar, False en caso contrario.
    L1 solo valida, no modifica órdenes.
    """
    # Validación de parámetros básicos
    if amount <= 0:
        logger.warning(f"Orden rechazada: cantidad inválida {amount}")
        return False
    
    if price and price <= 0:
        logger.warning(f"Orden rechazada: precio inválido {price}")
        return False

    # Validación de tamaño mínimo/máximo por operación
    if price:
        order_value_usdt = amount * price
    else:
        # Para órdenes market, usar precio estimado del ticker
        from l1_operational.data_feed import get_ticker
        ticker = get_ticker(symbol)
        if not ticker:
            logger.warning(f"Orden rechazada: no se puede obtener precio para {symbol}")
            return False
        order_value_usdt = amount * ticker["last"]

    if order_value_usdt < RISK_LIMITS["MIN_ORDER_SIZE_USDT"]:
        logger.warning(f"Orden rechazada: valor mínimo no alcanzado {order_value_usdt} < {RISK_LIMITS['MIN_ORDER_SIZE_USDT']} USDT")
        return False

    if order_value_usdt > RISK_LIMITS["MAX_ORDER_SIZE_USDT"]:
        logger.warning(f"Orden rechazada: valor máximo excedido {order_value_usdt} > {RISK_LIMITS['MAX_ORDER_SIZE_USDT']} USDT")
        return False

    # Validación específica por símbolo
    base_currency = symbol.split("/")[0]
    if base_currency == "BTC" and amount > RISK_LIMITS["MAX_ORDER_SIZE_BTC"]:
        logger.warning(f"Orden rechazada: tamaño BTC excede límite {amount} > {RISK_LIMITS['MAX_ORDER_SIZE_BTC']}")
        return False
    
    if base_currency == "ETH" and amount > RISK_LIMITS["MAX_ORDER_SIZE_ETH"]:
        logger.warning(f"Orden rechazada: tamaño ETH excede límite {amount} > {RISK_LIMITS['MAX_ORDER_SIZE_ETH']}")
        return False

    # Validación de saldo disponible
    if side == "buy":
        balance_usdt = get_balance("USDT")
        if balance_usdt < PORTFOLIO_LIMITS["MIN_BALANCE_USDT"]:
            logger.warning(f"Orden rechazada: balance insuficiente en USDT ({balance_usdt})")
            return False
        
        if order_value_usdt > balance_usdt:
            logger.warning(f"Orden rechazada: orden excede saldo USDT disponible {order_value_usdt} > {balance_usdt}")
            return False

    elif side == "sell":
        balance_base = get_balance(base_currency)
        if amount > balance_base:
            logger.warning(f"Orden rechazada: cantidad excede saldo {base_currency} disponible {amount} > {balance_base}")
            return False

    # Validación de exposición del portafolio (simplificada)
    # En un sistema real, esto vendría de L2/L3
    portfolio_exposure = _get_portfolio_exposure()
    max_exposure = PORTFOLIO_LIMITS[f"MAX_PORTFOLIO_EXPOSURE_{base_currency}"]
    if max_exposure and portfolio_exposure > max_exposure:
        logger.warning(f"Orden rechazada: exposición del portafolio excede límite {portfolio_exposure} > {max_exposure}")
        return False

    # Validación de drawdown diario
    daily_dd = _get_daily_drawdown()
    if daily_dd > PORTFOLIO_LIMITS["MAX_DAILY_DRAWDOWN"]:
        logger.warning(f"Orden rechazada: drawdown diario excede límite {daily_dd} > {PORTFOLIO_LIMITS['MAX_DAILY_DRAWDOWN']}")
        return False

    logger.info(f"Orden validada exitosamente: {symbol} {side} {amount}")
    return True

def _get_portfolio_exposure() -> float:
    """
    Obtiene la exposición actual del portafolio.
    En un sistema real, esto vendría de L2/L3.
    """
    # Placeholder - en implementación real esto vendría del estado del sistema
    return 0.1  # 10% por defecto

def _get_daily_drawdown() -> float:
    """
    Obtiene el drawdown diario actual.
    En un sistema real, esto vendría de L2/L3.
    """
    # Placeholder - en implementación real esto vendría del estado del sistema
    return 0.02  # 2% por defecto
