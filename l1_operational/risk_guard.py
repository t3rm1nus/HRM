# l1_operational/risk_guard.py
"""
Risk Guard central para L1.
Valida órdenes antes de ejecución: stops, límites por trade, saldo, exposición y drawdown.
Este módulo NO modifica órdenes, solo rechaza las que no cumplen reglas mínimas.
"""

from l1_operational.data_feed import get_balance, get_ticker
from l1_operational.config import RISK_LIMITS, PORTFOLIO_LIMITS
from loguru import logger


def validate_order(
    symbol: str,
    side: str,
    amount: float,
    price: float = None,
    stop_loss: float = None,
) -> bool:
    """
    Valida que la orden cumpla todos los límites de riesgo.
    Retorna True si la orden es segura para ejecutar, False en caso contrario.
    """

    # 1️⃣ Validación de parámetros básicos
    if amount <= 0:
        logger.warning(f"Orden rechazada: cantidad inválida {amount}")
        return False
    if price is not None and price <= 0:
        logger.warning(f"Orden rechazada: precio inválido {price}")
        return False

    # 2️⃣ Stop-loss obligatorio
    if stop_loss is None or stop_loss <= 0:
        logger.warning("Orden rechazada: falta stop-loss obligatorio")
        return False
    if price:
        if side.lower() == "buy" and stop_loss >= price:
            logger.warning(f"Orden rechazada: stop-loss {stop_loss} >= precio de compra {price}")
            return False
        elif side.lower() == "sell" and stop_loss <= price:
            logger.warning(f"Orden rechazada: stop-loss {stop_loss} <= precio de venta {price}")
            return False

    # 3️⃣ Calcular valor total de la orden en USDT
    order_value_usdt = _calc_order_value(symbol, amount, price)
    if order_value_usdt is None:
        logger.warning(f"Orden rechazada: no se puede calcular valor de la orden para {symbol}")
        return False

    # 4️⃣ Validación de tamaño mínimo y máximo por trade
    if order_value_usdt < RISK_LIMITS["MIN_ORDER_SIZE_USDT"]:
        logger.warning(
            f"Orden rechazada: valor mínimo no alcanzado "
            f"{order_value_usdt} < {RISK_LIMITS['MIN_ORDER_SIZE_USDT']} USDT"
        )
        return False
    if order_value_usdt > RISK_LIMITS["MAX_ORDER_SIZE_USDT"]:
        logger.warning(
            f"Orden rechazada: valor máximo excedido "
            f"{order_value_usdt} > {RISK_LIMITS['MAX_ORDER_SIZE_USDT']} USDT"
        )
        return False

    # 5️⃣ Validación específica por símbolo
    base_currency = symbol.split("/")[0].upper()
    if base_currency == "BTC" and amount > RISK_LIMITS["MAX_ORDER_SIZE_BTC"]:
        logger.warning(f"Orden rechazada: tamaño BTC excede límite {amount} > {RISK_LIMITS['MAX_ORDER_SIZE_BTC']}")
        return False
    if base_currency == "ETH" and amount > RISK_LIMITS["MAX_ORDER_SIZE_ETH"]:
        logger.warning(f"Orden rechazada: tamaño ETH excede límite {amount} > {RISK_LIMITS['MAX_ORDER_SIZE_ETH']}")
        return False

    # 6️⃣ Chequeo de saldo y liquidez
    if side.lower() == "buy":
        if not _check_buy_balance(order_value_usdt):
            return False
    elif side.lower() == "sell":
        if not _check_sell_balance(base_currency, amount):
            return False
    else:
        logger.warning(f"Orden rechazada: side inválido {side}")
        return False

    # 7️⃣ Validación de exposición del portafolio
    if not _check_portfolio_exposure(base_currency, order_value_usdt):
        return False

    # 8️⃣ Validación de drawdown diario
    if _get_daily_drawdown() > PORTFOLIO_LIMITS["MAX_DAILY_DRAWDOWN"]:
        logger.warning(
            f"Orden rechazada: drawdown diario excede límite "
            f"{_get_daily_drawdown()} > {PORTFOLIO_LIMITS['MAX_DAILY_DRAWDOWN']}"
        )
        return False

    logger.info(f"Orden validada exitosamente: {symbol} {side} {amount} @ {price}, SL {stop_loss}")
    return True


# 🔹 Funciones auxiliares
def _calc_order_value(symbol: str, amount: float, price: float = None) -> float:
    """Calcula el valor en USDT de una orden."""
    if price:
        return amount * price
    ticker = get_ticker(symbol)
    if not ticker or "last" not in ticker:
        return None
    return amount * ticker["last"]


def _check_buy_balance(order_value_usdt: float) -> bool:
    """Chequea que haya saldo suficiente para comprar."""
    balance_usdt = get_balance("USDT") or 0
    if balance_usdt < PORTFOLIO_LIMITS["MIN_BALANCE_USDT"]:
        logger.warning(f"Orden rechazada: balance insuficiente en USDT ({balance_usdt})")
        return False
    if order_value_usdt > balance_usdt:
        logger.warning(f"Orden rechazada: orden excede saldo USDT disponible {order_value_usdt} > {balance_usdt}")
        return False
    return True


def _check_sell_balance(base_currency: str, amount: float) -> bool:
    """Chequea que haya saldo suficiente para vender."""
    balance_base = get_balance(base_currency) or 0
    if amount > balance_base:
        logger.warning(f"Orden rechazada: cantidad excede saldo {base_currency} disponible {amount} > {balance_base}")
        return False
    return True


def _check_portfolio_exposure(base_currency: str, order_value_usdt: float) -> bool:
    """Valida exposición máxima por moneda."""
    portfolio_exposure = _get_portfolio_exposure()
    max_exposure = PORTFOLIO_LIMITS.get(f"MAX_PORTFOLIO_EXPOSURE_{base_currency.upper()}")
    if max_exposure and portfolio_exposure + order_value_usdt > max_exposure:
        logger.warning(f"Orden rechazada: exposición del portafolio excede límite {portfolio_exposure + order_value_usdt} > {max_exposure}")
        return False
    return True


def _get_portfolio_exposure() -> float:
    """Obtiene exposición actual del portafolio. Hard-coded por ahora."""
    return 0.1  # 10% por defecto


def _get_daily_drawdown() -> float:
    """Obtiene drawdown diario actual. Hard-coded por ahora."""
    return 0.02  # 2% por defecto
