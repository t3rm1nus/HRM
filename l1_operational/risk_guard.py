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
    logger.info(f"[RiskGuard] Iniciando validación para {side} {amount} {symbol} @ {price}, SL={stop_loss}")

    # 1️⃣ Validación de parámetros básicos
    if amount <= 0:
        logger.warning(f"[RiskGuard] Rechazado: cantidad inválida {amount}")
        return False
    if price is not None and price <= 0:
        logger.warning(f"[RiskGuard] Rechazado: precio inválido {price}")
        return False

    # 2️⃣ Stop-loss obligatorio
    if stop_loss is None or stop_loss <= 0:
        logger.warning("[RiskGuard] Rechazado: falta stop-loss obligatorio")
        return False
    if price:
        if side.lower() == "buy" and stop_loss >= price:
            logger.warning(f"[RiskGuard] Rechazado: stop-loss {stop_loss} >= precio de compra {price}")
            return False
        elif side.lower() == "sell" and stop_loss <= price:
            logger.warning(f"[RiskGuard] Rechazado: stop-loss {stop_loss} <= precio de venta {price}")
            return False
    logger.info("[RiskGuard] Stop-loss validado OK")

    # 3️⃣ Calcular valor total de la orden en USDT
    order_value_usdt = _calc_order_value(symbol, amount, price)
    if order_value_usdt is None:
        logger.warning(f"[RiskGuard] Rechazado: no se puede calcular valor de la orden para {symbol}")
        return False

    # 4️⃣ Validación de tamaño mínimo y máximo por trade
    if order_value_usdt < RISK_LIMITS["MIN_ORDER_SIZE_USDT"]:
        logger.warning(
            f"[RiskGuard] Rechazado: valor mínimo no alcanzado "
            f"{order_value_usdt} < {RISK_LIMITS['MIN_ORDER_SIZE_USDT']} USDT"
        )
        return False
    if order_value_usdt > RISK_LIMITS["MAX_ORDER_SIZE_USDT"]:
        logger.warning(
            f"[RiskGuard] Rechazado: valor máximo excedido "
            f"{order_value_usdt} > {RISK_LIMITS['MAX_ORDER_SIZE_USDT']} USDT"
        )
        return False
    logger.info(f"[RiskGuard] Tamaño de orden validado OK ({order_value_usdt} USDT)")

    # 5️⃣ Validación específica por símbolo
    base_currency = symbol.split("/")[0].upper()
    if base_currency == "BTC" and amount > RISK_LIMITS["MAX_ORDER_SIZE_BTC"]:
        logger.warning(f"[RiskGuard] Rechazado: tamaño BTC excede límite {amount} > {RISK_LIMITS['MAX_ORDER_SIZE_BTC']}")
        return False
    if base_currency == "ETH" and amount > RISK_LIMITS["MAX_ORDER_SIZE_ETH"]:
        logger.warning(f"[RiskGuard] Rechazado: tamaño ETH excede límite {amount} > {RISK_LIMITS['MAX_ORDER_SIZE_ETH']}")
        return False
    logger.info(f"[RiskGuard] Límites por símbolo validados OK")

    # 6️⃣ Chequeo de saldo y liquidez
    if side.lower() == "buy":
        if not _check_buy_balance(order_value_usdt):
            return False
    elif side.lower() == "sell":
        if not _check_sell_balance(base_currency, amount):
            return False
    else:
        logger.warning(f"[RiskGuard] Rechazado: side inválido {side}")
        return False
    logger.info("[RiskGuard] Saldo y liquidez validados OK")

    # 7️⃣ Validación de exposición del portafolio
    if not _check_portfolio_exposure(base_currency, order_value_usdt):
        return False
    logger.info("[RiskGuard] Exposición de portafolio validada OK")

    # 8️⃣ Validación de drawdown diario
    if _get_daily_drawdown() > PORTFOLIO_LIMITS["MAX_DAILY_DRAWDOWN"]:
        logger.warning(
            f"[RiskGuard] Rechazado: drawdown diario excede límite "
            f"{_get_daily_drawdown()} > {PORTFOLIO_LIMITS['MAX_DAILY_DRAWDOWN']}"
        )
        return False
    logger.info("[RiskGuard] Drawdown validado OK")

    logger.info(f"[RiskGuard] Orden validada completamente OK")
    return True


# 🔹 Funciones auxiliares
def _calc_order_value(symbol: str, amount: float, price: float = None) -> float:
    """Calcula el valor en USDT de una orden."""
    if price:
        value = amount * price
        logger.debug(f"[RiskGuard] Valor calculado con price: {value}")
        return value
    ticker = get_ticker(symbol)
    if not ticker or "last" not in ticker:
        logger.warning("[RiskGuard] No ticker disponible para calcular valor")
        return None
    value = amount * ticker["last"]
    logger.debug(f"[RiskGuard] Valor calculado con ticker last: {value}")
    return value


def _check_buy_balance(order_value_usdt: float) -> bool:
    """Chequea que haya saldo suficiente para comprar."""
    balance_usdt = get_balance("USDT") or 0
    if balance_usdt < PORTFOLIO_LIMITS["MIN_BALANCE_USDT"]:
        logger.warning(f"[RiskGuard] Rechazado: balance insuficiente en USDT ({balance_usdt})")
        return False
    if order_value_usdt > balance_usdt:
        logger.warning(f"[RiskGuard] Rechazado: orden excede saldo USDT disponible {order_value_usdt} > {balance_usdt}")
        return False
    return True


def _check_sell_balance(base_currency: str, amount: float) -> bool:
    """Chequea que haya saldo suficiente para vender."""
    balance_base = get_balance(base_currency) or 0
    if amount > balance_base:
        logger.warning(f"[RiskGuard] Rechazado: cantidad excede saldo {base_currency} disponible {amount} > {balance_base}")
        return False
    return True


def _check_portfolio_exposure(base_currency: str, order_value_usdt: float) -> bool:
    """Valida exposición máxima por moneda."""
    portfolio_exposure = _get_portfolio_exposure()
    max_exposure = PORTFOLIO_LIMITS.get(f"MAX_PORTFOLIO_EXPOSURE_{base_currency.upper()}")
    if max_exposure and portfolio_exposure + order_value_usdt > max_exposure:
        logger.warning(f"[RiskGuard] Rechazado: exposición del portafolio excede límite {portfolio_exposure + order_value_usdt} > {max_exposure}")
        return False
    return True


def _get_portfolio_exposure() -> float:
    """Obtiene exposición actual del portafolio. Hard-coded por ahora."""
    exposure = 0.1  # 10% por defecto
    logger.debug(f"[RiskGuard] Exposición actual: {exposure}")
    return exposure


def _get_daily_drawdown() -> float:
    """Obtiene drawdown diario actual. Hard-coded por ahora."""
    drawdown = 0.02  # 2% por defecto
    logger.debug(f"[RiskGuard] Drawdown diario: {drawdown}")
    return drawdown