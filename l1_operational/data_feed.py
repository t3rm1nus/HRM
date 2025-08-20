# l1_operational/data_feed.py
from loguru import logger
from l1_operational.binance_client import exchange

def get_ticker(symbol: str):
    """Obtiene último precio y spread."""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "last": ticker["last"],
            "bid": ticker["bid"],
            "ask": ticker["ask"],
        }
    except Exception as e:
        logger.error(f"Error obteniendo ticker: {e}")
        return None

def get_balance(asset: str = "USDT"):
    """Obtiene balance disponible de un activo."""
    try:
        balance = exchange.fetch_balance()
        return balance["free"].get(asset, 0.0)
    except Exception as e:
        logger.error(f"Error obteniendo balance: {e}")
        return 0.0
