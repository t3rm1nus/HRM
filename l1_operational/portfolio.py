"""
Helper para consultar saldos y exposición de forma robusta.
Incluye retry y manejo de fallos de API.
"""

import asyncio
from l1_operational.binance_client import exchange
from loguru import logger

MAX_RETRIES = 3
BACKOFF = 1  # segundos


async def get_available_balance(symbol: str):
    """
    Retorna: (base_amount, quote_amount)
    Ej: BTC/USDT → (BTC disponible, USDT disponible)
    """
    base, quote = symbol.split("/")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            balances = await asyncio.to_thread(exchange.fetch_balance)
            base_amt = balances["free"].get(base, 0)
            quote_amt = balances["free"].get(quote, 0)
            return float(base_amt), float(quote_amt)
        except Exception as e:
            logger.warning(f"[Portfolio] Error fetch_balance: {e}, retry {attempt}/{MAX_RETRIES}")
            await asyncio.sleep(BACKOFF * attempt)
    raise RuntimeError("No se pudo obtener balance tras varios intentos")
