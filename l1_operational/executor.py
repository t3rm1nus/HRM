"""
Executor robusto L1 con:
- Validación de liquidez/saldo
- Manejo de errores, timeout, retry
- Métricas centralizadas
- Trazabilidad de client_order_id
"""

import asyncio
import ccxt
import time
import uuid
from loguru import logger
from l1_operational.binance_client import exchange
from l1_operational import metrics, portfolio

ORDER_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_BASE = 2  # segundos


async def execute_order(
    symbol: str,
    side: str,
    amount: float,
    price: float = None,
    order_type: str = "market"
) -> dict:
    client_order_id = str(uuid.uuid4())

    # 1️⃣ Validación de saldo/liquidez
    try:
        available, quote_available = await portfolio.get_available_balance(symbol)
        if side.lower() == "buy" and quote_available < amount * (price or 1):
            metrics.increment("orders_rejected")
            msg = f"Saldo insuficiente: {quote_available} para comprar {amount} {symbol}"
            logger.warning(f"[Executor] {msg} (client_id={client_order_id})")
            return {"status": "rejected", "message": msg, "client_order_id": client_order_id}

        if side.lower() == "sell" and available < amount:
            metrics.increment("orders_rejected")
            msg = f"Saldo insuficiente: {available} para vender {amount} {symbol}"
            logger.warning(f"[Executor] {msg} (client_id={client_order_id})")
            return {"status": "rejected", "message": msg, "client_order_id": client_order_id}

    except Exception as e:
        metrics.increment("orders_failed")
        logger.error(f"[Executor] Error consultando saldo: {e} (client_id={client_order_id})")
        return {"status": "error", "message": f"Error saldo: {e}", "client_order_id": client_order_id}

    # 2️⃣ Intento de ejecución con retry
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"[Executor] Intento {attempt}/{MAX_RETRIES} → "
                f"{order_type.upper()} {side} {amount} {symbol} @ {price or 'MARKET'} "
                f"(client_id={client_order_id})"
            )

            def _place_order():
                if order_type == "market":
                    return exchange.create_market_order(
                        symbol, side, amount, params={"clientOrderId": client_order_id}
                    )
                elif order_type == "limit":
                    if price is None:
                        raise ValueError("Las órdenes limit requieren un precio")
                    return exchange.create_limit_order(
                        symbol, side, amount, price, params={"clientOrderId": client_order_id}
                    )
                else:
                    raise ValueError(f"Tipo de orden no soportado: {order_type}")

            start_time = time.perf_counter()
            order = await asyncio.wait_for(asyncio.to_thread(_place_order), timeout=ORDER_TIMEOUT)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # 3️⃣ Métricas
            metrics.increment("orders_success")
            metrics.observe_latency(latency_ms)
            await metrics.update_portfolio(symbol)

            # Manejo de órdenes parciales
            filled = float(order.get("filled", 0))
            if filled < amount:
                metrics.increment("orders_partial")
                logger.warning(f"[Executor] Orden parcialmente llenada: {filled}/{amount} (client_id={client_order_id})")

            logger.success(f"[Executor] Orden ejecutada OK (id={order.get('id')}, latency={latency_ms:.2f}ms)")
            return {
                "status": "success",
                "order": order,
                "latency_ms": latency_ms,
                "filled": filled,
                "client_order_id": client_order_id,
                "metrics": metrics.snapshot(),
            }

        except asyncio.TimeoutError:
            metrics.increment("orders_failed")
            return {"status": "error", "message": "Timeout ejecutando orden", "client_order_id": client_order_id}

        except ccxt.NetworkError as e:
            logger.warning(f"[Executor] NetworkError: {e} → reintentando")
            await asyncio.sleep(BACKOFF_BASE ** attempt)
            try:
                exchange.load_markets()
            except Exception as recon_err:
                logger.error(f"[Executor] Reconexión fallida: {recon_err}")

        except ccxt.ExchangeError as e:
            metrics.increment("orders_rejected")
            return {"status": "rejected", "message": str(e), "client_order_id": client_order_id}

        except Exception as e:
            metrics.increment("orders_failed")
            logger.exception(f"[Executor] Error inesperado: {e}")
            return {"status": "error", "message": str(e), "client_order_id": client_order_id}

    metrics.increment("orders_failed")
    return {"status": "error", "message": "Max retries exceeded", "client_order_id": client_order_id}
