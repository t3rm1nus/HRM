# core/portfolio_manager.py - Gestión del portfolio
import os
import csv
from datetime import datetime
import logging
import pandas as pd
from core.logging import logger

async def update_portfolio_from_orders(state, orders):
    """Actualiza el portfolio basado en las órdenes ejecutadas"""
    try:
        # Obtener portfolio actual
        portfolio = state.get("portfolio", {"positions": {}, "drawdown": 0.0, "peak_value": 0.0})
        positions = portfolio.get("positions", {})
        btc_balance = float(positions.get("BTCUSDT", {}).get("size", 0.0))
        eth_balance = float(positions.get("ETHUSDT", {}).get("size", 0.0))
        usdt_balance = float(portfolio.get("USDT", 3000.0))
        total_fees = float(portfolio.get("total_fees", 0.0))  # Tracking de fees acumulados

        # Obtener precios actuales del mercado
        market_data = state.get("mercado", {})
        btc_price = None
        eth_price = None
        
        # Manejar posibles Series o DataFrames en market_data
        btc_market = market_data.get("BTCUSDT", {})
        if isinstance(btc_market, dict):
            btc_price = float(btc_market.get("close", 50000.0))
        elif isinstance(btc_market, (pd.Series, pd.DataFrame)) and 'close' in btc_market:
            btc_price = float(btc_market['close'].iloc[-1] if isinstance(btc_market, pd.DataFrame) else btc_market['close'])
        else:
            btc_price = 50000.0

        eth_market = market_data.get("ETHUSDT", {})
        if isinstance(eth_market, dict):
            eth_price = float(eth_market.get("close", 4327.46))
        elif isinstance(eth_market, (pd.Series, pd.DataFrame)) and 'close' in eth_market:
            eth_price = float(eth_market['close'].iloc[-1] if isinstance(eth_market, pd.DataFrame) else eth_market['close'])
        else:
            eth_price = 4327.46

        # Procesar órdenes ejecutadas
        for order in orders:
            if order.get("status") != "filled":
                logger.warning(f"⚠️ Orden no procesada: {order.get('symbol', 'unknown')} - Status: {order.get('status')}")
                continue
            symbol = order.get("symbol")
            side = order.get("side")
            quantity = float(order.get("quantity", 0.0))
            # Usar filled_price de la orden si está disponible
            price = float(order.get("filled_price", btc_price if symbol == "BTCUSDT" else eth_price))

            if not price:
                logger.warning(f"⚠️ Precio no disponible para {symbol}, omitiendo orden")
                continue

            # Calcular costos de trading
            order_value = quantity * price
            trading_fee_rate = 0.001  # 0.1% comisión de Binance
            trading_fee = order_value * trading_fee_rate
            
            if symbol == "BTCUSDT":
                if side.lower() == "buy":
                    btc_balance += quantity
                    usdt_balance -= order_value + trading_fee  # Precio + comisión
                elif side.lower() == "sell":
                    btc_balance -= quantity
                    usdt_balance += order_value - trading_fee  # Precio - comisión
            elif symbol == "ETHUSDT":
                if side.lower() == "buy":
                    eth_balance += quantity
                    usdt_balance -= order_value + trading_fee  # Precio + comisión
                elif side.lower() == "sell":
                    eth_balance -= quantity
                    usdt_balance += order_value - trading_fee  # Precio - comisión
            
            logger.info(f"📈 Orden procesada: {symbol} {side} {quantity} @ {price} (fee: {trading_fee:.4f} USDT)")
            
            # Acumular fees totales
            total_fees += trading_fee

        # Validar balances
        if usdt_balance < 0:
            logger.warning(f"⚠️ Balance USDT negativo: {usdt_balance}, ajustando a 0")
            usdt_balance = 0.0
        if btc_balance < 0:
            logger.warning(f"⚠️ Balance BTC negativo: {btc_balance}, ajustando a 0")
            btc_balance = 0.0
        if eth_balance < 0:
            logger.warning(f"⚠️ Balance ETH negativo: {eth_balance}, ajustando a 0")
            eth_balance = 0.0

        # Calcular valores
        btc_value = btc_balance * btc_price if btc_price else 0.0
        eth_value = eth_balance * eth_price if eth_price else 0.0
        total_value = btc_value + eth_value + usdt_balance

        # Calcular P&L desde capital inicial
        initial_capital = state.get("initial_capital", 1000.0)
        pnl_absolute = total_value - initial_capital
        pnl_percentage = (pnl_absolute / initial_capital) * 100 if initial_capital > 0 else 0.0

        # Calcular drawdown
        peak_value = portfolio.get("peak_value", initial_capital)
        peak_value = max(peak_value, total_value)
        drawdown = (peak_value - total_value) / peak_value if peak_value > 0 else 0.0

        # Actualizar state
        state["portfolio"] = {
            "positions": {
                "BTCUSDT": {"size": btc_balance, "entry_price": btc_price or 0.0},
                "ETHUSDT": {"size": eth_balance, "entry_price": eth_price or 0.0}
            },
            "USDT": usdt_balance,
            "drawdown": drawdown,
            "peak_value": peak_value,
            "total_fees": total_fees
        }
        state["btc_balance"] = btc_balance
        state["btc_value"] = btc_value
        state["eth_balance"] = eth_balance
        state["eth_value"] = eth_value
        state["usdt_balance"] = usdt_balance
        state["total_value"] = total_value

        # Log con color según comparación con capital inicial - NEGRITA Y TAMAÑO AUMENTADO
        if total_value > initial_capital:
            logger.info(f"")
            logger.info(f"********************************************************************************************")
            logger.info(f"\x1b[32m\x1b[1m\x1b[2m💰 Portfolio actualizado: Total={total_value:.2f} USDT, BTC={btc_balance:.5f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}\x1b[0m")
            logger.info(f"********************************************************************************************")            
            logger.info(f"")
        elif total_value < initial_capital:
            logger.info(f"")
            logger.info(f"********************************************************************************************")
            logger.info(f"\x1b[31m\x1b[1m\x1b[2m💰 Portfolio actualizado: Total={total_value:.2f} USDT, BTC={btc_balance:.5f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}\x1b[0m")
            logger.info(f"********************************************************************************************")
            logger.info(f"")
        else:
            logger.info(f"")
            logger.info(f"********************************************************************************************")
            logger.info(f"\x1b[34m\x1b[1m\x1b[2m💰 Portfolio actualizado: Total={total_value:.2f} USDT, BTC={btc_balance:.5f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}\x1b[0m")
            logger.info(f"********************************************************************************************")
            logger.info(f"")

        logger.info(f"📊 P&L: {pnl_absolute:+.2f} USDT ({pnl_percentage:+.2f}%), Drawdown={drawdown:.4f}")
        logger.info(f"💸 Fees acumulados: {total_fees:.4f} USDT")

    except Exception as e:
        logger.error(f"❌ Error actualizando portfolio: {e}", exc_info=True)

async def save_portfolio_to_csv(state):
    """Guarda la línea del portfolio en un CSV externo usando los valores de state."""
    try:
        total_value = state.get("total_value", 0.0)
        btc_balance = state.get("btc_balance", 0.0)
        btc_value = state.get("btc_value", 0.0)
        eth_balance = state.get("eth_balance", 0.0)
        eth_value = state.get("eth_value", 0.0)
        usdt_balance = state.get("usdt_balance", 0.0)
        cycle_id = state.get("cycle_id", 0)

        output_dir = "data/portfolios"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "portfolio_log.csv")

        file_exists = os.path.isfile(csv_path)
        headers = [
            "timestamp", "cycle_id", "total_value",
            "btc_balance", "btc_value",
            "eth_balance", "eth_value",
            "usdt_balance"
        ]

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "cycle_id": cycle_id,
                "total_value": total_value,
                "btc_balance": btc_balance,
                "btc_value": btc_value,
                "eth_balance": eth_balance,
                "eth_value": eth_value,
                "usdt_balance": usdt_balance
            })
        logger.info(f"📝 Portfolio guardado en {csv_path}")

    except Exception as e:
        logger.error(f"❌ Error guardando portfolio en CSV: {e}", exc_info=True)