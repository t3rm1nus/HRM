# l3_strategy/exposure_manager.py
import logging
import pandas as pd
from core.logging import logger

def gestionar_exposicion(universo, portfolio_state, market_data, regime):
    """
    Gestiona la exposición del portfolio basado en capital disponible real,
    régimen de mercado y datos de mercado.
    Mantiene liquidez mínima y permite rebalanceo dinámico.
    """
    try:
        # Obtener estado actual del portfolio
        total_value = portfolio_state.get("total_value", 0.0)
        usdt_balance = portfolio_state.get("usdt_balance", 0.0)
        btc_balance = portfolio_state.get("btc_balance", 0.0)
        eth_balance = portfolio_state.get("eth_balance", 0.0)

        # Obtener precios actuales - CRITICAL FIX: Extract scalar values properly from DataFrames/Series
        btc_data = market_data.get("BTCUSDT", {})
        if isinstance(btc_data, pd.DataFrame):
            btc_price = float(btc_data["close"].iloc[-1]) if not btc_data.empty and "close" in btc_data.columns else 50000.0
        elif isinstance(btc_data, dict):
            btc_close = btc_data.get("close", 50000.0)
            if isinstance(btc_close, list):
                btc_price = float(btc_close[-1]) if btc_close else 50000.0
            else:
                btc_price = float(btc_close)
        else:
            btc_price = 50000.0

        eth_data = market_data.get("ETHUSDT", {})
        if isinstance(eth_data, pd.DataFrame):
            eth_price = float(eth_data["close"].iloc[-1]) if not eth_data.empty and "close" in eth_data.columns else 4327.46
        elif isinstance(eth_data, dict):
            eth_close = eth_data.get("close", 4327.46)
            if isinstance(eth_close, list):
                eth_price = float(eth_close[-1]) if eth_close else 4327.46
            else:
                eth_price = float(eth_close)
        else:
            eth_price = 4327.46

        # Calcular valores actuales de posiciones
        btc_value = btc_balance * btc_price
        eth_value = eth_balance * eth_price

        # Definir porcentajes de asignación según régimen
        if regime.lower() == "bear":
            # En bear market: reducir exposición, aumentar liquidez
            target_btc_pct = 0.20  # 20% BTC
            target_eth_pct = 0.20  # 20% ETH
            target_usdt_pct = 0.60  # 60% liquidez
            logger.info("🐻 Régimen bajista: exposición reducida, liquidez aumentada al 60%")
        elif regime.lower() == "bull":
            # En bull market: aumentar exposición
            target_btc_pct = 0.60  # 60% BTC
            target_eth_pct = 0.30  # 30% ETH
            target_usdt_pct = 0.10  # 10% liquidez
            logger.info("🚀 Régimen alcista: exposición aumentada")
        else:
            # Neutral: asignación balanceada
            target_btc_pct = 0.50  # 50% BTC
            target_eth_pct = 0.30  # 30% ETH
            target_usdt_pct = 0.20  # 20% liquidez

        # Calcular valores objetivo para cada activo
        btc_target_value = total_value * target_btc_pct
        eth_target_value = total_value * target_eth_pct
        usdt_target_value = total_value * target_usdt_pct

        # Calcular cantidades objetivo
        btc_target_qty = btc_target_value / btc_price if btc_price > 0 else 0.0
        eth_target_qty = eth_target_value / eth_price if eth_price > 0 else 0.0

        # Calcular ajustes necesarios (diferencia entre objetivo y actual)
        btc_adjustment = btc_target_qty - btc_balance
        eth_adjustment = eth_target_qty - eth_balance
        usdt_adjustment = usdt_target_value - usdt_balance

        # Determinar acciones basadas en ajustes
        exposure_decisions = {
            "BTCUSDT": {
                "current_position": btc_balance,
                "current_value": btc_value,
                "target_position": btc_target_qty,
                "target_value": btc_target_value,
                "adjustment": btc_adjustment,
                "action": "buy" if btc_adjustment > 0.0001 else "sell" if btc_adjustment < -0.0001 else "hold"
            },
            "ETHUSDT": {
                "current_position": eth_balance,
                "current_value": eth_value,
                "target_position": eth_target_qty,
                "target_value": eth_target_value,
                "adjustment": eth_adjustment,
                "action": "buy" if eth_adjustment > 0.0001 else "sell" if eth_adjustment < -0.0001 else "hold"
            },
            "USDT": {
                "current_balance": usdt_balance,
                "target_balance": usdt_target_value,
                "adjustment": usdt_adjustment,
                "min_liquidity_pct": target_usdt_pct,
                "is_liquidity": True
            }
        }

        logger.info(f"📊 Rebalanceo calculado - BTC: {btc_balance:.6f} → {btc_target_qty:.6f} ({btc_adjustment:+.6f})")
        logger.info(f"📊 Rebalanceo calculado - ETH: {eth_balance:.3f} → {eth_target_qty:.3f} ({eth_adjustment:+.3f})")
        logger.info(f"📊 Rebalanceo calculado - USDT: {usdt_balance:.2f} → {usdt_target_value:.2f} (liquidez {target_usdt_pct*100:.0f}%)")

        return exposure_decisions

    except Exception as e:
        logger.error(f"❌ Error en gestión de exposición: {e}")
        return {}
