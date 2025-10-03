"""
Sistema de Rotación de Posiciones - GESTIÓN AUTOMÁTICA DE CAPITAL

Funcionalidades:
- Monitoreo continuo de límites de posición (40% máximo por activo)
- Rotación automática cuando se exceden límites
- Liberación de capital cuando USDT < $500
- Rebalanceo automático cada hora
- Rotación por rendimiento (activos con pérdidas > 3%)
"""

import asyncio
import time
import threading
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from core.logging import logger
from l2_tactic.utils import safe_float
from l1_operational.config import ConfigObject
from core.config import HRM_PATH_MODE


def _extract_current_price(symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
    """
    Extract current price from market data with comprehensive error handling.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        market_data: Market data dictionary

    Returns:
        Current price as float, or None if extraction fails
    """
    if not market_data or symbol not in market_data:
        logger.warning(f"⚠️ No market data available for {symbol}")
        return None

    data = market_data[symbol]

    # Handle different data structures
    try:
        if isinstance(data, dict):
            if 'close' in data:
                price = safe_float(data['close'])
                logger.debug(f"Extracted {symbol} price from dict: ${price:,.2f}")
                return price
            elif 'price' in data:
                price = safe_float(data['price'])
                logger.debug(f"Extracted {symbol} price from dict (price field): ${price:,.2f}")
                return price
            else:
                logger.warning(f"⚠️ Dict structure for {symbol} missing 'close' or 'price' field: {list(data.keys())}")

        elif isinstance(data, pd.DataFrame):
            if 'close' in data.columns:
                price = safe_float(data['close'].iloc[-1])
                logger.debug(f"Extracted {symbol} price from DataFrame: ${price:,.2f}")
                return price
            else:
                logger.warning(f"⚠️ DataFrame for {symbol} missing 'close' column: {list(data.columns)}")

        elif isinstance(data, pd.Series):
            if len(data) > 0:
                price = safe_float(data.iloc[-1])
                logger.debug(f"Extracted {symbol} price from Series: ${price:,.2f}")
                return price

        else:
            logger.warning(f"⚠️ Unsupported data type for {symbol}: {type(data)}")

    except Exception as e:
        logger.error(f"❌ Error extracting price for {symbol}: {e}")

    return None


def generate_initial_deployment(capital: float, btc_target: float = 0.40, eth_target: float = 0.30,
                               market_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    LEGACY FUNCTION - Use PositionRotator.calculate_initial_deployment instead.
    This function is kept for backward compatibility.
    """
    logger.warning("⚠️ Using legacy generate_initial_deployment function. Consider migrating to PositionRotator.calculate_initial_deployment()")
    return []


class PositionRotator:
    """
    Sistema automático de rotación y rebalanceo de posiciones.

    Monitorea el portfolio y ejecuta rotaciones automáticas para:
    - Mantener límites de posición (40% máximo por activo)
    - Liberar capital cuando USDT < $500
    - Rebalancear cada hora si USDT < 15% del total
    - Rotar capital de activos con pérdidas > 3%
    """

    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.logger = logger

    def calculate_initial_deployment(self, market_data: Dict[str, pd.DataFrame]):
        """
        Calculate initial capital deployment orders using market data.

        Args:
            market_data: Dictionary with DataFrames containing market data for each symbol

        Returns:
            List of buy orders to execute, or empty list if data is invalid
        """
        # Extraer precios actuales del DataFrame
        try:
            btc_price = market_data['BTCUSDT']['close'].iloc[-1]
            eth_price = market_data['ETHUSDT']['close'].iloc[-1]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"❌ Error extracting prices from market_data: {e}")
            return []

        if pd.isna(btc_price) or pd.isna(eth_price):
            logger.error("🚨 Invalid prices detected")
            return []

        logger.info(f"✅ Prices loaded: BTC=${btc_price}, ETH=${eth_price}")

        # Get capital from portfolio manager
        capital = self.portfolio_manager.get_total_value()
        btc_target = 0.40  # 40% BTC
        eth_target = 0.30  # 30% ETH
        # 30% reserve in USDT

        # Calculate deployment amounts
        btc_amount = capital * btc_target
        eth_amount = capital * eth_target

        orders = []

        # Create BTC buy order
        if btc_amount >= 10.0:  # Minimum order size check
            btc_quantity = btc_amount / btc_price
            btc_order = {
                "symbol": "BTCUSDT",
                "side": "buy",
                "type": "MARKET",
                "quantity": btc_quantity,
                "price": btc_price,
                "timestamp": datetime.utcnow().isoformat(),
                "signal_source": "initial_deployment",
                "reason": "L3_initial_rebalance",
                "allocation_pct": btc_target,
                "status": "pending",
                "order_type": "ENTRY",
                "execution_type": "MARKET"
            }
            orders.append(btc_order)
            logger.info(f"✅ INITIAL DEPLOYMENT ORDER: BTC {btc_quantity:.4f} @ market (target: ${btc_amount:.2f})")

        # Create ETH buy order
        if eth_amount >= 10.0:  # Minimum order size check
            eth_quantity = eth_amount / eth_price
            eth_order = {
                "symbol": "ETHUSDT",
                "side": "buy",
                "type": "MARKET",
                "quantity": eth_quantity,
                "price": eth_price,
                "timestamp": datetime.utcnow().isoformat(),
                "signal_source": "initial_deployment",
                "reason": "L3_initial_rebalance",
                "allocation_pct": eth_target,
                "status": "pending",
                "order_type": "ENTRY",
                "execution_type": "MARKET"
            }
            orders.append(eth_order)
            logger.info(f"✅ INITIAL DEPLOYMENT ORDER: ETH {eth_quantity:.4f} @ market (target: ${eth_amount:.2f})")

        total_deployed = btc_amount + eth_amount
        logger.info(f"🚀 INITIAL DEPLOYMENT READY: {len(orders)} orders, ${total_deployed:.2f} deployed, ${capital - total_deployed:.2f} USDT reserved")

        return orders

    async def check_and_rotate_positions(self, state, market_data):
        """
        Check for position rotation needs and execute rotations if necessary.
        This is a placeholder implementation that returns an empty list.
        """
        # TODO: Implement actual position rotation logic
        return []
