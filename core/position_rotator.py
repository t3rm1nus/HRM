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


class AutoRebalancer:
    """
    Sistema automático de rebalanceo de portfolio.

    Monitorea el portfolio y ejecuta rebalanceos automáticos para:
    - Mantener las asignaciones objetivo de L3
    - Rebalancear cuando las desviaciones superan umbrales
    - Liberar capital en condiciones de riesgo excesivo
    """

    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.logger = logger

    async def check_and_execute_rebalance(self, market_data: Dict[str, Any], l3_active: bool = False,
                                        l3_asset_allocation: Dict[str, float] = None,
                                        l3_decision: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Check if portfolio rebalancing is needed and generate rebalance orders.

        SOLUCIÓN CRÍTICA: AutoRebalancer filtro obligatorio
        - Sistema global de estados (NORMAL, DEGRADED, BLIND, PANIC)
        - Distingue fallback vs decisión estratégica real
        - Distingue balance stale vs balance synced

        Args:
            market_data: Current market data for all symbols
            l3_active: Whether L3 strategy is active
            l3_asset_allocation: Asset allocation from L3 strategy
            l3_decision: Full L3 decision object with strategic_control metadata

        Returns:
            List of orders to execute for rebalancing, or empty list if no rebalancing needed
        """
        try:
            # ========================================================================================
            # FIX 4 - AutoRebalancer permitido en simulated mode
            # ========================================================================================
            # Check if we're in simulated mode and allow rebalancing regardless of other restrictions
            from core.config import get_config
            try:
                config = get_config("live")
                system_mode = getattr(config, 'mode', 'unknown')
            except Exception:
                system_mode = 'unknown'

            if system_mode == "simulated":
                logger.info("🛡️ FIX 4: Modo simulated detectado - AutoRebalancer habilitado en paper trading")
                # En modo simulated, permitir rebalancing incluso con baja confianza
                allow_rebalance = True
            else:
                allow_rebalance = False

            # ========================================================================================
            # CRÍTICO: SISTEMA GLOBAL DE ESTADOS - PRIORITY 1
            # ========================================================================================
            from core.state_manager import get_system_state, can_system_rebalance

            current_system_state = get_system_state()
            if not can_system_rebalance() and not allow_rebalance:
                logger.warning(f"🚫 SYSTEM STATE {current_system_state}: AutoRebalancer DISABLED - rebalancing not allowed in {current_system_state} state")
                return []

            # ========================================================================================
            # RULE 3: GLOBAL COOLDOWN POST-TRADE - AutoRebalancer skips if trade happened recently
            # ========================================================================================
            # Check if any trade happened in the last 30 seconds (prevents thrashing)
            current_time = time.time()
            last_trade_time = getattr(self.portfolio_manager, 'last_trade_timestamp', 0) if hasattr(self.portfolio_manager, 'last_trade_timestamp') else 0

            if last_trade_time > 0:
                time_since_last_trade = current_time - last_trade_time
                cooldown_seconds = 30  # 30 second cooldown after any trade

                if time_since_last_trade < cooldown_seconds:
                    logger.info(f"⏰ RULE 3 COOLDOWN: Skipping AutoRebalancer - {time_since_last_trade:.1f}s since last trade (wait {cooldown_seconds - time_since_last_trade:.1f}s)")
                    return []

                logger.debug(f"✅ RULE 3 COOLDOWN: {time_since_last_trade:.1f}s since last trade - AutoRebalancer allowed")

            # ========================================================================================
            # SOLUCIÓN CRÍTICA: AutoRebalancer filtro obligatorio
            # ========================================================================================
            # FILTER 1: Check L3 decision origin - if fallback, NO rebalancing
            if l3_decision:
                strategic_control = l3_decision.get('strategic_control', {})
                l3_mode = strategic_control.get('l3_mode')
                decision_origin = strategic_control.get('decision_origin', 'strategic')

                # CRÍTICO: If L3 is in BLIND mode or fallback, NO rebalancing
                if l3_mode == 'BLIND' or decision_origin == 'fallback':
                    logger.warning(f"🚫 L3 {l3_mode} MODE: AutoRebalancer DISABLED - L3 in blind/fallback mode (decision_origin: {decision_origin})")
                    return []

                # FILTER 2: Check if L3 explicitly blocks AutoRebalancer
                if strategic_control.get('block_autorebalancer', False):
                    logger.warning("🚫 L3 BLOCKS AUTOREBALANCER: L3 strategic_control.block_autorebalancer = True")
                    return []

                # FILTER 3: Check freeze_positions flag
                if strategic_control.get('freeze_positions', False):
                    logger.warning("🚫 POSITIONS FROZEN: L3 strategic_control.freeze_positions = True")
                    return []

            # FILTER 4: Check L3 active status (legacy compatibility)
            if l3_active:
                logger.info("🚫 L3 ACTIVE: AutoRebalancer disabled - L3 has absolute control over allocations")
                return []

            # Get current portfolio allocations
            total_value = self.portfolio_manager.get_total_value(market_data)

            if total_value <= 0:
                logger.warning("⚠️ Cannot rebalance: Invalid total portfolio value")
                return []

            # Get current balances and calculate percentages
            btc_balance = self.portfolio_manager.get_balance("BTCUSDT")
            eth_balance = self.portfolio_manager.get_balance("ETHUSDT")
            usdt_balance = self.portfolio_manager.get_balance("USDT")

            # Get current prices
            btc_price = _extract_current_price("BTCUSDT", market_data)
            eth_price = _extract_current_price("ETHUSDT", market_data)

            if not btc_price or not eth_price:
                logger.warning("⚠️ Cannot rebalance: Missing price data")
                return []

            # Calculate current allocations as percentages
            btc_value = btc_balance * btc_price
            eth_value = eth_balance * eth_price
            usdt_value = usdt_balance

            current_btc_pct = btc_value / total_value
            current_eth_pct = eth_value / total_value
            current_usdt_pct = usdt_value / total_value

            # ========================================================================================
            # CRÍTICO FIX 3: Use L3 asset allocation if provided, otherwise use defaults
            # ========================================================================================
            if l3_asset_allocation:
                target_btc_pct = l3_asset_allocation.get('BTCUSDT', 0.40)
                target_eth_pct = l3_asset_allocation.get('ETHUSDT', 0.30)
                target_usdt_pct = l3_asset_allocation.get('USDT', 0.30)
                logger.info("🎯 Using L3 asset allocation targets")
            else:
                # Default target allocations
                target_btc_pct = 0.40  # 40% BTC
                target_eth_pct = 0.30  # 30% ETH
                target_usdt_pct = 0.30  # 30% USDT

            # Check for rebalancing needs (deviation > 5%)
            btc_deviation = abs(current_btc_pct - target_btc_pct)
            eth_deviation = abs(current_eth_pct - target_eth_pct)
            usdt_deviation = abs(current_usdt_pct - target_usdt_pct)

            max_deviation = max(btc_deviation, eth_deviation, usdt_deviation)

            if max_deviation < 0.05:  # Less than 5% deviation
                logger.debug("🔄 Rebalance not needed: All allocations within tolerance")
                return []

            logger.info("🔄 PORTFOLIO IMBALANCED - Starting rebalance:")
            logger.info(f"📊 Current: BTC={current_btc_pct*100:.1f}%, ETH={current_eth_pct*100:.1f}%, USDT={current_usdt_pct*100:.1f}%")
            logger.info(f"🎯 Target: BTC={target_btc_pct*100:.1f}%, ETH={target_eth_pct*100:.1f}%, USDT={target_usdt_pct*100:.1f}%")
            # Generate rebalancing orders
            orders = []

            # Calculate target values for each asset
            target_btc_value = total_value * target_btc_pct
            target_eth_value = total_value * target_eth_pct
            target_usdt_value = total_value * target_usdt_pct

            # ========================================================================================
            # SOLUTION 2: Rebalance Grace Period - Skip if position too new
            # ========================================================================================
            # BTC adjustments
            if btc_deviation >= 0.05:
                btc_adjustment = target_btc_value - btc_value
                if abs(btc_adjustment) >= 10.0:  # Minimum trade size
                    # SOLUTION 2: Check position age before rebalancing
                    if btc_adjustment > 0:  # Need to buy BTC
                        if target_btc_pct > 0:  # Target is not 0%
                            # SOLUTION 2: For BUY orders, check if position can be rebalanced
                            if self.portfolio_manager.can_rebalance_position("BTCUSDT"):
                                quantity = btc_adjustment / btc_price
                                order = {
                                    "symbol": "BTCUSDT",
                                    "side": "buy",
                                    "type": "MARKET",
                                    "quantity": quantity,
                                    "price": btc_price,
                                    "reason": "auto_rebalance",
                                    "status": "pending",
                                    "allocation_target": target_btc_pct
                                }
                                orders.append(order)
                                logger.info(f"📈 REBALANCE ORDER: BUY {quantity:.4f} BTC (${btc_adjustment:.2f})")
                            else:
                                logger.info(f"⏰ SOLUTION 2: BTC BUY REBALANCE SKIPPED - Position too new")
                        else:
                            logger.debug(f"🔄 BTC target is 0%, skipping buy")
                    elif btc_adjustment < 0:  # Need to sell BTC
                        # SOLUTION 2: For SELL orders, always allow (no age restriction)
                        quantity = abs(btc_adjustment) / btc_price
                        order = {
                            "symbol": "BTCUSDT",
                            "side": "sell",
                            "type": "MARKET",
                            "quantity": quantity,
                            "price": btc_price,
                            "reason": "auto_rebalance",
                            "status": "pending",
                            "allocation_target": target_btc_pct
                        }
                        orders.append(order)
                        logger.info(f"📉 REBALANCE ORDER: SELL {quantity:.4f} BTC (${abs(btc_adjustment):.2f})")

            # ETH adjustments
            if eth_deviation >= 0.05:
                eth_adjustment = target_eth_value - eth_value
                if abs(eth_adjustment) >= 10.0:  # Minimum trade size
                    # SOLUTION 2: Check position age before rebalancing
                    if eth_adjustment > 0:  # Need to buy ETH
                        if target_eth_pct > 0:  # Target is not 0%
                            # SOLUTION 2: For BUY orders, check if position can be rebalanced
                            if self.portfolio_manager.can_rebalance_position("ETHUSDT"):
                                quantity = eth_adjustment / eth_price
                                order = {
                                    "symbol": "ETHUSDT",
                                    "side": "buy",
                                    "type": "MARKET",
                                    "quantity": quantity,
                                    "price": eth_price,
                                    "reason": "auto_rebalance",
                                    "status": "pending",
                                    "allocation_target": target_eth_pct
                                }
                                orders.append(order)
                                logger.info(f"📈 REBALANCE ORDER: BUY {quantity:.2f} ETH (${eth_adjustment:.2f})")
                            else:
                                logger.info(f"⏰ SOLUTION 2: ETH BUY REBALANCE SKIPPED - Position too new")
                        else:
                            logger.debug(f"🔄 ETH target is 0%, skipping buy")
                    elif eth_adjustment < 0:  # Need to sell ETH
                        # SOLUTION 2: For SELL orders, always allow (no age restriction)
                        quantity = abs(eth_adjustment) / eth_price
                        order = {
                            "symbol": "ETHUSDT",
                            "side": "sell",
                            "type": "MARKET",
                            "quantity": quantity,
                            "price": eth_price,
                            "reason": "auto_rebalance",
                            "status": "pending",
                            "allocation_target": target_eth_pct
                        }
                        orders.append(order)
                        logger.info(f"📉 REBALANCE ORDER: SELL {quantity:.2f} ETH (${abs(eth_adjustment):.2f})")

            if orders:
                logger.info(f"✅ AutoRebalance: Generated {len(orders)} orders to restore balance")
            else:
                logger.debug("🔄 AutoRebalance: No orders needed after threshold check")

            return orders

        except Exception as e:
            logger.error(f"❌ Error in auto rebalancing: {e}")
            return []
