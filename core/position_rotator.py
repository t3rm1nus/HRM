# core/position_rotator.py
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
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from core.logging import logger
from l2_tactic.utils import safe_float
from l1_operational.config import ConfigObject


class PositionRotator:
    """
    Sistema automático de rotación y rebalanceo de posiciones.

    Monitorea el portfolio y ejecuta rotaciones automáticas para:
    - Mantener límites de posición (40% máximo por activo)
    - Liberar capital cuando USDT < $500
    - Rebalancear cada hora si USDT < 15% del total
    - Rotar capital de activos con pérdidas > 3%
    """

    def __init__(self, portfolio_manager=None):
        """Inicializar el sistema de rotación"""
        self.config = ConfigObject
        self.portfolio_manager = portfolio_manager
        self.last_rebalance_check = datetime.utcnow()
        self.rotation_history = []
        self.monitoring_active = True

        # Configuración de límites
        self.max_exposure_btc = self.config.PORTFOLIO_LIMITS["MAX_PORTFOLIO_EXPOSURE_BTC"]  # 0.40
        self.max_exposure_eth = self.config.PORTFOLIO_LIMITS["MAX_PORTFOLIO_EXPOSURE_ETH"]  # 0.40
        self.max_position_usdt = self.config.PORTFOLIO_LIMITS["MAX_POSITION_SIZE_USDT"]    # 1200
        self.min_usdt_reserve = self.config.PORTFOLIO_LIMITS["MIN_USDT_RESERVE"]          # 0.20
        self.rebalance_threshold = self.config.PORTFOLIO_LIMITS["REBALANCE_THRESHOLD"]    # 0.15
        self.rotation_amount = self.config.PORTFOLIO_LIMITS["ROTATION_AMOUNT"]            # 0.25
        self.min_usdt_balance = self.config.PORTFOLIO_LIMITS["MIN_ACCOUNT_BALANCE_USDT"]  # 500

        logger.info("🔄 PositionRotator inicializado")
        logger.info(f"   Límites: BTC {self.max_exposure_btc*100:.0f}%, ETH {self.max_exposure_eth*100:.0f}%, Posición máx ${self.max_position_usdt}")
        logger.info(f"   Reserva USDT: {self.min_usdt_reserve*100:.0f}% mínimo, ${self.min_usdt_balance} absoluto")

    async def check_and_rotate_positions(self, state: Dict[str, Any], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verificar y ejecutar rotaciones automáticas de posiciones.

        Returns:
            Lista de órdenes de rotación a ejecutar
        """
        if not self.monitoring_active:
            return []

        rotation_orders = []

        try:
            # Obtener estado actual del portfolio
            portfolio = state.get("portfolio", {})
            total_value = self._calculate_total_value(portfolio, market_data)

            if total_value <= 0:
                logger.warning("⚠️ Valor total del portfolio inválido para rotación")
                return []

            # Extraer balances actuales
            btc_balance = safe_float(portfolio.get("BTCUSDT", {}).get("position", 0.0))
            eth_balance = safe_float(portfolio.get("ETHUSDT", {}).get("position", 0.0))
            usdt_balance = safe_float(portfolio.get("USDT", {}).get("free", 0.0))

            # Calcular valores actuales
            btc_price = self._get_price(market_data, "BTCUSDT", 50000.0)
            eth_price = self._get_price(market_data, "ETHUSDT", 3000.0)

            btc_value = btc_balance * btc_price
            eth_value = eth_balance * eth_price

            # Calcular porcentajes de exposición
            btc_exposure = btc_value / total_value if total_value > 0 else 0.0
            eth_exposure = eth_value / total_value if total_value > 0 else 0.0
            usdt_exposure = usdt_balance / total_value if total_value > 0 else 0.0

            logger.info("🔍 ROTATION CHECK:")
            logger.info(f"   Total: ${total_value:.2f} | BTC: ${btc_value:.2f} ({btc_exposure*100:.1f}%) | ETH: ${eth_value:.2f} ({eth_exposure*100:.1f}%) | USDT: ${usdt_balance:.2f} ({usdt_exposure*100:.1f}%)")

            # =================================================================
            # REGLA 1: VERIFICAR LÍMITES DE POSICIÓN INDIVIDUAL ($1200 máximo)
            # =================================================================
            individual_rotation = self._check_individual_position_limits(
                btc_value, eth_value, btc_balance, eth_balance, btc_price, eth_price
            )
            rotation_orders.extend(individual_rotation)

            # =================================================================
            # REGLA 2: VERIFICAR EXPOSICIÓN MÁXIMA POR ACTIVO (40%)
            # =================================================================
            exposure_rotation = self._check_exposure_limits(
                btc_exposure, eth_exposure, btc_value, eth_value,
                btc_balance, eth_balance, btc_price, eth_price, total_value
            )
            rotation_orders.extend(exposure_rotation)

            # =================================================================
            # REGLA 3: LIBERAR CAPITAL SI USDT < $500
            # =================================================================
            if usdt_balance < self.min_usdt_balance:
                logger.warning(f"🚨 USDT BAJO DETECTADO: ${usdt_balance:.2f} < ${self.min_usdt_balance:.2f}")
                capital_rotation = self._free_up_capital(
                    btc_value, eth_value, btc_balance, eth_balance,
                    btc_price, eth_price, usdt_balance
                )
                rotation_orders.extend(capital_rotation)

            # =================================================================
            # REGLA 4: REBALANCEO AUTOMÁTICO CADA HORA
            # =================================================================
            if self._should_rebalance_hourly(usdt_exposure):
                logger.info("⏰ REBALANCEO HORARIO - Verificando necesidad...")
                hourly_rotation = self._perform_hourly_rebalance(
                    btc_exposure, eth_exposure, usdt_exposure,
                    btc_balance, eth_balance, btc_price, eth_price, total_value
                )
                rotation_orders.extend(hourly_rotation)

            # =================================================================
            # REGLA 5: ROTACIÓN POR RENDIMIENTO (24h)
            # =================================================================
            performance_rotation = await self._check_performance_rotation(
                state, market_data, btc_balance, eth_balance, btc_price, eth_price
            )
            rotation_orders.extend(performance_rotation)

            # =================================================================
            # VALIDACIÓN CRÍTICA DE PRECIOS - PREVENIR VENTAS A PRECIOS ERRÓNEOS
            # =================================================================
            if rotation_orders:
                validated_orders = []
                for order in rotation_orders:
                    if self._validate_rotation_order(order, market_data):
                        validated_orders.append(order)
                    else:
                        logger.error(f"🚨 ORDER REJECTED: {order['symbol']} {order['side']} @ ${order['price']:.2f} - PRICE VALIDATION FAILED")

                rotation_orders = validated_orders

            # Registrar rotaciones ejecutadas
            if rotation_orders:
                self.rotation_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "orders_count": len(rotation_orders),
                    "total_value": total_value,
                    "usdt_balance": usdt_balance,
                    "reason": "automatic_rotation"
                })

                logger.info(f"🔄 ROTACIONES GENERADAS: {len(rotation_orders)} órdenes")
                for i, order in enumerate(rotation_orders):
                    logger.info(f"   {i+1}. {order['symbol']} {order['side']} {order['quantity']:.4f} @ ${order['price']:.2f}")

            return rotation_orders

        except Exception as e:
            logger.error(f"❌ Error en check_and_rotate_positions: {e}")
            return []

    def _check_individual_position_limits(self, btc_value: float, eth_value: float,
                                        btc_balance: float, eth_balance: float,
                                        btc_price: float, eth_price: float) -> List[Dict[str, Any]]:
        """Verificar límites individuales de posición ($1200 máximo)"""
        orders = []

        # Verificar BTC
        if btc_value > self.max_position_usdt:
            excess_value = btc_value - self.max_position_usdt
            excess_quantity = excess_value / btc_price

            if excess_quantity > 0.0001:  # Mínimo significativo
                order = {
                    "symbol": "BTCUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": excess_quantity,
                    "price": btc_price,
                    "reason": "individual_limit_exceeded",
                    "excess_value": excess_value,
                    "status": "pending"
                }
                orders.append(order)
                logger.warning(f"🚨 BTC POSITION LIMIT: ${btc_value:.2f} > ${self.max_position_usdt:.2f}, vendiendo {excess_quantity:.4f} BTC")

        # Verificar ETH
        if eth_value > self.max_position_usdt:
            excess_value = eth_value - self.max_position_usdt
            excess_quantity = excess_value / eth_price

            if excess_quantity > 0.001:  # Mínimo significativo
                order = {
                    "symbol": "ETHUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": excess_quantity,
                    "price": eth_price,
                    "reason": "individual_limit_exceeded",
                    "excess_value": excess_value,
                    "status": "pending"
                }
                orders.append(order)
                logger.warning(f"🚨 ETH POSITION LIMIT: ${eth_value:.2f} > ${self.max_position_usdt:.2f}, vendiendo {excess_quantity:.4f} ETH")

        return orders

    def _check_exposure_limits(self, btc_exposure: float, eth_exposure: float,
                             btc_value: float, eth_value: float,
                             btc_balance: float, eth_balance: float,
                             btc_price: float, eth_price: float, total_value: float) -> List[Dict[str, Any]]:
        """Verificar límites de exposición por activo (40%)"""
        orders = []

        # Verificar BTC exposure
        if btc_exposure > self.max_exposure_btc:
            excess_exposure = btc_exposure - self.max_exposure_btc
            excess_value = excess_exposure * total_value
            excess_quantity = excess_value / btc_price

            if excess_quantity > 0.0001:
                order = {
                    "symbol": "BTCUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": excess_quantity,
                    "price": btc_price,
                    "reason": "exposure_limit_exceeded",
                    "excess_exposure": excess_exposure,
                    "status": "pending"
                }
                orders.append(order)
                logger.warning(f"🚨 BTC EXPOSURE LIMIT: {btc_exposure*100:.1f}% > {self.max_exposure_btc*100:.1f}%, vendiendo {excess_quantity:.4f} BTC")

        # Verificar ETH exposure
        if eth_exposure > self.max_exposure_eth:
            excess_exposure = eth_exposure - self.max_exposure_eth
            excess_value = excess_exposure * total_value
            excess_quantity = excess_value / eth_price

            if excess_quantity > 0.001:
                order = {
                    "symbol": "ETHUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": excess_quantity,
                    "price": eth_price,
                    "reason": "exposure_limit_exceeded",
                    "excess_exposure": excess_exposure,
                    "status": "pending"
                }
                orders.append(order)
                logger.warning(f"🚨 ETH EXPOSURE LIMIT: {eth_exposure*100:.1f}% > {self.max_exposure_eth*100:.1f}%, vendiendo {excess_quantity:.4f} ETH")

        return orders

    def _free_up_capital(self, btc_value: float, eth_value: float,
                        btc_balance: float, eth_balance: float,
                        btc_price: float, eth_price: float, usdt_balance: float) -> List[Dict[str, Any]]:
        """Liberar capital cuando USDT < $500 - vender 25% de la posición más grande"""
        orders = []

        # Determinar cuál posición es más grande
        if btc_value > eth_value and btc_balance > 0.0001:
            # Vender 25% de BTC
            sell_quantity = btc_balance * self.rotation_amount
            if sell_quantity > 0.0001:
                order = {
                    "symbol": "BTCUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": sell_quantity,
                    "price": btc_price,
                    "reason": "free_up_capital",
                    "usdt_shortfall": self.min_usdt_balance - usdt_balance,
                    "status": "pending"
                }
                orders.append(order)
                logger.info(f"💰 FREEING CAPITAL: USDT ${usdt_balance:.2f} < ${self.min_usdt_balance:.2f}, vendiendo {sell_quantity:.4f} BTC (25%)")

        elif eth_balance > 0.001:
            # Vender 25% de ETH
            sell_quantity = eth_balance * self.rotation_amount
            if sell_quantity > 0.001:
                order = {
                    "symbol": "ETHUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": sell_quantity,
                    "price": eth_price,
                    "reason": "free_up_capital",
                    "usdt_shortfall": self.min_usdt_balance - usdt_balance,
                    "status": "pending"
                }
                orders.append(order)
                logger.info(f"💰 FREEING CAPITAL: USDT ${usdt_balance:.2f} < ${self.min_usdt_balance:.2f}, vendiendo {sell_quantity:.4f} ETH (25%)")

        return orders

    def _should_rebalance_hourly(self, usdt_exposure: float) -> bool:
        """Verificar si debe ejecutarse rebalanceo horario"""
        now = datetime.utcnow()
        time_since_last_check = now - self.last_rebalance_check

        # Rebalancear cada hora si USDT < 15%
        should_rebalance = (time_since_last_check >= timedelta(hours=1) and
                          usdt_exposure < self.rebalance_threshold)

        if should_rebalance:
            self.last_rebalance_check = now
            logger.info(f"⏰ HOURLY REBALANCE TRIGGERED: USDT {usdt_exposure*100:.1f}% < {self.rebalance_threshold*100:.1f}%")

        return should_rebalance

    def _perform_hourly_rebalance(self, btc_exposure: float, eth_exposure: float, usdt_exposure: float,
                                btc_balance: float, eth_balance: float,
                                btc_price: float, eth_price: float, total_value: float) -> List[Dict[str, Any]]:
        """Ejecutar rebalanceo horario - vender 10% de la posición más grande"""
        orders = []

        # Encontrar la posición más grande
        btc_value = btc_balance * btc_price
        eth_value = eth_balance * eth_price

        if btc_value > eth_value and btc_balance > 0.0001:
            # Vender 10% de BTC
            sell_quantity = btc_balance * 0.10
            if sell_quantity > 0.0001:
                order = {
                    "symbol": "BTCUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": sell_quantity,
                    "price": btc_price,
                    "reason": "hourly_rebalance",
                    "usdt_exposure": usdt_exposure,
                    "status": "pending"
                }
                orders.append(order)
                logger.info(f"⏰ HOURLY REBALANCE: Vendiendo 10% de BTC ({sell_quantity:.4f}) para aumentar liquidez")

        elif eth_balance > 0.001:
            # Vender 10% de ETH
            sell_quantity = eth_balance * 0.10
            if sell_quantity > 0.001:
                order = {
                    "symbol": "ETHUSDT",
                    "side": "sell",
                    "type": "MARKET",
                    "quantity": sell_quantity,
                    "price": eth_price,
                    "reason": "hourly_rebalance",
                    "usdt_exposure": usdt_exposure,
                    "status": "pending"
                }
                orders.append(order)
                logger.info(f"⏰ HOURLY REBALANCE: Vendiendo 10% de ETH ({sell_quantity:.4f}) para aumentar liquidez")

        return orders

    async def _check_performance_rotation(self, state: Dict[str, Any], market_data: Dict[str, Any],
                                        btc_balance: float, eth_balance: float,
                                        btc_price: float, eth_price: float) -> List[Dict[str, Any]]:
        """Verificar rotación por rendimiento (activos con pérdidas > 3% en 24h)"""
        orders = []

        try:
            # Obtener datos históricos de 24h para calcular rendimiento
            # Esto requeriría acceso a datos históricos - por ahora usamos lógica simplificada
            # En implementación real, comparar con precio de hace 24h

            # Lógica simplificada: si tenemos datos de rendimiento en el state
            performance_data = state.get("performance_24h", {})

            btc_performance = performance_data.get("BTCUSDT", 0.0)
            eth_performance = performance_data.get("ETHUSDT", 0.0)

            # Si algún activo tiene pérdidas > 3%, considerar reducir 30%
            loss_threshold = -0.03  # -3%

            if btc_performance < loss_threshold and btc_balance > 0.0001:
                sell_quantity = btc_balance * 0.30  # 30%
                if sell_quantity > 0.0001:
                    order = {
                        "symbol": "BTCUSDT",
                        "side": "sell",
                        "type": "MARKET",
                        "quantity": sell_quantity,
                        "price": btc_price,
                        "reason": "performance_rotation",
                        "performance_24h": btc_performance,
                        "status": "pending"
                    }
                    orders.append(order)
                    logger.info(f"📈 PERFORMANCE ROTATION: BTC -{btc_performance*100:.1f}% en 24h, vendiendo 30% ({sell_quantity:.4f})")

            if eth_performance < loss_threshold and eth_balance > 0.001:
                sell_quantity = eth_balance * 0.30  # 30%
                if sell_quantity > 0.001:
                    order = {
                        "symbol": "ETHUSDT",
                        "side": "sell",
                        "type": "MARKET",
                        "quantity": sell_quantity,
                        "price": eth_price,
                        "reason": "performance_rotation",
                        "performance_24h": eth_performance,
                        "status": "pending"
                    }
                    orders.append(order)
                    logger.info(f"📈 PERFORMANCE ROTATION: ETH -{eth_performance*100:.1f}% en 24h, vendiendo 30% ({sell_quantity:.4f})")

        except Exception as e:
            logger.error(f"❌ Error en performance rotation: {e}")

        return orders

    def _calculate_total_value(self, portfolio: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calcular valor total del portfolio"""
        try:
            btc_balance = safe_float(portfolio.get("BTCUSDT", {}).get("position", 0.0))
            eth_balance = safe_float(portfolio.get("ETHUSDT", {}).get("position", 0.0))
            usdt_balance = safe_float(portfolio.get("USDT", {}).get("free", 0.0))

            btc_price = self._get_price(market_data, "BTCUSDT", 50000.0)
            eth_price = self._get_price(market_data, "ETHUSDT", 3000.0)

            total_value = (btc_balance * btc_price) + (eth_balance * eth_price) + usdt_balance
            return max(0.0, total_value)

        except Exception as e:
            logger.error(f"❌ Error calculando valor total: {e}")
            return 0.0

    def _get_price(self, market_data: Dict[str, Any], symbol: str, default_price: float) -> float:
        """Obtener precio actual de un símbolo con validación robusta"""
        try:
            if not market_data or symbol not in market_data:
                logger.warning(f"⚠️ No market data available for {symbol}")
                return default_price

            symbol_data = market_data[symbol]

            # Handle DataFrame format (most common)
            if hasattr(symbol_data, 'iloc') and len(symbol_data) > 0:
                if 'close' in symbol_data.columns:
                    price = symbol_data['close'].iloc[-1]
                    return safe_float(price)
                else:
                    logger.warning(f"⚠️ No 'close' column in DataFrame for {symbol}")
                    return default_price

            # Handle dict format
            elif isinstance(symbol_data, dict):
                if 'close' in symbol_data:
                    return safe_float(symbol_data['close'])
                else:
                    logger.warning(f"⚠️ No 'close' key in dict for {symbol}")
                    return default_price

            # Handle list/array format (fallback)
            elif isinstance(symbol_data, (list, tuple)) and len(symbol_data) > 0:
                # Assume last element is most recent price
                return safe_float(symbol_data[-1])

            else:
                logger.warning(f"⚠️ Unsupported market data format for {symbol}: {type(symbol_data)}")
                return default_price

        except Exception as e:
            logger.error(f"❌ Error getting price for {symbol}: {e}")
            return default_price

    def _validate_rotation_order(self, order: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Validar que una orden de rotación tiene precios razonables"""
        try:
            symbol = order.get("symbol", "")
            order_price = order.get("price", 0.0)
            side = order.get("side", "")

            if not symbol or order_price <= 0:
                logger.error(f"🚨 Invalid order data: symbol={symbol}, price={order_price}")
                return False

            # Obtener precio actual del mercado
            current_price = self._get_price(market_data, symbol, 0.0)

            if current_price <= 0:
                logger.error(f"🚨 No current price available for {symbol}")
                return False

            # Calcular diferencia porcentual
            price_diff_pct = abs(order_price - current_price) / current_price

            # Límites de validación según el activo
            if symbol == "BTCUSDT":
                max_diff_pct = 0.50  # 50% máximo (BTC puede ser volátil)
                min_price = 10000.0  # Precio mínimo razonable
                max_price = 200000.0  # Precio máximo razonable
            elif symbol == "ETHUSDT":
                max_diff_pct = 0.50  # 50% máximo
                min_price = 500.0    # Precio mínimo razonable
                max_price = 10000.0  # Precio máximo razonable
            else:
                max_diff_pct = 0.30  # 30% para otros activos
                min_price = 0.01
                max_price = 1000000.0

            # Validaciones críticas
            if order_price < min_price or order_price > max_price:
                logger.error(f"🚨 PRICE OUT OF RANGE: {symbol} order=${order_price:.2f}, range=[${min_price:.2f}, ${max_price:.2f}]")
                return False

            if price_diff_pct > max_diff_pct:
                logger.error(f"🚨 PRICE DIFFERENCE TOO LARGE: {symbol} order=${order_price:.2f}, current=${current_price:.2f}, diff={price_diff_pct*100:.1f}% > {max_diff_pct*100:.1f}%")
                return False

            # Validación adicional: precio no debe ser exactamente el precio por defecto
            if symbol == "BTCUSDT" and abs(order_price - 50000.0) < 1000:
                logger.error(f"🚨 DEFAULT PRICE DETECTED: {symbol} order=${order_price:.2f} too close to default $50,000")
                return False
            elif symbol == "ETHUSDT" and abs(order_price - 3000.0) < 300:
                logger.error(f"🚨 DEFAULT PRICE DETECTED: {symbol} order=${order_price:.2f} too close to default $3,000")
                return False

            logger.info(f"✅ Order validation PASSED: {symbol} {side} @ ${order_price:.2f} (current: ${current_price:.2f}, diff: {price_diff_pct*100:.1f}%)")
            return True

        except Exception as e:
            logger.error(f"❌ Error validating rotation order: {e}")
            return False

    def get_rotation_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rotaciones ejecutadas"""
        total_rotations = len(self.rotation_history)
        recent_rotations = [r for r in self.rotation_history if (datetime.utcnow() - datetime.fromisoformat(r["timestamp"])).days < 1]

        return {
            "total_rotations": total_rotations,
            "rotations_today": len(recent_rotations),
            "monitoring_active": self.monitoring_active,
            "last_check": self.last_rebalance_check.isoformat()
        }

    def enable_monitoring(self):
        """Activar monitoreo de rotaciones"""
        self.monitoring_active = True
        logger.info("✅ Position rotation monitoring ENABLED")

    def disable_monitoring(self):
        """Desactivar monitoreo de rotaciones"""
        self.monitoring_active = False
        logger.info("⏸️ Position rotation monitoring DISABLED")

    async def emergency_free_capital(self, state: Dict[str, Any], market_data: Dict[str, Any],
                                   target_usdt: float = 600.0) -> List[Dict[str, Any]]:
        """Liberación de emergencia de capital para alcanzar target USDT"""
        orders = []

        try:
            portfolio = state.get("portfolio", {})
            usdt_balance = safe_float(portfolio.get("USDT", {}).get("free", 0.0))

            if usdt_balance >= target_usdt:
                logger.info(f"💰 Emergency free capital: Already have ${usdt_balance:.2f} >= ${target_usdt:.2f}")
                return []

            shortfall = target_usdt - usdt_balance
            logger.warning(f"🚨 EMERGENCY CAPITAL FREE: Need ${shortfall:.2f} more USDT")

            # Vender posiciones para liberar capital
            btc_balance = safe_float(portfolio.get("BTCUSDT", {}).get("position", 0.0))
            eth_balance = safe_float(portfolio.get("ETHUSDT", {}).get("position", 0.0))

            btc_price = self._get_price(market_data, "BTCUSDT", 50000.0)
            eth_price = self._get_price(market_data, "ETHUSDT", 3000.0)

            btc_value = btc_balance * btc_price
            eth_value = eth_balance * eth_price

            # Vender de la posición más grande primero
            if btc_value >= shortfall and btc_balance > 0.0001:
                sell_quantity = shortfall / btc_price
                if sell_quantity > 0.0001:
                    order = {
                        "symbol": "BTCUSDT",
                        "side": "sell",
                        "type": "MARKET",
                        "quantity": sell_quantity,
                        "price": btc_price,
                        "reason": "emergency_capital",
                        "target_usdt": target_usdt,
                        "shortfall": shortfall,
                        "status": "pending"
                    }
                    orders.append(order)
                    logger.warning(f"🚨 EMERGENCY: Vendiendo {sell_quantity:.4f} BTC para liberar ${shortfall:.2f}")

            elif eth_value >= shortfall and eth_balance > 0.001:
                sell_quantity = shortfall / eth_price
                if sell_quantity > 0.001:
                    order = {
                        "symbol": "ETHUSDT",
                        "side": "sell",
                        "type": "MARKET",
                        "quantity": sell_quantity,
                        "price": eth_price,
                        "reason": "emergency_capital",
                        "target_usdt": target_usdt,
                        "shortfall": shortfall,
                        "status": "pending"
                    }
                    orders.append(order)
                    logger.warning(f"🚨 EMERGENCY: Vendiendo {sell_quantity:.4f} ETH para liberar ${shortfall:.2f}")

        except Exception as e:
            logger.error(f"❌ Error en emergency capital free: {e}")

        return orders
