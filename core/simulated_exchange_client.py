"""
SimulatedExchangeClient - Cliente de intercambio simulado para backtesting y testing

Este cliente proporciona una implementación completa de un exchange simulado
que puede ser utilizado para backtesting, testing y desarrollo sin riesgo real.
"""

import asyncio
import random
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from core.logging import logger


class OrderStatus(Enum):
    """Estados posibles de una orden"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Tipos de órdenes soportadas"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class Order:
    """Representa una orden en el exchange simulado"""
    id: str
    symbol: str
    side: str  # "buy" o "sell"
    type: OrderType
    quantity: float
    price: Optional[float] = None  # Para órdenes limitadas
    stop_price: Optional[float] = None  # Para órdenes stop
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    timestamp: float = 0.0
    fees: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class SimulatedExchangeClient:
    """
    Cliente de intercambio simulado que replica el comportamiento de Binance
    pero con datos internos y sin riesgo real.
    """
    _instance = None
    _initialized = False

    def __new__(cls, initial_balances: Dict[str, float] = None,
                 enable_commissions: bool = True,
                 enable_slippage: bool = True,
                 volatility_factor: float = 0.02):
        """Singleton pattern to maintain state between instances"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, initial_balances: Dict[str, float] = None,
                 enable_commissions: bool = True,
                 enable_slippage: bool = True,
                 volatility_factor: float = 0.02):
        """
        Inicializa el cliente simulado.
        
        Args:
            initial_balances: Balances iniciales para cada activo
            enable_commissions: Habilitar comisiones de trading
            enable_slippage: Habilitar slippage en órdenes
            volatility_factor: Factor de volatilidad para simulación de precios
        """
        if SimulatedExchangeClient._initialized:
            logger.debug("🎮 SimulatedExchangeClient already initialized - maintaining state")
            # Si se proporcionan balances iniciales diferentes, actualizar (solo para pruebas)
            if initial_balances and initial_balances != self.initial_balances:
                logger.warning("⚠️ SimulatedExchangeClient already initialized - ignoring new initial balances")
            return
        
        SimulatedExchangeClient._initialized = True
        
        if initial_balances is None:
            initial_balances = {
                "BTC": 0.01549,
                "ETH": 0.385,
                "USDT": 3000.0
            }
        
        self.initial_balances = initial_balances.copy()
        self.enable_commissions = enable_commissions
        self.enable_slippage = enable_slippage
        self.volatility_factor = volatility_factor
        
        # Estado interno
        self.balances = initial_balances.copy()
        self.orders: Dict[str, Order] = {}
        self.order_counter = 1
        
        # Precios de mercado simulados
        self.market_prices = {}
        self.price_history = {}
        
        # Configuración de trading
        self.maker_fee = 0.001  # 0.1%
        self.taker_fee = 0.001  # 0.1%
        self.slippage_bps = 2   # 2 basis points (0.02%)
        
        # Inicializar precios basados en balances
        self._initialize_market_prices()
        
        logger.info(f"🎮 SimulatedExchangeClient inicializado")
        logger.info(f"   Balances iniciales: {self.balances}")
        logger.info(f"   Comisiones: {'Habilitadas' if self.enable_commissions else 'Deshabilitadas'}")
        logger.info(f"   Slippage: {'Habilitado' if self.enable_slippage else 'Deshabilitado'}")
        logger.info(f"   Volatilidad: {volatility_factor}")
        logger.info(f"   SIM_INIT_ONCE=True")

    def _initialize_market_prices(self):
        """Inicializa precios de mercado basados en balances o valores por defecto"""
        # Precios base para diferentes símbolos
        base_prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "BNBUSDT": 300.0,
            "SOLUSDT": 100.0,
            "ADAUSDT": 0.5,
            "XRPUSDT": 0.5,
            "DOGEUSDT": 0.1
        }
        
        for symbol in self.balances.keys():
            if symbol == "USDT":
                continue
                
            # Extraer el par (ej: BTCUSDT -> BTC)
            base_asset = symbol.replace("USDT", "")
            
            if base_asset in base_prices:
                base_price = base_prices[base_asset]
            else:
                # Precio base aleatorio para activos no definidos
                base_price = random.uniform(1.0, 1000.0)
            
            # Ajustar precio basado en el balance (mayor balance = menor precio)
            balance_factor = min(1.0, self.balances.get(symbol, 0.0) / 10.0)
            initial_price = base_price * (1.0 - balance_factor * 0.1)
            
            self.market_prices[symbol] = initial_price
            self.price_history[symbol] = [initial_price]
            
            logger.debug(f"   Precio inicial {symbol}: {initial_price:.6f}")

    def get_market_price(self, symbol: str) -> float:
        """Obtiene el precio actual del mercado para un símbolo"""
        if symbol not in self.market_prices:
            # Si no existe, crear con precio base
            self._initialize_symbol_price(symbol)
        
        return self.market_prices[symbol]

    def _initialize_symbol_price(self, symbol: str):
        """Inicializa el precio para un símbolo no existente"""
        base_price = random.uniform(10.0, 5000.0)
        self.market_prices[symbol] = base_price
        self.price_history[symbol] = [base_price]
        logger.debug(f"   Nuevo símbolo {symbol}: precio inicial {base_price:.6f}")

    def simulate_price_movement(self, symbol: str):
        """Simula el movimiento del precio basado en volatilidad"""
        if symbol not in self.market_prices:
            self._initialize_symbol_price(symbol)
            
        current_price = self.market_prices[symbol]
        
        # Movimiento aleatorio basado en volatilidad
        volatility = self.volatility_factor
        random_factor = random.uniform(-volatility, volatility)
        
        # Simular noticias o eventos del mercado
        news_factor = random.choice([1.0, 1.0, 1.0, 1.0, 1.0,  # Normal
                                   0.98, 1.02,  # Pequeños movimientos
                                   0.95, 1.05,  # Movimientos moderados
                                   0.90, 1.10])  # Movimientos grandes
        
        new_price = current_price * (1 + random_factor) * news_factor
        new_price = max(new_price, 0.0001)  # Evitar precios negativos o cero
        
        self.market_prices[symbol] = new_price
        self.price_history[symbol].append(new_price)
        
        # Mantener solo los últimos 1000 precios para evitar consumo de memoria
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]

    async def get_account_balances(self) -> Dict[str, float]:
        """Obtiene los balances de la cuenta (simulados) - versión asincrónica para compatibilidad con BinanceClient"""
        return self.balances.copy()

    def get_account_balances_sync(self) -> Dict[str, float]:
        """Obtiene los balances de la cuenta (simulados) - versión sincrónica"""
        return self.balances.copy()

    def get_balance(self, asset: str) -> float:
        """Obtiene el balance de un activo específico"""
        return self.balances.get(asset, 0.0)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, List[List[float]]]:
        """Obtiene el order book simulado"""
        current_price = self.get_market_price(symbol)
        
        # Generar bids (precios de compra)
        bids = []
        for i in range(limit):
            price = current_price * (0.999 - i * 0.0001)
            quantity = random.uniform(0.001, 0.1)
            bids.append([price, quantity])
        
        # Generar asks (precios de venta)
        asks = []
        for i in range(limit):
            price = current_price * (1.001 + i * 0.0001)
            quantity = random.uniform(0.001, 0.1)
            asks.append([price, quantity])
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        }

    def calculate_slippage(self, symbol: str, quantity: float, side: str) -> float:
        """Calcula el slippage basado en el tamaño de la orden y la liquidez"""
        if not self.enable_slippage:
            return 0.0
        
        current_price = self.get_market_price(symbol)
        order_value = quantity * current_price
        
        # Simular impacto en el mercado basado en el tamaño de la orden
        # Cuanto mayor sea la orden, mayor será el slippage
        market_impact = min(0.05, order_value / 100000.0)  # Máximo 5% de slippage
        
        # Factor aleatorio para simular condiciones del mercado
        random_slippage = random.uniform(-0.001, 0.005)  # -0.1% a +0.5%
        
        total_slippage = market_impact + random_slippage
        
        # Ajustar según la dirección de la orden
        if side.lower() == "buy":
            # Para compras, el slippage es positivo (precio más alto)
            return max(0.0, total_slippage)
        else:
            # Para ventas, el slippage es negativo (precio más bajo)
            return min(0.0, -total_slippage)

    def calculate_fees(self, order_value: float) -> float:
        """Calcula las comisiones de trading"""
        if not self.enable_commissions:
            return 0.0
        
        # Comisiones maker/taker (en simulación, asumimos taker)
        return order_value * self.taker_fee

    def validate_order(self, symbol: str, side: str, quantity: float, 
                      price: Optional[float] = None) -> tuple[bool, str, float]:
        """Valida una orden antes de colocarla y ajusta la cantidad si es necesario"""
        # Validar símbolo - permitir cualquier símbolo que termine en USDT
        if not symbol.endswith("USDT") and symbol != "USDT":
            return False, f"Símbolo no soportado: {symbol}", quantity
        
        # Extract base asset for symbol (e.g., BTCUSDT -> BTC)
        base_asset = symbol.replace("USDT", "") if symbol != "USDT" else "USDT"
        
        # Validar lado
        if side.lower() not in ["buy", "sell"]:
            return False, f"Lado inválido: {side}", quantity
        
        # Validar cantidad
        if quantity <= 0:
            return False, f"Cantidad inválida: {quantity}", quantity
        
        # Validar balance para compras
        if side.lower() == "buy":
            current_price = self.get_market_price(symbol)
            order_value = quantity * current_price
            
            # Calcular slippage y comisiones
            slippage = self.calculate_slippage(symbol, quantity, side)
            fees = self.calculate_fees(order_value)
            
            total_cost = order_value * (1 + slippage) + fees
            
            if self.get_balance("USDT") < total_cost:
                logger.warning(f"⚠️ Fondos insuficientes para compra: USDT={self.get_balance('USDT'):.2f} < {total_cost:.2f}")
                # Calcular cantidad máxima posible
                max_quantity = (self.get_balance("USDT") / (current_price * (1 + slippage) + current_price * self.taker_fee))
                if max_quantity > 0:
                    logger.info(f"📊 Ajustando cantidad de compra: {quantity:.6f} → {max_quantity:.6f} unidades")
                    return True, "Cantidad ajustada por fondos insuficientes", max_quantity
                else:
                    return False, "No hay suficientes fondos para comprar incluso la cantidad mínima", 0.0
        
        # Validar balance para ventas
        else:
            if self.get_balance(base_asset) < quantity:
                logger.warning(f"⚠️ Balance insuficiente para venta: {base_asset}={self.get_balance(base_asset):.6f} < {quantity:.6f}")
                logger.info(f"📊 Ajustando cantidad de venta: {quantity:.6f} → {self.get_balance(base_asset):.6f} unidades")
                return True, "Cantidad ajustada por balance insuficiente", self.get_balance(base_asset)
        
        return True, "OK", quantity

    async def execute_order(self, symbol: str, side: str, qty: float, market_price: float) -> Dict[str, Any]:
        """Execute an order with automatic quantity adjustment if funds are insufficient"""
        # Validate and adjust order quantity if needed
        is_valid, validation_msg, adjusted_qty = self.validate_order(symbol, side, qty, market_price)
        
        if not is_valid:
            logger.error(f"❌ Order execution failed: {validation_msg}")
            return {
                "status": "failed",
                "error": validation_msg,
                "quantity": qty,
                "execution_price": market_price,
                "cost": 0.0,
                "fee": 0.0,
                "slippage_cost": 0.0,
                "trade_id": f"sim_{time.time()}"
            }
        
        # Execute with adjusted quantity if needed
        if adjusted_qty != qty:
            logger.warning(f"⚠️ Executing adjusted order quantity: {adjusted_qty:.6f} instead of {qty:.6f}")
        
        return await self.create_order(symbol, side, adjusted_qty, market_price, order_type="market")

    async def create_order(self, symbol: str, side: str, quantity: float,
                          price: Optional[float] = None, 
                          order_type: str = "market") -> Dict[str, Any]:
        """
        Crea una nueva orden en el exchange simulado.
        
        Args:
            symbol: Símbolo a operar (ej: "BTCUSDT")
            side: "buy" o "sell"
            quantity: Cantidad a operar
            price: Precio para órdenes limitadas (None para market)
            order_type: "market", "limit", "stop_loss", "take_profit"
        
        Returns:
            Dict con información de la orden creada
        """
        try:
            # Validar la orden y ajustar cantidad si es necesario
            is_valid, error_msg, adjusted_quantity = self.validate_order(symbol, side, quantity, price)
            if not is_valid:
                logger.warning(f"❌ Orden rechazada: {error_msg}")
                return {
                    'id': None,
                    'status': 'rejected',
                    'error': error_msg,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price
                }
            
            # Crear la orden con la cantidad ajustada
            order_id = f"sim_{self.order_counter:06d}"
            self.order_counter += 1
            
            order = Order(
                id=order_id,
                symbol=symbol,
                side=side.lower(),
                type=OrderType(order_type.lower()),
                quantity=adjusted_quantity,
                price=price
            )
            
            self.orders[order_id] = order
            
            # Procesar la orden inmediatamente si es market
            if order_type.lower() == "market":
                await self._execute_market_order(order)
            elif order_type.lower() == "limit":
                await self._execute_limit_order(order)
            
            logger.info(f"✅ Orden creada: {order_id} - {side.upper()} {adjusted_quantity:.6f} {symbol}")
            
            return {
                'id': order.id,
                'status': order.status.value,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'price': order.price,
                'filled_price': order.filled_price,
                'fees': order.fees,
                'timestamp': order.timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Error creando orden: {e}")
            return {
                'id': None,
                'status': 'error',
                'error': str(e),
                'symbol': symbol,
                'side': side,
                'quantity': quantity
            }

    async def _execute_market_order(self, order: Order):
        """Ejecuta una orden market inmediatamente"""
        current_price = self.get_market_price(order.symbol)
        
        # Calcular slippage
        slippage = self.calculate_slippage(order.symbol, order.quantity, order.side)
        execution_price = current_price * (1 + slippage)
        
        # Calcular comisiones
        order_value = order.quantity * execution_price
        fees = self.calculate_fees(order_value)
        
        # Extract base asset for symbol (e.g., BTCUSDT -> BTC)
        base_asset = order.symbol.replace("USDT", "") if order.symbol != "USDT" else "USDT"
        
        # Actualizar balances
        if order.side == "buy":
            # Comprar: reducir USDT, aumentar base asset
            total_cost = order_value + fees
            self.balances["USDT"] -= total_cost
            self.balances[base_asset] += order.quantity
            
        else:
            # Vender: reducir base asset, aumentar USDT
            proceeds = order_value - fees
            self.balances[base_asset] -= order.quantity
            self.balances["USDT"] += proceeds
        
        # Actualizar orden
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.fees = fees
        
        logger.info(f"🎯 Market order ejecutada: {order.symbol} {order.side} {order.quantity} @ {execution_price:.6f} (fees: {fees:.4f})")

    async def _execute_limit_order(self, order: Order):
        """Ejecuta una orden limitada (puede quedar pendiente)"""
        current_price = self.get_market_price(order.symbol)
        
        # Para órdenes limitadas, verificar si el precio actual cumple con el límite
        if order.side == "buy":
            # Para compras, el precio actual debe ser <= precio límite
            if current_price <= order.price:
                await self._execute_market_order(order)
            else:
                order.status = OrderStatus.PENDING
                logger.info(f"⏳ Limit order pendiente: {order.symbol} BUY {order.quantity} @ {order.price:.6f} (actual: {current_price:.6f})")
                
        else:
            # Para ventas, el precio actual debe ser >= precio límite
            if current_price >= order.price:
                await self._execute_market_order(order)
            else:
                order.status = OrderStatus.PENDING
                logger.info(f"⏳ Limit order pendiente: {order.symbol} SELL {order.quantity} @ {order.price:.6f} (actual: {current_price:.6f})")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancela una orden pendiente"""
        if order_id not in self.orders:
            logger.warning(f"❌ Orden no encontrada para cancelar: {order_id}")
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            logger.warning(f"❌ No se puede cancelar orden {order.status.value}: {order_id}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"❌ Orden cancelada: {order_id}")
        return True

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de una orden específica"""
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        return {
            'id': order.id,
            'status': order.status.value,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'filled_quantity': order.filled_quantity,
            'price': order.price,
            'filled_price': order.filled_price,
            'fees': order.fees,
            'timestamp': order.timestamp
        }

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene órdenes abiertas"""
        open_orders = []
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                if symbol is None or order.symbol == symbol:
                    open_orders.append({
                        'id': order.id,
                        'status': order.status.value,
                        'symbol': order.symbol,
                        'side': order.side,
                        'quantity': order.quantity,
                        'filled_quantity': order.filled_quantity,
                        'price': order.price,
                        'timestamp': order.timestamp
                    })
        return open_orders

    def get_closed_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene órdenes cerradas (filled o cancelled)"""
        closed_orders = []
        for order in self.orders.values():
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
                if symbol is None or order.symbol == symbol:
                    closed_orders.append({
                        'id': order.id,
                        'status': order.status.value,
                        'symbol': order.symbol,
                        'side': order.side,
                        'quantity': order.quantity,
                        'filled_quantity': order.filled_quantity,
                        'price': order.price,
                        'filled_price': order.filled_price,
                        'fees': order.fees,
                        'timestamp': order.timestamp
                    })
        return closed_orders

    def get_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene trades ejecutados (órdenes filled)"""
        trades = []
        for order in self.orders.values():
            if order.status == OrderStatus.FILLED:
                if symbol is None or order.symbol == symbol:
                    trades.append({
                        'id': order.id,
                        'symbol': order.symbol,
                        'side': order.side,
                        'quantity': order.quantity,
                        'price': order.filled_price,
                        'fees': order.fees,
                        'timestamp': order.timestamp
                    })
        return trades

    def get_total_fees(self) -> float:
        """Obtiene el total de comisiones pagadas"""
        return sum(order.fees for order in self.orders.values() if order.status == OrderStatus.FILLED)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del rendimiento"""
        initial_value = sum(
            self.initial_balances.get(asset, 0) * self.get_market_price(asset) 
            if asset != "USDT" else self.initial_balances.get(asset, 0)
            for asset in self.initial_balances.keys()
        )
        
        current_value = sum(
            self.balances.get(asset, 0) * self.get_market_price(asset)
            if asset != "USDT" else self.balances.get(asset, 0)
            for asset in self.balances.keys()
        )
        
        total_trades = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        total_fees = self.get_total_fees()
        
        return {
            'initial_value': initial_value,
            'current_value': current_value,
            'pnl': current_value - initial_value,
            'pnl_percentage': ((current_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0,
            'total_trades': total_trades,
            'total_fees': total_fees,
            'balances': self.balances.copy(),
            'orders_count': len(self.orders)
        }

    def reset(self):
        """Reinicia el cliente a su estado inicial"""
        logger.critical("🚨 FATAL: Attempt to reset SimulatedExchangeClient state - this should never happen in paper mode")
        raise RuntimeError("Resetting SimulatedExchangeClient state is prohibited in paper mode")
        
    @classmethod
    def force_reset(cls, initial_balances: Dict[str, float] = None):
        """Force reset only for testing purposes - should NOT be used in production"""
        # Reset both the instance and the initialized flag
        cls._initialized = False
        if initial_balances:
            cls._instance = cls(initial_balances)
        else:
            cls._instance = cls()
        logger.warning("⚠️ SimulatedExchangeClient forcefully reset - testing only")

    def advance_time(self, steps: int = 1):
        """Avanza el tiempo simulado y actualiza precios"""
        for _ in range(steps):
            for symbol in list(self.market_prices.keys()):
                self.simulate_price_movement(symbol)
    
    def get_price_history(self, symbol: str, limit: int = 100) -> List[float]:
        """Obtiene el historial de precios para un símbolo"""
        if symbol not in self.price_history:
            return []
        return self.price_history[symbol][-limit:]

    async def close(self):
        """Cierra el cliente (no hace nada en simulación)"""
        logger.info("🎮 SimulatedExchangeClient cerrado")


# Ejemplo de uso
if __name__ == "__main__":
    # Crear cliente simulado
    fake_client = SimulatedExchangeClient(
        initial_balances={
            "BTC": 0.01549,
            "ETH": 0.385,
            "USDT": 3000.0
        },
        enable_commissions=True,
        enable_slippage=True
    )
    
    # Simular algunas operaciones
    import asyncio
    
    async def demo():
        print("🎮 Demostración del SimulatedExchangeClient")
        print(f"Balances iniciales: {fake_client.get_account_balances()}")
        
        # Crear algunas órdenes
        await fake_client.create_order("BTCUSDT", "buy", 0.001, order_type="market")
        await fake_client.create_order("ETHUSDT", "sell", 0.1, order_type="market")
        
        # Avanzar tiempo y ver cambios
        fake_client.advance_time(10)
        
        print(f"Balances después de operaciones: {fake_client.get_account_balances()}")
        print(f"Resumen de rendimiento: {fake_client.get_performance_summary()}")
        
        await fake_client.close()
    
    asyncio.run(demo())