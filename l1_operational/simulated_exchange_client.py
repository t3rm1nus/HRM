import asyncio
import time
from typing import Dict, List, Any, Optional
from core.logging import logger

class SimulatedExchangeClient:
    """
    Cliente de intercambio simulado para paper trading.
    
    Este es el punto CLAVE que ahora no tienes.
    
    Debe:
    - mantener balances locales
    - ejecutar órdenes ficticias
    - aplicar slippage + fees
    - guardar historial de trades
    - NUNCA hacer requests HTTP
    """
    
    def __init__(self, initial_balances: Dict[str, float], 
                 fee: float = 0.001, 
                 slippage: float = 0.0005):
        """
        Inicializa el cliente simulado.
        
        Args:
            initial_balances: Balances iniciales (ej: {"USDT": 10000, "BTC": 0})
            fee: Comisión de trading (0.001 = 0.1%)
            slippage: Deslizamiento de precio (0.0005 = 0.05%)
        """
        self.balances = initial_balances.copy()
        self.fee = fee
        self.slippage = slippage
        self.trades = []
        self.order_history = []
        self._trade_id_counter = 1
        
        logger.info(f"✅ SimulatedExchangeClient inicializado")
        logger.info(f"   Balances iniciales: {self.balances}")
        logger.info(f"   Comisión: {fee*100:.2f}%")
        logger.info(f"   Slippage: {slippage*100:.2f}%")
    
    def get_balances(self) -> Dict[str, float]:
        """
        Obtiene balances actuales.
        
        Returns:
            Copia de los balances actuales
        """
        return self.balances.copy()
    
    def execute_order(self, symbol: str, side: str, qty: float, market_price: float) -> Dict[str, Any]:
        """
        Ejecuta una orden ficticia con slippage y fees.
        
        Args:
            symbol: Símbolo de trading (ej: "BTCUSDT")
            side: Lado de la orden ("BUY" o "SELL")
            qty: Cantidad a operar
            market_price: Precio de mercado actual
            
        Returns:
            Diccionario con detalles de la ejecución
        """
        # Validar parámetros
        if qty <= 0:
            raise ValueError(f"Cantidad inválida: {qty}")
        if market_price <= 0:
            raise ValueError(f"Precio inválido: {market_price}")
        if side.upper() not in ["BUY", "SELL"]:
            raise ValueError(f"Lado inválido: {side}")
        
        # Calcular precio con slippage
        if side.upper() == "BUY":
            # En compra, el precio es peor (más alto)
            execution_price = market_price * (1 + self.slippage)
        else:
            # En venta, el precio es peor (más bajo)
            execution_price = market_price * (1 - self.slippage)
        
        # Calcular costos
        cost = qty * execution_price
        fee = cost * self.fee
        
        # Obtener nombres de assets
        base_asset = symbol.replace("USDT", "")
        quote_asset = "USDT"
        
        # Validar suficiente balance antes de ejecutar
        if side.upper() == "BUY":
            required_funds = cost + fee
            if self.balances[quote_asset] < required_funds:
                raise ValueError(f"Fondos insuficientes. Necesita {required_funds}, tiene {self.balances[quote_asset]}")
        else:
            if self.balances[base_asset] < qty:
                raise ValueError(f"Balance insuficiente de {base_asset}. Necesita {qty}, tiene {self.balances[base_asset]}")
        
        # Ejecutar la orden
        trade_id = self._trade_id_counter
        self._trade_id_counter += 1
        
        if side.upper() == "BUY":
            self.balances[quote_asset] -= cost + fee
            self.balances[base_asset] += qty
        else:
            self.balances[base_asset] -= qty
            self.balances[quote_asset] += cost - fee
        
        # Registrar trade
        trade = {
            "trade_id": trade_id,
            "timestamp": time.time(),
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty,
            "market_price": market_price,
            "execution_price": execution_price,
            "fee": fee,
            "cost": cost,
            "slippage_cost": abs(execution_price - market_price) * qty
        }
        
        self.trades.append(trade)
        
        # Registrar en historial de órdenes
        order_record = {
            "order_id": f"simulated_{trade_id}",
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty,
            "price": execution_price,
            "status": "filled",
            "fee": fee,
            "timestamp": trade["timestamp"]
        }
        self.order_history.append(order_record)
        
        # Logging
        logger.info(f"✅ Orden simulada ejecutada: {side.upper()} {qty} {symbol}")
        logger.info(f"   Precio mercado: {market_price:.2f}")
        logger.info(f"   Precio ejecución: {execution_price:.2f} (slippage: {self.slippage*100:.2f}%)")
        logger.info(f"   Comisión: {fee:.2f}")
        logger.info(f"   Costo total: {cost + fee:.2f}")
        logger.info(f"   Balances actuales: {self.balances}")
        
        return trade
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene el historial completo de trades.
        
        Returns:
            Lista de trades ejecutados
        """
        return self.trades.copy()
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Obtiene el historial completo de órdenes.
        
        Returns:
            Lista de órdenes ejecutadas
        """
        return self.order_history.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Genera un resumen de performance de trading.
        
        Returns:
            Diccionario con métricas de performance
        """
        if not self.trades:
            return {"total_trades": 0, "total_fees": 0, "total_slippage_cost": 0}
        
        total_fees = sum(trade["fee"] for trade in self.trades)
        total_slippage_cost = sum(trade["slippage_cost"] for trade in self.trades)
        buy_trades = [t for t in self.trades if t["side"] == "BUY"]
        sell_trades = [t for t in self.trades if t["side"] == "SELL"]
        
        summary = {
            "total_trades": len(self.trades),
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "total_fees": total_fees,
            "total_slippage_cost": total_slippage_cost,
            "average_slippage": total_slippage_cost / len(self.trades) if self.trades else 0,
            "current_balances": self.balances.copy()
        }
        
        return summary
    
    def reset(self, new_balances: Optional[Dict[str, float]] = None):
        """
        Reinicia el cliente con nuevos balances.
        
        Args:
            new_balances: Nuevos balances (si no se proporciona, usa los iniciales)
        """
        if new_balances:
            self.balances = new_balances.copy()
        else:
            # Mantener los balances actuales pero limpiar historial
            pass
        
        self.trades = []
        self.order_history = []
        self._trade_id_counter = 1
        
        logger.info(f"🔄 SimulatedExchangeClient reiniciado")
        logger.info(f"   Balances: {self.balances}")
    
    async def close(self):
        """
        Cierra el cliente (no hace nada en simulación, pero mantiene interfaz).
        """
        logger.info("✅ SimulatedExchangeClient cerrado (simulación)")
    
    # Métodos compatibles con BinanceClient para interoperabilidad
    
    async def get_klines(self, symbol: str, timeframe: str = '1m', limit: int = 50) -> list:
        """
        NO IMPLEMENTADO - Este método no debe ser llamado en modo simulado.
        """
        raise NotImplementedError("get_klines no está disponible en SimulatedExchangeClient. Use RealTimeDataLoader para datos de mercado.")
    
    async def get_ticker_price(self, symbol: str) -> float:
        """
        NO IMPLEMENTADO - Este método no debe ser llamado en modo simulado.
        """
        raise NotImplementedError("get_ticker_price no está disponible en SimulatedExchangeClient. Use RealTimeDataLoader para precios de mercado.")
    
    async def get_account_balances(self) -> Dict[str, float]:
        """
        Obtiene balances de cuenta (compatibilidad con BinanceClient).
        """
        return self.get_balances()
    
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         price: Optional[float] = None, order_type: str = "MARKET") -> Dict[str, Any]:
        """
        Coloca una orden (compatibilidad con BinanceClient).
        
        Args:
            symbol: Símbolo de trading
            side: Lado de la orden
            quantity: Cantidad
            price: Precio (solo para órdenes limitadas, no implementado)
            order_type: Tipo de orden ("MARKET" soportado)
        """
        if order_type.upper() != "MARKET":
            raise NotImplementedError("Solo órdenes MARKET están implementadas en modo simulado")
        
        # Para órdenes market, necesitamos el precio de mercado actual
        # Esto debe ser proporcionado externamente o obtenido de otro cliente
        raise NotImplementedError("Para órdenes MARKET, use execute_order directamente con el precio de mercado")
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancela una orden (no implementado en simulación).
        """
        logger.warning(f"⚠️ Cancelación de orden no implementada en modo simulado: {order_id}")
        return False
    
    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Obtiene órdenes abiertas (no implementado en simulación).
        """
        logger.warning("⚠️ Órdenes abiertas no implementadas en modo simulado")
        return []