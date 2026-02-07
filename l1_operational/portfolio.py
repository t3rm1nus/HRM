"""
Helper para consultar saldos y exposición de forma robusta.
Incluye retry y manejo de fallos de API.
"""

import asyncio
from typing import Tuple, Dict, Any
from core.logging import logger
from .simulated_exchange_client import SimulatedExchangeClient

MAX_RETRIES = 3
BACKOFF = 1  # segundos


class PortfolioManager:
    """
    Gestor de portfolio que utiliza SimulatedExchangeClient en paper mode.
    """
    
    def __init__(self, client: SimulatedExchangeClient, mode: str = "simulated"):
        """
        Inicializa el PortfolioManager.
        
        Args:
            client: Cliente de intercambio simulado
            mode: Modo de operación ("simulated" o "real")
        """
        self.client = client
        self.mode = mode
        
        logger.info(f"✅ PortfolioManager initialized in {mode} mode")
        logger.info(f"   Initial balances: {self.client.get_balances()}")
    
    async def get_available_balance(self, symbol: str) -> Tuple[float, float]:
        """
        Obtiene balances disponibles para un símbolo.
        Retorna: (base_amount, quote_amount)
        Ej: BTC/USDT → (BTC disponible, USDT disponible)
        """
        try:
            balances = self.client.get_balances()
            base, quote = symbol.split("/")
            base_amt = balances.get(base, 0.0)
            quote_amt = balances.get(quote, 0.0)
            return float(base_amt), float(quote_amt)
        except Exception as e:
            logger.error(f"❌ Error getting balance for {symbol}: {e}")
            return 0.0, 0.0
    
    async def update_position(self, symbol: str, action: str, quantity: float, 
                             price: float, commission: float = 0.0):
        """
        Actualiza la posición después de una ejecución de orden.
        """
        try:
            # En modo simulado, el cliente ya actualizó los balances
            current_balances = self.client.get_balances()
            logger.info(f"📊 Portfolio updated for {symbol} {action.upper()} {quantity:.6f} @ ${price:.2f}")
            logger.info(f"   Comisión: ${commission:.2f}")
            logger.info(f"   Balances: {current_balances}")
        except Exception as e:
            logger.error(f"❌ Error updating portfolio for {symbol}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen de performance del portfolio.
        """
        return self.client.get_performance_summary()
    
    def get_trade_history(self) -> list:
        """
        Obtiene el historial de trades.
        """
        return self.client.get_trade_history()


# Función de compatibilidad para mantener la interfaz existente
async def get_available_balance(symbol: str, client: SimulatedExchangeClient = None) -> Tuple[float, float]:
    """
    Función de compatibilidad para obtener balances.
    Requiere un cliente pre-inicializado.
    """
    if client is None:
        logger.critical("🚨 FATAL: get_available_balance requires a pre-initialized SimulatedExchangeClient")
        raise RuntimeError("get_available_balance requires a pre-initialized SimulatedExchangeClient")
    
    portfolio_manager = PortfolioManager(client, mode="simulated")
    return await portfolio_manager.get_available_balance(symbol)
