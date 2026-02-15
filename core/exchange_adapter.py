from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class ExchangeAdapter(ABC):
    """
    Abstract base class for exchange adapters.
    
    Provides a unified interface for all exchange operations across different modes.
    No implementation logic should be present in this class - only method signatures.
    """

    @abstractmethod
    async def get_balances(self) -> Dict[str, float]:
        """
        Get account balances from the exchange.
        
        Returns:
            Dict[str, float]: Dictionary of asset balances
            
        Note: This method is async and MUST be awaited
        """
        pass

    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Current market price
            
        Note: This method is async and MUST be awaited
        """
        pass

    @abstractmethod
    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a trading order.
        
        Args:
            order: Order parameters dictionary
            
        Returns:
            Dict[str, Any]: Order execution result
            
        Note: This method is async and MUST be awaited
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            bool: True if successfully cancelled, False otherwise
            
        Note: This method is async and MUST be awaited
        """
        pass

    @abstractmethod
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get list of open orders.
        
        Returns:
            List[Dict[str, Any]]: List of open orders
            
        Note: This method is async and MUST be awaited
        """
        pass

    @abstractmethod
    async def sync_positions(self) -> Dict[str, float]:
        """
        Synchronize current positions with exchange.
        
        Returns:
            Dict[str, float]: Current position quantities
            
        Note: This method is async and MUST be awaited
        """
        pass

    @abstractmethod
    def get_balance_sync(self, asset: str) -> float:
        """
        Synchronous version of get_balances for a single asset.
        Should ONLY be called from sync contexts.
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH', 'USDT')
            
        Returns:
            float: Balance for the specified asset
            
        Note: This method is sync and should NOT be called from async contexts
        """
        pass

    @abstractmethod
    def get_price_sync(self, symbol: str) -> float:
        """
        Synchronous version of get_price.
        Should ONLY be called from sync contexts.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Current market price
            
        Note: This method is sync and should NOT be called from async contexts
        """
        pass