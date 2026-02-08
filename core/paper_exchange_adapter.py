from typing import Dict, List, Optional, Any
from core.exchange_adapter import ExchangeAdapter
from l1_operational.binance_client import BinanceClient
from core.logging import logger


class PaperExchangeAdapter(ExchangeAdapter):
    """
    Paper trading adapter that uses BinanceClient as backend for real market data
    while maintaining paper trading logic (no real orders executed).
    
    This adapter provides:
    - Real market data from Binance testnet
    - Paper trading simulation (no real orders)
    - Testnet validation
    - Mock balances for paper trading
    """

    def __init__(self, use_testnet: bool = True):
        """
        Initialize the paper trading adapter with Binance backend.
        
        Args:
            use_testnet: Force testnet mode for safety
        """
        self.use_testnet = use_testnet
        self._binance_client = None
        self._initialized = False
        
        # Paper trading balances (simulated)
        self._paper_balances = {
            "BTC": 0.01538,
            "ETH": 0.381,
            "USDT": 897.9
        }
        
        # Track paper positions for simulation
        self._paper_positions = self._paper_balances.copy()

    async def initialize(self):
        """Initialize the Binance client backend."""
        if self._initialized:
            return
            
        try:
            # Create Binance client with testnet configuration
            self._binance_client = BinanceClient()
            
            # Validate testnet mode
            if not self._is_testnet_configured():
                raise ValueError("Binance client not configured for testnet - unsafe for paper trading")
            
            self._initialized = True
            logger.info("✅ PaperExchangeAdapter initialized with Binance testnet backend")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize PaperExchangeAdapter: {e}")
            raise

    def _is_testnet_configured(self) -> bool:
        """Validate that Binance client is configured for testnet."""
        if not self._binance_client:
            return False
            
        # Check if testnet is enabled in config
        config = getattr(self._binance_client, 'config', {})
        use_testnet = config.get('USE_TESTNET', False)
        
        if not use_testnet:
            logger.error("❌ Binance client not in testnet mode - unsafe for paper trading")
            return False
            
        return True

    async def get_balances(self) -> Dict[str, float]:
        """
        Get paper trading balances (simulated, not real).
        
        Returns:
            Dict[str, float]: Paper account balances
        """
        if not self._initialized:
            await self.initialize()
            
        return self._paper_balances.copy()

    async def get_price(self, symbol: str) -> float:
        """
        Get real market price from Binance testnet.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            float: Real market price from Binance testnet
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Get real price from Binance testnet
            klines = await self._binance_client.get_klines(symbol, timeframe='1m', limit=1)
            if klines and len(klines) > 0:
                # Return close price of latest candle
                return float(klines[-1][4])  # Close price
            else:
                logger.warning(f"⚠️ No price data available for {symbol}, using fallback")
                return 1000.0
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting price from Binance: {e}, using fallback")
            # Fallback prices
            fallback_prices = {
                "BTCUSDT": 52000.0,
                "ETHUSDT": 3200.0,
                "BTCETH": 16.25
            }
            return fallback_prices.get(symbol, 1000.0)

    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate placing a trading order (paper trading only).
        
        Args:
            order: Order parameters dictionary
            
        Returns:
            Dict[str, Any]: Simulated order execution result
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Get real market price for simulation
            symbol = order.get("symbol", "BTCUSDT")
            current_price = await self.get_price(symbol)
            
            # Simulate order execution with real market price
            execution_price = current_price
            quantity = order.get("quantity", 0.0)
            side = order.get("side", "BUY")
            
            # Update paper balances based on simulated execution
            await self._update_paper_balances(symbol, side, quantity, execution_price)
            
            result = {
                "status": "FILLED",
                "price": execution_price,
                "quantity": quantity,
                "symbol": symbol,
                "side": side,
                "type": "PAPER_TRADE"
            }
            
            logger.info(f"📝 Paper order executed: {side} {quantity} {symbol} @ {execution_price}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in paper order simulation: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "symbol": order.get("symbol", "UNKNOWN")
            }

    async def _update_paper_balances(self, symbol: str, side: str, quantity: float, price: float):
        """
        Update paper trading balances based on simulated order execution.
        
        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Order quantity
            price: Execution price
        """
        if side.upper() == "BUY":
            # Buy: reduce USDT, increase asset
            cost = quantity * price
            if self._paper_balances["USDT"] >= cost:
                self._paper_balances["USDT"] -= cost
                self._paper_balances[symbol.replace("USDT", "")] += quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = self._paper_balances["USDT"] / price
                logger.warning(f"🧪 PAPER MODE: Adjusting buy quantity - requested {quantity:.6f} exceeds available USDT, using {max_quantity:.6f}")
                if max_quantity > 0:
                    self._paper_balances["USDT"] -= max_quantity * price
                    self._paper_balances[symbol.replace("USDT", "")] += max_quantity
                
        elif side.upper() == "SELL":
            # Sell: reduce asset, increase USDT
            asset = symbol.replace("USDT", "")
            if self._paper_balances[asset] >= quantity:
                proceeds = quantity * price
                self._paper_balances[asset] -= quantity
                self._paper_balances["USDT"] += proceeds
            else:
                logger.warning(f"🧪 PAPER MODE: Adjusting sell quantity - requested {quantity:.6f} exceeds available {asset}, using {self._paper_balances[asset]:.6f}")
                if self._paper_balances[asset] > 0:
                    proceeds = self._paper_balances[asset] * price
                    self._paper_balances[asset] = 0.0
                    self._paper_balances["USDT"] += proceeds

    async def cancel_order(self, order_id: str) -> bool:
        """
        Simulate cancelling an order (always successful in paper trading).
        
        Args:
            order_id: Order identifier
            
        Returns:
            bool: True (always successful in paper trading)
        """
        logger.info(f"📝 Paper order cancelled: {order_id}")
        return True

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get mock list of open orders (paper trading).
        
        Returns:
            List[Dict[str, Any]]: Empty list (no real open orders in paper trading)
        """
        # In paper trading, we don't have real open orders
        # All orders are simulated as filled immediately
        return []

    async def sync_positions(self) -> Dict[str, float]:
        """
        Get current paper trading positions.
        
        Returns:
            Dict[str, float]: Current paper position quantities
        """
        if not self._initialized:
            await self.initialize()
            
        return self._paper_balances.copy()

    async def get_real_balances(self) -> Dict[str, float]:
        """
        Get real balances from Binance testnet (for monitoring only).
        
        Returns:
            Dict[str, float]: Real testnet balances
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            real_balances = await self._binance_client.get_account_balances()
            logger.info("📊 Real testnet balances retrieved")
            return real_balances
        except Exception as e:
            logger.warning(f"⚠️ Could not get real balances: {e}")
            return {}

    async def close(self):
        """Clean up resources."""
        if self._binance_client:
            await self._binance_client.close()
        self._initialized = False
        logger.info("🧹 PaperExchangeAdapter closed")
