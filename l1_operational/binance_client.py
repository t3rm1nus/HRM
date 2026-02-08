import asyncio
import aiohttp
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Any
from core.logging import logger

try:
    from comms.config import config
    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)
    config = {
        "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
        "BINANCE_API_SECRET": os.getenv("BINANCE_API_SECRET"),
        "USE_TESTNET": os.getenv("USE_TESTNET", "false").lower() == "true",
        "PAPER_MODE": os.getenv("PAPER_MODE", "true").lower() == "true"
    }
except ImportError:
    logger.error("❌ No se pudo importar config desde comms.config")
    config = {
        "BINANCE_API_KEY": "",
        "BINANCE_API_SECRET": "",
        "USE_TESTNET": False,
        "PAPER_MODE": True
    }


class BinanceClient:
    def __init__(self, config_dict: dict = None):
        from dotenv import load_dotenv
        import os
        load_dotenv(override=True)
        self._closed = False
        self.config = config_dict or {
            "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
            "BINANCE_API_SECRET": os.getenv("BINANCE_API_SECRET"),
            "USE_TESTNET": os.getenv("USE_TESTNET", "false").lower() == "true",
            "PAPER_MODE": os.getenv("PAPER_MODE", "true").lower() == "true"
        }
        api_key = self.config.get('BINANCE_API_KEY', '')
        api_secret = self.config.get('BINANCE_API_SECRET', '')
        self.paper_mode = self.config.get('PAPER_MODE', True)
        use_testnet = self.config.get('USE_TESTNET', False) and not self.paper_mode

        logger.info(f"Inicializando BinanceClient: paper_mode={self.paper_mode}, use_testnet={use_testnet}")

        self.simulated_client = None
        if self.paper_mode:
            logger.info("🧪 Paper trading with simulated execution")
            from l1_operational.simulated_exchange_client import SimulatedExchangeClient
            initial_balances = {"BTC": 0.01549, "ETH": 0.385, "USDT": 3000.0}
            self.simulated_client = SimulatedExchangeClient.initialize_once(initial_balances)
        elif not api_key or not api_secret:
            logger.warning("⚠️ Claves API no configuradas - usando modo simulado")
            self.paper_mode = True
            from l1_operational.simulated_exchange_client import SimulatedExchangeClient
            initial_balances = {"BTC": 0.01549, "ETH": 0.385, "USDT": 3000.0}
            self.simulated_client = SimulatedExchangeClient.initialize_once(initial_balances)

        urls = {
            'api': 'https://testnet.binance.vision/api' if use_testnet else 'https://api.binance.com/api',
            'test': 'https://testnet.binance.vision/api' if use_testnet else 'https://api.binance.com/api'
        }

        options = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot', 'test': use_testnet},
            'urls': urls
        }

        self.exchange = ccxt.binance(options)
        self.exchange.set_sandbox_mode(use_testnet)

        connector = aiohttp.TCPConnector(limit=30, ttl_dns_cache=300, enable_cleanup_closed=True, force_close=False)
        timeout = aiohttp.ClientTimeout(total=10, connect=5, sock_read=5)
        self.exchange.session = aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True)

        self.public_exchange = ccxt.binance({
            'apiKey': '', 'secret': '', 'enableRateLimit': True,
            'options': {'defaultType': 'spot', 'test': False},
            'urls': {'api': 'https://api.binance.com/api', 'test': 'https://api.binance.com/api'}
        })
        self.public_exchange.set_sandbox_mode(False)
        self.public_exchange.session = aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True)

        logger.info(f"✅ Modo {'sandbox/testnet' if use_testnet else 'mainnet'} habilitado")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Cierra apropiadamente todas las conexiones y sesiones aiohttp"""
        if not self._closed:
            try:
                if self.exchange:
                    if hasattr(self.exchange, 'session') and self.exchange.session:
                        try:
                            if not self.exchange.session.closed:
                                await self.exchange.session.close()
                                logger.info("✅ Sesión exchange principal cerrada")
                        except Exception as session_err:
                            logger.warning(f"⚠️ Error cerrando sesión exchange: {session_err}")
                    try:
                        await self.exchange.close()
                    except Exception as exchange_err:
                        logger.warning(f"⚠️ Error cerrando exchange: {exchange_err}")
                
                if self.public_exchange:
                    if hasattr(self.public_exchange, 'session') and self.public_exchange.session:
                        try:
                            if not self.public_exchange.session.closed:
                                await self.public_exchange.session.close()
                                logger.info("✅ Sesión public_exchange cerrada")
                        except Exception as session_err:
                            logger.warning(f"⚠️ Error cerrando sesión public_exchange: {session_err}")
                    try:
                        await self.public_exchange.close()
                    except Exception as pub_err:
                        logger.warning(f"⚠️ Error cerrando public_exchange: {pub_err}")
                
                self._closed = True
                logger.info("✅ BinanceClient cerrado correctamente")
            except Exception as e:
                logger.error(f"❌ Error cerrando BinanceClient: {e}")

    async def get_klines(self, symbol: str, timeframe: str = '1m', limit: int = 50) -> list:
        logger.info(f"📊 Obteniendo datos OHLCV para {symbol} (mainnet público)")
        return await self._get_real_klines(symbol, timeframe, limit)

    async def _get_real_klines(self, symbol: str, timeframe: str, limit: int) -> list:
        import aiohttp
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.warning(f"Reintento #{attempt} para {symbol}...")
                    await asyncio.sleep(retry_delay * attempt)

                url = f"https://api.binance.com/api/v3/klines"
                params = {'symbol': symbol.upper(), 'interval': timeframe, 'limit': limit}
                logger.info(f"Solicitando OHLCV: symbol={symbol}, timeframe={timeframe}, limit={limit}")

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                                logger.debug(f"📊 Klines para {symbol}: {len(data)} filas")
                                return data
                            else:
                                logger.error(f"❌ Respuesta OHLCV inválida: {repr(data)}")
                        else:
                            logger.error(f"❌ Error HTTP {response.status} al obtener klines")
            except aiohttp.ClientError as e:
                if "10053" in str(e):
                    logger.warning(f"Conexión abortada (10053), reintentando...")
                else:
                    logger.error(f"Error de red en intento {attempt+1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("❌ Máximo de reintentos alcanzado")
                    return []
            except Exception as e:
                logger.error(f"❌ Error inesperado: {str(e)}", exc_info=True)
                return []
        return []

    def _get_mock_klines(self, symbol: str, timeframe: str, limit: int) -> list:
        import random
        import time
        base_prices = {'BTCUSDT': 50000.0, 'ETHUSDT': 3000.0, 'BNBUSDT': 300.0, 'SOLUSDT': 100.0}
        base_price = base_prices.get(symbol, 50000.0)
        current_time = int(time.time() * 1000)
        timeframe_ms = {'1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000, '1h': 3600000}.get(timeframe, 60000)

        klines = []
        current_price = base_price
        for i in range(limit):
            price_change = random.uniform(-0.02, 0.02) * current_price
            current_price = max(0.01, current_price + price_change)
            timestamp = current_time - (limit - i - 1) * timeframe_ms
            klines.append([
                timestamp,
                float(current_price),
                float(current_price * random.uniform(1.0, 1.02)),
                float(current_price * random.uniform(0.98, 1.0)),
                float(random.uniform(current_price * 0.98, current_price * 1.02)),
                float(random.uniform(1.0, 100.0)),
            ])
        return klines

    def _get_timeframe_ms(self, timeframe: str) -> int:
        return {'1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000, '1h': 3600000}.get(timeframe, 60000)

    async def get_ticker_price(self, symbol: str) -> float:
        logger.info(f"💰 Obteniendo precio actual para {symbol} (mainnet público)")
        return await self._get_real_price(symbol)

    async def _get_real_price(self, symbol: str) -> float:
        try:
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': symbol.upper()}
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'price' in data:
                            return float(data['price'])
                    else:
                        logger.error(f"❌ Error HTTP {response.status} al obtener precio")
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error obteniendo precio para {symbol}: {str(e)}")
            return 0.0

    def _get_mock_price(self, symbol: str) -> float:
        import random
        base_prices = {'BTCUSDT': 50000.0, 'ETHUSDT': 3000.0}
        base_price = base_prices.get(symbol, 50000.0)
        return max(0.01, base_price + random.uniform(-0.02, 0.02) * base_price)

    async def get_account_balances(self) -> Dict[str, float]:
        if self.paper_mode:
            if self.simulated_client:
                return self.simulated_client.get_balances()
            return {}
        try:
            account = await self.exchange.fetch_balance()
            balances = {}
            for key in ['free', 'used']:
                if key in account:
                    for asset, amount in account[key].items():
                        if amount > 0.00000001:
                            balances[asset] = balances.get(asset, 0) + amount
            return balances
        except Exception as e:
            logger.error(f"❌ Error obteniendo balances: {e}")
            return {}

    def force_reset(self, initial_balances: Dict[str, float] = None):
        if self.paper_mode and self.simulated_client:
            self.simulated_client.force_reset(initial_balances or {"BTC": 0.01549, "ETH": 0.385, "USDT": 3000.0})

    async def place_stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price: float, limit_price: Optional[float] = None) -> Dict[str, Any]:
        if self.paper_mode:
            return {'id': f'simulated_sl_{symbol}_{side}', 'status': 'simulated', 'symbol': symbol, 'side': side, 'quantity': quantity, 'stop_price': stop_price, 'limit_price': limit_price}
        return {}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.paper_mode:
            return True
        return False

    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        if self.paper_mode:
            return []
        return []

    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float, stop_price: Optional[float] = None, order_type: str = "LIMIT") -> Dict[str, Any]:
        if self.paper_mode:
            return {'id': f'simulated_limit_{symbol}_{side}_{price:.6f}', 'status': 'simulated', 'symbol': symbol, 'side': side, 'quantity': quantity, 'price': price}
        return {}
