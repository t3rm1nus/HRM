import ccxt.async_support as ccxt
from core.logging import logger
try:
    from comms.config import config
except ImportError:
    logger.error("❌ No se pudo importar config desde comms.config")
    config = {
        "BINANCE_API_KEY": "",
        "BINANCE_API_SECRET": "",
        "USE_TESTNET": False
    }

class BinanceClient:
    def __init__(self, config_dict: dict = None):
        self.config = config_dict or config
        api_key = self.config.get('BINANCE_API_KEY', '')
        api_secret = self.config.get('BINANCE_API_SECRET', '')
        use_testnet = self.config.get('USE_TESTNET', False)

        options = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}  # Para spot trading
        }

        if use_testnet:
            options['urls'] = {'api': 'https://testnet.binance.vision/api'}
            options['options']['test'] = True
            logger.info("✅ Usando Testnet de Binance con URLs configuradas")

        self.exchange = ccxt.binance(options)
        if use_testnet:
            self.exchange.set_sandbox_mode(True)  # Habilitar modo sandbox/testnet
            logger.info("✅ Modo sandbox/testnet habilitado")

        logger.info("✅ BinanceClient inicializado")

    async def get_klines(self, symbol: str, timeframe: str = '1m', limit: int = 50) -> list:
        """
        Obtiene datos OHLCV para un símbolo.
        """
        try:
            await self.exchange.load_markets()
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            logger.debug(f"📊 Klines para {symbol}: {len(ohlcv)} filas")
            return ohlcv
        except ccxt.AuthenticationError as e:
            logger.error(f"❌ Error de autenticación para {symbol}: {str(e)} (verifique claves de testnet)", exc_info=True)
            return []
        except ccxt.NetworkError as e:
            logger.error(f"❌ Error de red para {symbol}: {str(e)} (verifique conexión o URLs de testnet)", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"❌ Error obteniendo klines para {symbol}: {str(e)}", exc_info=True)
            return []

    async def close(self):
        """
        Cierra la conexión.
        """
        try:
            await self.exchange.close()
            logger.info("✅ BinanceClient cerrado")
        except Exception as e:
            logger.error(f"❌ Error cerrando BinanceClient: {e}", exc_info=True)