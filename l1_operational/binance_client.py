import ccxt.async_support as ccxt
from core.logging import logger
try:
    from comms.config import config
    from dotenv import load_dotenv
    import os
    
    # Forzar recarga del .env
    load_dotenv(override=True)
    
    # Sobreescribir la configuración con valores del .env
    config = {
        "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
        "BINANCE_API_SECRET": os.getenv("BINANCE_API_SECRET"),
        "USE_TESTNET": os.getenv("USE_TESTNET", "false").lower() == "true"  # Forzar evaluación explícita
    }
except ImportError:
    logger.error("❌ No se pudo importar config desde comms.config")
    config = {
        "BINANCE_API_KEY": "",
        "BINANCE_API_SECRET": "",
        "USE_TESTNET": False
    }

class BinanceClient:
    def __init__(self, config_dict: dict = None):
        from dotenv import load_dotenv
        import os
        load_dotenv(override=True)
        self.config = config_dict or {
            "BINANCE_API_KEY": os.getenv("BINANCE_API_KEY"),
            "BINANCE_API_SECRET": os.getenv("BINANCE_API_SECRET"),
            "USE_TESTNET": os.getenv("USE_TESTNET", "false").lower() == "true"
        }
        api_key = self.config.get('BINANCE_API_KEY', '')
        api_secret = self.config.get('BINANCE_API_SECRET', '')
        # Forzar testnet siempre
        use_testnet = True

        logger.info(f"Inicializando BinanceClient con: api_key={'SET' if api_key else 'NOT SET'}, api_secret={'SET' if api_secret else 'NOT SET'}, use_testnet={use_testnet}")

        options = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot', 'test': True},
            'urls': {'api': 'https://testnet.binance.vision/api'}
        }
        logger.info("✅ Usando Testnet de Binance con URLs configuradas")

        self.exchange = ccxt.binance(options)
        self.exchange.set_sandbox_mode(True)
        logger.info("✅ Modo sandbox/testnet habilitado")

        logger.info(f"✅ BinanceClient inicializado con opciones: {options}")

    async def get_klines(self, symbol: str, timeframe: str = '1m', limit: int = 50) -> list:
        """
        Obtiene datos OHLCV para un símbolo. Loguea la respuesta cruda para diagnóstico.
        """
        try:
            await self.exchange.load_markets()
            logger.info(f"Solicitando OHLCV: symbol={symbol}, timeframe={timeframe}, limit={limit}")
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            # Si la respuesta es una lista, procesar normalmente
            if isinstance(ohlcv, list) and len(ohlcv) > 0 and isinstance(ohlcv[0], list):
                logger.debug(f"📊 Klines para {symbol}: {len(ohlcv)} filas")
                return ohlcv
            # Si la respuesta es un dict con error, loguear y devolver []
            if isinstance(ohlcv, dict):
                logger.error(f"❌ Error en respuesta de Binance: {repr(ohlcv)}")
                if 'msg' in ohlcv:
                    logger.error(f"❌ Mensaje de Binance: {ohlcv['msg']}")
                return []
            # Si la respuesta es vacía o inesperada
            logger.error(f"❌ Respuesta OHLCV inesperada: {repr(ohlcv)}")
            return []
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