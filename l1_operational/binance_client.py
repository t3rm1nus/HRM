import asyncio
import aiohttp
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
        self._closed = False
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
            'options': {
                'defaultType': 'spot', 
                'test': True,
                # Configuración de timeouts
                'timeout': 10000,  # 10 segundos
                'connectTimeout': 5000,  # 5 segundos para conexión
                # Reintentos automáticos
                'retry': {
                    'enabled': True,
                    'max': 3,  # Máximo 3 intentos
                    'delay': 1000  # 1 segundo entre intentos
                },
                # Pool de conexiones
                'pool': {
                    'maxsize': 30,  # Máximo 30 conexiones concurrentes
                    'use_dns_cache': True,
                    'ttl_dns_cache': 300  # 5 minutos de cache DNS
                }
            },
            'urls': {'api': 'https://testnet.binance.vision/api'}
        }
        logger.info("✅ Usando Testnet de Binance con URLs configuradas")

        self.exchange = ccxt.binance(options)
        self.exchange.set_sandbox_mode(True)
        
        # Configurar aiohttp específicamente
        import aiohttp
        connector = aiohttp.TCPConnector(
            limit=30,  # Máximo 30 conexiones concurrentes
            ttl_dns_cache=300,  # Cache DNS por 5 minutos
            enable_cleanup_closed=True,  # Limpiar conexiones cerradas
            force_close=False  # Mantener conexiones vivas
        )
        timeout = aiohttp.ClientTimeout(
            total=10,  # Timeout total
            connect=5,  # Timeout de conexión
            sock_read=5  # Timeout de lectura
        )
        self.exchange.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True  # Respetar configuración de proxy del sistema
        )
        
        logger.info("✅ Modo sandbox/testnet habilitado")
        

    async def __aenter__(self):
        """Soporte para context manager async"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup al salir del context manager"""
        await self.close()

    async def close(self):
        """Cierra apropiadamente todas las conexiones"""
        if not self._closed:
            try:
                if self.exchange:
                    if hasattr(self.exchange, 'session') and self.exchange.session and not self.exchange.session.closed:
                        await self.exchange.session.close()
                    await self.exchange.close()
                self._closed = True
                logger.info("✅ BinanceClient cerrado correctamente")
            except Exception as e:
                logger.error(f"❌ Error cerrando BinanceClient: {e}")

    async def get_klines(self, symbol: str, timeframe: str = '1m', limit: int = 50) -> list:
        """
        Obtiene datos OHLCV para un símbolo con manejo robusto de errores y reintentos.
        """
        max_retries = 3
        retry_delay = 1  # segundos
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.warning(f"Reintento #{attempt} para {symbol}...")
                    await asyncio.sleep(retry_delay * attempt)  # Backoff exponencial
                
                # Verificar y reiniciar sesión si es necesario
                if self.exchange.session is None or self.exchange.session.closed:
                    logger.warning("Sesión cerrada, reiniciando...")
                    await self.exchange.close()
                    # Recrear el exchange con las mismas opciones
                    self.exchange = ccxt.binance(self.exchange.options)
                    self.exchange.set_sandbox_mode(True)
                
                await self.exchange.load_markets()
                logger.info(f"Solicitando OHLCV: symbol={symbol}, timeframe={timeframe}, limit={limit}")
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                # Validar respuesta
                if isinstance(ohlcv, list) and len(ohlcv) > 0 and isinstance(ohlcv[0], list):
                    logger.debug(f"📊 Klines para {symbol}: {len(ohlcv)} filas")
                    return ohlcv
                    
                # Manejar respuesta inválida
                logger.error(f"❌ Respuesta OHLCV inválida: {repr(ohlcv)}")
                if isinstance(ohlcv, dict) and 'msg' in ohlcv:
                    logger.error(f"❌ Mensaje de Binance: {ohlcv['msg']}")
                    
            except ccxt.AuthenticationError as e:
                logger.error(f"❌ Error de autenticación para {symbol}: {str(e)} (verifique claves de testnet)")
                return []  # Error de auth no se reintenta
                
            except (ccxt.NetworkError, ccxt.ExchangeError, aiohttp.ClientError) as e:
                if "10053" in str(e):  # WinError 10053
                    logger.warning(f"Conexión abortada (10053), reintentando... {str(e)}")
                else:
                    logger.error(f"Error de red en intento {attempt+1}/{max_retries}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("❌ Máximo de reintentos alcanzado")
                    return []
                    
            except Exception as e:
                logger.error(f"❌ Error inesperado: {str(e)}", exc_info=True)
                return []
                
        return []  # Si llegamos aquí, todos los intentos fallaron
        
    async def close(self):
        """Cierra apropiadamente las conexiones del cliente."""
        try:
            if self.exchange:
                if self.exchange.session and not self.exchange.session.closed:
                    await self.exchange.session.close()
                await self.exchange.close()
                logger.info("Cliente cerrado correctamente")
        except Exception as e:
            logger.error(f"Error cerrando el cliente: {e}", exc_info=True)
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