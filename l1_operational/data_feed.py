import asyncio
import pandas as pd
from typing import Dict, Any
from core.logging import logger
try:
    from .binance_client import BinanceClient
except ImportError:
    logger.warning("⚠️ No se pudo importar BinanceClient, usando ccxt como fallback")
    BinanceClient = None

class DataFeed:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbols = config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        self.binance_client = None
        self.ccxt_exchange = None
        self._closed = False
        
    async def _init_binance(self):
        """Inicializa el cliente de Binance de forma segura"""
        if not self.binance_client and BinanceClient:
            self.binance_client = BinanceClient(self.config)
        if not self.binance_client:
            try:
                import ccxt.async_support as ccxt
                api_key = config.get('BINANCE_API_KEY', '')
                api_secret = config.get('BINANCE_API_SECRET', '')
                use_testnet = config.get('USE_TESTNET', False)

                options = {
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}  # Para spot trading
                }

                if use_testnet:
                    options['urls'] = {'api': 'https://testnet.binance.vision/api'}
                    options['options']['test'] = True
                    logger.info("✅ Usando Testnet de Binance en ccxt fallback")

                self.ccxt_exchange = ccxt.binance(options)
                if use_testnet:
                    self.ccxt_exchange.set_sandbox_mode(True)  # Habilitar modo sandbox/testnet
                    logger.info("✅ Modo sandbox/testnet habilitado en fallback")

                logger.info("✅ Usando ccxt.binance como fallback para DataFeed")
            except ImportError:
                logger.error("❌ ccxt no instalado. Instale con: pip install ccxt")
                raise ImportError("Falta ccxt para el fallback de BinanceClient")

    async def close(self):
        """Cierra apropiadamente las conexiones"""
        if not self._closed:
            try:
                if self.binance_client:
                    await self.binance_client.close()
                if self.ccxt_exchange:
                    await self.ccxt_exchange.close()
                self._closed = True
                logger.info("✅ DataFeed cerrado correctamente")
            except Exception as e:
                logger.error(f"❌ Error cerrando DataFeed: {e}")
                
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1000) -> pd.DataFrame:
        """
        Obtiene datos OHLCV para un símbolo.
        """
        # Asegurar que tenemos un cliente inicializado
        if not self.binance_client and not self.ccxt_exchange:
            await self._init_binance()
            
        try:
            if self.binance_client:
                data = await self.binance_client.get_klines(symbol, timeframe, limit=limit)
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            else:
                await self.ccxt_exchange.load_markets()
                ohlcv = await self.ccxt_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            logger.debug(f"📊 OHLCV para {symbol}: shape={df.shape}")
            return df

        except ccxt.AuthenticationError as e:
            logger.error(f"❌ Error de autenticación para {symbol}: {str(e)} (verifique claves de testnet)", exc_info=True)
            return pd.DataFrame()
        except ccxt.NetworkError as e:
            logger.error(f"❌ Error de red para {symbol}: {str(e)} (verifique conexión o URLs de testnet)", exc_info=True)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error obteniendo OHLCV para {symbol}: {str(e)}", exc_info=True)
            return pd.DataFrame()

    async def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos de mercado para todos los símbolos.
        """
        try:
            tasks = [self.fetch_ohlcv(symbol) for symbol in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_data = {}
            for symbol, result in zip(self.symbols, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    market_data[symbol] = result
                    logger.info(f"✅ Market data {symbol} shape: {result.shape}")
                else:
                    logger.warning(f"⚠️ No se obtuvieron datos para {symbol}")
            
            if not market_data:
                logger.warning("⚠️ No se obtuvieron datos de mercado válidos")
            
            return market_data

        except Exception as e:
            logger.error(f"❌ Error en get_market_data: {e}", exc_info=True)
            return {}

    async def close(self):
        """
        Cierra conexiones abiertas.
        """
        try:
            if self.binance_client:
                await self.binance_client.close()
            if self.ccxt_exchange:
                await self.ccxt_exchange.close()
            logger.info("✅ Conexiones de DataFeed cerradas")
        except Exception as e:
            logger.error(f"❌ Error cerrando DataFeed: {e}", exc_info=True)
