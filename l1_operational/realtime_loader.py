import asyncio
import pandas as pd
from typing import Dict, Any
from core.logging import logger
from comms.config import config
from .binance_client import BinanceClient

class RealTimeDataLoader:
    def __init__(self, config_dict: dict):
        self.config = config_dict
        self.symbols = self.config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        self.binance_client = BinanceClient(self.config)
        logger.info("✅ RealTimeDataLoader inicializado")

    async def fetch_realtime_data(self, symbol: str, timeframe: str = '1m', limit: int = 50) -> pd.DataFrame:
        """
        Obtiene datos OHLCV en tiempo real para un símbolo.
        """
        try:
            data = await self.binance_client.get_klines(symbol, timeframe, limit)
            if not data:
                logger.warning(f"⚠️ No se obtuvieron datos para {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            logger.debug(f"📊 Datos en tiempo real para {symbol}: shape={df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Error obteniendo datos en tiempo real para {symbol}: {str(e)} (detalle completo)", exc_info=True)
            return pd.DataFrame()

    async def get_realtime_data(self) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos en tiempo real para todos los símbolos.
        """
        try:
            tasks = [self.fetch_realtime_data(symbol) for symbol in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_data = {}
            for symbol, result in zip(self.symbols, results):
                if isinstance(result, pd.DataFrame) and not result.empty:
                    market_data[symbol] = result
                    logger.info(f"✅ Datos en tiempo real {symbol} shape: {result.shape}")
                else:
                    logger.warning(f"⚠️ No se obtuvieron datos para {symbol}")
            
            return market_data
        except Exception as e:
            logger.error(f"❌ Error en get_realtime_data: {e}", exc_info=True)
            return {}

    async def close(self):
        """
        Cierra conexiones abiertas.
        """
        try:
            await self.binance_client.close()
            logger.info("✅ RealTimeDataLoader cerrado")
        except Exception as e:
            logger.error(f"❌ Error cerrando RealTimeDataLoader: {e}", exc_info=True)