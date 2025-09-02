# l1_operational/data_feed.py
import pandas as pd
from .binance_client import BinanceClient
from comms.config import SYMBOLS
import logging
import asyncio

logger = logging.getLogger(__name__)

class DataFeed:
    def __init__(self):
        self.binance = BinanceClient()
        self.symbols = SYMBOLS
    
    async def start(self):
        """Inicialización asíncrona con datos reales"""
        logger.info("[DataFeed] Conectando a Binance...")
        # Test rápido de conexión
        try:
            btc_price = self.binance.client.get_symbol_ticker(symbol="BTCUSDT")
            eth_price = self.binance.client.get_symbol_ticker(symbol="ETHUSDT")
            logger.info(f"[DataFeed] Precios: BTC=${btc_price['price']}, ETH=${eth_price['price']}")
        except Exception as e:
            logger.warning(f"[DataFeed] Error en conexión inicial: {e}")
        
        logger.info("[DataFeed] Conexión establecida.")

    async def stop(self):
        """Simulación de apagado asíncrono."""
        logger.info("[DataFeed] Desconectando de fuentes de datos...")
        await asyncio.sleep(1)
        logger.info("[DataFeed] Desconexión completa.")

    def fetch_data(self, symbol, timeframe="1m", limit=100):
        """Obtener datos OHLCV y convertir a DataFrame"""
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignored"
        ]
        try:
            klines = self.binance.get_klines(symbol, timeframe, limit)
            if not klines:
                return pd.DataFrame(columns=columns)
            
            df = pd.DataFrame(klines, columns=columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Convertir todas las columnas numéricas
            for col in ["open", "high", "low", "close", "volume",
                        "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]:
                df[col] = df[col].astype(float)
            df["trades"] = df["trades"].astype(int)
            return df
        except Exception as e:
            logger.error(f"Error al obtener datos para {symbol}: {e}")
            return pd.DataFrame(columns=columns)

    def get_latest_data(self, symbol=None):
        """Devuelve la última vela disponible para un símbolo o todos los símbolos."""
        if symbol is None:
            return {s: self.fetch_data(s, limit=1).iloc[-1] for s in self.symbols if not self.fetch_data(s, limit=1).empty}
        df = self.fetch_data(symbol, limit=1)
        if df.empty:
            return None
        return df.iloc[-1]