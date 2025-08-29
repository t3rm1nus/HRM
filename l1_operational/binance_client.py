# l1_operational/binance_client.py
import os
from binance.client import Client
from binance.enums import *
from comms.config import BINANCE_API_KEY, BINANCE_API_SECRET, USE_TESTNET, MODE
import logging

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        self.client = Client(
            BINANCE_API_KEY, 
            BINANCE_API_SECRET,
            testnet=USE_TESTNET
        )
        self.mode = MODE
        logger.info(f"BinanceClient inicializado en modo: {self.mode}, Testnet: {USE_TESTNET}")
    
    def get_klines(self, symbol, interval, limit=100):
        """Obtener datos OHLCV de Binance"""
        try:
            if self.mode == "PAPER":
                # Simular en modo paper o usar datos reales
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                return klines
            else:
                # Modo LIVE - implementar lógica real
                return self._get_real_klines(symbol, interval, limit)
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return None
    
    def _get_real_klines(self, symbol, interval, limit):
        """Lógica para modo LIVE"""
        # Implementar cuando estés listo para trading real
        pass