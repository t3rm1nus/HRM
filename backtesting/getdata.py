#!/backtesting/getdata.py
"""
Recolector de Datos de Binance para HRM Backtesting
Conecta directamente a la API de Binance para obtener datos históricos
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import time
import hmac
import hashlib
from urllib.parse import urlencode

from core.logging import logger  # ✅ Usar logger centralizado


class BinanceDataCollector:
    """Recolector de datos históricos de Binance"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', True)
        
        # URLs base
        if self.testnet:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"
            
        # ✅ Usar logger centralizado
        self.logger = logger
        
        # Cache para evitar requests repetidos
        self.data_cache = {}
        
        # Rate limiting
        self.last_request = 0
        self.request_interval = 0.1  # 100ms entre requests

    async def collect_historical_data(self, symbols: List[str], start_date: str, 
                                    end_date: str, intervals: List[str]) -> Dict:
        """Recolectar datos históricos para múltiples símbolos e intervalos"""
        
        self.logger.info(f"Recolectando datos para {len(symbols)} símbolos, {len(intervals)} intervalos")
        
        data = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                data[symbol] = {}
                
                for interval in intervals:
                    self.logger.info(f"  Obteniendo {symbol} {interval}...")
                    
                    cache_key = f"{symbol}_{interval}_{start_date}_{end_date}"
                    if cache_key in self.data_cache:
                        self.logger.info(f"    Usando datos de cache para {cache_key}")
                        data[symbol][interval] = self.data_cache[cache_key]
                        continue
                        
                    try:
                        klines = await self._get_historical_klines(
                            session, symbol, interval, start_date, end_date
                        )
                        
                        if klines:
                            df = self._process_klines_to_dataframe(klines)
                            df = self._add_technical_indicators(df, symbol, interval)
                            data[symbol][interval] = df
                            self.data_cache[cache_key] = df
                            
                            self.logger.info(f"    ✅ {len(df)} velas obtenidas")
                        else:
                            self.logger.warning(f"    ⚠️  Sin datos para {symbol} {interval}")
                            
                    except Exception as e:
                        self.logger.error(f"    ❌ Error obteniendo {symbol} {interval}: {e}", exc_info=True)
                        
                    # Rate limiting
                    await asyncio.sleep(self.request_interval)
        
        # Añadir datos adicionales
        data = await self._enrich_data(data)
        
        self.logger.info("✅ Recolección de datos completada")
        return data

    # ... (todas las funciones internas igual, pero usando self.logger en vez de logging) ...


# Función de utilidad para testing rápido
async def quick_test():
    """Test rápido del recolector"""
    
    config = {
        'api_key': '',  # Dejar vacío para test sin auth
        'api_secret': '',
        'testnet': True,
        'symbols': ['BTCUSDT', 'ETHUSDT'],
        'intervals': ['1h', '5m']
    }
    
    collector = BinanceDataCollector(config)
    
    # Test básico - últimas 24 horas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    data = await collector.collect_historical_data(
        symbols=config['symbols'],
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        intervals=config['intervals']
    )
    
    # Guardar datos
    collector.save_data_to_csv(data)
    
    # Test precios en vivo
    prices = await collector.get_live_prices(config['symbols'])
    logger.info(f"Precios en vivo: {prices}")
    
    # Test info cuenta (si hay credenciales)
    if config['api_key'] and config['api_secret']:
        account_info = await collector.get_account_info()
        logger.info(f"Info cuenta: {account_info}")

if __name__ == "__main__":
    asyncio.run(quick_test())
