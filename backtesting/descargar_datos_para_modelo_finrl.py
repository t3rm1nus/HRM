#!/usr/bin/env python3
"""
Script para descargar datos históricos de Binance de los últimos 5 años
Guarda los datos en formato CSV compatible con el backtesting HRM
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
from typing import List, Dict, Optional
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class BinanceHistoricalDownloader:
    """Descargador de datos históricos de Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = None
        self.rate_limit_delay = 0.2  # 200ms entre requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000) -> List:
        """Obtiene klines (velas) de Binance"""
        url = f"{self.base_url}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': limit
        }
        
        try:
            await asyncio.sleep(self.rate_limit_delay)  # Rate limiting
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded for {symbol}. Waiting 60 seconds...")
                    await asyncio.sleep(60)
                    return await self.get_klines(symbol, interval, start_time, end_time, limit)
                else:
                    logger.error(f"Error {response.status} for {symbol}: {await response.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []
    
    async def download_symbol_data(self, symbol: str, interval: str = '5m', years: int = 5) -> pd.DataFrame:
        """Descarga datos históricos para un símbolo específico"""
        logger.info(f"Descargando datos de {symbol} con intervalo {interval} para {years} años...")
        
        # Calcular fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        # Convertir a timestamps en milisegundos
        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        all_klines = []
        current_start = start_timestamp
        
        # Binance permite máximo 1000 klines por request
        # Para 5m, 1000 klines = ~3.47 días
        batch_size = 1000
        interval_ms = self._interval_to_milliseconds(interval)
        batch_duration = batch_size * interval_ms
        
        batch_count = 0
        while current_start < end_timestamp:
            current_end = min(current_start + batch_duration, end_timestamp)
            
            logger.info(f"  Batch {batch_count + 1}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} a {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}")
            
            klines = await self.get_klines(symbol, interval, current_start, current_end, batch_size)
            
            if klines:
                all_klines.extend(klines)
                logger.info(f"    Obtenidas {len(klines)} velas")
            else:
                logger.warning(f"    No se obtuvieron datos para este batch")
                break
            
            current_start = current_end + interval_ms  # Evitar solapamiento
            batch_count += 1
            
            # Progreso cada 10 batches
            if batch_count % 10 == 0:
                progress = (current_start - start_timestamp) / (end_timestamp - start_timestamp) * 100
                logger.info(f"  Progreso: {progress:.1f}%")
        
        if not all_klines:
            logger.error(f"No se obtuvieron datos para {symbol}")
            return pd.DataFrame()
        
        # Convertir a DataFrame
        df = self._klines_to_dataframe(all_klines, symbol)
        
        # Remover duplicados por timestamp
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        logger.info(f"  Completado: {len(df)} velas descargadas para {symbol}")
        return df
    
    def _interval_to_milliseconds(self, interval: str) -> int:
        """Convierte intervalo a milisegundos"""
        intervals = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000
        }
        return intervals.get(interval, 5 * 60 * 1000)  # Default 5m
    
    def _klines_to_dataframe(self, klines: List, symbol: str) -> pd.DataFrame:
        """Convierte klines a DataFrame normalizado"""
        if not klines:
            return pd.DataFrame()
        
        # Estructura de klines de Binance:
        # [open_time, open, high, low, close, volume, close_time, quote_asset_volume, 
        #  number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Añadir columna de símbolo
        df['symbol'] = symbol
        
        # Seleccionar solo las columnas necesarias
        df = df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
        
        return df.dropna()

async def main():
    """Función principal"""
    
    # Configuración
    symbols = ['BTCUSDT', 'ETHUSDT']  # Añadir más símbolos si se desea
    interval = '5m'
    years = 5
    
    # Rutas
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / 'final_combinado.csv'
    
    logger.info(f"Iniciando descarga de datos históricos...")
    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Intervalo: {interval}")
    logger.info(f"Años: {years}")
    logger.info(f"Archivo de salida: {output_file}")
    
    all_data = []
    
    async with BinanceHistoricalDownloader() as downloader:
        for symbol in symbols:
            try:
                df = await downloader.download_symbol_data(symbol, interval, years)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Datos de {symbol} añadidos: {len(df)} registros")
                else:
                    logger.warning(f"No se obtuvieron datos para {symbol}")
            except Exception as e:
                logger.error(f"Error descargando {symbol}: {e}")
                continue
    
    if not all_data:
        logger.error("No se descargaron datos de ningún símbolo")
        return
    
    # Combinar todos los datos
    logger.info("Combinando y guardando datos...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ordenar por símbolo y timestamp
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Estadísticas
    logger.info(f"Dataset final:")
    logger.info(f"  Total de registros: {len(combined_df)}")
    logger.info(f"  Símbolos: {combined_df['symbol'].unique()}")
    logger.info(f"  Rango de fechas: {combined_df['timestamp'].min()} a {combined_df['timestamp'].max()}")
    logger.info(f"  Registros por símbolo:")
    for symbol in combined_df['symbol'].unique():
        count = len(combined_df[combined_df['symbol'] == symbol])
        logger.info(f"    {symbol}: {count}")
    
    # Guardar en formato CSV
    try:
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Datos guardados exitosamente en: {output_file}")
        logger.info(f"Tamaño del archivo: {output_file.stat().st_size / (1024*1024):.1f} MB")
    except Exception as e:
        logger.error(f"Error guardando el archivo CSV: {e}")
        # Fallback a pickle si CSV falla
        fallback_file = data_dir / 'normalized_grok.pkl'
        combined_df.to_pickle(fallback_file)
        logger.info(f"Datos guardados como fallback en: {fallback_file}")
    
    # Verificación final
    logger.info("Verificando archivo guardado...")
    try:
        test_df = pd.read_csv(output_file, parse_dates=['timestamp'])
        logger.info(f"Verificación exitosa: {len(test_df)} registros leídos")
        logger.info(f"Columnas: {list(test_df.columns)}")
        logger.info(f"Primeros registros:")
        print(test_df.head())
    except Exception as e:
        logger.error(f"Error en verificación: {e}")

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import aiohttp
    except ImportError as e:
        logger.error(f"Dependencia faltante: {e}")
        logger.error("Instalar con: pip install aiohttp pandas")
        exit(1)
    
    logger.info("Iniciando descarga...")
    asyncio.run(main())