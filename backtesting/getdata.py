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
                        if klines is not None:
                            df = self._process_klines_to_dataframe(klines)
                            if df is not None and not df.empty:
                                df = self._add_technical_indicators(df, symbol, interval)
                                data[symbol][interval] = df
                                self.data_cache[cache_key] = df
                                self.logger.info(f"    ✅ {len(df)} velas obtenidas")
                            else:
                                self.logger.warning(f"    ⚠️  Sin datos para {symbol} {interval}")
                        else:
                            self.logger.warning(f"    ⚠️  Sin datos para {symbol} {interval}")
                    except Exception as e:
                        self.logger.error(f"    ❌ Error obteniendo {symbol} {interval}: {e}", exc_info=True)
                        
                    # Rate limiting
                    await asyncio.sleep(self.request_interval)
        
        # Añadir datos adicionales (placeholder)
        data = await self._enrich_data(data)
        
        self.logger.info("✅ Recolección de datos completada")
        return data

    # -------------------------
    # Helpers internos (fallback local)
    # -------------------------
    async def _get_historical_klines(self, session, symbol: str, interval: str,
                                     start_date: str, end_date: str):
        """Lee desde parquet consolidado (datos/normalized_grok.parquet) y re-muestrea.
        Si no existe, intenta data/normalized_grok.parquet. Columnas esperadas: symbol, open, high, low, close, volume, timestamp.
        """
        try:
            import pandas as pd
            from pathlib import Path

            # Cachea la carga para no re-leer el parquet por cada símbolo/intervalo
            if not hasattr(self, "_normalized_cache"):
                self._normalized_cache = None

            if self._normalized_cache is None:
                root = Path(__file__).resolve().parents[1]
                parquet_paths = [
                    root / 'data' / 'normalized_grok.parquet',
                    root / 'datos' / 'normalized_grok.parquet',
                ]
                parquet_fp = next((p for p in parquet_paths if p.exists()), None)
                if parquet_fp is None:
                    self.logger.warning("    ⚠️  Parquet normalizado no encontrado en datos/ o data/")
                    return None
                self.logger.info(f"    📁 Leyendo parquet desde: {parquet_fp}")
                df_all = pd.read_parquet(parquet_fp)
                self.logger.info(f"    📊 Parquet cargado: {df_all.shape[0]} filas, {df_all.shape[1]} columnas")
                # Normalización mínima
                time_col = None
                for c in ['timestamp', 'time', 'date', 'datetime', 'open_time']:
                    if c in df_all.columns:
                        time_col = c
                        break
                if time_col is None:
                    self.logger.warning("    ⚠️  Parquet sin columna temporal reconocible")
                    return None
                df_all[time_col] = pd.to_datetime(df_all[time_col])
                df_all = df_all.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                })
                # Estandariza símbolo: mapear ticker a symbol
                if 'ticker' in df_all.columns:
                    df_all['symbol'] = df_all['ticker'].replace({'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}).astype(str)
                elif 'symbol' in df_all.columns:
                    df_all['symbol'] = df_all['symbol'].replace({'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}).astype(str)
                else:
                    # Si no hay columna de símbolo, asume BTCUSDT
                    df_all['symbol'] = 'BTCUSDT'

                self._normalized_cache = df_all[[time_col, 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                self._normalized_cache = self._normalized_cache.dropna(subset=['open', 'high', 'low', 'close']).copy()

            df_all = self._normalized_cache

            # Filtra símbolo
            sym = symbol
            if sym not in ['BTCUSDT', 'ETHUSDT']:
                # intenta mapear por si vienen tickers básicos
                mapping = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}
                sym = mapping.get(sym, sym)
            self.logger.info(f"    🔍 Buscando símbolo: {sym} en {df_all['symbol'].unique()}")
            df = df_all[df_all['symbol'] == sym].copy()
            if df.empty:
                self.logger.warning(f"    ⚠️  Sin datos en parquet para {symbol} (símbolos disponibles: {df_all['symbol'].unique()})")
                return None
            self.logger.info(f"    ✅ Encontrados {len(df)} registros para {symbol}")

            # Índice temporal
            idx_col = [c for c in df.columns if str(df[c].dtype).startswith('datetime')]
            idx_col = idx_col[0] if idx_col else None
            if idx_col is None:
                self.logger.warning("    ⚠️  No se pudo identificar la columna temporal tras carga")
                return None
            df = df.set_index(idx_col).sort_index()

            # Recorte por fechas
            try:
                self.logger.info(f"    📅 Rango de fechas solicitado: {start_date} a {end_date}")
                self.logger.info(f"    📅 Rango de fechas en datos: {df.index.min()} a {df.index.max()}")
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date) + pd.Timedelta(days=1)]
                self.logger.info(f"    📅 Después del filtro: {len(df)} registros")
            except Exception as e:
                self.logger.error(f"    ❌ Error en filtro de fechas: {e}")
                pass

            # Llenar NaN en volume con 0 para evitar problemas en resampling
            df['volume'] = df['volume'].fillna(0)

            # El parquet está a 5m; re-muestrea a lo solicitado
            if interval:
                rule = interval
                if rule == '1m':
                    # Upsample no fiable; avisamos y devolvemos vacío para evitar conclusiones erróneas
                    self.logger.warning("    ⚠️  1m solicitado pero fuente es 5m; se omite 1m")
                    return None
                if rule != '5m':
                    df = df.resample(rule).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum',
                    }).dropna(subset=['open', 'high', 'low', 'close'])

            # Selecciona columnas OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            return df
        except Exception as e:
            self.logger.error(f"    ❌ Lectura desde parquet falló: {e}")
            return None

    def _process_klines_to_dataframe(self, klines):
        """Normaliza klines a DataFrame OHLCV con índice datetime."""
        import pandas as pd
        if isinstance(klines, pd.DataFrame):
            return klines
        # Si llegaran como lista de listas [openTime, open, high, low, close, volume, closeTime, ...]
        try:
            cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
            df = pd.DataFrame(klines, columns=cols[:len(klines[0])])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df = df.set_index('open_time')
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            return df[['open', 'high', 'low', 'close', 'volume']].dropna()
        except Exception:
            return None

    def _add_technical_indicators(self, df, symbol: str, interval: str):
        """Añade indicadores básicos (RSI, SMA20, Bollinger)."""
        import pandas as pd
        import numpy as np
        out = df.copy()

        # SMA
        out['sma_20'] = out['close'].rolling(20, min_periods=5).mean()
        # Bollinger
        std = out['close'].rolling(20, min_periods=5).std()
        out['bollinger_middle'] = out['sma_20']
        out['bollinger_upper'] = out['sma_20'] + 2 * std
        out['bollinger_lower'] = out['sma_20'] - 2 * std

        # RSI simple
        delta = out['close'].diff()
        up = delta.clip(lower=0).rolling(14, min_periods=5).mean()
        down = -delta.clip(upper=0).rolling(14, min_periods=5).mean()
        rs = up / (down.replace(0, np.nan))
        out['rsi'] = 100 - (100 / (1 + rs.replace({np.inf: np.nan})))

        return out.dropna().copy() if len(out) else out

    async def _enrich_data(self, data: Dict) -> Dict:
        """Placeholder para enriquecer datos (retorna tal cual por ahora)."""
        return data


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
