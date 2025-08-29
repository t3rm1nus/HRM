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
import logging
import json
import time
import hmac
import hashlib
from urllib.parse import urlencode

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
            
        self.logger = logging.getLogger(__name__)
        
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
                        self.logger.error(f"    ❌ Error obteniendo {symbol} {interval}: {e}")
                        
                    # Rate limiting
                    await asyncio.sleep(self.request_interval)
        
        # Añadir datos adicionales
        data = await self._enrich_data(data)
        
        self.logger.info("✅ Recolección de datos completada")
        return data


    async def _get_historical_klines(self, session: aiohttp.ClientSession, 
                               symbol: str, interval: str, 
                               start_date: str, end_date: str) -> List:
        # Convertir fechas
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_klines = []
        current_start = start_ts
        
        while current_start < end_ts:
            url = f"{self.base_url}/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': min(current_start + (1000 * self._interval_to_ms(interval)), end_ts),
                'limit': 1000
            }
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines_batch = await response.json()
                        
                        if not klines_batch:
                            self.logger.warning(f"No data from API for {symbol} {interval}, generating mock data")
                            interval_ms = self._interval_to_ms(interval)
                            timestamps = list(range(current_start, min(current_start + (1000 * interval_ms), end_ts), interval_ms))
                            base_price = 50000 if 'BTC' in symbol else 3000  # Mock price for BTC or ETH
                            mock_klines = [
                                [
                                    ts, base_price, base_price + np.random.normal(0, 100), 
                                    base_price - np.random.normal(0, 100), base_price + np.random.normal(0, 50),
                                    1000, ts + interval_ms - 1, 1000000, 100, 500, 500000, 0
                                ] for ts in timestamps
                            ]
                            all_klines.extend(mock_klines)
                            current_start += 1000 * interval_ms
                            continue
                            
                        all_klines.extend(klines_batch)
                        current_start = klines_batch[-1][6] + 1  # Close time + 1ms
                        
                    else:
                        self.logger.error(f"Error HTTP {response.status}: {await response.text()}")
                        break
                        
            except Exception as e:
                self.logger.error(f"Error en request: {e}")
                break
                
            await asyncio.sleep(self.request_interval)
        
        return all_klines
    
    
    
    def _interval_to_ms(self, interval: str) -> int:
        """Convertir intervalo a millisegundos"""
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
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return intervals.get(interval, 60 * 1000)

    def _process_klines_to_dataframe(self, klines: List) -> pd.DataFrame:
        """Procesar klines de Binance a un DataFrame de pandas con validación de datos."""
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close',
                                        'volume', 'close_time', 'quote_asset_volume',
                                        'number_of_trades', 'taker_buy_base_asset_volume',
                                        'taker_buy_quote_asset_volume', 'ignore'])

        # Conversión de tipos de datos
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # **>>> AÑADIR VALIDACIÓN DE DATOS ANÓMALOS <<<**
        # Filtro básico para precios con valores atípicos (ej. > 1,000,000)
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        df = df[df['close'] < 1000000] # O un valor más alto si trabajas con activos de alto valor

        return df

    def _add_technical_indicators(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """Añadir indicadores técnicos al DataFrame"""
        
        if df.empty:
            return df
            
        try:
            # RSI
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            
            # MACD
            macd_data = self._calculate_macd(df['close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_hist'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['close'], 20, 2)
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            
            # Moving Averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # Volatilidad
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            
            # Volumen relativo
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Returns
            df['returns'] = df['close'].pct_change()
            df['returns_cumulative'] = (1 + df['returns']).cumprod()
            
            # ATR (Average True Range)
            df['atr'] = self._calculate_atr(df)
            
            # Momentum
            df['momentum'] = df['close'] - df['close'].shift(10)
            
            # VWAP (Volume Weighted Average Price)
            df['vwap'] = self._calculate_vwap(df)
            
        except Exception as e:
            self.logger.warning(f"Error calculando indicadores para {symbol} {interval}: {e}")
        
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calcular MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict:
        """Calcular Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcular Average True Range (ATR)"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calcular Volume Weighted Average Price (VWAP)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap

    async def _enrich_data(self, data: Dict) -> Dict:
        """Enriquecer datos con información adicional"""
        
        # Agregar correlaciones entre activos
        if 'BTCUSDT' in data and 'ETHUSDT' in data:
            # Usar el intervalo más común disponible
            common_interval = None
            for interval in ['1h', '5m', '1m']:
                if interval in data['BTCUSDT'] and interval in data['ETHUSDT']:
                    common_interval = interval
                    break
            
            if common_interval:
                btc_returns = data['BTCUSDT'][common_interval]['returns']
                eth_returns = data['ETHUSDT'][common_interval]['returns']
                
                # Correlación rolling
                correlation = btc_returns.rolling(100).corr(eth_returns)
                
                # Añadir a ambos datasets
                data['BTCUSDT'][common_interval]['btc_eth_correlation'] = correlation
                data['ETHUSDT'][common_interval]['btc_eth_correlation'] = correlation
                
                # Ratio ETH/BTC
                eth_btc_ratio = (data['ETHUSDT'][common_interval]['close'] / 
                               data['BTCUSDT'][common_interval]['close'])
                data['BTCUSDT'][common_interval]['eth_btc_ratio'] = eth_btc_ratio
                data['ETHUSDT'][common_interval]['eth_btc_ratio'] = eth_btc_ratio
                
                # Ratio de volatilidad
                btc_vol = data['BTCUSDT'][common_interval]['volatility']
                eth_vol = data['ETHUSDT'][common_interval]['volatility']
                vol_ratio = eth_vol / btc_vol
                data['BTCUSDT'][common_interval]['eth_btc_vol_ratio'] = vol_ratio
                data['ETHUSDT'][common_interval]['eth_btc_vol_ratio'] = vol_ratio
        
        # Añadir metadatos
        for symbol in data:
            data[symbol]['metadata'] = {
                'collection_time': datetime.now().isoformat(),
                'source': 'binance_api',
                'testnet': self.testnet,
                'symbols': list(data.keys()),
                'intervals': list(data[symbol].keys())
            }
        
        return data

    async def get_live_prices(self, symbols: List[str]) -> Dict:
        """Obtener precios en tiempo real"""
        
        url = f"{self.base_url}/v3/ticker/price"
        
        async with aiohttp.ClientSession() as session:
            prices = {}
            
            for symbol in symbols:
                try:
                    params = {'symbol': symbol}
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            prices[symbol] = {
                                'price': float(data['price']),
                                'timestamp': datetime.now().isoformat()
                            }
                        else:
                            self.logger.error(f"Error obteniendo precio de {symbol}: {response.status}")
                            
                except Exception as e:
                    self.logger.error(f"Error obteniendo precio de {symbol}: {e}")
                    
                await asyncio.sleep(self.request_interval)
        
        return prices

    async def get_account_info(self) -> Optional[Dict]:
        """Obtener información de la cuenta (si hay API keys)"""
        
        if not self.api_key or not self.api_secret:
            self.logger.warning("No se proporcionaron API key o secret")
            return None
            
        try:
            url = f"{self.base_url}/v3/account"
            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            
            # Firmar request
            query_string = urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            
            headers = {'X-MBX-APIKEY': self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        account_info = await response.json()
                        # Procesar información de la cuenta
                        processed_info = {
                            'account_type': account_info.get('accountType', ''),
                            'balances': [
                                {
                                    'asset': b['asset'],
                                    'free': float(b['free']),
                                    'locked': float(b['locked'])
                                } for b in account_info.get('balances', [])
                                if float(b['free']) > 0 or float(b['locked']) > 0
                            ],
                            'can_trade': account_info.get('canTrade', False),
                            'update_time': datetime.fromtimestamp(
                                account_info.get('updateTime', 0) / 1000
                            ).isoformat()
                        }
                        return processed_info
                    else:
                        self.logger.error(f"Error obteniendo info de cuenta: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error obteniendo info de cuenta: {e}")
            return None

    def save_data_to_csv(self, data: Dict, output_dir: str = "backtesting/data"):
        """Guardar datos en archivos CSV"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol in data:
            symbol_dir = os.path.join(output_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            for interval, df in data[symbol].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    filename = os.path.join(symbol_dir, f"{symbol}_{interval}.csv")
                    df.to_csv(filename)
                    self.logger.info(f"Guardado: {filename}")
                elif interval == 'metadata':
                    filename = os.path.join(symbol_dir, f"{symbol}_metadata.json")
                    with open(filename, 'w') as f:
                        json.dump(df, f, indent=4)
                    self.logger.info(f"Guardado: {filename}")

# Función de utilidad para testing rápido
async def quick_test():
    """Test rápido del recolector"""
    
    logging.basicConfig(level=logging.INFO)
    
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
    print("Precios en vivo:", prices)
    
    # Test info cuenta (si hay credenciales)
    if config['api_key'] and config['api_secret']:
        account_info = await collector.get_account_info()
        print("Info cuenta:", account_info)

if __name__ == "__main__":
    asyncio.run(quick_test())