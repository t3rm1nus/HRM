import asyncio
import pandas as pd
import socket
from typing import Dict, Any
from core.logging import logger
from comms.config import config
from binance import AsyncClient, BinanceSocketManager

class RealTimeDataLoader:
    def __init__(self, config_dict: dict):
        self.config = config_dict
        self.symbols = self.config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        self.binance_client = None
        self.bm = None
        self._closed = False
        self._stream_tasks = []
        logger.info("✅ RealTimeDataLoader inicializado")

    async def _init_binance(self):
        """Inicializa el cliente de Binance de forma segura"""
        if not self.binance_client:
            # ✅ CRITICAL: Market Data SIEMPRE desde mainnet o feed externo
            # Configurar cliente para datos de mercado en mainnet (sin API keys)
            self.binance_client = await AsyncClient.create()
            self.bm = BinanceSocketManager(self.binance_client)
            logger.info("✅ Cliente de datos de mercado configurado en mainnet (sin API keys)")

    async def close(self):
        """Cierra apropiadamente las conexiones"""
        if not self._closed:
            try:
                # Cancelar tareas de stream
                for task in self._stream_tasks:
                    task.cancel()
                if self.binance_client:
                    await self.binance_client.close_connection()
                self._closed = True
                logger.info("✅ RealTimeDataLoader cerrado correctamente")
            except Exception as e:
                logger.error(f"❌ Error cerrando RealTimeDataLoader: {e}")

    async def start_realtime(self):
        """Inicia streams WebSocket para todos los símbolos"""
        await self._init_binance()
        
        for symbol in self.symbols:
            task = asyncio.create_task(self._handle_kline_stream(symbol, '1m'))
            self._stream_tasks.append(task)
            logger.info(f"✅ Stream iniciado para {symbol}")

        logger.info("✅ Todos los streams WebSocket iniciados")

    async def _handle_kline_stream(self, symbol: str, interval: str):
        """Maneja el stream de kline para un símbolo"""
        try:
            ts = self.bm.kline_socket(symbol=symbol, interval=interval)
            async with ts as tscm:
                while not self._closed:
                    res = await tscm.recv()
                    kline = res['k']
                    logger.debug(f"📊 Kline recibido para {symbol}: "
                                f"Open={kline['o']}, High={kline['h']}, Low={kline['l']}, "
                                f"Close={kline['c']}, Volume={kline['v']}, Cerrado={kline['x']}")
                    
                    # TODO: Implementar lógica para procesar el kline en tiempo real
                    # Por ejemplo: actualizar el datafeed, generar señales, etc.
                    
        except Exception as e:
            logger.opt(exception=True).error(f"❌ Error en stream de {symbol}: {str(e)}")

    async def fetch_realtime_data(self, symbol: str, timeframe: str = '1m', limit: int = 200) -> pd.DataFrame:
        """
        Obtiene datos OHLCV en tiempo real para un símbolo.
        Usa aiohttp directamente con manejo correcto de sesiones.
        """
        import aiohttp
        
        url = f"https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol.upper(), 'interval': timeframe, 'limit': limit}
        
        try:
            # ✅ CRITICAL FIX: Crear sesión en el método y cerrarla correctamente
            # Usar async with para garantizar cierre de sesión
            connector = aiohttp.TCPConnector(family=socket.AF_INET, limit=10)
            timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=10)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        if isinstance(klines, list) and len(klines) > 0 and isinstance(klines[0], list):
                            # Formatear datos
                            if len(klines) > 0 and len(klines[0]) > 6:
                                klines = [row[:6] for row in klines]
                            
                            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            df.set_index('timestamp', inplace=True)
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = df[col].astype(float)
                            
                            logger.debug(f"📊 Datos en tiempo real para {symbol}: shape={df.shape}")
                            return df

            logger.warning(f"⚠️ No se obtuvieron datos para {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.opt(exception=True).error(f"❌ Error obteniendo datos en tiempo real para {symbol}: {str(e)} (detalle completo)")
            return pd.DataFrame()

    async def get_realtime_data(self) -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos en tiempo real para todos los símbolos.
        """
        try:
            tasks = [self.fetch_realtime_data(symbol, limit=200) for symbol in self.symbols]
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
            logger.opt(exception=True).error(f"❌ Error en get_realtime_data: {e}")
            return {}

    async def simulate_realtime_data(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Simula datos en tiempo real agregando ruido gaussiano a datos históricos.
        Útil como alternativa temporal si WebSocket falla.
        """
        import numpy as np
        simulated_data = {}
        
        for symbol, df in historical_data.items():
            if df.empty:
                simulated_data[symbol] = pd.DataFrame()
                continue
                
            # Agregar ruido gaussiano pequeño a las columnas numéricas
            noise = np.random.normal(0, 0.001, size=df.shape)
            noisy_df = df.copy()
            noisy_df[['open', 'high', 'low', 'close', 'volume']] += noisy_df[['open', 'high', 'low', 'close', 'volume']] * noise
            
            # Asegurar que high >= close >= low >= 0
            noisy_df['high'] = np.maximum(noisy_df['high'], noisy_df['close'])
            noisy_df['low'] = np.minimum(noisy_df['low'], noisy_df['close'])
            noisy_df = noisy_df.clip(lower=0)
            
            simulated_data[symbol] = noisy_df
            logger.debug(f"✅ Datos simulados para {symbol}: shape={noisy_df.shape}")
            
        return simulated_data
