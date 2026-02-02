import asyncio
import aiohttp
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Any
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
        
        # ✅ CRITICAL: Force testnet mode for paper trading safety
        use_testnet = True
        self.config['USE_TESTNET'] = True  # Ensure config reflects this

        logger.info(f"Inicializando BinanceClient con: api_key={'SET' if api_key else 'NOT SET'}, api_secret={'SET' if api_secret else 'NOT SET'}, use_testnet={use_testnet}")

        # ✅ CRITICAL: Validate testnet configuration
        if not api_key or not api_secret:
            logger.warning("⚠️ Advertencia: Claves API no configuradas - usando modo simulado")
            # Could set to simulated mode here if needed
        
        options = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot', 
                'test': True,  # ✅ CRITICAL: Force testnet
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
            # ✅ CRITICAL: Explicit testnet URLs
            'urls': {
                'api': 'https://testnet.binance.vision/api',
                'test': 'https://testnet.binance.vision/api'
            }
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
        # ✅ CRITICAL: PROHIBIR Binance testnet para market data
        if self.config.get('USE_TESTNET', True):
            logger.warning(f"🧪 MODO PAPER: Retornando datos simulados para {symbol}")
            return self._get_mock_klines(symbol, timeframe, limit)
        
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

    def _get_mock_klines(self, symbol: str, timeframe: str, limit: int) -> list:
        """
        Genera datos OHLCV simulados para modo paper trading.
        """
        import random
        import time
        
        # Precios base según el símbolo
        base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'BNBUSDT': 300.0,
            'SOLUSDT': 100.0
        }
        
        base_price = base_prices.get(symbol, 50000.0)
        current_time = int(time.time() * 1000)
        timeframe_ms = self._get_timeframe_ms(timeframe)
        
        klines = []
        current_price = base_price
        
        for i in range(limit):
            # Generar precios aleatorios con tendencia suave
            price_change = random.uniform(-0.02, 0.02) * current_price  # ±2%
            current_price = max(0.01, current_price + price_change)
            
            open_price = current_price
            high_price = open_price * random.uniform(1.0, 1.02)
            low_price = open_price * random.uniform(0.98, 1.0)
            close_price = random.uniform(low_price, high_price)
            volume = random.uniform(1.0, 100.0)
            
            timestamp = current_time - (limit - i - 1) * timeframe_ms
            
            klines.append([
                timestamp,
                float(open_price),
                float(high_price),
                float(low_price),
                float(close_price),
                float(volume)
            ])
        
        logger.info(f"📊 Datos simulados generados para {symbol}: {len(klines)} velas")
        return klines

    def _get_timeframe_ms(self, timeframe: str) -> int:
        """Convierte timeframe a milisegundos"""
        timeframe_map = {
            '1m': 60000,
            '3m': 180000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '2h': 7200000,
            '4h': 14400000,
            '1d': 86400000
        }
        return timeframe_map.get(timeframe, 60000)

    async def get_ticker_price(self, symbol: str) -> float:
        """
        Obtiene el precio actual de un símbolo.
        """
        # ✅ CRITICAL: PROHIBIR Binance testnet para market data
        if self.config.get('USE_TESTNET', True):
            logger.warning(f"🧪 MODO PAPER: Retornando precio simulado para {symbol}")
            return self._get_mock_price(symbol)
        
        try:
            if self.exchange.session is None or self.exchange.session.closed:
                logger.warning("Sesión cerrada, reiniciando...")
                await self.exchange.close()
                # Recrear el exchange con las mismas opciones
                self.exchange = ccxt.binance(self.exchange.options)
                self.exchange.set_sandbox_mode(True)
            
            await self.exchange.load_markets()
            ticker = await self.exchange.fetch_ticker(symbol)
            
            if 'last' in ticker:
                price = float(ticker['last'])
                logger.debug(f"💰 Precio actual {symbol}: {price}")
                return price
            else:
                logger.error(f"❌ No se encontró precio en ticker para {symbol}")
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo precio para {symbol}: {str(e)}")
            return 0.0

    def _get_mock_price(self, symbol: str) -> float:
        """
        Genera un precio simulado para modo paper trading.
        """
        import random
        
        # Precios base según el símbolo
        base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'BNBUSDT': 300.0,
            'SOLUSDT': 100.0
        }
        
        base_price = base_prices.get(symbol, 50000.0)
        # Variación aleatoria del ±2%
        price_change = random.uniform(-0.02, 0.02) * base_price
        price = max(0.01, base_price + price_change)
        
        logger.info(f"💰 Precio simulado para {symbol}: {price:.2f}")
        return price
        
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

    async def get_account_balances(self) -> Dict[str, float]:
        """
        Obtiene los balances reales de la cuenta de Binance.
        CRÍTICO para sincronización en modo producción.
        """
        try:
            if not hasattr(self, 'exchange') or self.exchange is None:
                logger.error("❌ Exchange no inicializado")
                return {}

            # Obtener balances de la cuenta
            account = await self.exchange.fetch_balance()

            # Extraer balances no cero
            balances = {}
            if 'free' in account:
                for asset, amount in account['free'].items():
                    if amount > 0.00000001:  # Ignorar cantidades insignificantes
                        balances[asset] = amount

            if 'used' in account:
                for asset, amount in account['used'].items():
                    if amount > 0.00000001:
                        # Agregar a balances existentes o crear nuevos
                        if asset in balances:
                            balances[asset] += amount
                        else:
                            balances[asset] = amount

            logger.info(f"✅ Balances obtenidos de Binance: {len(balances)} activos")
            for asset, amount in balances.items():
                logger.debug(f"   {asset}: {amount}")

            return balances

        except Exception as e:
            logger.error(f"❌ Error obteniendo balances de Binance: {e}")
            return {}

    async def place_stop_loss_order(self, symbol: str, side: str, quantity: float,
                                   stop_price: float, limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Coloca una orden STOP_LOSS en Binance.
        CRÍTICO para protección de posiciones en modo producción.
        """
        try:
            if self.config.get('USE_TESTNET', True):
                logger.warning("🧪 MODO TESTNET: Stop-loss orders simulados (no se envían a exchange)")
                return {
                    'id': f'simulated_sl_{symbol}_{side}',
                    'status': 'simulated',
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'stop_price': stop_price,
                    'limit_price': limit_price
                }

            # Validar parámetros
            if quantity <= 0:
                raise ValueError(f"Cantidad inválida: {quantity}")
            if stop_price <= 0:
                raise ValueError(f"Precio stop inválido: {stop_price}")

            # Preparar orden
            order_params = {
                'symbol': symbol,
                'type': 'STOP_LOSS_LIMIT' if limit_price else 'STOP_LOSS',
                'side': side.upper(),
                'amount': quantity,
                'params': {
                    'stopPrice': stop_price
                }
            }

            if limit_price:
                order_params['price'] = limit_price

            # Colocar orden
            order = await self.exchange.create_order(**order_params)

            logger.info(f"🛡️ STOP-LOSS colocado: {symbol} {side} {quantity} @ stop={stop_price}")
            return order

        except Exception as e:
            logger.error(f"❌ Error colocando stop-loss {symbol}: {e}")
            raise

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancela una orden específica.
        """
        try:
            await self.exchange.cancel_order(order_id, symbol)
            logger.info(f"❌ Orden cancelada: {symbol} {order_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cancelando orden {order_id}: {e}")
            return False

    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Obtiene órdenes abiertas.
        """
        try:
            orders = await self.exchange.fetch_open_orders(symbol)
            logger.debug(f"📋 Órdenes abiertas: {len(orders)}")
            return orders
        except Exception as e:
            logger.error(f"❌ Error obteniendo órdenes abiertas: {e}")
            return []

    async def place_limit_order(self, symbol: str, side: str, quantity: float,
                               price: float, stop_price: Optional[float] = None,
                               order_type: str = "LIMIT") -> Dict[str, Any]:
        """
        Coloca una orden LIMIT en Binance.
        CRÍTICO para órdenes de profit-taking en modo producción.
        """
        try:
            if self.config.get('USE_TESTNET', True):
                logger.warning("🧪 MODO TESTNET: Limit orders simulados (no se envían a exchange)")
                return {
                    'id': f'simulated_limit_{symbol}_{side}_{price:.6f}',
                    'status': 'simulated',
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'stop_price': stop_price,
                    'order_type': order_type
                }

            # Validar parámetros
            if quantity <= 0:
                raise ValueError(f"Cantidad inválida: {quantity}")
            if price <= 0:
                raise ValueError(f"Precio inválido: {price}")

            # Preparar orden
            if stop_price:
                # STOP_LOSS_LIMIT order
                order_params = {
                    'symbol': symbol,
                    'type': 'STOP_LOSS_LIMIT',
                    'side': side.upper(),
                    'amount': quantity,
                    'price': price,
                    'params': {
                        'stopPrice': stop_price
                    }
                }
            else:
                # Regular LIMIT order
                order_params = {
                    'symbol': symbol,
                    'type': 'LIMIT',
                    'side': side.upper(),
                    'amount': quantity,
                    'price': price
                }

            # Colocar orden
            order = await self.exchange.create_order(**order_params)

            logger.info(f"💰 LIMIT ORDER colocado: {symbol} {side} {quantity} @ limit={price}")
            return order

        except Exception as e:
            logger.error(f"❌ Error colocando limit order {symbol}: {e}")
            raise

    async def close(self):
        """
        Cierra la conexión.
        """
        try:
            await self.exchange.close()
            logger.info("✅ BinanceClient cerrado")
        except Exception as e:
            logger.error(f"❌ Error cerrando BinanceClient: {e}", exc_info=True)
