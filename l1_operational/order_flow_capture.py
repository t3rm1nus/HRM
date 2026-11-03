# -*- coding: utf-8 -*-
"""
OrderFlowCapture - Sistema de captura de Order Flow real de Binance

CAPTURA DUAL:
1. MODO WEBSOCKET (default): 3 streams WebSocket simultáneos
   - @bookTicker: Best bid/ask en tiempo real
   - @depth5: Order book top 5 niveles
   - @aggTrade: Trades agregados en tiempo real

2. MODO POLLING (fallback): REST API cada 5 segundos
   - GET /api/v3/depth
   - GET /api/v3/ticker/bookTicker
   - GET /api/v3/trades

Calcula métricas de flujo de órdenes y persiste en CSV para análisis posterior.
"""

import asyncio
import aiohttp
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import deque
from pathlib import Path
import threading

from core.logging import logger

# REPO_ROOT para paths absolutos
def get_repo_root() -> Path:
    """Obtiene el directorio raíz del repositorio"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent

REPO_ROOT = get_repo_root()

# URLs de Binance REST API (para polling fallback)
BINANCE_REST_URL = "https://api.binance.com"

# Flag para usar polling en lugar de websockets
USE_POLLING_FALLBACK = os.environ.get('ORDERFLOW_USE_POLLING', 'true').lower() == 'true'
POLLING_INTERVAL = int(os.environ.get('ORDERFLOW_POLL_INTERVAL', '5'))  # segundos

try:
    from binance import AsyncClient, BinanceSocketManager
    BINANCE_AVAILABLE = True
except ImportError:
    logger.error("❌ python-binance no está instalado")
    BINANCE_AVAILABLE = False


class OrderFlowMetrics:
    """Métricas de Order Flow para un símbolo"""
    
    def __init__(self):
        # Book Ticker
        self.best_bid_price: float = 0.0
        self.best_bid_qty: float = 0.0
        self.best_ask_price: float = 0.0
        self.best_ask_qty: float = 0.0
        self.spread_bps: float = 0.0
        self.bid_ask_imbalance: float = 0.0
        
        # Depth
        self.book_imbalance_5: float = 1.0
        self.weighted_mid_price: float = 0.0
        
        # AggTrades
        self.delta_volume_1m: float = 0.0
        self.buy_pressure_1m: float = 0.5
        self.large_trade_detected: bool = False
        
        # Timestamp
        self.last_update: datetime = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.last_update.isoformat(),
            'bid_ask_imbalance': self.bid_ask_imbalance,
            'book_imbalance_5': self.book_imbalance_5,
            'delta_volume_1m': self.delta_volume_1m,
            'buy_pressure_1m': self.buy_pressure_1m,
            'spread_bps': self.spread_bps,
            'weighted_mid_price': self.weighted_mid_price,
            'large_trade_detected': self.large_trade_detected,
            'best_bid_price': self.best_bid_price,
            'best_ask_price': self.best_ask_price,
            'best_bid_qty': self.best_bid_qty,
            'best_ask_qty': self.best_ask_qty,
        }


class TradeBuffer:
    """Buffer circular para almacenar trades recientes"""
    
    def __init__(self, max_size: int = 100):
        self.trades: deque = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add_trade(self, qty: float, is_buyer_maker: bool, timestamp: datetime):
        with self.lock:
            self.trades.append({
                'qty': qty,
                'is_buyer_maker': is_buyer_maker,
                'timestamp': timestamp
            })
    
    def get_stats(self, window_seconds: int = 60) -> tuple:
        """Retorna (buy_volume, sell_volume, total_volume, avg_qty)"""
        with self.lock:
            cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
            recent_trades = [t for t in self.trades if t['timestamp'] > cutoff]
            
            if not recent_trades:
                return 0.0, 0.0, 0.0, 0.0
            
            buy_volume = sum(t['qty'] for t in recent_trades if not t['is_buyer_maker'])
            sell_volume = sum(t['qty'] for t in recent_trades if t['is_buyer_maker'])
            total_volume = buy_volume + sell_volume
            avg_qty = total_volume / len(recent_trades)
            
            return buy_volume, sell_volume, total_volume, avg_qty
    
    def is_large_trade(self, qty: float) -> bool:
        """Detecta si un trade es grande (> 3x la media)"""
        with self.lock:
            if len(self.trades) < 10:
                return qty > 1.0  # Umbral mínimo inicial
            
            avg_qty = sum(t['qty'] for t in self.trades) / len(self.trades)
            return qty > (avg_qty * 3.0)


class SymbolOrderFlow:
    """Gestiona el order flow para un símbolo específico"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.metrics = OrderFlowMetrics()
        self.trade_buffer = TradeBuffer(max_size=100)
        self._lock = asyncio.Lock()
        
        # Contadores de mensajes recibidos
        self.depth_count = 0
        self.agg_trade_count = 0
        self.book_ticker_count = 0
        
        # Flags para log de primer mensaje
        self._first_depth_logged = False
        self._first_agg_trade_logged = False
        self._first_book_ticker_logged = False
        
    async def update_book_ticker(self, bid_price: float, bid_qty: float, 
                                  ask_price: float, ask_qty: float):
        """Actualiza métricas desde bookTicker"""
        async with self._lock:
            self.book_ticker_count += 1
            
            # CRITICAL FIX: Log cuando se recibe el PRIMER mensaje bookTicker
            if not self._first_book_ticker_logged:
                print(f"✅ MSG RECEIVED: bookTicker {self.symbol} - bid={bid_price}, ask={ask_price}", flush=True)
                logger.info(f"✅ Primer mensaje bookTicker recibido para {self.symbol}")
                self._first_book_ticker_logged = True
            
            self.metrics.best_bid_price = bid_price
            self.metrics.best_bid_qty = bid_qty
            self.metrics.best_ask_price = ask_price
            self.metrics.best_ask_qty = ask_qty
            
            # Calcular spread en basis points
            mid_price = (bid_price + ask_price) / 2
            if mid_price > 0:
                self.metrics.spread_bps = ((ask_price - bid_price) / mid_price) * 10000
            
            # Calcular imbalance bid/ask
            total_qty = bid_qty + ask_qty
            if total_qty > 0:
                self.metrics.bid_ask_imbalance = (bid_qty - ask_qty) / total_qty
            
            self.metrics.last_update = datetime.utcnow()
    
    async def update_depth(self, bids: List[List], asks: List[List]):
        """Actualiza métricas desde depth5"""
        async with self._lock:
            self.depth_count += 1
            
            # CRITICAL FIX: Log cuando se recibe el PRIMER mensaje depth
            if not self._first_depth_logged:
                print(f"✅ MSG RECEIVED: depth {self.symbol} - bids={len(bids)}, asks={len(asks)}", flush=True)
                logger.info(f"✅ Primer mensaje depth recibido para {self.symbol}")
                self._first_depth_logged = True
            
            # Log cada 100 mensajes
            if self.depth_count % 100 == 0:
                logger.debug(f"📨 depth messages received: {self.depth_count}")
            
            # Sumar top 5 niveles
            bid_qty_sum = sum(float(b[1]) for b in bids[:5])
            ask_qty_sum = sum(float(a[1]) for a in asks[:5])
            
            # Book imbalance
            if ask_qty_sum > 0:
                self.metrics.book_imbalance_5 = bid_qty_sum / ask_qty_sum
            
            # Weighted mid price por liquidez
            total_liquidity = bid_qty_sum + ask_qty_sum
            if total_liquidity > 0:
                bid_weighted = sum(float(b[0]) * float(b[1]) for b in bids[:5])
                ask_weighted = sum(float(a[0]) * float(a[1]) for a in asks[:5])
                self.metrics.weighted_mid_price = (bid_weighted + ask_weighted) / total_liquidity
            
            self.metrics.last_update = datetime.utcnow()
    
    async def process_agg_trade(self, qty: float, price: float, 
                                 is_buyer_maker: bool, trade_time: int):
        """Procesa un trade agregado"""
        timestamp = datetime.utcfromtimestamp(trade_time / 1000)
        
        # Detectar trade grande
        is_large = self.trade_buffer.is_large_trade(qty)
        
        async with self._lock:
            self.agg_trade_count += 1
            
            # CRITICAL FIX: Log cuando se recibe el PRIMER mensaje aggTrade
            if not self._first_agg_trade_logged:
                print(f"✅ MSG RECEIVED: aggTrade {self.symbol} - qty={qty}, price={price}", flush=True)
                logger.info(f"✅ Primer mensaje aggTrade recibido para {self.symbol}")
                self._first_agg_trade_logged = True
            
            # Log cada 100 mensajes
            if self.agg_trade_count % 100 == 0:
                logger.debug(f"📨 aggTrade messages received: {self.agg_trade_count}")
            
            self.trade_buffer.add_trade(qty, is_buyer_maker, timestamp)
            self.metrics.large_trade_detected = is_large
            
            # Calcular delta de volumen último minuto
            buy_vol, sell_vol, total_vol, _ = self.trade_buffer.get_stats(window_seconds=60)
            self.metrics.delta_volume_1m = buy_vol - sell_vol
            
            if total_vol > 0:
                self.metrics.buy_pressure_1m = buy_vol / total_vol
            
            self.metrics.last_update = datetime.utcnow()
    
    def get_metrics(self) -> OrderFlowMetrics:
        """Retorna copia de las métricas actuales"""
        return self.metrics


class OrderFlowCapture:
    """
    Captura Order Flow real de Binance mediante:
    - 3 streams WebSocket simultáneos (default)
    - O polling REST API cada 5 segundos (fallback)
    
    Streams WebSocket:
    - @bookTicker: Best bid/ask tiempo real
    - @depth5: Top 5 niveles del libro
    - @aggTrade: Trades agregados
    """
    
    def __init__(self, symbols: List[str] = None, data_dir: str = None, use_polling: bool = None):
        """
        Inicializa el capturador de Order Flow.
        
        Args:
            symbols: Lista de símbolos a monitorear (default: BTCUSDT, ETHUSDT)
            data_dir: Directorio para guardar CSVs (default: backtesting/data/orderflow/)
            use_polling: Si True, usa REST API polling en lugar de websockets
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.data_dir = data_dir or "backtesting/data/orderflow"
        self.use_polling = use_polling if use_polling is not None else USE_POLLING_FALLBACK
        
        # CRITICAL FIX: Crear directorio con path absoluto relativo a REPO_ROOT
        if not os.path.isabs(self.data_dir):
            abs_data_dir = os.path.join(REPO_ROOT, self.data_dir)
        else:
            abs_data_dir = self.data_dir
        
        # Asegurar que existe el directorio
        os.makedirs(abs_data_dir, exist_ok=True)
        
        # CRITICAL FIX: Log inmediato confirmando path de escritura
        print(f"💾 OrderFlow CSV path: {abs_data_dir}", flush=True)
        logger.info(f"💾 OrderFlow CSV path: {abs_data_dir}")
        logger.info(f"📁 Directorio OrderFlow verificado: {abs_data_dir}")
        
        # Actualizar data_dir al path absoluto para uso interno
        self.data_dir = abs_data_dir
        
        # Cliente de Binance
        self.client: Optional[AsyncClient] = None
        self.bm: Optional[BinanceSocketManager] = None
        
        # Session para REST API polling
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Order flow por símbolo
        self.symbol_flows: Dict[str, SymbolOrderFlow] = {
            s: SymbolOrderFlow(s) for s in self.symbols
        }
        
        # Tareas de stream
        self._stream_tasks: List[asyncio.Task] = []
        self._running = False
        self._closed = False
        
        # Persistencia
        self._buffer: List[Dict] = []
        self._buffer_lock = asyncio.Lock()
        self._current_date = datetime.utcnow().date()
        self._last_flush = datetime.utcnow()
        
        # Estadísticas
        self._stream_health = {s: {'book_ticker': 0, 'depth': 0, 'agg_trade': 0, 'polls': 0} 
                               for s in self.symbols}
        self._last_health_log = datetime.utcnow()
        
        # Reconexión
        self._reconnect_attempts = {s: 0 for s in self.symbols}
        self._max_reconnects = 3
        
        mode_str = "POLLING REST API" if self.use_polling else "WEBSOCKET"
        print(f"✅ OrderFlowCapture inicializado en modo {mode_str} para {self.symbols}", flush=True)
        logger.info(f"✅ OrderFlowCapture inicializado en modo {mode_str} para {self.symbols}")
    
    async def initialize(self):
        """Inicializa el cliente de Binance"""
        if not BINANCE_AVAILABLE and not self.use_polling:
            raise RuntimeError("python-binance no está instalado y no se usa polling")
        
        if self.use_polling:
            # Inicializar session aiohttp para REST API
            self._session = aiohttp.ClientSession()
            print("✅ Cliente REST API inicializado", flush=True)
            logger.info("✅ Cliente REST API inicializado")
        else:
            # Crear cliente websocket sin API keys (solo datos de mercado)
            self.client = await AsyncClient.create()
            self.bm = BinanceSocketManager(self.client)
            print("✅ Cliente Binance WebSocket inicializado (mainnet - solo lectura)", flush=True)
            logger.info("✅ Cliente Binance WebSocket inicializado (mainnet - solo lectura)")
    
    async def start(self):
        """Inicia todos los streams o polling"""
        if self._running:
            logger.warning("⚠️ OrderFlowCapture ya está corriendo")
            return
        
        await self.initialize()
        self._running = True
        self._closed = False
        
        if self.use_polling:
            # Iniciar polling REST API para cada símbolo
            for symbol in self.symbols:
                task = asyncio.create_task(
                    self._poll_order_flow(symbol),
                    name=f"poll_{symbol}"
                )
                self._stream_tasks.append(task)
                logger.info(f"✅ Polling REST API iniciado para {symbol}")
            
            print(f"🔄 Polling REST API iniciado para {self.symbols} cada {POLLING_INTERVAL}s", flush=True)
        else:
            # Iniciar streams WebSocket para cada símbolo
            for symbol in self.symbols:
                # Stream 1: Book Ticker
                task_bt = asyncio.create_task(
                    self._handle_book_ticker_stream(symbol),
                    name=f"book_ticker_{symbol}"
                )
                self._stream_tasks.append(task_bt)
                
                # Stream 2: Depth
                task_d = asyncio.create_task(
                    self._handle_depth_stream(symbol),
                    name=f"depth_{symbol}"
                )
                self._stream_tasks.append(task_d)
                
                # Stream 3: AggTrades
                task_at = asyncio.create_task(
                    self._handle_agg_trade_stream(symbol),
                    name=f"agg_trade_{symbol}"
                )
                self._stream_tasks.append(task_at)
                
                logger.info(f"✅ Streams WebSocket iniciados para {symbol}")
        
        # Iniciar tareas de mantenimiento
        self._stream_tasks.append(
            asyncio.create_task(self._flush_loop(), name="flush_loop")
        )
        self._stream_tasks.append(
            asyncio.create_task(self._health_check_loop(), name="health_check")
        )
        
        print("✅ OrderFlowCapture completamente iniciado", flush=True)
        logger.info("✅ OrderFlowCapture completamente iniciado")
    
    async def _poll_order_flow(self, symbol: str):
        """
        POLLING FALLBACK: Obtiene datos via REST API cada 5 segundos.
        
        Endpoints:
        - GET /api/v3/depth?symbol={symbol}&limit=5  → book_imbalance
        - GET /api/v3/ticker/bookTicker?symbol={symbol} → spread, bid_ask_imbalance  
        - GET /api/v3/trades?symbol={symbol}&limit=50 → últimos trades para delta_volume
        """
        print(f"🔄 Iniciando polling REST para {symbol}", flush=True)
        
        while self._running and not self._closed:
            try:
                poll_start = datetime.utcnow()
                
                # 1. Obtener depth (order book)
                depth_url = f"{BINANCE_REST_URL}/api/v3/depth?symbol={symbol}&limit=5"
                async with self._session.get(depth_url) as resp:
                    if resp.status == 200:
                        depth_data = await resp.json()
                        bids = [[b[0], b[1]] for b in depth_data.get('bids', [])]
                        asks = [[a[0], a[1]] for a in depth_data.get('asks', [])]
                        
                        if bids and asks:
                            await self.symbol_flows[symbol].update_depth(bids, asks)
                            print(f"📊 POLL depth {symbol}: {len(bids)} bids, {len(asks)} asks", flush=True)
                    else:
                        print(f"⚠️ Error depth {symbol}: HTTP {resp.status}", flush=True)
                
                # 2. Obtener bookTicker
                ticker_url = f"{BINANCE_REST_URL}/api/v3/ticker/bookTicker?symbol={symbol}"
                async with self._session.get(ticker_url) as resp:
                    if resp.status == 200:
                        ticker_data = await resp.json()
                        await self.symbol_flows[symbol].update_book_ticker(
                            bid_price=float(ticker_data['bidPrice']),
                            bid_qty=float(ticker_data['bidQty']),
                            ask_price=float(ticker_data['askPrice']),
                            ask_qty=float(ticker_data['askQty'])
                        )
                        print(f"💰 POLL ticker {symbol}: bid={ticker_data['bidPrice']}, ask={ticker_data['askPrice']}", flush=True)
                    else:
                        print(f"⚠️ Error ticker {symbol}: HTTP {resp.status}", flush=True)
                
                # 3. Obtener trades recientes
                trades_url = f"{BINANCE_REST_URL}/api/v3/trades?symbol={symbol}&limit=50"
                async with self._session.get(trades_url) as resp:
                    if resp.status == 200:
                        trades_data = await resp.json()
                        
                        # Procesar cada trade
                        for trade in trades_data:
                            await self.symbol_flows[symbol].process_agg_trade(
                                qty=float(trade['qty']),
                                price=float(trade['price']),
                                is_buyer_maker=trade['isBuyerMaker'],
                                trade_time=trade['time']
                            )
                        
                        print(f"🔄 POLL trades {symbol}: {len(trades_data)} trades", flush=True)
                    else:
                        print(f"⚠️ Error trades {symbol}: HTTP {resp.status}", flush=True)
                
                # Guardar snapshot inmediatamente
                await self.capture_snapshot()
                self._stream_health[symbol]['polls'] += 1
                self._reconnect_attempts[symbol] = 0
                
                # Calcular tiempo de espera restante
                elapsed = (datetime.utcnow() - poll_start).total_seconds()
                sleep_time = max(0.1, POLLING_INTERVAL - elapsed)
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self._reconnect_attempts[symbol] += 1
                print(f"❌ Error en polling {symbol}: {e}", flush=True)
                logger.error(f"❌ Error en polling {symbol}: {e}")
                
                if self._reconnect_attempts[symbol] >= self._max_reconnects:
                    print(f"🚨 Máximos reintentos alcanzados para {symbol} polling", flush=True)
                    logger.critical(f"🚨 Máximos reintentos alcanzados para {symbol} polling")
                    await asyncio.sleep(60)  # Esperar antes de reintentar
                    self._reconnect_attempts[symbol] = 0
                else:
                    wait_time = 2 ** self._reconnect_attempts[symbol]
                    print(f"🔄 Reintentando polling {symbol} en {wait_time}s...", flush=True)
                    logger.info(f"🔄 Reintentando polling {symbol} en {wait_time}s...")
                    await asyncio.sleep(wait_time)
    
    async def _handle_book_ticker_stream(self, symbol: str):
        """Maneja el stream de bookTicker (@bookTicker)"""
        while self._running and not self._closed:
            try:
                ts = self.bm.symbol_book_ticker_socket(symbol=symbol)
                async with ts as tscm:
                    print(f"✅ WebSocket bookTicker CONECTADO para {symbol}", flush=True)
                    
                    while self._running and not self._closed:
                        msg = await tscm.recv()
                        
                        # Actualizar métricas
                        await self.symbol_flows[symbol].update_book_ticker(
                            bid_price=float(msg['b']),
                            bid_qty=float(msg['B']),
                            ask_price=float(msg['a']),
                            ask_qty=float(msg['A'])
                        )
                        
                        self._stream_health[symbol]['book_ticker'] += 1
                        self._reconnect_attempts[symbol] = 0  # Reset contador
                        
            except Exception as e:
                self._reconnect_attempts[symbol] += 1
                print(f"❌ Error en bookTicker {symbol}: {e}", flush=True)
                logger.error(f"❌ Error en bookTicker {symbol}: {e}")
                
                if self._reconnect_attempts[symbol] >= self._max_reconnects:
                    print(f"🚨 Máximos reintentos alcanzados para {symbol} bookTicker", flush=True)
                    logger.critical(f"🚨 Máximos reintentos alcanzados para {symbol} bookTicker")
                    await asyncio.sleep(60)  # Esperar antes de reintentar
                    self._reconnect_attempts[symbol] = 0
                else:
                    wait_time = 2 ** self._reconnect_attempts[symbol]
                    print(f"🔄 Reconectando bookTicker {symbol} en {wait_time}s...", flush=True)
                    logger.info(f"🔄 Reconectando bookTicker {symbol} en {wait_time}s...")
                    await asyncio.sleep(wait_time)
    
    async def _handle_depth_stream(self, symbol: str):
        """Maneja el stream de depth (@depth5)"""
        while self._running and not self._closed:
            try:
                ts = self.bm.depth_socket(symbol=symbol, depth="5")
                async with ts as tscm:
                    print(f"✅ WebSocket depth CONECTADO para {symbol}", flush=True)
                    
                    while self._running and not self._closed:
                        msg = await tscm.recv()
                        
                        # Actualizar métricas
                        await self.symbol_flows[symbol].update_depth(
                            bids=msg.get('bids', []),
                            asks=msg.get('asks', [])
                        )
                        
                        self._stream_health[symbol]['depth'] += 1
                        self._reconnect_attempts[symbol] = 0
                        
            except Exception as e:
                self._reconnect_attempts[symbol] += 1
                print(f"❌ Error en depth {symbol}: {e}", flush=True)
                logger.error(f"❌ Error en depth {symbol}: {e}")
                
                if self._reconnect_attempts[symbol] >= self._max_reconnects:
                    print(f"🚨 Máximos reintentos alcanzados para {symbol} depth", flush=True)
                    logger.critical(f"🚨 Máximos reintentos alcanzados para {symbol} depth")
                    await asyncio.sleep(60)
                    self._reconnect_attempts[symbol] = 0
                else:
                    wait_time = 2 ** self._reconnect_attempts[symbol]
                    print(f"🔄 Reconectando depth {symbol} en {wait_time}s...", flush=True)
                    logger.info(f"🔄 Reconectando depth {symbol} en {wait_time}s...")
                    await asyncio.sleep(wait_time)
    
    async def _handle_agg_trade_stream(self, symbol: str):
        """Maneja el stream de aggTrades (@aggTrade)"""
        while self._running and not self._closed:
            try:
                ts = self.bm.aggtrade_socket(symbol=symbol)
                async with ts as tscm:
                    print(f"✅ WebSocket aggTrade CONECTADO para {symbol}", flush=True)
                    
                    while self._running and not self._closed:
                        msg = await tscm.recv()
                        
                        # is_buyer_maker: True = venta agresiva, False = compra agresiva
                        await self.symbol_flows[symbol].process_agg_trade(
                            qty=float(msg['q']),
                            price=float(msg['p']),
                            is_buyer_maker=msg['m'],
                            trade_time=msg['T']
                        )
                        
                        self._stream_health[symbol]['agg_trade'] += 1
                        self._reconnect_attempts[symbol] = 0
                        
            except Exception as e:
                self._reconnect_attempts[symbol] += 1
                print(f"❌ Error en aggTrade {symbol}: {e}", flush=True)
                logger.error(f"❌ Error en aggTrade {symbol}: {e}")
                
                if self._reconnect_attempts[symbol] >= self._max_reconnects:
                    print(f"🚨 Máximos reintentos alcanzados para {symbol} aggTrade", flush=True)
                    logger.critical(f"🚨 Máximos reintentos alcanzados para {symbol} aggTrade")
                    await asyncio.sleep(60)
                    self._reconnect_attempts[symbol] = 0
                else:
                    wait_time = 2 ** self._reconnect_attempts[symbol]
                    print(f"🔄 Reconectando aggTrade {symbol} en {wait_time}s...", flush=True)
                    logger.info(f"🔄 Reconectando aggTrade {symbol} en {wait_time}s...")
                    await asyncio.sleep(wait_time)
    
    async def _flush_loop(self):
        """Loop para persistir datos cada 60 segundos"""
        print(f"💾 Flush loop iniciado (cada 60s)", flush=True)
        
        while self._running and not self._closed:
            try:
                await asyncio.sleep(60)
                await self._flush_buffer()
                
                # Verificar rotación de archivo (cambio de día)
                current_date = datetime.utcnow().date()
                if current_date != self._current_date:
                    print(f"📅 Rotación de archivo: {self._current_date} -> {current_date}", flush=True)
                    logger.info(f"📅 Rotación de archivo: {self._current_date} -> {current_date}")
                    self._current_date = current_date
                    
            except Exception as e:
                print(f"❌ Error en flush_loop: {e}", flush=True)
                logger.error(f"❌ Error en flush_loop: {e}")
    
    async def _flush_buffer(self):
        """Escribe el buffer a disco"""
        print(f"CSV WRITE CALLED: {datetime.now()}", flush=True)
        
        async with self._buffer_lock:
            if not self._buffer:
                print(f"⚠️ CSV WRITE: Buffer vacío, nada que escribir", flush=True)
                return
            
            print(f"📝 CSV WRITE: {len(self._buffer)} filas en buffer", flush=True)
            
            # Agrupar por símbolo
            by_symbol: Dict[str, List[Dict]] = {}
            for row in self._buffer:
                sym = row['symbol']
                if sym not in by_symbol:
                    by_symbol[sym] = []
                by_symbol[sym].append(row)
            
            # Escribir cada símbolo
            for symbol, rows in by_symbol.items():
                filename = f"orderflow_{symbol}_{self._current_date.strftime('%Y%m%d')}.csv"
                filepath = os.path.join(self.data_dir, filename)
                
                try:
                    df = pd.DataFrame(rows)
                    
                    # Si existe, append; sino, crear nuevo
                    if os.path.exists(filepath):
                        df.to_csv(filepath, mode='a', header=False, index=False)
                        print(f"💾 CSV APPEND: {filename} ({len(rows)} filas)", flush=True)
                    else:
                        df.to_csv(filepath, index=False)
                        print(f"✅ CSV CREADO: {filepath} ({len(rows)} filas)", flush=True)
                    
                    logger.info(f"💾 Escribiendo {len(rows)} filas a {filepath}")
                    logger.info(f"✅ CSV actualizado: {filename} ({len(rows)} filas)")
                    
                except Exception as e:
                    print(f"❌ Error escribiendo CSV {filepath}: {e}", flush=True)
                    logger.error(f"❌ Error escribiendo CSV {filepath}: {e}")
            
            # Limpiar buffer
            total_rows = len(self._buffer)
            self._buffer.clear()
            self._last_flush = datetime.utcnow()
            print(f"✅ CSV WRITE COMPLETED: {total_rows} filas escritas total", flush=True)
    
    async def _health_check_loop(self):
        """Log de salud de streams cada 5 minutos"""
        # CRITICAL FIX: Log inmediato confirmando inicio del scheduler
        print("🏥 Health Check scheduler INICIADO (cada 5 minutos)", flush=True)
        logger.info("🏥 Health Check scheduler INICIADO (cada 5 minutos)")
        
        # Esperar un poco para que los streams comiencen a recibir datos
        await asyncio.sleep(5)
        
        while self._running and not self._closed:
            try:
                await asyncio.sleep(300)  # 5 minutos
                
                # Construir mensaje de salud
                health_msgs = []
                for symbol in self.symbols:
                    h = self._stream_health[symbol]
                    flow = self.symbol_flows[symbol]
                    metrics = flow.metrics
                    
                    if self.use_polling:
                        health_msgs.append(
                            f"{symbol} spread={metrics.spread_bps:.2f}bps "
                            f"imbalance={metrics.bid_ask_imbalance:.3f} "
                            f"(polls:{h['polls']})"
                        )
                    else:
                        health_msgs.append(
                            f"{symbol} spread={metrics.spread_bps:.2f}bps "
                            f"imbalance={metrics.bid_ask_imbalance:.3f} "
                            f"(bt:{h['book_ticker']} d:{h['depth']} at:{h['agg_trade']})"
                        )
                    
                    # Reset contadores
                    h['book_ticker'] = 0
                    h['depth'] = 0
                    h['agg_trade'] = 0
                    h['polls'] = 0
                
                msg = f"✅ OrderFlow streams activos | {' | '.join(health_msgs)}"
                print(msg, flush=True)
                logger.info(msg)
                
            except Exception as e:
                print(f"❌ Error en health_check: {e}", flush=True)
                logger.error(f"❌ Error en health_check: {e}")
    
    async def capture_snapshot(self):
        """Captura snapshot actual de todas las métricas"""
        snapshot = []
        timestamp = datetime.utcnow().isoformat()
        
        for symbol, flow in self.symbol_flows.items():
            metrics = flow.get_metrics()
            row = {
                'timestamp': timestamp,
                'symbol': symbol,
                'bid_ask_imbalance': metrics.bid_ask_imbalance,
                'book_imbalance_5': metrics.book_imbalance_5,
                'delta_volume_1m': metrics.delta_volume_1m,
                'buy_pressure_1m': metrics.buy_pressure_1m,
                'spread_bps': metrics.spread_bps,
                'weighted_mid_price': metrics.weighted_mid_price,
                'large_trade_detected': metrics.large_trade_detected,
            }
            snapshot.append(row)
            
            # Agregar al buffer para persistencia
            async with self._buffer_lock:
                self._buffer.append(row)
                print(f"📥 BUFFER ADD: {symbol} - buffer size: {len(self._buffer)}", flush=True)
        
        return snapshot
    
    def get_current_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retorna las métricas calculadas actuales para un símbolo.
        
        Args:
            symbol: Símbolo (e.g., 'BTCUSDT', 'ETHUSDT')
            
        Returns:
            Dict con todas las métricas o None si el símbolo no existe
        """
        if symbol not in self.symbol_flows:
            logger.warning(f"⚠️ Símbolo {symbol} no está siendo monitoreado")
            return None
        
        flow = self.symbol_flows[symbol]
        metrics = flow.get_metrics()
        
        return {
            'timestamp': metrics.last_update.isoformat(),
            'symbol': symbol,
            'bid_ask_imbalance': metrics.bid_ask_imbalance,
            'book_imbalance_5': metrics.book_imbalance_5,
            'delta_volume_1m': metrics.delta_volume_1m,
            'buy_pressure_1m': metrics.buy_pressure_1m,
            'spread_bps': metrics.spread_bps,
            'weighted_mid_price': metrics.weighted_mid_price,
            'large_trade_detected': metrics.large_trade_detected,
            'best_bid_price': metrics.best_bid_price,
            'best_ask_price': metrics.best_ask_price,
        }
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """Retorna métricas para todos los símbolos"""
        return {s: self.get_current_features(s) for s in self.symbols}
    
    async def close(self):
        """Cierra todas las conexiones de forma limpia"""
        if self._closed:
            return
        
        print("🛑 Cerrando OrderFlowCapture...", flush=True)
        logger.info("🛑 Cerrando OrderFlowCapture...")
        self._running = False
        self._closed = True
        
        # Flush final
        try:
            await self._flush_buffer()
            print("💾 Flush final completado", flush=True)
            logger.info("💾 Flush final completado")
        except Exception as e:
            print(f"❌ Error en flush final: {e}", flush=True)
            logger.error(f"❌ Error en flush final: {e}")
        
        # Cancelar tareas
        for task in self._stream_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._stream_tasks.clear()
        
        # Cerrar cliente REST si existe
        if self._session:
            try:
                await self._session.close()
                print("✅ Session REST API cerrada", flush=True)
                logger.info("✅ Session REST API cerrada")
            except Exception as e:
                print(f"⚠️ Error cerrando session REST: {e}", flush=True)
                logger.warning(f"⚠️ Error cerrando session REST: {e}")
        
        # Cerrar cliente WebSocket si existe
        if self.client:
            try:
                await self.client.close_connection()
                print("✅ Conexión Binance WebSocket cerrada", flush=True)
                logger.info("✅ Conexión Binance WebSocket cerrada")
            except Exception as e:
                print(f"⚠️ Error cerrando conexión WebSocket: {e}", flush=True)
                logger.warning(f"⚠️ Error cerrando conexión WebSocket: {e}")
        
        print("✅ OrderFlowCapture cerrado correctamente", flush=True)
        logger.info("✅ OrderFlowCapture cerrado correctamente")
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Instancia global para acceso desde main.py
_order_flow_capture: Optional[OrderFlowCapture] = None


def get_order_flow_capture() -> Optional[OrderFlowCapture]:
    """Retorna la instancia global de OrderFlowCapture"""
    return _order_flow_capture


def set_order_flow_capture(capture: OrderFlowCapture):
    """Establece la instancia global"""
    global _order_flow_capture
    _order_flow_capture = capture
    print("✅ OrderFlowCapture registrado globalmente", flush=True)
    logger.info("✅ OrderFlowCapture registrado globalmente")


async def create_and_start_order_flow_capture(
    symbols: List[str] = None,
    data_dir: str = None,
    use_polling: bool = None
) -> OrderFlowCapture:
    """
    Factory function para crear e iniciar OrderFlowCapture.
    
    Args:
        symbols: Lista de símbolos a monitorear
        data_dir: Directorio para guardar CSVs
        use_polling: Si True, usa REST API polling en lugar de websockets
    
    Uso:
        order_flow = await create_and_start_order_flow_capture()
        # o con polling:
        order_flow = await create_and_start_order_flow_capture(use_polling=True)
    """
    capture = OrderFlowCapture(symbols=symbols, data_dir=data_dir, use_polling=use_polling)
    await capture.start()
    set_order_flow_capture(capture)
    return capture
