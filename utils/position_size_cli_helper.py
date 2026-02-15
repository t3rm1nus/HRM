# -*- coding: utf-8 -*-
"""
Position Size CLI Helper

Helper de CLI que para cada señal de compra o venta:
1. Lee el balance disponible del asset desde balance_cache
2. Lee el porcentaje de allocation deseado y el precio actual del mercado
3. Calcula la cantidad (qty) a operar redondeando al mínimo decimal permitido
4. Si qty < min_qty: marca la señal como REJECTED y no genera orden
5. Si qty es válido: permite la creación de la orden con ese qty

Uso:
    from utils.position_size_cli_helper import PositionSizeCLIHelper
    
    helper = PositionSizeCLIHelper(portfolio_manager)
    result = await helper.calculate_position_size(
        signal=signal,
        current_price=50000.0,
        allocation_pct=0.10,  # 10% del balance disponible
        min_order_value=2.0   # Mínimo $2 USDT
    )
    
    if result.is_valid:
        order_qty = result.qty
    else:
        # Señal rechazada - no generar orden
        print(f"Señal inválida: {result.rejection_reason}")
"""

import asyncio
import threading
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime
from concurrent.futures import Future

# Lazy import del logger para evitar circular imports
logger = None

def get_logger():
    global logger
    if logger is None:
        try:
            from core.logging import logger as core_logger
            logger = core_logger
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
    return logger


@dataclass
class PositionSizeResult:
    """
    Resultado del cálculo de tamaño de posición.
    
    Attributes:
        is_valid: True si el qty calculado es válido para operar
        qty: Cantidad calculada (0.0 si no es válida)
        qty_raw: Cantidad antes del redondeo (para debugging)
        order_value_usd: Valor de la orden en USD
        balance_used: Balance utilizado para el cálculo
        allocation_pct: Porcentaje de allocation usado
        min_qty_threshold: Umbral mínimo de qty aplicado
        min_order_value: Valor mínimo de orden requerido
        min_order_qty: Cantidad mínima de orden requerida
        rejection_reason: Razón del rechazo (si is_valid es False)
        signal_status: Estado de la señal ('VALID', 'REJECTED')
        metadata: Metadatos adicionales del cálculo
    """
    is_valid: bool
    qty: float
    qty_raw: float
    order_value_usd: float
    balance_used: float
    allocation_pct: float
    min_qty_threshold: float
    min_order_value: float = 2.0
    min_order_qty: float = 0.0
    rejection_reason: Optional[str] = None
    signal_status: str = "VALID"  # 'VALID' o 'REJECTED'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Actualizar signal_status basado en is_valid
        self.signal_status = "VALID" if self.is_valid else "REJECTED"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el resultado a diccionario para logging/JSON"""
        return {
            "is_valid": self.is_valid,
            "signal_status": self.signal_status,
            "qty": round(self.qty, 8),
            "qty_raw": round(self.qty_raw, 8),
            "order_value_usd": round(self.order_value_usd, 2),
            "balance_used": round(self.balance_used, 8),
            "allocation_pct": round(self.allocation_pct, 4),
            "min_qty_threshold": round(self.min_qty_threshold, 8),
            "min_order_value": round(self.min_order_value, 2),
            "min_order_qty": round(self.min_order_qty, 8),
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata
        }


class AsyncLoopDetector:
    """
    Detector de estado del asyncio event loop.
    Utilidad para determinar si el loop está corriendo y en qué thread.
    """
    
    @staticmethod
    def is_loop_running() -> bool:
        """
        Detecta si hay un asyncio event loop corriendo en el thread actual.
        
        Returns:
            True si el loop está corriendo, False en caso contrario
        """
        try:
            loop = asyncio.get_running_loop()
            return loop.is_running()
        except RuntimeError:
            # No hay loop corriendo en este thread
            return False
    
    @staticmethod
    def get_loop() -> Optional[asyncio.AbstractEventLoop]:
        """
        Obtiene el asyncio event loop actual si existe.
        
        Returns:
            El event loop o None si no hay ninguno
        """
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.get_event_loop()
            except RuntimeError:
                return None
    
    @staticmethod
    def is_in_event_loop_thread() -> bool:
        """
        Verifica si el código se está ejecutando en el thread del event loop.
        
        Returns:
            True si estamos en el thread del event loop
        """
        try:
            loop = asyncio.get_running_loop()
            return loop._thread_id == threading.current_thread().ident
        except RuntimeError:
            return False


class PositionSizeCLIHelper:
    """
    Helper de CLI para cálculo de tamaños de posición con validación completa.
    
    Integra con:
    - balance_cache del PortfolioManager
    - Sistema de allocation porcentual
    - Validación de mínimos de orden
    - Detección de asyncio loop para ejecución segura
    """
    
    # Configuración por defecto de símbolos soportados
    SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    
    # Precisión decimal por asset (mínimo decimal permitido en exchanges)
    ASSET_PRECISION = {
        "BTC": 6,   # 0.000001 BTC mínimo
        "ETH": 5,   # 0.00001 ETH mínimo
        "USDT": 2   # 0.01 USDT mínimo
    }
    
    # Valor mínimo de orden por defecto (en USDT)
    DEFAULT_MIN_ORDER_VALUE = 2.0
    
    # Cantidad mínima de orden por defecto (en unidades del asset)
    DEFAULT_MIN_ORDER_QTY = 0.00001
    
    def __init__(
        self,
        portfolio_manager: Any,
        min_order_value: float = DEFAULT_MIN_ORDER_VALUE,
        min_order_qty: float = DEFAULT_MIN_ORDER_QTY,
        default_allocation_pct: float = 0.10  # 10% por defecto
    ):
        """
        Inicializa el helper.
        
        Args:
            portfolio_manager: Instancia de PortfolioManager con acceso a balance_cache
            min_order_value: Valor mínimo de orden en USDT (default: $2.0)
            min_order_qty: Cantidad mínima de orden en unidades del asset (default: 0.00001)
            default_allocation_pct: Allocation porcentual por defecto (default: 10%)
        """
        self.portfolio_manager = portfolio_manager
        self.min_order_value = min_order_value
        self.min_order_qty = min_order_qty
        self.default_allocation_pct = default_allocation_pct
        
        get_logger().info(
            f"✅ PositionSizeCLIHelper initialized | "
            f"min_order_value=${min_order_value:.2f} | "
            f"min_order_qty={min_order_qty:.8f} | "
            f"default_allocation={default_allocation_pct:.1%}"
        )
    
    def _run_async_safely(self, coro: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta una coroutine de forma segura considerando el estado del event loop.
        
        Si el loop está corriendo:
        - Usa asyncio.create_task si estamos en el thread del loop
        - Usa asyncio.run_coroutine_threadsafe si estamos en otro thread
        
        Si no hay loop corriendo:
        - Usa asyncio.run para crear uno nuevo
        
        Args:
            coro: Coroutine a ejecutar
            *args, **kwargs: Argumentos para la coroutine
            
        Returns:
            Resultado de la coroutine
        """
        detector = AsyncLoopDetector()
        
        if detector.is_loop_running():
            loop = detector.get_loop()
            
            if detector.is_in_event_loop_thread():
                # Estamos en el thread del event loop
                get_logger().debug("[ASYNC] Loop running in current thread, using create_task")
                try:
                    # Crear task y esperar su resultado
                    task = asyncio.create_task(coro(*args, **kwargs))
                    # Para obtener el resultado necesitamos await, pero estamos en sync context
                    # Devolvemos un future que puede ser awaited por el caller
                    return task
                except Exception as e:
                    get_logger().error(f"[ASYNC] Error creating task: {e}")
                    raise
            else:
                # Estamos en otro thread, usar run_coroutine_threadsafe
                get_logger().debug("[ASYNC] Loop running in different thread, using run_coroutine_threadsafe")
                try:
                    future = asyncio.run_coroutine_threadsafe(coro(*args, **kwargs), loop)
                    # Esperar el resultado con timeout
                    return future.result(timeout=30.0)
                except Exception as e:
                    get_logger().error(f"[ASYNC] Error in run_coroutine_threadsafe: {e}")
                    raise
        else:
            # No hay loop corriendo, crear uno nuevo
            get_logger().debug("[ASYNC] No loop running, using asyncio.run")
            try:
                return asyncio.run(coro(*args, **kwargs))
            except Exception as e:
                get_logger().error(f"[ASYNC] Error in asyncio.run: {e}")
                raise
    
    async def get_balance_from_cache(self, asset: str) -> Optional[float]:
        """
        Obtiene el balance desde el balance_cache del PortfolioManager.
        
        Args:
            asset: Asset a consultar (BTC, ETH, USDT)
            
        Returns:
            Balance disponible o None si no está en cache
        """
        try:
            # Normalizar asset
            asset = asset.upper().replace("USDT", "")
            
            # Intentar obtener desde el balance_cache del portfolio_manager
            if hasattr(self.portfolio_manager, '_balance_cache'):
                cache = self.portfolio_manager._balance_cache
                if asset in cache:
                    balance = cache[asset]
                    get_logger().debug(
                        f"[BALANCE_CACHE] {asset}: {balance:.8f} "
                        f"(source: _balance_cache)"
                    )
                    return float(balance)
            
            # Fallback: usar método async del portfolio_manager
            if hasattr(self.portfolio_manager, 'get_asset_balance_async'):
                balance = await self.portfolio_manager.get_asset_balance_async(asset)
                get_logger().debug(
                    f"[BALANCE_CACHE] {asset}: {balance:.8f} "
                    f"(source: get_asset_balance_async)"
                )
                return float(balance)
            
            # Último fallback: usar portfolio dict
            if hasattr(self.portfolio_manager, 'portfolio'):
                portfolio = self.portfolio_manager.portfolio
                if asset == "USDT":
                    balance = portfolio.get("USDT", {}).get("free", 0.0)
                else:
                    symbol = f"{asset}USDT"
                    balance = portfolio.get(symbol, {}).get("free", 0.0)
                get_logger().debug(
                    f"[BALANCE_CACHE] {asset}: {balance:.8f} "
                    f"(source: portfolio dict)"
                )
                return float(balance)
            
            get_logger().warning(f"⚠️ No balance source available for {asset}")
            return None
            
        except Exception as e:
            get_logger().error(f"❌ Error getting balance from cache for {asset}: {e}")
            return None
    
    def get_asset_precision(self, asset: str) -> int:
        """
        Obtiene la precisión decimal para un asset.
        
        Args:
            asset: Asset (BTC, ETH, USDT)
            
        Returns:
            Número de decimales permitidos
        """
        asset = asset.upper().replace("USDT", "")
        return self.ASSET_PRECISION.get(asset, 6)  # Default 6 decimales
    
    def round_to_precision(self, qty: float, asset: str) -> float:
        """
        Redondea la cantidad al mínimo decimal permitido para el asset.
        
        Args:
            qty: Cantidad a redondear
            asset: Asset para determinar precisión
            
        Returns:
            Cantidad redondeada hacia abajo
        """
        precision = self.get_asset_precision(asset)
        
        # Usar Decimal para precisión exacta
        decimal_places = Decimal(10) ** (-precision)
        qty_decimal = Decimal(str(qty))
        rounded = qty_decimal.quantize(decimal_places, rounding=ROUND_DOWN)
        
        return float(rounded)
    
    def calculate_min_qty_threshold(
        self,
        current_price: float,
        asset: str,
        min_order_value: Optional[float] = None
    ) -> float:
        """
        Calcula el threshold mínimo de qty basado en el valor mínimo de orden.
        
        Args:
            current_price: Precio actual del mercado
            asset: Asset a operar
            min_order_value: Valor mínimo de orden (usa default si None)
            
        Returns:
            Cantidad mínima permitida
        """
        min_value = min_order_value or self.min_order_value
        
        if current_price <= 0:
            get_logger().warning(f"⚠️ Invalid current_price: {current_price}")
            return 0.0
        
        # Calcular qty mínima teórica
        min_qty_raw = min_value / current_price
        
        # Redondear a la precisión del asset
        min_qty = self.round_to_precision(min_qty_raw, asset)
        
        return min_qty
    
    def _validate_qty(
        self,
        qty: float,
        min_qty: float,
        min_value: float,
        order_value: float,
        current_price: float
    ) -> tuple[bool, Optional[str]]:
        """
        Valida que el qty cumpla con todos los requisitos mínimos.
        
        Args:
            qty: Cantidad calculada
            min_qty: Cantidad mínima requerida
            min_value: Valor mínimo de orden requerido
            order_value: Valor de la orden calculada
            current_price: Precio actual
            
        Returns:
            Tuple (is_valid, rejection_reason)
        """
        # Validar contra qty mínimo
        if qty <= 0:
            return False, f"QTY_ZERO: Calculated qty={qty:.8f} is zero or negative"
        
        if qty < min_qty:
            return False, (
                f"QTY_BELOW_MIN_ORDER_QTY: qty={qty:.8f} < min_order_qty={min_qty:.8f} "
                f"(minimum quantity required)"
            )
        
        # Validar contra valor mínimo de orden
        if order_value < min_value:
            return False, (
                f"ORDER_VALUE_BELOW_MIN: order_value=${order_value:.2f} < min_order_value=${min_value:.2f} "
                f"(qty={qty:.8f} @ ${current_price:.2f})"
            )
        
        return True, None
    
    async def calculate_position_size(
        self,
        symbol: str,
        side: str,
        current_price: float,
        allocation_pct: Optional[float] = None,
        min_order_value: Optional[float] = None,
        min_order_qty: Optional[float] = None,
        use_balance_cache: bool = True,
        custom_balance: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Calcula el tamaño de posición para una señal de trading.
        
        Este es el método principal que:
        1. Lee balance desde cache (detectando asyncio loop)
        2. Aplica allocation porcentual
        3. Calcula qty y redondea
        4. Valida contra mínimos (qty y value)
        5. Marca señal como REJECTED si es inválida
        
        Args:
            symbol: Símbolo (BTCUSDT, ETHUSDT)
            side: Lado de la operación ('buy' o 'sell')
            current_price: Precio actual del mercado
            allocation_pct: Porcentaje de allocation (usa default si None)
            min_order_value: Valor mínimo de orden (usa default si None)
            min_order_qty: Cantidad mínima de orden (usa default si None)
            use_balance_cache: Si es True, usa balance_cache; si es False, fuerza fetch
            custom_balance: Balance custom para override (ignora cache si se proporciona)
            
        Returns:
            PositionSizeResult con toda la información del cálculo
        """
        start_time = datetime.utcnow()
        
        # Normalizar inputs
        symbol = symbol.upper()
        side = side.lower()
        asset = symbol.replace("USDT", "")
        
        allocation = allocation_pct or self.default_allocation_pct
        min_value = min_order_value or self.min_order_value
        min_qty = min_order_qty or self.min_order_qty
        
        # Validar inputs básicos
        if current_price is None or current_price <= 0:
            return PositionSizeResult(
                is_valid=False,
                qty=0.0,
                qty_raw=0.0,
                order_value_usd=0.0,
                balance_used=0.0,
                allocation_pct=allocation,
                min_qty_threshold=0.0,
                min_order_value=min_value,
                min_order_qty=min_qty,
                rejection_reason=f"INVALID_PRICE: {current_price}",
                signal_status="REJECTED",
                metadata={"symbol": symbol, "side": side}
            )
        
        if side not in ["buy", "sell"]:
            return PositionSizeResult(
                is_valid=False,
                qty=0.0,
                qty_raw=0.0,
                order_value_usd=0.0,
                balance_used=0.0,
                allocation_pct=allocation,
                min_qty_threshold=0.0,
                min_order_value=min_value,
                min_order_qty=min_qty,
                rejection_reason=f"INVALID_SIDE: {side}",
                signal_status="REJECTED",
                metadata={"symbol": symbol, "side": side}
            )
        
        # ========================================================================
        # PASO 1: Obtener balance disponible (con detección de asyncio loop)
        # ========================================================================
        
        if custom_balance is not None:
            # Usar balance custom proporcionado
            available_balance = custom_balance
            balance_source = "custom_override"
            get_logger().info(f"[BALANCE] {symbol} {side}: Using custom balance = {available_balance:.8f}")
        elif use_balance_cache:
            # Usar balance_cache (modo normal)
            if side == "buy":
                # Para BUY, necesitamos USDT
                available_balance = await self.get_balance_from_cache("USDT")
                balance_source = "balance_cache_USDT"
            else:
                # Para SELL, necesitamos el asset base
                available_balance = await self.get_balance_from_cache(asset)
                balance_source = f"balance_cache_{asset}"
        else:
            # Fuerza fetch fresco (sin cache)
            if hasattr(self.portfolio_manager, 'get_asset_balance_async'):
                if side == "buy":
                    available_balance = await self.portfolio_manager.get_asset_balance_async("USDT")
                else:
                    available_balance = await self.portfolio_manager.get_asset_balance_async(asset)
                balance_source = "fresh_fetch"
            else:
                available_balance = await self.get_balance_from_cache("USDT" if side == "buy" else asset)
                balance_source = "cache_fallback"
        
        # Validar que obtuvimos un balance
        if available_balance is None:
            return PositionSizeResult(
                is_valid=False,
                qty=0.0,
                qty_raw=0.0,
                order_value_usd=0.0,
                balance_used=0.0,
                allocation_pct=allocation,
                min_qty_threshold=0.0,
                min_order_value=min_value,
                min_order_qty=min_qty,
                rejection_reason="BALANCE_NOT_AVAILABLE",
                signal_status="REJECTED",
                metadata={"symbol": symbol, "side": side, "source": balance_source}
            )
        
        if available_balance <= 0:
            return PositionSizeResult(
                is_valid=False,
                qty=0.0,
                qty_raw=0.0,
                order_value_usd=0.0,
                balance_used=available_balance,
                allocation_pct=allocation,
                min_qty_threshold=0.0,
                min_order_value=min_value,
                min_order_qty=min_qty,
                rejection_reason=f"INSUFFICIENT_BALANCE: {available_balance:.8f}",
                signal_status="REJECTED",
                metadata={"symbol": symbol, "side": side, "source": balance_source}
            )
        
        # ========================================================================
        # PASO 2: Calcular qty basado en allocation
        # ========================================================================
        
        # Calcular valor a asignar
        allocated_value = available_balance * allocation
        
        if side == "buy":
            # Para BUY: qty = (USDT disponible * allocation) / precio
            qty_raw = allocated_value / current_price
        else:
            # Para SELL: qty = asset disponible * allocation
            qty_raw = allocated_value
        
        # ========================================================================
        # PASO 3: Redondear al mínimo decimal permitido
        # ========================================================================
        
        qty_rounded = self.round_to_precision(qty_raw, asset)
        
        # ========================================================================
        # PASO 4: Calcular thresholds mínimos y validar
        # ========================================================================
        
        min_qty_threshold = self.calculate_min_qty_threshold(
            current_price=current_price,
            asset=asset,
            min_order_value=min_value
        )
        
        # Usar el máximo entre min_qty_threshold calculado y min_order_qty configurado
        effective_min_qty = max(min_qty_threshold, min_qty)
        
        order_value = qty_rounded * current_price
        
        # Validar contra mínimos usando el nuevo método de validación
        is_valid, rejection_reason = self._validate_qty(
            qty=qty_rounded,
            min_qty=effective_min_qty,
            min_value=min_value,
            order_value=order_value,
            current_price=current_price
        )
        
        # Calcular tiempo de procesamiento
        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Construir metadata
        metadata = {
            "symbol": symbol,
            "side": side,
            "asset": asset,
            "balance_source": balance_source,
            "current_price": current_price,
            "allocation_pct": allocation,
            "available_balance": available_balance,
            "allocated_value": allocated_value,
            "processing_time_ms": round(processing_time_ms, 2),
            "precision": self.get_asset_precision(asset),
            "async_loop_detected": AsyncLoopDetector.is_loop_running(),
            "in_loop_thread": AsyncLoopDetector.is_in_event_loop_thread()
        }
        
        # Log del resultado
        if is_valid:
            get_logger().info(
                f"✅ [POSITION_SIZE] {symbol} {side.upper()}: "
                f"qty={qty_rounded:.8f} (raw={qty_raw:.8f}) | "
                f"value=${order_value:.2f} | "
                f"allocation={allocation:.1%} of {available_balance:.8f} | "
                f"min_threshold={effective_min_qty:.8f}"
            )
        else:
            get_logger().warning(
                f"❌ [POSITION_SIZE_REJECTED] {symbol} {side.upper()}: "
                f"{rejection_reason} | "
                f"balance={available_balance:.8f} | "
                f"allocation={allocation:.1%} | "
                f"STATUS=REJECTED"
            )
        
        return PositionSizeResult(
            is_valid=is_valid,
            qty=qty_rounded if is_valid else 0.0,
            qty_raw=qty_raw,
            order_value_usd=order_value,
            balance_used=available_balance,
            allocation_pct=allocation,
            min_qty_threshold=effective_min_qty,
            min_order_value=min_value,
            min_order_qty=min_qty,
            rejection_reason=rejection_reason,
            signal_status="VALID" if is_valid else "REJECTED",
            metadata=metadata
        )
    
    def calculate_position_size_sync(
        self,
        symbol: str,
        side: str,
        current_price: float,
        allocation_pct: Optional[float] = None,
        min_order_value: Optional[float] = None,
        min_order_qty: Optional[float] = None,
        use_balance_cache: bool = True,
        custom_balance: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Versión síncrona del cálculo de posición que detecta el asyncio loop
        y ejecuta de forma segura usando create_task o run_coroutine_threadsafe.
        
        Args:
            symbol: Símbolo (BTCUSDT, ETHUSDT)
            side: Lado de la operación ('buy' o 'sell')
            current_price: Precio actual del mercado
            allocation_pct: Porcentaje de allocation (usa default si None)
            min_order_value: Valor mínimo de orden (usa default si None)
            min_order_qty: Cantidad mínima de orden (usa default si None)
            use_balance_cache: Si es True, usa balance_cache; si es False, fuerza fetch
            custom_balance: Balance custom para override (ignora cache si se proporciona)
            
        Returns:
            PositionSizeResult con toda la información del cálculo
        """
        detector = AsyncLoopDetector()
        
        if detector.is_loop_running():
            loop = detector.get_loop()
            
            if detector.is_in_event_loop_thread():
                # Estamos en el thread del event loop
                get_logger().debug("[SYNC_WRAPPER] Loop running in current thread, using create_task")
                # No podemos hacer await aquí, así que creamos la task y devolvemos un resultado parcial
                # El caller debe manejar esto apropiadamente
                task = asyncio.create_task(
                    self.calculate_position_size(
                        symbol=symbol,
                        side=side,
                        current_price=current_price,
                        allocation_pct=allocation_pct,
                        min_order_value=min_order_value,
                        min_order_qty=min_order_qty,
                        use_balance_cache=use_balance_cache,
                        custom_balance=custom_balance
                    )
                )
                # Retornar un resultado indicando que es un task pendiente
                return PositionSizeResult(
                    is_valid=False,
                    qty=0.0,
                    qty_raw=0.0,
                    order_value_usd=0.0,
                    balance_used=0.0,
                    allocation_pct=allocation_pct or self.default_allocation_pct,
                    min_qty_threshold=0.0,
                    min_order_value=min_order_value or self.min_order_value,
                    min_order_qty=min_order_qty or self.min_order_qty,
                    rejection_reason="ASYNC_TASK_PENDING",
                    signal_status="PENDING",
                    metadata={
                        "symbol": symbol,
                        "side": side,
                        "async_task": True,
                        "task": task
                    }
                )
            else:
                # Estamos en otro thread, usar run_coroutine_threadsafe
                get_logger().debug("[SYNC_WRAPPER] Loop running in different thread, using run_coroutine_threadsafe")
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.calculate_position_size(
                            symbol=symbol,
                            side=side,
                            current_price=current_price,
                            allocation_pct=allocation_pct,
                            min_order_value=min_order_value,
                            min_order_qty=min_order_qty,
                            use_balance_cache=use_balance_cache,
                            custom_balance=custom_balance
                        ),
                        loop
                    )
                    return future.result(timeout=30.0)
                except Exception as e:
                    get_logger().error(f"[SYNC_WRAPPER] Error: {e}")
                    return PositionSizeResult(
                        is_valid=False,
                        qty=0.0,
                        qty_raw=0.0,
                        order_value_usd=0.0,
                        balance_used=0.0,
                        allocation_pct=allocation_pct or self.default_allocation_pct,
                        min_qty_threshold=0.0,
                        min_order_value=min_order_value or self.min_order_value,
                        min_order_qty=min_order_qty or self.min_order_qty,
                        rejection_reason=f"ASYNC_EXECUTION_ERROR: {str(e)}",
                        signal_status="REJECTED",
                        metadata={"symbol": symbol, "side": side, "error": str(e)}
                    )
        else:
            # No hay loop corriendo, usar asyncio.run
            get_logger().debug("[SYNC_WRAPPER] No loop running, using asyncio.run")
            try:
                return asyncio.run(
                    self.calculate_position_size(
                        symbol=symbol,
                        side=side,
                        current_price=current_price,
                        allocation_pct=allocation_pct,
                        min_order_value=min_order_value,
                        min_order_qty=min_order_qty,
                        use_balance_cache=use_balance_cache,
                        custom_balance=custom_balance
                    )
                )
            except Exception as e:
                get_logger().error(f"[SYNC_WRAPPER] Error in asyncio.run: {e}")
                return PositionSizeResult(
                    is_valid=False,
                    qty=0.0,
                    qty_raw=0.0,
                    order_value_usd=0.0,
                    balance_used=0.0,
                    allocation_pct=allocation_pct or self.default_allocation_pct,
                    min_qty_threshold=0.0,
                    min_order_value=min_order_value or self.min_order_value,
                    min_order_qty=min_order_qty or self.min_order_qty,
                    rejection_reason=f"ASYNC_RUN_ERROR: {str(e)}",
                    signal_status="REJECTED",
                    metadata={"symbol": symbol, "side": side, "error": str(e)}
                )
    
    async def validate_signal_for_order(
        self,
        signal: Any,
        current_price: float,
        allocation_pct: Optional[float] = None,
        min_order_value: Optional[float] = None,
        min_order_qty: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Valida una señal de trading completa y calcula el qty si es válida.
        
        Este método es un wrapper conveniente que extrae información del signal object.
        
        Args:
            signal: Objeto de señal (debe tener .symbol, .side)
            current_price: Precio actual del mercado
            allocation_pct: Porcentaje de allocation (opcional)
            min_order_value: Valor mínimo de orden (opcional)
            min_order_qty: Cantidad mínima de orden (opcional)
            
        Returns:
            PositionSizeResult con el resultado de la validación
        """
        # Extraer información del signal
        symbol = getattr(signal, 'symbol', None)
        side = getattr(signal, 'side', None)
        
        if symbol is None or side is None:
            return PositionSizeResult(
                is_valid=False,
                qty=0.0,
                qty_raw=0.0,
                order_value_usd=0.0,
                balance_used=0.0,
                allocation_pct=allocation_pct or self.default_allocation_pct,
                min_qty_threshold=0.0,
                min_order_value=min_order_value or self.min_order_value,
                min_order_qty=min_order_qty or self.min_order_qty,
                rejection_reason=f"INVALID_SIGNAL: missing symbol={symbol} or side={side}",
                signal_status="REJECTED",
                metadata={"signal_type": type(signal).__name__}
            )
        
        # Intentar extraer allocation_pct del signal si no se proporcionó
        if allocation_pct is None:
            # Buscar en metadata o attributes del signal
            if hasattr(signal, 'metadata') and signal.metadata:
                allocation_pct = signal.metadata.get('allocation_pct')
            if allocation_pct is None and hasattr(signal, 'confidence'):
                # Usar confidence como proxy de allocation (común en el sistema)
                allocation_pct = signal.confidence
        
        return await self.calculate_position_size(
            symbol=symbol,
            side=side,
            current_price=current_price,
            allocation_pct=allocation_pct,
            min_order_value=min_order_value,
            min_order_qty=min_order_qty,
            use_balance_cache=True
        )
    
    async def batch_calculate(
        self,
        signals: list,
        market_data: Dict[str, float],
        allocation_pct: Optional[float] = None,
        min_order_value: Optional[float] = None,
        min_order_qty: Optional[float] = None
    ) -> Dict[str, PositionSizeResult]:
        """
        Calcula posiciones para múltiples señales en batch.
        
        Args:
            signals: Lista de señales a procesar
            market_data: Dict con precios {symbol: price}
            allocation_pct: Allocation por defecto
            min_order_value: Valor mínimo por defecto
            min_order_qty: Cantidad mínima por defecto
            
        Returns:
            Dict mapeando symbol a PositionSizeResult
        """
        results = {}
        
        for signal in signals:
            symbol = getattr(signal, 'symbol', None)
            if symbol is None:
                continue
            
            current_price = market_data.get(symbol)
            if current_price is None:
                results[symbol] = PositionSizeResult(
                    is_valid=False,
                    qty=0.0,
                    qty_raw=0.0,
                    order_value_usd=0.0,
                    balance_used=0.0,
                    allocation_pct=allocation_pct or self.default_allocation_pct,
                    min_qty_threshold=0.0,
                    min_order_value=min_order_value or self.min_order_value,
                    min_order_qty=min_order_qty or self.min_order_qty,
                    rejection_reason=f"NO_PRICE_DATA for {symbol}",
                    signal_status="REJECTED",
                    metadata={}
                )
                continue
            
            result = await self.validate_signal_for_order(
                signal=signal,
                current_price=current_price,
                allocation_pct=allocation_pct,
                min_order_value=min_order_value,
                min_order_qty=min_order_qty
            )
            results[symbol] = result
        
        return results
    
    def format_result_for_cli(self, result: PositionSizeResult) -> str:
        """
        Formatea un resultado para mostrar en CLI.
        
        Args:
            result: PositionSizeResult a formatear
            
        Returns:
            String formateado para CLI
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"📊 POSITION SIZE CALCULATION RESULT")
        lines.append("=" * 70)
        
        meta = result.metadata
        lines.append(f"Symbol:     {meta.get('symbol', 'N/A')} {meta.get('side', 'N/A').upper()}")
        lines.append(f"Price:      ${meta.get('current_price', 0):.2f}")
        lines.append(f"Balance:    {result.balance_used:.8f} (source: {meta.get('balance_source', 'unknown')})")
        lines.append(f"Allocation: {result.allocation_pct:.1%}")
        lines.append("-" * 70)
        lines.append(f"Qty (raw):  {result.qty_raw:.8f}")
        lines.append(f"Qty (rounded): {result.qty:.8f} (precision: {meta.get('precision', 'N/A')} decimals)")
        lines.append(f"Order Value: ${result.order_value_usd:.2f}")
        lines.append(f"Min Threshold: {result.min_qty_threshold:.8f}")
        lines.append(f"Min Order Value: ${result.min_order_value:.2f}")
        lines.append(f"Min Order Qty: {result.min_order_qty:.8f}")
        lines.append("-" * 70)
        
        if result.is_valid:
            lines.append(f"✅ RESULT: VALID - Order can be created with qty={result.qty:.8f}")
        else:
            lines.append(f"❌ RESULT: {result.signal_status} - {result.rejection_reason}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# =============================================================================
# FUNCIONES HELPER INDEPENDIENTES
# =============================================================================

def is_async_loop_running() -> bool:
    """
    Función helper para detectar si hay un asyncio loop corriendo.
    
    Returns:
        True si el loop está corriendo, False en caso contrario
    """
    return AsyncLoopDetector.is_loop_running()


def validate_qty_requirements(
    qty: float,
    min_order_qty: float,
    min_order_value: float,
    order_value: float,
    current_price: float
) -> tuple[bool, Optional[str]]:
    """
    Valida que el qty cumpla con todos los requisitos mínimos.
    
    Args:
        qty: Cantidad calculada
        min_order_qty: Cantidad mínima requerida
        min_order_value: Valor mínimo de orden requerido
        order_value: Valor de la orden calculada
        current_price: Precio actual
        
    Returns:
        Tuple (is_valid, rejection_reason)
    """
    # Validar contra qty mínimo
    if qty <= 0:
        return False, f"QTY_ZERO: Calculated qty={qty:.8f} is zero or negative"
    
    if qty < min_order_qty:
        return False, (
            f"QTY_BELOW_MIN_ORDER_QTY: qty={qty:.8f} < min_order_qty={min_order_qty:.8f} "
            f"(minimum quantity required)"
        )
    
    # Validar contra valor mínimo de orden
    if order_value < min_order_value:
        return False, (
            f"ORDER_VALUE_BELOW_MIN: order_value=${order_value:.2f} < min_order_value=${min_order_value:.2f} "
            f"(qty={qty:.8f} @ ${current_price:.2f})"
        )
    
    return True, None


# =============================================================================
# CLI INTERFACE (para uso directo desde línea de comandos)
# =============================================================================

async def main():
    """
    Función main para testing del helper desde CLI.
    """
    print("=" * 70)
    print("🧪 Position Size CLI Helper - Test Mode")
    print("=" * 70)
    
    # Crear portfolio manager mock para testing (inline para evitar imports)
    class MockPortfolioManager:
        def __init__(self):
            self._balance_cache = {
                "BTC": 0.01549,
                "ETH": 0.385,
                "USDT": 3000.0
            }
            self.portfolio = {
                "BTCUSDT": {"position": 0.01549, "free": 0.01549},
                "ETHUSDT": {"position": 0.385, "free": 0.385},
                "USDT": {"free": 3000.0}
            }
        
        async def get_asset_balance_async(self, asset: str) -> float:
            return self._balance_cache.get(asset, 0.0)
    
    pm = MockPortfolioManager()
    helper = PositionSizeCLIHelper(
        pm, 
        min_order_value=2.0,
        min_order_qty=0.00001
    )
    
    # Test cases
    test_cases = [
        {"symbol": "BTCUSDT", "side": "buy", "price": 50000.0, "allocation": 0.10},
        {"symbol": "ETHUSDT", "side": "buy", "price": 3000.0, "allocation": 0.10},
        {"symbol": "BTCUSDT", "side": "sell", "price": 50000.0, "allocation": 0.50},
        {"symbol": "BTCUSDT", "side": "buy", "price": 50000.0, "allocation": 0.0001},  # Too small
        {"symbol": "BTCUSDT", "side": "buy", "price": 50000.0, "allocation": 0.001, "min_order_value": 100.0},  # Below min value
    ]
    
    print(f"\n🔍 Async Loop Running: {is_async_loop_running()}")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}:")
        result = await helper.calculate_position_size(
            symbol=test["symbol"],
            side=test["side"],
            current_price=test["price"],
            allocation_pct=test["allocation"],
            min_order_value=test.get("min_order_value", 2.0)
        )
        print(helper.format_result_for_cli(result))
        
        # Verificar que el status es correcto
        if result.is_valid:
            assert result.signal_status == "VALID", f"Expected VALID, got {result.signal_status}"
            print(f"   ✅ Signal status correctly set to VALID")
        else:
            assert result.signal_status == "REJECTED", f"Expected REJECTED, got {result.signal_status}"
            print(f"   ✅ Signal status correctly set to REJECTED")
    
    # Test sync wrapper
    print("\n" + "=" * 70)
    print("🧪 Testing Sync Wrapper")
    print("=" * 70)
    
    result_sync = helper.calculate_position_size_sync(
        symbol="BTCUSDT",
        side="buy",
        current_price=50000.0,
        allocation_pct=0.10
    )
    print(helper.format_result_for_cli(result_sync))
    
    print("\n✅ All tests completed!")
    return 0


if __name__ == "__main__":
    asyncio.run(main())
