import copy
import math
import re
import threading
import time
import uuid

from typing import Dict, List, Any, Optional

from core.logging import logger


class SimulatedExchangeClient:
    """
    Exchange simulado de HRM para PAPER TRADING.

    Responsabilidades:
    - Mantener balances simulados.
    - Ejecutar operaciones BUY y SELL.
    - Aplicar comisiones y slippage.
    - Mantener historial de órdenes y operaciones.
    - Preservar el estado entre ciclos mediante Singleton.

    No realiza peticiones HTTP.
    No obtiene precios de mercado.
    No ejecuta operaciones reales.
    """

    _instance = None
    _initialized = False
    _just_initialized = False

    _singleton_lock = threading.RLock()

    QUOTE_ASSET = "USDT"

    ASSETS_BASE = {
        "BTC",
        "ETH"
    }

    def __new__(
        cls,
        initial_balances: Optional[Dict[str, float]] = None,
        fee: float = 0.001,
        slippage: float = 0.0005
    ):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)

            return cls._instance

    # ------------------------------------------------------------------
    # INICIALIZACIÓN
    # ------------------------------------------------------------------

    def __init__(
        self,
        initial_balances: Optional[Dict[str, float]] = None,
        fee: float = 0.001,
        slippage: float = 0.0005
    ):
        with self._singleton_lock:

            if SimulatedExchangeClient._initialized:
                logger.debug(
                    "SimulatedExchangeClient ya inicializado. "
                    "Se conserva el estado actual."
                )
                return

            balances = self._validar_balances(initial_balances)

            fee = self._validar_porcentaje(fee, "fee")
            slippage = self._validar_porcentaje(
                slippage,
                "slippage"
            )

            self._state_lock = threading.RLock()

            self.initial_balances = balances.copy()
            self.balances = balances.copy()

            self.fee = fee
            self.slippage = slippage

            self.trades: List[Dict[str, Any]] = []
            self.order_history: List[Dict[str, Any]] = []

            self._trade_id_counter = 1

            self._asegurar_activos_base()

            SimulatedExchangeClient._initialized = True
            SimulatedExchangeClient._just_initialized = True

            logger.info(
                "SimulatedExchangeClient inicializado"
            )
            logger.info(
                "Capital inicial: %s",
                self.initial_balances
            )
            logger.info(
                "Comisión: %.4f%% | Slippage: %.4f%%",
                self.fee * 100,
                self.slippage * 100
            )

    # ------------------------------------------------------------------
    # VALIDACIONES
    # ------------------------------------------------------------------

    @staticmethod
    def _validar_porcentaje(
        valor: float,
        nombre: str
    ) -> float:

        try:
            porcentaje = float(valor)
        except (TypeError, ValueError):
            raise ValueError(
                f"{nombre} debe ser numérico"
            )

        if (
            not math.isfinite(porcentaje)
            or porcentaje < 0
            or porcentaje >= 1
        ):
            raise ValueError(
                f"{nombre} debe estar entre 0 y 1"
            )

        return porcentaje

    @staticmethod
    def _validar_cantidad(
        valor: float,
        nombre: str
    ) -> float:

        try:
            cantidad = float(valor)
        except (TypeError, ValueError):
            raise ValueError(
                f"{nombre} debe ser numérico"
            )

        if not math.isfinite(cantidad) or cantidad <= 0:
            raise ValueError(
                f"{nombre} debe ser positivo y finito"
            )

        return cantidad

    @classmethod
    def _validar_balances(
        cls,
        initial_balances: Optional[Dict[str, float]]
    ) -> Dict[str, float]:

        if not isinstance(initial_balances, dict):
            raise RuntimeError(
                "Se requiere un diccionario de balances iniciales"
            )

        if not initial_balances:
            raise RuntimeError(
                "Los balances iniciales no pueden estar vacíos"
            )

        balances = {}

        for activo, cantidad in initial_balances.items():

            if (
                not isinstance(activo, str)
                or not re.fullmatch(r"[A-Z0-9]{2,15}", activo.upper())
            ):
                raise ValueError(
                    f"Activo no válido: {activo}"
                )

            activo = activo.upper()

            if activo in balances:
                raise ValueError(
                    f"Activo duplicado: {activo}"
                )

            try:
                cantidad = float(cantidad)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Balance inválido para {activo}"
                )

            if not math.isfinite(cantidad) or cantidad < 0:
                raise ValueError(
                    f"Balance negativo o no finito: {activo}"
                )

            balances[activo] = cantidad

        if balances.get(cls.QUOTE_ASSET, 0) <= 0:
            raise RuntimeError(
                "El capital inicial USDT debe ser positivo"
            )

        balances.setdefault("BTC", 0.0)
        balances.setdefault("ETH", 0.0)

        return balances

    def _asegurar_activos_base(self) -> None:
        for activo in self.ASSETS_BASE:
            self.balances.setdefault(activo, 0.0)

        self.balances.setdefault(
            self.QUOTE_ASSET,
            0.0
        )

    def _parse_symbol(self, symbol: str):
        if not isinstance(symbol, str):
            raise ValueError("El símbolo debe ser texto")

        symbol = symbol.strip().upper()

        if not symbol.endswith(self.QUOTE_ASSET):
            raise ValueError(
                f"Par no soportado: {symbol}. "
                f"Se requiere cotización en {self.QUOTE_ASSET}."
            )

        base_asset = symbol[:-len(self.QUOTE_ASSET)]

        if not base_asset or not re.fullmatch(
            r"[A-Z0-9]{2,15}",
            base_asset
        ):
            raise ValueError(
                f"Símbolo inválido: {symbol}"
            )

        self.balances.setdefault(base_asset, 0.0)

        return base_asset, self.QUOTE_ASSET

    # ------------------------------------------------------------------
    # SINGLETON
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        initial_balances: Optional[Dict[str, float]] = None,
        fee: float = 0.001,
        slippage: float = 0.0005
    ):
        with cls._singleton_lock:

            if cls._instance is not None:
                return cls._instance

            if initial_balances is None:
                raise RuntimeError(
                    "get_instance requiere initial_balances "
                    "en la primera inicialización"
                )

            return cls(
                initial_balances,
                fee,
                slippage
            )

    @classmethod
    def initialize_once(
        cls,
        initial_balances: Dict[str, float],
        fee: float = 0.001,
        slippage: float = 0.0005
    ):
        return cls.get_instance(
            initial_balances,
            fee,
            slippage
        )

    @classmethod
    def reset_instance(cls):
        with cls._singleton_lock:
            cls._instance = None
            cls._initialized = False
            cls._just_initialized = False

    @classmethod
    def cleanup(cls) -> bool:
        with cls._singleton_lock:
            cls._instance = None
            cls._initialized = False
            cls._just_initialized = False

        logger.info(
            "SimulatedExchangeClient cleanup completado"
        )

        return True

    # ------------------------------------------------------------------
    # BALANCES
    # ------------------------------------------------------------------

    def get_balances(self) -> Dict[str, float]:
        with self._state_lock:
            return self.balances.copy()

    async def get_account_balances(self) -> Dict[str, float]:
        return self.get_balances()

    def set_balance(
        self,
        asset: str,
        amount: float
    ) -> None:

        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("Activo inválido")

        asset = asset.strip().upper()

        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ValueError("Balance inválido")

        if not math.isfinite(amount) or amount < 0:
            raise ValueError(
                "El balance debe ser finito y no negativo"
            )

        with self._state_lock:
            anterior = self.balances.get(asset, 0.0)
            self.balances[asset] = amount

        logger.info(
            "Balance actualizado: %s %.8f -> %.8f",
            asset,
            anterior,
            amount
        )

    # ------------------------------------------------------------------
    # EJECUCIÓN DE ÓRDENES
    # ------------------------------------------------------------------

    def execute_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        market_price: float
    ) -> Dict[str, Any]:

        side = str(side).upper()

        if side not in {"BUY", "SELL"}:
            raise ValueError(
                f"Operación no soportada: {side}"
            )

        qty = self._validar_cantidad(
            qty,
            "qty"
        )

        market_price = self._validar_cantidad(
            market_price,
            "market_price"
        )

        with self._state_lock:

            base_asset, quote_asset = self._parse_symbol(
                symbol
            )

            execution_price = self._calcular_precio_ejecucion(
                side,
                market_price
            )

            cost = qty * execution_price

            if not math.isfinite(cost):
                raise ValueError(
                    "El coste de la operación no es válido"
                )

            fee = cost * self.fee

            if side == "BUY":
                self._validar_compra(
                    quote_asset,
                    cost,
                    fee
                )
            else:
                self._validar_venta(
                    base_asset,
                    qty
                )

            trade = self._registrar_ejecucion(
                symbol=symbol.upper(),
                side=side,
                base_asset=base_asset,
                quote_asset=quote_asset,
                qty=qty,
                market_price=market_price,
                execution_price=execution_price,
                cost=cost,
                fee=fee
            )

            self._actualizar_balances(
                side,
                base_asset,
                quote_asset,
                qty,
                cost,
                fee
            )

            return copy.deepcopy(trade)

    def _calcular_precio_ejecucion(
        self,
        side: str,
        market_price: float
    ) -> float:

        if side == "BUY":
            precio = market_price * (1 + self.slippage)
        else:
            precio = market_price * (1 - self.slippage)

        if not math.isfinite(precio) or precio <= 0:
            raise ValueError(
                "Precio de ejecución inválido"
            )

        return precio

    def _validar_compra(
        self,
        quote_asset: str,
        cost: float,
        fee: float
    ) -> None:

        requerido = cost + fee
        disponible = self.balances[quote_asset]

        if disponible < requerido:
            raise ValueError(
                f"Fondos insuficientes {quote_asset}: "
                f"requiere {requerido:.8f}, "
                f"disponible {disponible:.8f}"
            )

    def _validar_venta(
        self,
        base_asset: str,
        qty: float
    ) -> None:

        disponible = self.balances[base_asset]

        if disponible < qty:
            raise ValueError(
                f"Balance insuficiente {base_asset}: "
                f"requiere {qty:.8f}, "
                f"disponible {disponible:.8f}"
            )

    def _actualizar_balances(
        self,
        side: str,
        base_asset: str,
        quote_asset: str,
        qty: float,
        cost: float,
        fee: float
    ) -> None:

        if side == "BUY":
            self.balances[quote_asset] -= cost + fee
            self.balances[base_asset] += qty
        else:
            self.balances[base_asset] -= qty
            self.balances[quote_asset] += cost - fee

    def _registrar_ejecucion(
        self,
        symbol: str,
        side: str,
        base_asset: str,
        quote_asset: str,
        qty: float,
        market_price: float,
        execution_price: float,
        cost: float,
        fee: float
    ) -> Dict[str, Any]:

        trade_id = self._trade_id_counter
        self._trade_id_counter += 1

        timestamp = time.time()
        order_id = f"simulated_{uuid.uuid4().hex}"

        trade = {
            "trade_id": trade_id,
            "order_id": order_id,
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "market_price": market_price,
            "execution_price": execution_price,
            "fee": fee,
            "cost": cost,
            "slippage_cost": abs(
                execution_price - market_price
            ) * qty
        }

        self.trades.append(trade)

        self.order_history.append({
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": execution_price,
            "status": "FILLED",
            "fee": fee,
            "timestamp": timestamp
        })

        logger.info(
            "PAPER ORDER | %s %s %.8f @ %.8f",
            side,
            symbol,
            qty,
            execution_price
        )

        return trade

    # ------------------------------------------------------------------
    # HISTORIAL
    # ------------------------------------------------------------------

    def get_trade_history(self) -> List[Dict[str, Any]]:
        with self._state_lock:
            return copy.deepcopy(self.trades)

    def get_order_history(self) -> List[Dict[str, Any]]:
        with self._state_lock:
            return copy.deepcopy(self.order_history)

    def get_performance_summary(self) -> Dict[str, Any]:

        with self._state_lock:

            total_trades = len(self.trades)

            total_fees = sum(
                trade["fee"]
                for trade in self.trades
            )

            total_slippage = sum(
                trade["slippage_cost"]
                for trade in self.trades
            )

            compras = sum(
                1
                for trade in self.trades
                if trade["side"] == "BUY"
            )

            ventas = sum(
                1
                for trade in self.trades
                if trade["side"] == "SELL"
            )

            return {
                "total_trades": total_trades,
                "buy_trades": compras,
                "sell_trades": ventas,
                "total_fees": total_fees,
                "total_slippage_cost": total_slippage,
                "avg_slippage": (
                    total_slippage / total_trades
                    if total_trades
                    else 0.0
                ),
                "balances": self.balances.copy()
            }

    # ------------------------------------------------------------------
    # RESET Y CONTROL
    # ------------------------------------------------------------------

    def reset(
        self,
        new_balances: Optional[Dict[str, float]] = None
    ) -> None:

        with self._state_lock:

            if new_balances is None:
                balances = self.initial_balances.copy()
            else:
                balances = self._validar_balances(
                    new_balances
                )

            self.balances = balances
            self._asegurar_activos_base()

            self.trades.clear()
            self.order_history.clear()

            self._trade_id_counter = 1

        logger.info(
            "SimulatedExchangeClient reiniciado"
        )

    def force_reset(
        self,
        initial_balances: Optional[Dict[str, float]] = None
    ) -> None:

        if initial_balances is None:
            raise RuntimeError(
                "force_reset requiere initial_balances"
            )

        balances = self._validar_balances(
            initial_balances
        )

        with self._state_lock:

            self.initial_balances = balances.copy()
            self.balances = balances.copy()

            self._asegurar_activos_base()

            self.trades.clear()
            self.order_history.clear()

            self._trade_id_counter = 1

        logger.warning(
            "SimulatedExchangeClient reiniciado forzosamente"
        )

    @classmethod
    def force_reset_class(
        cls,
        initial_balances: Optional[Dict[str, float]] = None
    ):

        if initial_balances is None:
            raise RuntimeError(
                "force_reset_class requiere initial_balances"
            )

        with cls._singleton_lock:

            if cls._instance is None:
                raise RuntimeError(
                    "No existe instancia para reiniciar"
                )

            cls._instance.force_reset(initial_balances)

            cls._initialized = True
            cls._just_initialized = True

    # ------------------------------------------------------------------
    # CONTROL DE CICLOS
    # ------------------------------------------------------------------

    @classmethod
    def mark_cycle_completed(cls) -> bool:

        with cls._singleton_lock:

            estaba_recien_inicializado = cls._just_initialized

            cls._just_initialized = False

            return estaba_recien_inicializado

    @classmethod
    def is_just_initialized(cls) -> bool:

        with cls._singleton_lock:
            return cls._just_initialized

    # ------------------------------------------------------------------
    # COMPATIBILIDAD
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """
        Cierre compatible con clientes asíncronos.
        El simulador no mantiene conexiones externas.
        """
        logger.debug(
            "SimulatedExchangeClient cerrado"
        )

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "MARKET"
    ):

        raise NotImplementedError(
            "Utiliza execute_order() para ejecutar operaciones "
            "simuladas. Las órdenes pendientes requieren un "
            "gestor específico."
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str
    ) -> bool:

        logger.warning(
            "No se pueden cancelar órdenes ya ejecutadas en PAPER"
        )

        return False

    async def get_open_orders(
        self,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        return []

    def get_market_price(self, symbol: str) -> float:
        """
        Los precios pertenecen al sistema de datos de mercado.

        No se deben generar precios artificiales desde el exchange
        simulado.
        """
        raise NotImplementedError(
            "Utiliza MarketDataManager para obtener precios reales"
        )