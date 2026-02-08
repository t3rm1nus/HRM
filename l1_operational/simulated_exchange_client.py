import time
from typing import Dict, List, Any, Optional
from core.logging import logger


class SimulatedExchangeClient:
    """
    Cliente de intercambio simulado para paper trading.

    - Mantiene balances locales
    - Ejecuta órdenes ficticias
    - Aplica slippage + fees
    - Guarda historial de trades
    - NUNCA hace requests HTTP
    - Singleton pattern to maintain state between cycles
    """
    _instance = None
    _initialized = False

    def __new__(cls, initial_balances: Dict[str, float] = None,
                 fee: float = 0.001,
                 slippage: float = 0.0005):
        """Singleton pattern to maintain state between instances"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        initial_balances: Dict[str, float] = None,
        fee: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Inicializa el cliente simulado.
        
        Args:
            initial_balances: Balances iniciales para cada activo (REQUERIDO)
            fee: Comisión por trade (0.001 = 0.1%)
            slippage: Slippage por trade (0.0005 = 0.05%)
        
        Raises:
            RuntimeError: Si se intenta inicializar sin balances iniciales válidos
        """
        if SimulatedExchangeClient._initialized:
            logger.debug("🎮 SimulatedExchangeClient already initialized - maintaining state")
            # Si ya está inicializado, no hacer nada más
            return
        
        # Validar balances iniciales (REQUERIDOS)
        if initial_balances is None or not isinstance(initial_balances, dict) or len(initial_balances) == 0:
            logger.critical("🚨 FATAL: SimulatedExchangeClient requires valid initial_balances (non-empty dict)", exc_info=True)
            raise RuntimeError("SimulatedExchangeClient cannot be initialized without valid initial_balances")
        
        # Validar que los balances sean positivos
        invalid_balances = [asset for asset, balance in initial_balances.items() if balance <= 0]
        if invalid_balances:
            logger.critical(f"🚨 FATAL: SimulatedExchangeClient balances must be positive. Invalid: {invalid_balances}", exc_info=True)
            raise RuntimeError("SimulatedExchangeClient cannot be initialized with non-positive balances")
        
        SimulatedExchangeClient._initialized = True
        
        self.initial_balances = initial_balances.copy()
        self.balances: Dict[str, float] = initial_balances.copy()
        self.fee = fee
        self.slippage = slippage

        self.trades: List[Dict[str, Any]] = []
        self.order_history: List[Dict[str, Any]] = []
        self._trade_id_counter = 1

        # 🔐 Garantizar assets base comunes
        self._ensure_asset("USDT")
        self._ensure_asset("BTC")
        self._ensure_asset("ETH")

        logger.info("✅ SimulatedExchangeClient inicializado")
        logger.info(f"   Balances iniciales: {self.balances}")
        logger.info(f"   Comisión: {fee*100:.2f}%")
        logger.info(f"   Slippage: {slippage*100:.2f}%")
        logger.info(f"   SIM_INIT_ONCE=True")

    @classmethod
    def initialize_once(cls, initial_balances: Dict[str, float],
                       fee: float = 0.001,
                       slippage: float = 0.0005):
        """
        Inicializa el SimulatedExchangeClient solo una vez por proceso.
        
        Args:
            initial_balances: Balances iniciales para cada activo (REQUERIDO)
            fee: Comisión por trade (0.001 = 0.1%)
            slippage: Slippage por trade (0.0005 = 0.05%)
        
        Returns:
            SimulatedExchangeClient: Instancia del cliente simulado
        
        Raises:
            RuntimeError: Si se intenta inicializar sin balances o más de una vez
        """
        if cls._initialized:
            logger.warning("⚠️ SimulatedExchangeClient already initialized - returning existing instance")
            return cls._instance
        
        return cls(initial_balances, fee, slippage)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _ensure_asset(self, asset: str):
        if asset not in self.balances:
            self.balances[asset] = 0.0

    def _parse_symbol(self, symbol: str):
        """
        Asume pares tipo BTCUSDT, ETHUSDT, etc.
        """
        if not symbol.endswith("USDT"):
            raise ValueError(f"Símbolo no soportado en simulado: {symbol}")

        base_asset = symbol.replace("USDT", "")
        quote_asset = "USDT"

        self._ensure_asset(base_asset)
        self._ensure_asset(quote_asset)

        return base_asset, quote_asset

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def get_balances(self) -> Dict[str, float]:
        return self.balances.copy()

    def execute_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        market_price: float
    ) -> Dict[str, Any]:

        side = side.upper()

        if qty <= 0:
            raise ValueError(f"Cantidad inválida: {qty}")
        if market_price <= 0:
            raise ValueError(f"Precio inválido: {market_price}")
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Lado inválido: {side}")

        base_asset, quote_asset = self._parse_symbol(symbol)

        # Slippage
        if side == "BUY":
            execution_price = market_price * (1 + self.slippage)
        else:
            execution_price = market_price * (1 - self.slippage)

        cost = qty * execution_price
        fee = cost * self.fee

        # Validaciones
        if side == "BUY":
            required = cost + fee
            if self.balances[quote_asset] < required:
                raise ValueError(
                    f"Fondos insuficientes {quote_asset}: "
                    f"requiere {required}, disponible {self.balances[quote_asset]}"
                )
        else:
            if self.balances[base_asset] < qty:
                raise ValueError(
                    f"Balance insuficiente {base_asset}: "
                    f"requiere {qty}, disponible {self.balances[base_asset]}"
                )

        # Ejecutar
        trade_id = self._trade_id_counter
        self._trade_id_counter += 1

        if side == "BUY":
            self.balances[quote_asset] -= (cost + fee)
            self.balances[base_asset] += qty
        else:
            self.balances[base_asset] -= qty
            self.balances[quote_asset] += (cost - fee)

        trade = {
            "trade_id": trade_id,
            "timestamp": time.time(),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "market_price": market_price,
            "execution_price": execution_price,
            "fee": fee,
            "cost": cost,
            "slippage_cost": abs(execution_price - market_price) * qty
        }

        self.trades.append(trade)

        self.order_history.append({
            "order_id": f"simulated_{trade_id}",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": execution_price,
            "status": "FILLED",
            "fee": fee,
            "timestamp": trade["timestamp"]
        })

        logger.info(
            f"✅ SIM ORDER | {side} {qty:.6f} {symbol} @ {execution_price:.2f}"
        )
        logger.info(f"   Fee: {fee:.4f} | Balances: {self.balances}")

        return trade

    # ------------------------------------------------------------------
    # Historial y métricas
    # ------------------------------------------------------------------

    def get_trade_history(self) -> List[Dict[str, Any]]:
        return self.trades.copy()

    def get_order_history(self) -> List[Dict[str, Any]]:
        return self.order_history.copy()

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                "total_trades": 0,
                "total_fees": 0.0,
                "total_slippage_cost": 0.0,
                "balances": self.balances.copy()
            }

        total_fees = sum(t["fee"] for t in self.trades)
        total_slippage = sum(t["slippage_cost"] for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "buy_trades": sum(1 for t in self.trades if t["side"] == "BUY"),
            "sell_trades": sum(1 for t in self.trades if t["side"] == "SELL"),
            "total_fees": total_fees,
            "total_slippage_cost": total_slippage,
            "avg_slippage": total_slippage / len(self.trades),
            "balances": self.balances.copy()
        }

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def reset(self, new_balances: Optional[Dict[str, float]] = None):
        if new_balances:
            self.balances = new_balances.copy()

        self.trades.clear()
        self.order_history.clear()
        self._trade_id_counter = 1

        logger.info("🔄 SimulatedExchangeClient reiniciado")
        logger.info(f"   Balances: {self.balances}")

    async def close(self):
        logger.info("✅ SimulatedExchangeClient cerrado (simulación)")

    # ------------------------------------------------------------------
    # Compatibilidad BinanceClient
    # ------------------------------------------------------------------

    async def get_account_balances(self) -> Dict[str, float]:
        """Devuelve los balances actuales - compatibilidad con BinanceClient"""
        if not hasattr(self, 'balances') or not self.balances:
            logger.warning("⚠️ SimulatedExchangeClient balances not initialized, returning initial balances")
            return self.initial_balances.copy()
        return self.get_balances()

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "MARKET"
    ):
        raise NotImplementedError(
            "Use execute_order(symbol, side, qty, market_price) en simulado"
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        logger.warning("⚠️ Cancelación no soportada en simulado")
        return False

    async def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        return []
    
    def force_reset(self, initial_balances: Dict[str, float] = None):
        """Force reset the client with new initial balances"""
        if initial_balances:
            self.balances = initial_balances.copy()
            self.initial_balances = initial_balances.copy()
            self.trades.clear()
            self.order_history.clear()
            self._trade_id_counter = 1
            logger.info(f"🔄 SimulatedExchangeClient forcefully reset with balances: {self.balances}")
        else:
            logger.critical("🚨 FATAL: force_reset requires initial_balances")
            raise RuntimeError("force_reset requires initial_balances")
    
    @classmethod
    def force_reset_class(cls, initial_balances: Dict[str, float] = None):
        """Force reset at class level - should be used only for testing"""
        if cls._instance is not None:
            cls._initialized = False
            if initial_balances:
                cls._instance.__init__(initial_balances)
            else:
                logger.critical("🚨 FATAL: force_reset requires initial_balances")
                raise RuntimeError("force_reset requires initial_balances")
            logger.warning("⚠️ SimulatedExchangeClient forcefully reset - testing only")
        else:
            logger.warning("⚠️ SimulatedExchangeClient not initialized - cannot reset")
            
    def reset(self):
        """Reinicia el cliente a su estado inicial"""
        self.balances = self.initial_balances.copy()
        self.trades.clear()
        self.order_history.clear()
        self._trade_id_counter = 1
        logger.info("🔄 SimulatedExchangeClient reiniciado a estado inicial")
        logger.info(f"   Balances: {self.balances}")
    
    @classmethod
    def cleanup(cls) -> bool:
        """
        Cleanup completo del singleton - USAR TRAS SystemCleanup.
        Resetea los flags para permitir nueva inicialización.
        
        Returns:
            True si cleanup fue exitoso
        """
        cls._instance = None
        cls._initialized = False
        logger.info("🔄 SimulatedExchangeClient.cleanup() completado - singleton listo para re-init")
        return True
