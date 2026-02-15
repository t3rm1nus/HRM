# core/portfolio_manager.py - Gestión del portfolio

import os
import csv
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, TYPE_CHECKING

import pandas as pd

from core.logging import logger
from l2_tactic.l2_utils import safe_float
from core.async_balance_helper import (
    AsyncContextDetector, 
    BalanceAccessLogger, 
    AsyncBalanceRequiredError,
    enforce_async_balance_access,
    BalanceVerificationStatus
)

if TYPE_CHECKING:
    from system.market_data_manager import MarketDataManager

# =========================
# OPTIONAL DEPENDENCIES
# =========================

try:
    from hacienda.tax_tracker import TaxTracker
    TAX_TRACKER_AVAILABLE = True
except ImportError:
    TAX_TRACKER_AVAILABLE = False
    logger.warning("⚠️ TaxTracker no disponible - seguimiento fiscal deshabilitado")

try:
    from storage.paper_trade_logger import get_paper_logger
    PAPER_LOGGER_AVAILABLE = True
except ImportError:
    PAPER_LOGGER_AVAILABLE = False
    logger.warning("⚠️ PaperTradeLogger no disponible")

try:
    from core.simulated_exchange_client import SimulatedExchangeClient
    SIMULATED_CLIENT_AVAILABLE = True
except ImportError:
    SIMULATED_CLIENT_AVAILABLE = False
    logger.warning("⚠️ SimulatedExchangeClient no disponible")

# =========================
# CSV LOGGER
# =========================

async def save_portfolio_to_csv(state: Dict[str, Any]):
    try:
        portfolio = state.get("portfolio", {})

        output_dir = "data/portfolios"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "portfolio_log.csv")

        headers = [
            "timestamp", "cycle_id", "total_value",
            "btc_balance", "eth_balance", "usdt_balance"
        ]

        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle_id": state.get("cycle_id", 0),
                "total_value": portfolio.get("total", 0.0),
                "btc_balance": portfolio.get("BTCUSDT", {}).get("position", 0.0),
                "eth_balance": portfolio.get("ETHUSDT", {}).get("position", 0.0),
                "usdt_balance": portfolio.get("USDT", {}).get("free", 0.0)
            })

        logger.info("📝 Portfolio guardado en CSV")

    except Exception as e:
        logger.error(f"❌ Error guardando portfolio CSV: {e}", exc_info=True)

# =========================
# PORTFOLIO MANAGER
# =========================

class PortfolioManager:
    """
    Portfolio Manager - Versión PURA con inyección de dependencias.
    
    No lee config, no crea clientes, depende solo de interfaces inyectadas.
    
    Args:
        exchange_client: Cliente de exchange (SimulatedExchangeClient o BinanceClient)
        market_data_manager: Gestor de datos de mercado para obtener precios
        mode: Modo de operación ("simulated" o "live")
        initial_balance: Balance inicial para modo simulated
        symbols: Lista de símbolos a operar
        enable_commissions: Habilitar comisiones
        enable_slippage: Habilitar slippage
    """
    
    _instance = None
    _lock = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of PortfolioManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None

    def __init__(
        self,
        exchange_client: Optional[Any] = None,
        market_data_manager: Optional["MarketDataManager"] = None,
        mode: str = "simulated",
        initial_balance: float = 500.0,
        symbols: Optional[list] = None,
        enable_commissions: bool = True,
        enable_slippage: bool = True
    ):
        # Evitar re-inicialización si ya existe
        if hasattr(self, '_initialized'):
            return
        
        self.exchange_client = exchange_client
        self.market_data_manager = market_data_manager
        self.mode = mode
        self.initial_balance = initial_balance
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]

        self.enable_commissions = enable_commissions
        self.enable_slippage = enable_slippage

        self.portfolio: Dict[str, Any] = {}
        self.peak_value = initial_balance
        self.total_fees = 0.0

        self.position_age: Dict[str, float] = {}
        self.MIN_HOLD_TIME = 60
        
        # Balance verification tracking
        self._balance_verification = BalanceVerificationStatus()
        self._balance_cache: Dict[str, float] = {}
        self._balance_cache_time: Dict[str, float] = {}
        self._balance_cache_ttl = 5.0  # Cache balances for 5 seconds

        # Initialize tax tracker and paper logger if available
        self.tax_tracker = None
        if TAX_TRACKER_AVAILABLE:
            try:
                self.tax_tracker = TaxTracker()
                logger.info("✅ TaxTracker inicializado para seguimiento fiscal")
            except Exception as e:
                logger.error(f"❌ Error inicializando TaxTracker: {e}")

        self.paper_logger = None
        if PAPER_LOGGER_AVAILABLE:
            try:
                self.paper_logger = get_paper_logger()
                logger.info("✅ PaperTradeLogger inicializado para registro de trades")
            except Exception as e:
                logger.error(f"❌ Error inicializando PaperTradeLogger: {e}")

        self._configure_trading_costs()
        
        # Initialize portfolio to empty state - actual initialization happens in initialize_async
        self._init_portfolio()
        
        self._initialized = True

        logger.info(f"✅ PortfolioManager created (await initialize_async() to complete initialization) | mode={self.mode} | initial_balance={initial_balance}")
    
    async def initialize_async(self):
        """Initialize portfolio manager asynchronously - use this instead of relying on __init__"""
        if self.exchange_client:
            if self.mode == "simulated" or (hasattr(self.exchange_client, 'paper_mode') and self.exchange_client.paper_mode):
                await self._init_portfolio_from_simulated_client_async()
            else:
                if not await self._sync_from_client_async():
                    self._init_portfolio()
        else:
            self._init_portfolio()
            
        logger.info(f"✅ PortfolioManager fully initialized | mode={self.mode} | balance={self.portfolio.get('total', self.initial_balance):.2f} USDT")

    # =========================
    # SYNC FROM EXCHANGE (CRITICAL FIX)
    # =========================

    def sync_from_exchange(self, exchange_client) -> bool:
        """
        CRITICAL METHOD: Synchronize portfolio balances from exchange client.
        
        This method pulls REAL balances from the exchange client and updates
        internal portfolio state atomically. MUST be called after EVERY executed order.
        
        Args:
            exchange_client: SimulatedExchangeClient or BinanceClient instance
            
        Returns:
            bool: True if sync successful, False otherwise
        """
        try:
            logger.info("[SYNC] Starting portfolio sync from exchange client...")
            
            # Validate exchange_client
            if exchange_client is None:
                logger.error("[SYNC] ❌ exchange_client is None")
                return False
            
            # Get balances from exchange client
            import inspect
            
            balances = None
            if hasattr(exchange_client, 'get_balances'):
                if inspect.iscoroutinefunction(exchange_client.get_balances):
                    # Can't call async from sync context
                    logger.warning("[SYNC] ⚠️ Cannot call async get_balances from sync context")
                    return False
                balances = exchange_client.get_balances()
            elif hasattr(exchange_client, 'get_account_balances'):
                if inspect.iscoroutinefunction(exchange_client.get_account_balances):
                    # Can't call async from sync context
                    logger.warning("[SYNC] ⚠️ Cannot call async get_account_balances from sync context")
                    return False
                balances = exchange_client.get_account_balances()
            else:
                logger.error("[SYNC] ❌ exchange_client has no get_balances or get_account_balances method")
                return False
            
            # Validate balances
            if not balances or not isinstance(balances, dict):
                logger.error(f"[SYNC] ❌ Invalid balances returned: {balances}")
                return False
            
            # Extract balances
            current_btc = balances.get("BTC", 0.0)
            current_eth = balances.get("ETH", 0.0)
            current_usdt = balances.get("USDT", 0.0)
            
            # Validate that not all balances are zero (state loss detection)
            if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
                logger.critical("[SYNC] 🚨 CRITICAL: All balances are zero - state loss detected!")
                return False
            
            # Update portfolio atomically with balances only - NAV will be calculated separately using MarketDataManager
            self.portfolio["BTCUSDT"] = {"position": current_btc, "free": current_btc}
            self.portfolio["ETHUSDT"] = {"position": current_eth, "free": current_eth}
            self.portfolio["USDT"] = {"free": current_usdt}
            
            # Update balance cache
            import time
            self._balance_cache = {
                "BTC": current_btc,
                "ETH": current_eth,
                "USDT": current_usdt
            }
            self._balance_cache_time = {
                "BTC": time.time(),
                "ETH": time.time(),
                "USDT": time.time()
            }
            
            # Log sync success with full details (without NAV - will be calculated separately)
            logger.info("=" * 70)
            logger.info("[SYNC] ✅ Portfolio synced from exchange client")
            logger.info(f"[SYNC]    BTC Balance:  {current_btc:.6f}")
            logger.info(f"[SYNC]    ETH Balance:  {current_eth:.6f}")
            logger.info(f"[SYNC]    USDT Balance: ${current_usdt:.2f}")
            logger.info(f"[SYNC]    {'─' * 50}")
            logger.info(f"[SYNC]    NAV calculation requires MarketDataManager")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"[SYNC] ❌ Error during sync_from_exchange: {e}", exc_info=True)
            return False

    async def sync_from_exchange_async(self, exchange_client) -> bool:
        """
        Async version of sync_from_exchange.
        
        Args:
            exchange_client: SimulatedExchangeClient or BinanceClient instance
            
        Returns:
            bool: True if sync successful, False otherwise
        """
        try:
            logger.info("[SYNC] Starting portfolio sync from exchange client (async)...")
            
            # Validate exchange_client
            if exchange_client is None:
                logger.error("[SYNC] ❌ exchange_client is None")
                return False
            
            # Get balances from exchange client
            import inspect
            balances = None
            
            if hasattr(exchange_client, 'get_account_balances'):
                if inspect.iscoroutinefunction(exchange_client.get_account_balances):
                    balances = await exchange_client.get_account_balances()
                else:
                    balances = exchange_client.get_account_balances()
            elif hasattr(exchange_client, 'get_balances'):
                if inspect.iscoroutinefunction(exchange_client.get_balances):
                    balances = await exchange_client.get_balances()
                else:
                    balances = exchange_client.get_balances()
            else:
                logger.error("[SYNC] ❌ exchange_client has no get_balances or get_account_balances method")
                return False
            
            # Validate balances
            if not balances or not isinstance(balances, dict):
                logger.error(f"[SYNC] ❌ Invalid balances returned: {balances}")
                return False
            
            # Extract balances
            current_btc = balances.get("BTC", 0.0)
            current_eth = balances.get("ETH", 0.0)
            current_usdt = balances.get("USDT", 0.0)
            
            # Validate that not all balances are zero (state loss detection)
            if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
                logger.critical("[SYNC] 🚨 CRITICAL: All balances are zero - state loss detected!")
                return False
            
            # Defensive safeguard: Check for negative balances
            if current_btc < 0:
                logger.critical(f"🚨 NEGATIVE BTC BALANCE DETECTED: {current_btc:.6f} - resetting to 0")
                current_btc = 0.0
            if current_eth < 0:
                logger.critical(f"🚨 NEGATIVE ETH BALANCE DETECTED: {current_eth:.4f} - resetting to 0")
                current_eth = 0.0
            if current_usdt < 0:
                logger.critical(f"🚨 NEGATIVE USDT BALANCE DETECTED: {current_usdt:.2f} - resetting to 0")
                current_usdt = 0.0
            
            # Update portfolio atomically with balances only - NAV will be calculated separately using MarketDataManager
            self.portfolio["BTCUSDT"] = {"position": current_btc, "free": current_btc}
            self.portfolio["ETHUSDT"] = {"position": current_eth, "free": current_eth}
            self.portfolio["USDT"] = {"free": current_usdt}
            
            # Update balance cache
            import time
            self._balance_cache = {
                "BTC": current_btc,
                "ETH": current_eth,
                "USDT": current_usdt
            }
            self._balance_cache_time = {
                "BTC": time.time(),
                "ETH": time.time(),
                "USDT": time.time()
            }
            self._balance_verification.mark_synced("exchange_async")
            
            # Log sync success with full details (without NAV - will be calculated separately)
            logger.info("=" * 70)
            logger.info("[SYNC] ✅ Portfolio synced from exchange client")
            logger.info(f"[SYNC]    BTC Balance:  {current_btc:.6f}")
            logger.info(f"[SYNC]    ETH Balance:  {current_eth:.6f}")
            logger.info(f"[SYNC]    USDT Balance: ${current_usdt:.2f}")
            logger.info(f"[SYNC]    {'─' * 50}")
            logger.info(f"[SYNC]    NAV calculation requires MarketDataManager")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"[SYNC] ❌ Error during sync_from_exchange_async: {e}", exc_info=True)
            return False

    async def update_nav(self) -> Dict[str, float]:
        """
        Update NAV using current market prices from MarketDataManager.
        
        Returns:
            Dict with NAV breakdown
        """
        nav = await self.calculate_nav_async()
        
        # Update portfolio total
        self.portfolio["total"] = nav["total_nav"]
        
        logger.info(f"[ASYNC] NAV updated: ${nav['total_nav']:.2f}")
        
        return nav

    async def update_nav_async(self, market_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Async version of update_nav.
        Fetches fresh balances before calculating NAV using MarketDataManager.
        
        Args:
            market_prices: Optional pre-fetched market prices. If None, fetches from MarketDataManager.
        
        Returns:
            Dict with NAV breakdown
        """
        # Get fresh balances
        balances = await self.get_balances_async()
        
        # Get market prices from MarketDataManager if not provided
        if market_prices is None:
            market_prices = await self._get_market_prices()
        
        # Calculate NAV with fresh balances
        usdt_balance = balances.get("USDT", 0.0)
        btc_balance = balances.get("BTC", 0.0)
        eth_balance = balances.get("ETH", 0.0)
        
        btc_price = market_prices.get("BTCUSDT", 0.0)
        eth_price = market_prices.get("ETHUSDT", 0.0)
        
        btc_value = btc_balance * btc_price
        eth_value = eth_balance * eth_price
        total_nav = usdt_balance + btc_value + eth_value
        
        # Defensive safeguard: Ensure NAV never drops to 0 unless portfolio is actually empty
        if total_nav <= 0 and (usdt_balance > 0 or btc_balance > 0 or eth_balance > 0):
            logger.critical(f"🚨 NAV CALCULATION ERROR: NAV={total_nav:.2f} with non-zero balances (USDT={usdt_balance:.2f}, BTC={btc_balance:.6f}, ETH={eth_balance:.4f})")
            # Calculate minimum possible NAV assuming $1 minimum price
            min_nav = usdt_balance + btc_balance * 1.0 + eth_balance * 1.0
            total_nav = max(min_nav, 0.01)
            logger.warning(f"⚠️ NAV adjusted to minimum: ${total_nav:.2f}")
        
        # Update portfolio with fresh values
        self.portfolio["BTCUSDT"] = {"position": btc_balance, "free": btc_balance}
        self.portfolio["ETHUSDT"] = {"position": eth_balance, "free": eth_balance}
        self.portfolio["USDT"] = {"free": usdt_balance}
        self.portfolio["total"] = total_nav
        
        # Update cache
        import time
        self._balance_cache = {
            "BTC": btc_balance,
            "ETH": eth_balance,
            "USDT": usdt_balance
        }
        self._balance_cache_time = {
            "BTC": time.time(),
            "ETH": time.time(),
            "USDT": time.time()
        }
        
        nav = {
            "total_nav": total_nav,
            "usdt": usdt_balance,
            "btc_balance": btc_balance,
            "btc_price": btc_price,
            "btc_value": btc_value,
            "eth_balance": eth_balance,
            "eth_price": eth_price,
            "eth_value": eth_value
        }
        
        logger.info(f"[ASYNC] NAV updated: ${total_nav:.2f}")
        return nav

    async def _get_market_prices(self) -> Dict[str, float]:
        """
        Obtiene precios de mercado desde MarketDataManager.
        
        Returns:
            Dict[str, float] con precios actuales
        """
        if self.market_data_manager is None:
            logger.error("❌ MarketDataManager no inyectado - no se pueden obtener precios")
            return {}
        
        try:
            prices = await self.market_data_manager.get_market_prices()
            return prices
        except Exception as e:
            logger.error(f"❌ Error obteniendo precios desde MarketDataManager: {e}")
            return {}

    # =========================
    # INITIALIZATION
    # =========================

    async def _sync_from_client_async(self):
        """Sincronizar portfolio con balances reales del cliente (single source of truth) - versión asíncrona"""
        # Si es modo paper, sincronizar con SimulatedExchangeClient
        if self.mode == "simulated" or (self.exchange_client and hasattr(self.exchange_client, 'paper_mode') and self.exchange_client.paper_mode):
            logger.debug("🧪 Paper mode: Synchronizing with SimulatedExchangeClient")
            try:
                if hasattr(self.exchange_client, 'get_account_balances'):
                    import inspect
                    if inspect.iscoroutinefunction(self.exchange_client.get_account_balances):
                        balances = await self.exchange_client.get_account_balances()
                    else:
                        balances = self.exchange_client.get_account_balances()
                elif hasattr(self.exchange_client, 'get_balance'):
                    import inspect
                    if inspect.iscoroutinefunction(self.exchange_client.get_balance):
                        current_btc = await self.exchange_client.get_balance("BTC")
                        current_eth = await self.exchange_client.get_balance("ETH")
                        current_usdt = await self.exchange_client.get_balance("USDT")
                        balances = {
                            "BTC": current_btc,
                            "ETH": current_eth,
                            "USDT": current_usdt
                        }
                    else:
                        balances = {
                            "BTC": self.exchange_client.get_balance("BTC"),
                            "ETH": self.exchange_client.get_balance("ETH"),
                            "USDT": self.exchange_client.get_balance("USDT")
                        }
                else:
                    logger.warning("⚠️ Simulated client has no get_account_balances or get_balance method")
                    return False

                logger.debug(f"🔄 Sincronizando portfolio desde SimulatedExchangeClient: {balances}")
                
                # Map client balances to portfolio structure
                self.portfolio = {
                    "BTCUSDT": {"position": balances.get("BTC", 0.0), "free": balances.get("BTC", 0.0)},
                    "ETHUSDT": {"position": balances.get("ETH", 0.0), "free": balances.get("ETH", 0.0)},
                    "USDT": {"free": balances.get("USDT", self.initial_balance)},
                    "total": self.initial_balance,
                    "peak_value": self.initial_balance,
                    "total_fees": 0.0,
                }
                
                # Update cache
                import time
                self._balance_cache = balances
                self._balance_cache_time = {
                    "BTC": time.time(),
                    "ETH": time.time(),
                    "USDT": time.time()
                }

                return True

            except Exception as e:
                logger.error(f"❌ Error sincronizando portfolio desde SimulatedExchangeClient: {e}")
                return False
        else:
            # Modo live: sincronizar con BinanceClient
            try:
                if hasattr(self.exchange_client, 'get_balances'):
                    balances = self.exchange_client.get_balances()
                elif hasattr(self.exchange_client, 'get_account_balances'):
                    balances = await self.exchange_client.get_account_balances()
                else:
                    logger.warning("⚠️ Client has no get_balances or get_account_balances method")
                    return False

                logger.debug(f"🔄 Sincronizando portfolio desde exchange client: {balances}")
                
                # Map client balances to portfolio structure
                self.portfolio = {
                    "BTCUSDT": {"position": balances.get("BTC", 0.0), "free": balances.get("BTC", 0.0)},
                    "ETHUSDT": {"position": balances.get("ETH", 0.0), "free": balances.get("ETH", 0.0)},
                    "USDT": {"free": balances.get("USDT", self.initial_balance)},
                    "total": self.initial_balance,
                    "peak_value": self.initial_balance,
                    "total_fees": 0.0,
                }
                
                # Update cache
                import time
                self._balance_cache = balances
                self._balance_cache_time = {
                    "BTC": time.time(),
                    "ETH": time.time(),
                    "USDT": time.time()
                }

                return True

            except Exception as e:
                logger.error(f"❌ Error sincronizando portfolio desde exchange client: {e}")
                return False

    async def _init_portfolio_from_simulated_client_async(self):
        """Initialize portfolio from simulated client balances to maintain state between cycles - async version"""
        if self.exchange_client:
            try:
                # Try to get balances from SimulatedExchangeClient (check if async)
                import inspect
                if hasattr(self.exchange_client, 'get_account_balances'):
                    if inspect.iscoroutinefunction(self.exchange_client.get_account_balances):
                        balances = await self.exchange_client.get_account_balances()
                    else:
                        balances = self.exchange_client.get_account_balances()
                elif hasattr(self.exchange_client, 'get_balance'):
                    # Fallback if only single balance method available
                    if inspect.iscoroutinefunction(self.exchange_client.get_balance):
                        current_btc = await self.exchange_client.get_balance("BTC")
                        current_eth = await self.exchange_client.get_balance("ETH")
                        current_usdt = await self.exchange_client.get_balance("USDT")
                        balances = {
                            "BTC": current_btc,
                            "ETH": current_eth,
                            "USDT": current_usdt
                        }
                    else:
                        balances = {
                            "BTC": self.exchange_client.get_balance("BTC"),
                            "ETH": self.exchange_client.get_balance("ETH"),
                            "USDT": self.exchange_client.get_balance("USDT")
                        }
                else:
                    raise AttributeError("Simulated client has no get_account_balances or get_balance method")
                
                logger.debug(f"📊 Paper mode: Using simulated client balances: {balances}")
                
                # Convert client balances to portfolio structure (using BTC/ETH as base assets)
                self.portfolio = {
                    "BTCUSDT": {"position": balances.get("BTC", 0.0), "free": balances.get("BTC", 0.0)},
                    "ETHUSDT": {"position": balances.get("ETH", 0.0), "free": balances.get("ETH", 0.0)},
                    "USDT": {"free": balances.get("USDT", self.initial_balance)},
                    "total": self.initial_balance,
                    "peak_value": self.initial_balance,
                    "total_fees": 0.0,
                }
                
                # Calculate initial total value using simulated client's prices
                if hasattr(self.exchange_client, 'get_market_price'):
                    btc_price = self.exchange_client.get_market_price("BTCUSDT")
                    eth_price = self.exchange_client.get_market_price("ETHUSDT")
                    self.portfolio["total"] = (
                        self.portfolio["USDT"]["free"] +
                        self.portfolio["BTCUSDT"]["position"] * btc_price +
                        self.portfolio["ETHUSDT"]["position"] * eth_price
                    )
                    self.portfolio["peak_value"] = self.portfolio["total"]
                
                self.peak_value = self.portfolio["peak_value"]
                self.total_fees = 0.0
                
                # Update cache
                import time
                self._balance_cache = balances
                self._balance_cache_time = {
                    "BTC": time.time(),
                    "ETH": time.time(),
                    "USDT": time.time()
                }
                
                logger.info(f"🎯 Portfolio initialized from simulated client ({self.portfolio['total']:.2f} USDT)")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to get simulated client balances: {e}, using fallback")
        
        # Fallback if simulated client not available or failed
        self._init_portfolio()

    def _init_portfolio(self):
        """Initialize portfolio with hardcoded values (only as fallback)"""
        self.portfolio = {
            "BTCUSDT": {"position": 0.0, "free": 0.0},
            "ETHUSDT": {"position": 0.0, "free": 0.0},
            "USDT": {"free": self.initial_balance},
            "total": self.initial_balance,
            "peak_value": self.initial_balance,
            "total_fees": 0.0,
        }
        
        # Initialize cache
        self._balance_cache = {
            "BTC": 0.0,
            "ETH": 0.0,
            "USDT": self.initial_balance
        }
        import time
        self._balance_cache_time = {
            "BTC": time.time(),
            "ETH": time.time(),
            "USDT": time.time()
        }

        self.peak_value = self.initial_balance
        self.total_fees = 0.0

        logger.info(f"🎯 Portfolio inicializado limpio ({self.initial_balance} USDT)")

    def reset_portfolio(self):
        """Reset portfolio to initial state - ONLY call this explicitly"""
        logger.warning("⚠️ Resetting portfolio - this will lose all paper trading history")
        self._init_portfolio()

    def _configure_trading_costs(self):
        if self.mode == "live":
            self.maker_fee = 0.001
            self.taker_fee = 0.001
            self.slippage_bps = 2
        elif self.mode == "simulated":
            self.maker_fee = 0.0005  # Reduced from 0.002 to 0.0005 (0.05%)
            self.taker_fee = 0.0005  # Reduced from 0.002 to 0.0005 (0.05%)
            self.slippage_bps = 0.2  # Reduced from 10 to 0.2 (0.02%)
        else:
            self.maker_fee = 0.001
            self.taker_fee = 0.001
            self.slippage_bps = 5

        if not self.enable_commissions:
            self.maker_fee = 0.0
            self.taker_fee = 0.0

        if not self.enable_slippage:
            self.slippage_bps = 0

    # =========================
    # BALANCES - ASYNC FIRST
    # =========================

    @enforce_async_balance_access
    def get_balance(self, symbol: str) -> float:
        """
        Get balance for a symbol. DEPRECATED in async contexts.
        
        ⚠️ WARNING: This method should NOT be called in async contexts.
        Use get_asset_balance_async() instead when in async mode.
        
        Args:
            symbol: Symbol to get balance for (e.g., 'BTCUSDT', 'USDT')
            
        Returns:
            Balance as float
            
        Raises:
            AsyncBalanceRequiredError: If called in async context
        """
        if AsyncContextDetector.is_in_async_context():
            raise AsyncBalanceRequiredError(
                f"get_balance('{symbol}') called in async context. "
                f"Use await get_asset_balance_async('{symbol}') instead."
            )
        
        # Try to get from cache first
        import time
        if symbol == "USDT":
            cache_key = "USDT"
        else:
            cache_key = symbol.replace("USDT", "")
        
        if cache_key in self._balance_cache:
            cache_time = self._balance_cache_time.get(cache_key, 0)
            if time.time() - cache_time < self._balance_cache_ttl:
                value = self._balance_cache[cache_key]
                BalanceAccessLogger.log_balance_access(cache_key, "SYNC", value, "balance_cache")
                return value
        
        # Fallback to portfolio cache
        if symbol == "USDT":
            value = safe_float(self.portfolio.get("USDT", {}).get("free", 0.0))
        else:
            value = safe_float(self.portfolio.get(symbol, {}).get("position", 0.0))
        
        BalanceAccessLogger.log_balance_access(symbol, "SYNC", value, "portfolio_cache")
        return value

    async def get_asset_balance_async(self, asset: str) -> float:
        """
        Async method to get balance for a specific asset.
        This is the PREFERRED method in async contexts.
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH', 'USDT', 'BTCUSDT')
            
        Returns:
            Balance as float
        """
        try:
            # Normalize asset symbol
            normalized_asset = asset
            if asset in ["BTCUSDT", "ETHUSDT"]:
                normalized_asset = asset.replace("USDT", "")
            
            # Check cache first
            import time
            if normalized_asset in self._balance_cache:
                cache_time = self._balance_cache_time.get(normalized_asset, 0)
                if time.time() - cache_time < self._balance_cache_ttl:
                    value = self._balance_cache[normalized_asset]
                    BalanceAccessLogger.log_balance_access(asset, "ASYNC", value, "balance_cache")
                    return value
            
            # First try to get from exchange client if available
            if self.exchange_client:
                import inspect
                
                # Try get_account_balances first
                if hasattr(self.exchange_client, 'get_account_balances'):
                    if inspect.iscoroutinefunction(self.exchange_client.get_account_balances):
                        balances = await self.exchange_client.get_account_balances()
                    else:
                        balances = self.exchange_client.get_account_balances()
                    
                    value = balances.get(normalized_asset, 0.0)
                    
                    # Update cache
                    self._balance_cache[normalized_asset] = value
                    self._balance_cache_time[normalized_asset] = time.time()
                    
                    BalanceAccessLogger.log_balance_access(asset, "ASYNC", value, "exchange")
                    return value
                
                # Fallback to get_balance
                elif hasattr(self.exchange_client, 'get_balance'):
                    if inspect.iscoroutinefunction(self.exchange_client.get_balance):
                        value = await self.exchange_client.get_balance(normalized_asset)
                    else:
                        value = self.exchange_client.get_balance(normalized_asset)
                    
                    # Update cache
                    self._balance_cache[normalized_asset] = value
                    self._balance_cache_time[normalized_asset] = time.time()
                    
                    BalanceAccessLogger.log_balance_access(asset, "ASYNC", value, "exchange")
                    return value
            
            # Fallback to portfolio cache
            if normalized_asset == "USDT":
                value = safe_float(self.portfolio.get("USDT", {}).get("free", 0.0))
            else:
                symbol = f"{normalized_asset}USDT" if not normalized_asset.endswith("USDT") else normalized_asset
                value = safe_float(self.portfolio.get(symbol, {}).get("position", 0.0))
            
            BalanceAccessLogger.log_balance_access(asset, "ASYNC", value, "portfolio_cache")
            return value
            
        except Exception as e:
            logger.error(f"❌ Error getting balance for {asset}: {e}")
            BalanceAccessLogger.log_balance_access(asset, "ASYNC", 0.0, f"error: {e}")
            return 0.0

    async def get_balances_async(self) -> Dict[str, float]:
        """
        Async method to get all balances.
        This is the PREFERRED method in async contexts.
        
        Returns:
            Dictionary mapping asset symbols to balances
        """
        try:
            # Check if all balances are fresh in cache
            import time
            all_cached = True
            current_time = time.time()
            
            for asset in ["BTC", "ETH", "USDT"]:
                if (asset not in self._balance_cache or 
                    asset not in self._balance_cache_time or
                    current_time - self._balance_cache_time[asset] >= self._balance_cache_ttl):
                    all_cached = False
                    break
            
            if all_cached:
                balances = {k: v for k, v in self._balance_cache.items() if k in ["BTC", "ETH", "USDT"]}
                logger.info(f"[BALANCE_ACCESS] ASYNC | All balances | Source: balance_cache | Values: {balances}")
                self._balance_verification.mark_synced("balance_cache")
                return balances
            
            # Try to get from exchange client if available
            if self.exchange_client:
                import inspect
                
                if hasattr(self.exchange_client, 'get_account_balances'):
                    if inspect.iscoroutinefunction(self.exchange_client.get_account_balances):
                        balances = await self.exchange_client.get_account_balances()
                    else:
                        balances = self.exchange_client.get_account_balances()
                    
                    logger.info(f"[BALANCE_ACCESS] ASYNC | All balances | Source: exchange | Values: {balances}")
                    
                    # Update cache
                    self._balance_cache.update(balances)
                    for asset in balances:
                        self._balance_cache_time[asset] = time.time()
                    
                    self._balance_verification.mark_synced("exchange_async")
                    return balances
                
                elif hasattr(self.exchange_client, 'get_balances'):
                    if inspect.iscoroutinefunction(self.exchange_client.get_balances):
                        balances = await self.exchange_client.get_balances()
                    else:
                        balances = self.exchange_client.get_balances()
                    
                    logger.info(f"[BALANCE_ACCESS] ASYNC | All balances | Source: exchange | Values: {balances}")
                    
                    # Update cache
                    self._balance_cache.update(balances)
                    for asset in balances:
                        self._balance_cache_time[asset] = time.time()
                    
                    self._balance_verification.mark_synced("exchange_async")
                    return balances
            
            # Fallback to portfolio
            balances = {
                "BTC": safe_float(self.portfolio.get("BTCUSDT", {}).get("position", 0.0)),
                "ETH": safe_float(self.portfolio.get("ETHUSDT", {}).get("position", 0.0)),
                "USDT": safe_float(self.portfolio.get("USDT", {}).get("free", 0.0))
            }
            
            logger.warning(f"[BALANCE_ACCESS] ASYNC | All balances | Source: portfolio_cache (FALLBACK) | Values: {balances}")
            self._balance_verification.mark_fallback("portfolio_cache_fallback")
            
            # Update cache
            self._balance_cache.update(balances)
            for asset in balances:
                self._balance_cache_time[asset] = time.time()
            
            return balances
            
        except Exception as e:
            logger.error(f"❌ Error getting balances: {e}")
            self._balance_verification.mark_fallback(f"error: {e}")
            return {
                "BTC": 0.0,
                "ETH": 0.0,
                "USDT": 0.0
            }

    def has_position(self, symbol: str, threshold: float = 1e-6) -> bool:
        """Check if there's a position for the given symbol"""
        balance = self.get_balance(symbol)
        return balance > threshold

    async def has_position_async(self, symbol: str, threshold: float = 1e-6) -> bool:
        """Async version of has_position"""
        balance = await self.get_asset_balance_async(symbol)
        return balance > threshold

    def get_all_positions(self) -> Dict[str, float]:
        """Get all positions (SYNC - deprecated in async contexts)"""
        if AsyncContextDetector.is_in_async_context():
            logger.warning("get_all_positions() called in async context. Use get_balances_async() instead.")
        return {s: self.get_balance(s) for s in self.symbols + ["USDT"]}

    def get_balance_verification_status(self) -> Dict[str, Any]:
        """Get the current balance verification status"""
        return self._balance_verification.get_status()

    def are_balances_verified(self) -> bool:
        """Check if current balances are from verified async sync"""
        return self._balance_verification.is_verified()

    # =========================
    # VALUE - PURA: Usa MarketDataManager inyectado
    # =========================

    async def get_total_value_async(self) -> float:
        """
        Async version of get_total_value - VERSIÓN PURA.
        
        Usa MarketDataManager inyectado para obtener precios.
        No requiere pasar market_data como argumento.
        
        Returns:
            float: Valor total del portfolio en USDT
        """
        # Get fresh balances
        balances = await self.get_balances_async()
        
        usdt_balance = balances.get("USDT", 0.0)
        
        # Get market prices from MarketDataManager
        market_prices = await self._get_market_prices()
        
        if not market_prices:
            logger.warning("⚠️ No market prices available from MarketDataManager, using cached portfolio total")
            return safe_float(self.portfolio.get("total", usdt_balance))
        
        # Calculate using market prices
        total = usdt_balance
        
        for symbol in self.symbols:
            asset = symbol.replace("USDT", "")
            bal = balances.get(asset, 0.0)
            if bal <= 0:
                continue

            price = market_prices.get(symbol, 0.0)
            if price > 0:
                total += bal * price

        return total

    async def get_total_crypto_value_async(self) -> float:
        """
        Async method to get total value of crypto assets (BTC + ETH) at market prices.
        Usa MarketDataManager inyectado para obtener precios.
        
        Returns:
            float: Valor total de criptos en USDT
        """
        # Get fresh balances
        balances = await self.get_balances_async()
        
        # Get market prices from MarketDataManager
        market_prices = await self._get_market_prices()
        
        if not market_prices:
            logger.warning("⚠️ No market prices available from MarketDataManager")
            return 0.0
        
        # Calculate using market prices
        total_crypto = 0.0
        
        for symbol in self.symbols:
            asset = symbol.replace("USDT", "")
            bal = balances.get(asset, 0.0)
            if bal <= 0:
                continue

            price = market_prices.get(symbol, 0.0)
            if price > 0:
                total_crypto += bal * price

        return total_crypto

    # =========================
    # SNAPSHOT
    # =========================

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Snapshot completo del portfolio, compatible con pipeline y paper trading.
        
        Note: El valor total es sincrónico y puede estar desactualizado.
        Para valores precisos, usar get_total_value_async().
        """
        balances = {}

        for asset, data in self.portfolio.items():
            if asset in ["BTCUSDT", "ETHUSDT", "USDT"]:
                balances[asset] = {
                    "free": data.get("free", 0.0),
                    "locked": data.get("locked", 0.0),
                    "position": data.get("position", data.get("free", 0.0))
                }

        return {
            **balances,
            "total": self.portfolio.get("total", self.initial_balance)
        }

    # =========================
    # ORDER UPDATES - FIXED SIGNATURE
    # =========================

    async def update_from_orders_async(self, orders: list, market_data: Optional[Dict] = None):
        """
        Update portfolio after order execution.
        
        CRITICAL: En modo paper, NO procesamos órdenes aquí.
        Las órdenes ya fueron ejecutadas por SimulatedExchangeClient.
        Solo sincronizamos el portfolio desde el exchange simulado.
        
        Args:
            orders: Lista de órdenes ejecutadas
            market_data: Optional market data dict (for backwards compatibility, not used)
        """
        try:
            filled = [o for o in orders if o.get("status") == "filled"]
            if not filled:
                return

            # Check if we're in paper mode
            is_paper_mode = self.mode == "simulated" or (self.exchange_client and hasattr(self.exchange_client, 'paper_mode') and self.exchange_client.paper_mode)
            
            if is_paper_mode:
                # 📝 PAPER MODE: NO procesar órdenes, NO llamar a create_order
                # Solo sincronizar desde SimulatedExchangeClient (single source of truth)
                logger.info("[PAPER_MODE] Processing update_from_orders_async – syncing from simulated exchange")
                
                # CRITICAL FIX: Log NAV before execution
                nav_before = await self.get_total_value_async()
                logger.info("=" * 80)
                logger.info(f"📊 NAV BEFORE ORDER EXECUTION: ${nav_before:.2f}")
                logger.info("=" * 80)
                
                # Sync portfolio from SimulatedExchangeClient (single source of truth)
                await self.sync_from_exchange_async(self.exchange_client)
                
                # CRITICAL FIX: Update NAV after sync and log
                nav_after = await self.update_nav_async()
                
                # CRITICAL FIX: Log NAV after execution with detailed breakdown
                logger.info("=" * 80)
                logger.info(f"📊 NAV AFTER ORDER EXECUTION: ${nav_after['total_nav']:.2f}")
                logger.info(f"   USDT: ${nav_after['usdt']:.2f}")
                logger.info(f"   BTC:  {nav_after['btc_balance']:.6f} @ ${nav_after['btc_price']:.2f} = ${nav_after['btc_value']:.2f}")
                logger.info(f"   ETH:  {nav_after['eth_balance']:.4f} @ ${nav_after['eth_price']:.2f} = ${nav_after['eth_value']:.2f}")
                logger.info("=" * 80)
                
                # Verify synchronization: ensure SimulatedExchangeClient, PortfolioManager, and NAV are perfectly aligned
                if self.exchange_client and self.market_data_manager:
                    try:
                        # Get balances from exchange client (handle both get_balances and get_account_balances)
                        import inspect
                        if hasattr(self.exchange_client, 'get_balances'):
                            sim_balances = self.exchange_client.get_balances()
                        elif hasattr(self.exchange_client, 'get_account_balances'):
                            if inspect.iscoroutinefunction(self.exchange_client.get_account_balances):
                                sim_balances = await self.exchange_client.get_account_balances()
                            else:
                                sim_balances = self.exchange_client.get_account_balances()
                        else:
                            logger.warning("⚠️ Exchange client has no get_balances or get_account_balances method, skipping sync verification")
                            sim_balances = None
                        
                        if sim_balances is None:
                            return
                        
                        # Get balances from PortfolioManager
                        pm_balances = await self.get_balances_async()
                        
                        # Get market prices
                        market_prices = await self._get_market_prices()
                        
                        # Calculate NAV using both methods
                        sim_nav = sim_balances.get('USDT', 0.0)
                        pm_nav = pm_balances.get('USDT', 0.0)
                        
                        for asset in ['BTC', 'ETH']:
                            sim_balance = sim_balances.get(asset, 0.0)
                            pm_balance = pm_balances.get(asset, 0.0)
                            price = market_prices.get(f'{asset}USDT', 0.0)
                            
                            sim_nav += sim_balance * price
                            pm_nav += pm_balance * price
                            
                            # Verify balances are synchronized
                            balance_tolerance = 1e-6  # Very small tolerance for floating point
                            if abs(sim_balance - pm_balance) > balance_tolerance:
                                logger.critical(f"🚨 BALANCE SYNCHRONIZATION ERROR: {asset} Sim={sim_balance:.6f}, PM={pm_balance:.6f}, Diff={abs(sim_balance - pm_balance):.6f}")
                                # Force synchronization
                                if hasattr(self.exchange_client, 'get_account_balances'):
                                    import inspect
                                    if inspect.iscoroutinefunction(self.exchange_client.get_account_balances):
                                        fresh_balances = await self.exchange_client.get_account_balances()
                                    else:
                                        fresh_balances = self.exchange_client.get_account_balances()
                                    await self.sync_from_exchange_async(self.exchange_client)
                                return
                        
                        # Verify NAV synchronization
                        nav_tolerance = 0.01  # 1 cent tolerance
                        if abs(sim_nav - pm_nav) > nav_tolerance:
                            logger.critical(f"🚨 NAV SYNCHRONIZATION ERROR: Sim={sim_nav:.2f}, PM={pm_nav:.2f}, Diff={abs(sim_nav - pm_nav):.2f}")
                            # Force NAV recalculation
                            await self.update_nav_async()
                        else:
                            logger.info(f"✅ Synchronization verified: NAVs aligned within tolerance")
                    
                    except Exception as sync_error:
                        logger.error(f"❌ Error during synchronization verification: {sync_error}")
                
                # Log filled orders for tax and paper trading
                for o in filled:
                    # Log to tax tracker if available
                    if self.tax_tracker:
                        try:
                            self.tax_tracker.record_operation(o)
                            logger.debug(f"📝 Operación registrada en TaxTracker: {o['symbol']} {o['side']} {o['quantity']}")
                        except Exception as e:
                            logger.error(f"❌ Error registrando operación en TaxTracker: {e}")
                    
                    # Log to paper trade logger if available
                    if self.paper_logger:
                        try:
                            # Get market prices from MarketDataManager si está disponible
                            market_prices = await self._get_market_prices() if self.market_data_manager else {}
                            self.paper_logger.log_paper_trade(o, market_prices)
                            logger.debug(f"📝 Trade registrado en PaperTradeLogger: {o['symbol']} {o['side']} {o['quantity']}")
                        except Exception as e:
                            logger.error(f"❌ Error registrando trade en PaperTradeLogger: {e}")
                
                return
            else:
                # Real mode: sync with Binance balances
                if await self._sync_from_client_async():
                    logger.info("✅ Portfolio sincronizado con balances reales del exchange")
                else:
                    logger.warning("⚠️ Fallback: Actualizando portfolio desde cálculo local")
                    # Actualización local fallback - sincronizar con exchange_client si existe
                    if self.exchange_client:
                        await self._sync_from_client_async()

        except Exception as e:
            logger.error(f"❌ Error update_from_orders_async: {e}", exc_info=True)

    def update_from_orders(self, orders: list, market_data: Optional[Dict] = None):
        """
        Sync version - DEPRECATED.
        
        Args:
            orders: Lista de órdenes
            market_data: DEPRECATED - ya no se usa
        """
        import asyncio
        asyncio.run(self.update_from_orders_async(orders, market_data))

    # =========================
    # REBALANCE SUPPORT
    # =========================

    def get_position_age_seconds(self, symbol: str) -> float:
        if symbol not in self.position_age:
            return float("inf")
        return max(0, datetime.now().timestamp() - self.position_age[symbol])

    def can_rebalance_position(self, symbol: str) -> bool:
        return self.get_position_age_seconds(symbol) >= self.MIN_HOLD_TIME
    
    async def can_rebalance_position_async(self, symbol: str) -> bool:
        """Async version of can_rebalance_position"""
        return self.get_position_age_seconds(symbol) >= self.MIN_HOLD_TIME

    # =========================
    # NAV CALCULATION - PURA
    # =========================

    async def calculate_nav_async(self) -> Dict[str, float]:
        """
        Calculate TOTAL NAV including all assets at current market prices (async version).
        VERSIÓN PURA - Usa MarketDataManager inyectado.
        
        Formula: TOTAL_NAV = USDT + BTC * market_price_BTC + ETH * market_price_ETH
        
        Returns:
            Dict with NAV breakdown
        """
        balances = await self.get_balances_async()
        
        usdt_balance = balances.get("USDT", 0.0)
        btc_balance = balances.get("BTC", 0.0)
        eth_balance = balances.get("ETH", 0.0)
        
        # Get market prices from MarketDataManager
        market_prices = await self._get_market_prices()
        
        # Get market prices
        btc_price = market_prices.get("BTCUSDT", 0.0)
        eth_price = market_prices.get("ETHUSDT", 0.0)
        
        # Calculate value of each asset
        btc_value = btc_balance * btc_price
        eth_value = eth_balance * eth_price
        
        # Calculate TOTAL NAV
        total_nav = usdt_balance + btc_value + eth_value
        
        # Defensive safeguard: Ensure NAV never drops to 0 unless portfolio is actually empty
        if total_nav <= 0 and (usdt_balance > 0 or btc_balance > 0 or eth_balance > 0):
            logger.critical(f"🚨 NAV CALCULATION ERROR in calculate_nav_async: NAV={total_nav:.2f} with non-zero balances")
            # Calculate minimum possible NAV assuming $1 minimum price
            min_nav = usdt_balance + btc_balance * 1.0 + eth_balance * 1.0
            total_nav = max(min_nav, 0.01)
            logger.warning(f"⚠️ NAV adjusted to minimum: ${total_nav:.2f}")
        
        return {
            "total_nav": total_nav,
            "usdt": usdt_balance,
            "btc_balance": btc_balance,
            "btc_price": btc_price,
            "btc_value": btc_value,
            "eth_balance": eth_balance,
            "eth_price": eth_price,
            "eth_value": eth_value
        }

    def calculate_nav(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate TOTAL NAV including all assets at current market prices (sync version).
        
        Args:
            market_prices: Dict with symbol -> price mapping (e.g., {'BTCUSDT': 50000, 'ETHUSDT': 3000})
        
        Returns:
            Dict with NAV breakdown
        """
        usdt_balance = self.get_balance("USDT")
        btc_balance = self.get_balance("BTCUSDT")
        eth_balance = self.get_balance("ETHUSDT")
        
        # Get market prices
        btc_price = market_prices.get("BTCUSDT", 0.0)
        eth_price = market_prices.get("ETHUSDT", 0.0)
        
        # Calculate value of each asset
        btc_value = btc_balance * btc_price
        eth_value = eth_balance * eth_price
        
        # Calculate TOTAL NAV
        total_nav = usdt_balance + btc_value + eth_value
        
        return {
            "total_nav": total_nav,
            "usdt": usdt_balance,
            "btc_balance": btc_balance,
            "btc_price": btc_price,
            "btc_value": btc_value,
            "eth_balance": eth_balance,
            "eth_price": eth_price,
            "eth_value": eth_value
        }

    def log_nav(self, client=None):
        """
        Log NAV with complete breakdown including crypto values at market prices.
        Usa MarketDataManager inyectado para obtener precios.
        
        Args:
            client: Optional client (DEPRECATED - ya no se usa, solo para compatibilidad)
        """
        try:
            # Get prices from MarketDataManager
            import asyncio
            market_prices = asyncio.run(self._get_market_prices()) if self.market_data_manager else {}
            
            # Calculate NAV
            nav = self.calculate_nav(market_prices)
            
            # Log complete NAV breakdown
            logger.info("=" * 70)
            logger.info(f"💰 NAV CALCULATION - Total Portfolio Value")
            logger.info(f"   USDT:     ${nav['usdt']:.2f}")
            logger.info(f"   BTC:      {nav['btc_balance']:.6f} @ ${nav['btc_price']:.2f} = ${nav['btc_value']:.2f}")
            logger.info(f"   ETH:      {nav['eth_balance']:.4f} @ ${nav['eth_price']:.2f} = ${nav['eth_value']:.2f}")
            logger.info(f"   {'─' * 50}")
            logger.info(f"   TOTAL NAV: ${nav['total_nav']:.2f}")
            logger.info(f"   {'=' * 50}")
            
            # Store prices for future reference
            self._last_market_prices = market_prices
            
            return nav
            
        except Exception as e:
            logger.error(f"❌ Error calculating NAV: {e}")
            # Fallback to basic logging
            self.log_status_basic()
            return None

    def log_status_basic(self):
        """Basic portfolio logging without market prices"""
        logger.info(
            f"📊 Portfolio | TOTAL={self.portfolio.get('total', 0):.2f} | "
            f"BTC={self.portfolio.get('BTCUSDT', {}).get('position', 0):.6f} | "
            f"ETH={self.portfolio.get('ETHUSDT', {}).get('position', 0):.4f} | "
            f"USDT={self.portfolio.get('USDT', {}).get('free', 0):.2f}"
        )

    async def check_portfolio_stop_loss(self) -> bool:
        """
        Check if portfolio has hit stop-loss threshold.
        Usa MarketDataManager inyectado para obtener precios.
        
        Returns:
            bool: True si se activó stop-loss
        """
        total_value = await self.get_total_value_async()
        initial_value = self.initial_balance
        
        loss_pct = ((total_value - initial_value) / initial_value) * 100
        
        # ✅ Stop-loss at -5%
        if loss_pct <= -5.0:
            logger.critical(f"🚨 PORTFOLIO STOP-LOSS HIT: {loss_pct:.2f}%")
            logger.critical(f"   Current: ${total_value:.2f}")
            logger.critical(f"   Initial: ${initial_value:.2f}")
            logger.critical(f"   Loss: ${total_value - initial_value:.2f}")
            
            # Force liquidate all positions to USDT
            await self._emergency_liquidation()
            return True
        
        # ⚠️ Warning at -3%
        elif loss_pct <= -3.0:
            logger.warning(f"⚠️ PORTFOLIO WARNING: {loss_pct:.2f}% loss")
            return False
        
        return False
    
    async def _emergency_liquidation(self):
        """
        Emergency liquidation of all positions to USDT.
        Usa MarketDataManager inyectado para obtener precios.
        """
        logger.critical("🚨 EXECUTING EMERGENCY LIQUIDATION")
        
        # Get current positions
        balances = await self.get_balances_async()
        btc_balance = balances.get("BTC", 0.0)
        eth_balance = balances.get("ETH", 0.0)
        
        # Get current prices from MarketDataManager
        market_prices = await self._get_market_prices()
        btc_price = market_prices.get("BTCUSDT", 0.0)
        eth_price = market_prices.get("ETHUSDT", 0.0)
        
        # Calculate liquidation values
        btc_value = btc_balance * btc_price
        eth_value = eth_balance * eth_price
        total_liquidation_value = balances.get("USDT", 0.0) + btc_value + eth_value
        
        logger.critical(f"📊 LIQUIDATION SUMMARY:")
        logger.critical(f"   BTC: {btc_balance:.6f} @ ${btc_price:.2f} = ${btc_value:.2f}")
        logger.critical(f"   ETH: {eth_balance:.4f} @ ${eth_price:.2f} = ${eth_value:.2f}")
        logger.critical(f"   USDT: ${balances.get('USDT', 0.0):.2f}")
        logger.critical(f"   TOTAL: ${total_liquidation_value:.2f}")
        
        # Update portfolio to all USDT
        self.portfolio["BTCUSDT"] = {"position": 0.0, "free": 0.0}
        self.portfolio["ETHUSDT"] = {"position": 0.0, "free": 0.0}
        self.portfolio["USDT"] = {"free": total_liquidation_value}
        self.portfolio["total"] = total_liquidation_value
        
        logger.critical("✅ EMERGENCY LIQUIDATION COMPLETED - All positions converted to USDT")

    def log_status(self, market_data: Optional[Dict[str, Any]] = None, client=None):
        """
        Log portfolio status with full NAV calculation.
        
        This is the main logging method - it includes complete NAV breakdown.
        
        Args:
            market_data: DEPRECATED - ya no se usa
            client: DEPRECATED - ya no se usa
        """
        self.log_nav(client)

    async def update_with_consistent_prices(self) -> None:
        """
        Actualiza los precios del portfolio con valores consistentes del MarketDataManager.
        VERSIÓN PURA - Usa MarketDataManager inyectado.
        """
        if self.market_data_manager:
            prices = await self._get_market_prices()
            self._prices = prices
            logger.info(f"💰 Precios del portfolio actualizados: {prices}")
    
    def update_current_prices(self, current_prices: Dict[str, float]) -> None:
        """Actualiza los precios actuales para cálculo de NAV."""
        if not hasattr(self, '_current_prices'):
            self._current_prices = {}
        self._current_prices.update(current_prices)
        logger.info(f"📊 Precios actualizados: {current_prices}")

    def calculate_nav_with_prices(self, prices: Dict[str, float] = None) -> float:
        """Calcula NAV con precios específicos."""
        try:
            # Usar precios proporcionados o los almacenados
            if prices is None:
                prices = getattr(self, '_current_prices', {})
            
            # Obtener balances usando métodos sync
            usdt_balance = self.get_balance("USDT")
            btc_balance = self.get_balance("BTCUSDT")
            eth_balance = self.get_balance("ETHUSDT")
            
            nav = usdt_balance
            
            # Valor de criptos
            if btc_balance > 0:
                btc_price = prices.get("BTCUSDT", 0.0)
                nav += btc_balance * btc_price
            
            if eth_balance > 0:
                eth_price = prices.get("ETHUSDT", 0.0)
                nav += eth_balance * eth_price
            
            logger.debug(f"NAV calculado: ${nav:.2f} con precios {prices}")
            return nav
            
        except Exception as e:
            logger.error(f"Error calculando NAV: {e}", exc_info=True)
            return 0.0
