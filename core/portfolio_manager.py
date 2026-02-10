# core/portfolio_manager.py - Gestión del portfolio

import os
import csv
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pandas as pd

from core.logging import logger
from l2_tactic.utils import safe_float
from core.async_balance_helper import (
    AsyncContextDetector, 
    BalanceAccessLogger, 
    AsyncBalanceRequiredError,
    enforce_async_balance_access,
    BalanceVerificationStatus
)

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
    """Portfolio Manager with singleton pattern for consistent state across the system."""
    
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
        mode: str = "simulated",
        initial_balance: float = 3000.0,
        client: Optional[Any] = None,
        symbols: Optional[list] = None,
        enable_commissions: bool = True,
        enable_slippage: bool = True
    ):
        self.mode = mode
        self.initial_balance = initial_balance
        self.client = client
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

        self._configure_trading_costs()
        
        # Initialize portfolio from client if available
        if self.client:
            self._init_portfolio_from_client()
        else:
            self._init_portfolio()

        logger.info(f"✅ PortfolioManager iniciado | mode={self.mode} | balance={initial_balance}")

    # =========================
    # SYNC FROM EXCHANGE (CRITICAL FIX)
    # =========================

    def sync_from_exchange(self, exchange_client) -> bool:
        """
        CRITICAL METHOD: Synchronize portfolio balances from SimulatedExchangeClient.
        
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
            
            # Get market prices for NAV calculation
            btc_price = 0.0
            eth_price = 0.0
            
            if hasattr(exchange_client, 'get_market_price'):
                try:
                    btc_price = exchange_client.get_market_price("BTCUSDT")
                    eth_price = exchange_client.get_market_price("ETHUSDT")
                except Exception as e:
                    logger.warning(f"[SYNC] Could not get market prices: {e}")
            
            # Calculate NAV
            btc_value = current_btc * btc_price
            eth_value = current_eth * eth_price
            total_nav = current_usdt + btc_value + eth_value
            
            # Update portfolio atomically
            self.portfolio["BTCUSDT"] = {"position": current_btc, "free": current_btc}
            self.portfolio["ETHUSDT"] = {"position": current_eth, "free": current_eth}
            self.portfolio["USDT"] = {"free": current_usdt}
            self.portfolio["total"] = total_nav
            
            # Update peak value if needed
            if total_nav > self.peak_value:
                self.peak_value = total_nav
                self.portfolio["peak_value"] = total_nav
            
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
            
            # Log sync success with full details
            logger.info("=" * 70)
            logger.info("[SYNC] ✅ Portfolio synced from SimulatedExchangeClient")
            logger.info(f"[SYNC]    BTC Balance:  {current_btc:.6f}")
            logger.info(f"[SYNC]    ETH Balance:  {current_eth:.6f}")
            logger.info(f"[SYNC]    USDT Balance: ${current_usdt:.2f}")
            logger.info(f"[SYNC]    BTC Price:    ${btc_price:.2f} (Value: ${btc_value:.2f})")
            logger.info(f"[SYNC]    ETH Price:    ${eth_price:.2f} (Value: ${eth_value:.2f})")
            logger.info(f"[SYNC]    {'─' * 50}")
            logger.info(f"[SYNC]    TOTAL NAV:    ${total_nav:.2f}")
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
            
            # Get market prices for NAV calculation
            btc_price = 0.0
            eth_price = 0.0
            
            if hasattr(exchange_client, 'get_market_price'):
                try:
                    btc_price = exchange_client.get_market_price("BTCUSDT")
                    eth_price = exchange_client.get_market_price("ETHUSDT")
                except Exception as e:
                    logger.warning(f"[SYNC] Could not get market prices: {e}")
            
            # Calculate NAV
            btc_value = current_btc * btc_price
            eth_value = current_eth * eth_price
            total_nav = current_usdt + btc_value + eth_value
            
            # Update portfolio atomically
            self.portfolio["BTCUSDT"] = {"position": current_btc, "free": current_btc}
            self.portfolio["ETHUSDT"] = {"position": current_eth, "free": current_eth}
            self.portfolio["USDT"] = {"free": current_usdt}
            self.portfolio["total"] = total_nav
            
            # Update peak value if needed
            if total_nav > self.peak_value:
                self.peak_value = total_nav
                self.portfolio["peak_value"] = total_nav
            
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
            
            # Log sync success with full details
            logger.info("=" * 70)
            logger.info("[SYNC] ✅ Portfolio synced from SimulatedExchangeClient")
            logger.info(f"[SYNC]    BTC Balance:  {current_btc:.6f}")
            logger.info(f"[SYNC]    ETH Balance:  {current_eth:.6f}")
            logger.info(f"[SYNC]    USDT Balance: ${current_usdt:.2f}")
            logger.info(f"[SYNC]    BTC Price:    ${btc_price:.2f} (Value: ${btc_value:.2f})")
            logger.info(f"[SYNC]    ETH Price:    ${eth_price:.2f} (Value: ${eth_value:.2f})")
            logger.info(f"[SYNC]    {'─' * 50}")
            logger.info(f"[SYNC]    TOTAL NAV:    ${total_nav:.2f}")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"[SYNC] ❌ Error during sync_from_exchange_async: {e}", exc_info=True)
            return False

    def update_nav(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Update NAV using current market prices.
        
        Args:
            market_prices: Dict with symbol -> price mapping
            
        Returns:
            Dict with NAV breakdown
        """
        nav = self.calculate_nav(market_prices)
        
        # Update portfolio total
        self.portfolio["total"] = nav["total_nav"]
        
        logger.info(f"[SYNC] NAV updated: ${nav['total_nav']:.2f}")
        
        return nav

    async def update_nav_async(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Async version of update_nav.
        Fetches fresh balances before calculating NAV.
        
        Args:
            market_prices: Dict with symbol -> price mapping
            
        Returns:
            Dict with NAV breakdown
        """
        # Get fresh balances from exchange
        balances = await self.get_balances_async()
        
        # Calculate NAV with fresh balances
        usdt_balance = balances.get("USDT", 0.0)
        btc_balance = balances.get("BTC", 0.0)
        eth_balance = balances.get("ETH", 0.0)
        
        btc_price = market_prices.get("BTCUSDT", 0.0)
        eth_price = market_prices.get("ETHUSDT", 0.0)
        
        btc_value = btc_balance * btc_price
        eth_value = eth_balance * eth_price
        total_nav = usdt_balance + btc_value + eth_value
        
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

    # =========================
    # INITIALIZATION
    # =========================

    async def _sync_from_client_async(self):
        """Sincronizar portfolio con balances reales del cliente (single source of truth) - versión asíncrona"""
        # Si es modo paper, sincronizar con SimulatedExchangeClient
        if self.mode == "simulated" or (self.client and hasattr(self.client, 'paper_mode') and self.client.paper_mode):
            logger.debug("🧪 Paper mode: Synchronizing with SimulatedExchangeClient")
            try:
                if hasattr(self.client, 'get_account_balances'):
                    import inspect
                    if inspect.iscoroutinefunction(self.client.get_account_balances):
                        balances = await self.client.get_account_balances()
                    else:
                        balances = self.client.get_account_balances()
                elif hasattr(self.client, 'get_balance'):
                    import inspect
                    if inspect.iscoroutinefunction(self.client.get_balance):
                        current_btc = await self.client.get_balance("BTC")
                        current_eth = await self.client.get_balance("ETH")
                        current_usdt = await self.client.get_balance("USDT")
                        balances = {
                            "BTC": current_btc,
                            "ETH": current_eth,
                            "USDT": current_usdt
                        }
                    else:
                        balances = {
                            "BTC": self.client.get_balance("BTC"),
                            "ETH": self.client.get_balance("ETH"),
                            "USDT": self.client.get_balance("USDT")
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
                if hasattr(self.client, 'get_balances'):
                    balances = self.client.get_balances()
                elif hasattr(self.client, 'get_account_balances'):
                    balances = await self.client.get_account_balances()
                else:
                    logger.warning("⚠️ Client has no get_balances or get_account_balances method")
                    return False

                logger.debug(f"🔄 Sincronizando portfolio desde BinanceClient: {balances}")
                
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
                logger.error(f"❌ Error sincronizando portfolio desde BinanceClient: {e}")
                return False

    def _sync_from_client(self):
        """Sincronizar portfolio con balances reales del cliente (single source of truth) - versión sincrónica"""
        # Si es modo paper, sincronizar con SimulatedExchangeClient
        if self.mode == "simulated" or (self.client and hasattr(self.client, 'paper_mode') and self.client.paper_mode):
            logger.debug("🧪 Paper mode: Synchronizing with SimulatedExchangeClient")
            try:
                if hasattr(self.client, 'get_account_balances'):
                    import inspect
                    if inspect.iscoroutinefunction(self.client.get_account_balances):
                        import asyncio
                        try:
                            if not asyncio.get_running_loop():
                                balances = asyncio.run(self.client.get_account_balances())
                            else:
                                logger.warning("⚠️ Cannot use sync sync from async context")
                                return False
                        except RuntimeError:
                            balances = asyncio.run(self.client.get_account_balances())
                    else:
                        balances = self.client.get_account_balances()
                elif hasattr(self.client, 'get_balance'):
                    import inspect
                    if inspect.iscoroutinefunction(self.client.get_balance):
                        import asyncio
                        try:
                            if not asyncio.get_running_loop():
                                current_btc = asyncio.run(self.client.get_balance("BTC"))
                                current_eth = asyncio.run(self.client.get_balance("ETH"))
                                current_usdt = asyncio.run(self.client.get_balance("USDT"))
                                balances = {
                                    "BTC": current_btc,
                                    "ETH": current_eth,
                                    "USDT": current_usdt
                                }
                            else:
                                logger.warning("⚠️ Cannot use sync sync from async context")
                                return False
                        except RuntimeError:
                            current_btc = asyncio.run(self.client.get_balance("BTC"))
                            current_eth = asyncio.run(self.client.get_balance("ETH"))
                            current_usdt = asyncio.run(self.client.get_balance("USDT"))
                            balances = {
                                "BTC": current_btc,
                                "ETH": current_eth,
                                "USDT": current_usdt
                            }
                    else:
                        balances = {
                            "BTC": self.client.get_balance("BTC"),
                            "ETH": self.client.get_balance("ETH"),
                            "USDT": self.client.get_balance("USDT")
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
                if hasattr(self.client, 'get_balances'):
                    balances = self.client.get_balances()
                elif hasattr(self.client, 'get_account_balances'):
                    import asyncio
                    try:
                        if not asyncio.get_running_loop():
                            balances = asyncio.run(self.client.get_account_balances())
                        else:
                            logger.warning("⚠️ Cannot use sync sync from async context")
                            return False
                    except RuntimeError:
                        balances = asyncio.run(self.client.get_account_balances())
                else:
                    logger.warning("⚠️ Client has no get_balances or get_account_balances method")
                    return False

                logger.debug(f"🔄 Sincronizando portfolio desde BinanceClient: {balances}")
                
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
                logger.error(f"❌ Error sincronizando portfolio desde BinanceClient: {e}")
                return False

    def _init_portfolio_from_client(self):
        """Initialize portfolio from client balances (single source of truth)"""
        # Si es modo paper, inicializar con valores del cliente simulado
        if self.mode == "simulated" or (self.client and hasattr(self.client, 'paper_mode') and self.client.paper_mode):
            logger.info("🧪 Paper mode: Initializing portfolio from simulated client balances")
            self._init_portfolio_from_simulated_client()
        else:
            if not self._sync_from_client():
                self._init_portfolio()

    async def _init_portfolio_from_simulated_client_async(self):
        """Initialize portfolio from simulated client balances to maintain state between cycles - async version"""
        if self.client:
            try:
                # Try to get balances from SimulatedExchangeClient (check if async)
                import inspect
                if hasattr(self.client, 'get_account_balances'):
                    if inspect.iscoroutinefunction(self.client.get_account_balances):
                        balances = await self.client.get_account_balances()
                    else:
                        balances = self.client.get_account_balances()
                elif hasattr(self.client, 'get_balance'):
                    # Fallback if only single balance method available
                    if inspect.iscoroutinefunction(self.client.get_balance):
                        current_btc = await self.client.get_balance("BTC")
                        current_eth = await self.client.get_balance("ETH")
                        current_usdt = await self.client.get_balance("USDT")
                        balances = {
                            "BTC": current_btc,
                            "ETH": current_eth,
                            "USDT": current_usdt
                        }
                    else:
                        balances = {
                            "BTC": self.client.get_balance("BTC"),
                            "ETH": self.client.get_balance("ETH"),
                            "USDT": self.client.get_balance("USDT")
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
                if hasattr(self.client, 'get_market_price'):
                    btc_price = self.client.get_market_price("BTCUSDT")
                    eth_price = self.client.get_market_price("ETHUSDT")
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

    def _init_portfolio_from_simulated_client(self):
        """Initialize portfolio from simulated client balances to maintain state between cycles - sync version"""
        if self.client:
            try:
                # Try to get balances from SimulatedExchangeClient (check if async)
                import inspect
                if hasattr(self.client, 'get_account_balances'):
                    if inspect.iscoroutinefunction(self.client.get_account_balances):
                        import asyncio
                        try:
                            if not asyncio.get_running_loop():
                                balances = asyncio.run(self.client.get_account_balances())
                            else:
                                # If we're already in an async loop, we should have used initialize_async instead
                                logger.warning("⚠️ Should use initialize_async instead of sync init in async context")
                                raise RuntimeError("Should use initialize_async instead of sync init in async context")
                        except Exception as e:
                            logger.warning(f"⚠️ Cannot get async balances: {e}")
                            raise RuntimeError(f"Cannot get async balances: {e}")
                    else:
                        balances = self.client.get_account_balances()
                elif hasattr(self.client, 'get_balance'):
                    # Fallback if only single balance method available
                    if inspect.iscoroutinefunction(self.client.get_balance):
                        import asyncio
                        try:
                            if not asyncio.get_running_loop():
                                current_btc = asyncio.run(self.client.get_balance("BTC"))
                                current_eth = asyncio.run(self.client.get_balance("ETH"))
                                current_usdt = asyncio.run(self.client.get_balance("USDT"))
                                balances = {
                                    "BTC": current_btc,
                                    "ETH": current_eth,
                                    "USDT": current_usdt
                                }
                            else:
                                # If we're already in an async loop, we should have used initialize_async instead
                                logger.warning("⚠️ Should use initialize_async instead of sync init in async context")
                                raise RuntimeError("Should use initialize_async instead of sync init in async context")
                        except Exception as e:
                            logger.warning(f"⚠️ Cannot get async balance: {e}")
                            raise RuntimeError(f"Cannot get async balance: {e}")
                    else:
                        balances = {
                            "BTC": self.client.get_balance("BTC"),
                            "ETH": self.client.get_balance("ETH"),
                            "USDT": self.client.get_balance("USDT")
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
                if hasattr(self.client, 'get_market_price'):
                    btc_price = self.client.get_market_price("BTCUSDT")
                    eth_price = self.client.get_market_price("ETHUSDT")
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

    async def initialize_async(self):
        """Initialize portfolio manager asynchronously - use this instead of __init__ in async contexts"""
        if self.client:
            if self.mode == "simulated" or (self.client and hasattr(self.client, 'paper_mode') and self.client.paper_mode):
                await self._init_portfolio_from_simulated_client_async()
            else:
                if not await self._sync_from_client_async():
                    self._init_portfolio()

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
            self.maker_fee = 0.002
            self.taker_fee = 0.002
            self.slippage_bps = 10
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
            if self.client:
                import inspect
                
                # Try get_account_balances first
                if hasattr(self.client, 'get_account_balances'):
                    if inspect.iscoroutinefunction(self.client.get_account_balances):
                        balances = await self.client.get_account_balances()
                    else:
                        balances = self.client.get_account_balances()
                    
                    value = balances.get(normalized_asset, 0.0)
                    
                    # Update cache
                    self._balance_cache[normalized_asset] = value
                    self._balance_cache_time[normalized_asset] = time.time()
                    
                    BalanceAccessLogger.log_balance_access(asset, "ASYNC", value, "exchange")
                    return value
                
                # Fallback to get_balance
                elif hasattr(self.client, 'get_balance'):
                    if inspect.iscoroutinefunction(self.client.get_balance):
                        value = await self.client.get_balance(normalized_asset)
                    else:
                        value = self.client.get_balance(normalized_asset)
                    
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
            if self.client:
                import inspect
                
                if hasattr(self.client, 'get_account_balances'):
                    if inspect.iscoroutinefunction(self.client.get_account_balances):
                        balances = await self.client.get_account_balances()
                    else:
                        balances = self.client.get_account_balances()
                    
                    logger.info(f"[BALANCE_ACCESS] ASYNC | All balances | Source: exchange | Values: {balances}")
                    
                    # Update cache
                    self._balance_cache.update(balances)
                    for asset in balances:
                        self._balance_cache_time[asset] = time.time()
                    
                    self._balance_verification.mark_synced("exchange_async")
                    return balances
                
                elif hasattr(self.client, 'get_balances'):
                    if inspect.iscoroutinefunction(self.client.get_balances):
                        balances = await self.client.get_balances()
                    else:
                        balances = self.client.get_balances()
                    
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
    # VALUE
    # =========================

    def get_total_value(self, market_data: Optional[Dict[str, Any]] = None) -> float:
        """Get total portfolio value (SYNC - deprecated in async contexts)"""
        if AsyncContextDetector.is_in_async_context():
            logger.warning("get_total_value() called in async context. Consider using async version.")
        
        # No sincronizar desde aquí para evitar error con asyncio.run()
        if not market_data:
            return safe_float(self.portfolio.get("total", 0.0))

        total = self.get_balance("USDT")

        for symbol in self.symbols:
            bal = self.get_balance(symbol)
            if bal <= 0:
                continue

            data = market_data.get(symbol)
            if isinstance(data, dict) and "close" in data:
                total += bal * safe_float(data["close"])
            elif isinstance(data, pd.DataFrame) and "close" in data.columns:
                total += bal * safe_float(data["close"].iloc[-1])

        return total

    async def get_total_value_async(self, market_data: Optional[Dict[str, Any]] = None) -> float:
        """Async version of get_total_value"""
        # Get fresh balances
        balances = await self.get_balances_async()
        
        usdt_balance = balances.get("USDT", 0.0)
        
        if not market_data:
            # Try to get prices from client
            btc_price = 0.0
            eth_price = 0.0
            
            if self.client and hasattr(self.client, 'get_market_price'):
                try:
                    btc_price = self.client.get_market_price("BTCUSDT")
                    eth_price = self.client.get_market_price("ETHUSDT")
                except Exception as e:
                    logger.debug(f"Could not get market prices: {e}")
            
            btc_value = balances.get("BTC", 0.0) * btc_price
            eth_value = balances.get("ETH", 0.0) * eth_price
            
            return usdt_balance + btc_value + eth_value
        
        # Calculate using provided market data
        total = usdt_balance
        
        for symbol in self.symbols:
            asset = symbol.replace("USDT", "")
            bal = balances.get(asset, 0.0)
            if bal <= 0:
                continue

            data = market_data.get(symbol)
            if isinstance(data, dict) and "close" in data:
                total += bal * safe_float(data["close"])
            elif isinstance(data, pd.DataFrame) and "close" in data.columns:
                total += bal * safe_float(data["close"].iloc[-1])

        return total

    # =========================
    # SNAPSHOT (🔥 FIX CLAVE)
    # =========================

    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Snapshot completo del portfolio, compatible con pipeline y paper trading.
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
            "total": self.get_total_value()
        }

    # =========================
    # ORDER UPDATES
    # =========================

    async def update_from_orders_async(self, orders: list, market_data: Dict[str, Any]):
        """
        Update portfolio after order execution.
        
        CRITICAL: En modo paper, NO procesamos órdenes aquí.
        Las órdenes ya fueron ejecutadas por SimulatedExchangeClient.
        Solo sincronizamos el portfolio desde el exchange simulado.
        """
        try:
            filled = [o for o in orders if o.get("status") == "filled"]
            if not filled:
                return

            # Check if we're in paper mode
            is_paper_mode = self.mode == "simulated" or (self.client and hasattr(self.client, 'paper_mode') and self.client.paper_mode)
            
            if is_paper_mode:
                # 📝 PAPER MODE: NO procesar órdenes, NO llamar a create_order
                # Solo sincronizar desde SimulatedExchangeClient (single source of truth)
                logger.info("[PAPER_MODE] Skipping update_from_orders_async – syncing from simulated exchange only")
                
                # Sync portfolio from SimulatedExchangeClient (single source of truth)
                await self.sync_from_exchange_async(self.client)
                
                return
            else:
                # Real mode: sync with Binance balances
                if await self._sync_from_client_async():
                    logger.info("✅ Portfolio sincronizado con balances reales del exchange")
                else:
                    logger.warning("⚠️ Fallback: Actualizando portfolio desde cálculo local")
                    btc = self.get_balance("BTCUSDT")
                    eth = self.get_balance("ETHUSDT")
                    usdt = self.get_balance("USDT")

                    for o in filled:
                        symbol = o["symbol"]
                        side = o["side"].lower()
                        qty = safe_float(o["quantity"])
                        price = safe_float(o["filled_price"])

                        value = qty * price
                        fee = value * self.taker_fee

                        if symbol == "BTCUSDT":
                            if side == "buy" and usdt >= value + fee:
                                btc += qty
                                usdt -= value + fee
                            elif side == "sell" and btc >= qty:
                                btc -= qty
                                usdt += value - fee

                        elif symbol == "ETHUSDT":
                            if side == "buy" and usdt >= value + fee:
                                eth += qty
                                usdt -= value + fee
                            elif side == "sell" and eth >= qty:
                                eth -= qty
                                usdt += value - fee

                        self.total_fees += fee
                        self.position_age[symbol] = datetime.now().timestamp()

                    self.portfolio = {
                        "BTCUSDT": {"position": btc, "free": btc},
                        "ETHUSDT": {"position": eth, "free": eth},
                        "USDT": {"free": usdt},
                        "total": self.get_total_value(market_data),
                        "peak_value": max(self.peak_value, self.get_total_value(market_data)),
                        "total_fees": self.total_fees,
                    }

                    self.peak_value = self.portfolio["peak_value"]

        except Exception as e:
            logger.error(f"❌ Error update_from_orders_async: {e}", exc_info=True)

    def update_from_orders(self, orders: list, market_data: Dict[str, Any]):
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
    # NAV CALCULATION & LOGGING (FIXED)
    # =========================

    def calculate_nav(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate TOTAL NAV including all assets at current market prices.
        
        Formula: TOTAL_NAV = USDT + BTC * market_price_BTC + ETH * market_price_ETH
        
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

    def log_nav(self, market_data: Optional[Dict[str, Any]] = None, client=None):
        """
        Log NAV with complete breakdown including crypto values at market prices.
        Uses MarketDataManager or DataFeed for latest prices.
        
        Args:
            market_data: Optional market data dict with prices
            client: Optional client (SimulatedExchangeClient or BinanceClient) with get_market_price method
        """
        try:
            market_prices = {}
            
            # Try to get prices from client first (most reliable)
            if client and hasattr(client, 'get_market_price'):
                try:
                    market_prices["BTCUSDT"] = client.get_market_price("BTCUSDT")
                    market_prices["ETHUSDT"] = client.get_market_price("ETHUSDT")
                except Exception as e:
                    logger.debug(f"Could not get prices from client: {e}")
            
            # Fallback to market_data
            if not market_prices and market_data:
                for symbol in ["BTCUSDT", "ETHUSDT"]:
                    if symbol in market_data:
                        data = market_data[symbol]
                        if isinstance(data, dict) and "close" in data:
                            market_prices[symbol] = safe_float(data["close"])
                        elif isinstance(data, pd.DataFrame) and "close" in data.columns:
                            market_prices[symbol] = safe_float(data["close"].iloc[-1])
            
            # Fallback to portfolio's stored prices
            if not market_prices and hasattr(self, '_last_market_prices'):
                market_prices = self._last_market_prices
            
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
            f"📊 Portfolio | TOTAL={self.get_total_value():.2f} | "
            f"BTC={self.get_balance('BTCUSDT'):.6f} | "
            f"ETH={self.get_balance('ETHUSDT'):.4f} | "
            f"USDT={self.get_balance('USDT'):.2f}"
        )

    def log_status(self, market_data: Optional[Dict[str, Any]] = None, client=None):
        """
        Log portfolio status with full NAV calculation.
        
        This is the main logging method - it includes complete NAV breakdown.
        """
        self.log_nav(market_data, client)

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
            
            # Obtener balances usando el método correcto
            # NO usar self.balances si no existe
            if hasattr(self, 'get_balances_async'):
                # En contexto async necesitamos manejar esto diferente
                # Por ahora, devolver 0 y loggear advertencia
                logger.warning("⚠️ calculate_nav_with_prices no puede usar async en contexto sync")
                return 0.0
                
            # Si hay un método sync para balances
            if hasattr(self, 'get_balances'):
                balances = self.get_balances()
            else:
                # Intentar acceder a atributo si existe
                balances = getattr(self, 'balances', {})
            
            nav = 0.0
            
            # Valor de criptos
            btc_balance = balances.get("BTC", 0.0)
            if btc_balance > 0:
                btc_price = prices.get("BTCUSDT", 0.0)
                nav += btc_balance * btc_price
            
            eth_balance = balances.get("ETH", 0.0)
            if eth_balance > 0:
                eth_price = prices.get("ETHUSDT", 0.0)
                nav += eth_balance * eth_price
            
            # Valor de USDT
            usdt_balance = balances.get("USDT", 0.0)
            nav += usdt_balance
            
            logger.debug(f"NAV calculado: ${nav:.2f} con precios {prices}")
            return nav
            
        except Exception as e:
            logger.error(f"Error calculando NAV: {e}", exc_info=True)
            return 0.0
