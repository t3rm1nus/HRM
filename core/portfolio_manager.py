# core/portfolio_manager.py - Gestión del portfolio

import os
import csv
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import pandas as pd

from core.logging import logger
from l2_tactic.utils import safe_float

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

        self._configure_trading_costs()
        
        # Initialize portfolio from client if available
        if self.client:
            self._init_portfolio_from_client()
        else:
            self._init_portfolio()

        logger.info(f"✅ PortfolioManager iniciado | mode={self.mode} | balance={initial_balance}")

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
    # BALANCES
    # =========================

    def get_balance(self, symbol: str) -> float:
        if symbol == "USDT":
            return safe_float(self.portfolio.get("USDT", {}).get("free", 0.0))
        return safe_float(self.portfolio.get(symbol, {}).get("position", 0.0))

    def has_position(self, symbol: str, threshold: float = 1e-6) -> bool:
        return self.get_balance(symbol) > threshold

    def get_all_positions(self) -> Dict[str, float]:
        return {s: self.get_balance(s) for s in self.symbols + ["USDT"]}

    # =========================
    # VALUE
    # =========================

    def get_total_value(self, market_data: Optional[Dict[str, Any]] = None) -> float:
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
        try:
            filled = [o for o in orders if o.get("status") == "filled"]
            if not filled:
                return

            # In paper mode, update SimulatedExchangeClient directly (single source of truth)
            if self.mode == "simulated" or (self.client and hasattr(self.client, 'paper_mode') and self.client.paper_mode):
                logger.debug("🧪 Paper mode: Updating portfolio from local calculation")
                
                # Send orders to SimulatedExchangeClient
                for o in filled:
                    symbol = o["symbol"]
                    side = o["side"].lower()
                    qty = safe_float(o["quantity"])
                    price = safe_float(o["filled_price"])
                    
                    await self.client.create_order(symbol, side, qty, price)
                
                # Sync portfolio from SimulatedExchangeClient (single source of truth)
                import inspect
                if inspect.iscoroutinefunction(self.client.get_account_balances):
                    balances = await self.client.get_account_balances()
                else:
                    balances = self.client.get_account_balances()
                    
                # Validate balances after order execution
                current_btc = balances.get("BTC", 0.0)
                current_eth = balances.get("ETH", 0.0)
                current_usdt = balances.get("USDT", 0.0)
                
                if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
                    logger.critical("🚨 FATAL: Balances volvieron a cero después de ejecutar órdenes - deteniendo sistema")
                    raise RuntimeError("Balances cero después de ejecución de órdenes")
                
                self.portfolio = {
                    "BTCUSDT": {"position": current_btc, "free": current_btc},
                    "ETHUSDT": {"position": current_eth, "free": current_eth},
                    "USDT": {"free": current_usdt},
                    "total": self.initial_balance,
                    "peak_value": self.initial_balance,
                    "total_fees": 0.0,
                }
                
                # Calculate total value
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
                
                logger.debug("✅ Portfolio updated from SimulatedExchangeClient")
                
                # Log portfolio comparison to show real changes
                logger.info("📊 PORTFOLIO COMPARISON - After order execution:")
                logger.info(f"   BTC: {self.portfolio['BTCUSDT']['position']:.6f}")
                logger.info(f"   ETH: {self.portfolio['ETHUSDT']['position']:.6f}")
                logger.info(f"   USDT: {self.portfolio['USDT']['free']:.2f}")
                logger.info(f"   Total Value: {self.portfolio['total']:.2f}")
                
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

    # =========================
    # LOGGING
    # =========================

    def log_status(self):
        logger.info(
            f"📊 Portfolio | TOTAL={self.get_total_value():.2f} | "
            f"BTC={self.get_balance('BTCUSDT'):.6f} | "
            f"ETH={self.get_balance('ETHUSDT'):.4f} | "
            f"USDT={self.get_balance('USDT'):.2f}"
        )