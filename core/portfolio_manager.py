# core/portfolio_manager.py - Gestión del portfolio

import os
import csv
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
        self._init_portfolio()

        logger.info(f"✅ PortfolioManager iniciado | mode={self.mode} | balance={initial_balance}")

    # =========================
    # INITIALIZATION
    # =========================

    def _init_portfolio(self):
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

            logger.info("✅ Portfolio actualizado desde órdenes")

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
