# l2_tactic/main_processor.py
import logging
from typing import Dict, Any, List
from dataclasses import asdict
import pandas as pd

from .signal_generator import L2TacticProcessor
from .signal_composer import SignalComposer
from .position_sizer import PositionSizerManager as PositionSizer
from .risk_controls import RiskControlManager
from .metrics import L2Metrics
from .config import L2Config
from .models import TacticalSignal, MarketFeatures, PositionSize

logger = logging.getLogger(__name__)

class L2MainProcessor:
    """
    Orquesta todo el flujo L2:
    señales → composición → sizing → control de riesgo → output final.
    """

    def __init__(self, config: L2Config, bus):
        self.config = config
        self.bus = bus
        self.generator = L2TacticProcessor(config)
        self.composer = SignalComposer(config)
        self.sizer = PositionSizer(config)
        self.risk = RiskControlManager(config)
        self.metrics = L2Metrics()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orquesta la ejecución táctica:
        - Obtiene señales brutas (raw_signals) del generador
        - Las compone y ajusta riesgo/sizing
        - Devuelve dict con órdenes listas para L1
        """
        portfolio = state.get("portfolio", {})
        market_data = state.get("mercado", {})
        features = state.get("features", {})
        correlation_matrix = state.get("correlation_matrix")

        # Estado de portfolio
        portfolio_state = {
            "total_capital": state.get("portfolio_value") or state.get("total_capital", 100_000.0),
            "available_capital": state.get("available_capital", state.get("cash", 100_000.0)),
            "daily_pnl": state.get("daily_pnl", 0.0),
        }

        # 1) Generar señales brutas
        raw_signals = await self.generator.process(
            portfolio=portfolio,
            market_data=market_data,
            features_by_symbol=features
        )

        # 2) Componer señales
        final_signals = self.composer.compose(raw_signals, market_data)


        # 3) Garantizar SL + sizing + riesgo
        orders: List[Dict] = []
        sizings: List[PositionSize] = []

        for sig in final_signals:
            mf: MarketFeatures = features.get(sig.symbol)
            if mf is None:
                logger.warning(f"No market features for {sig.symbol}. Skipping.")
                continue

            ensured = await self.sizer.ensure_stop_and_size(
                signal=sig,
                market_features=mf,
                portfolio_state=portfolio_state,
                corr_matrix=correlation_matrix,
            )
            if ensured is None:
                continue

            sig_ok, ps_ok = ensured
            self.risk.add_position(sig_ok, ps_ok, mf)

            order = self.sizer.to_l1_order(sig_ok, ps_ok)
            orders.append(order)
            sizings.append(ps_ok)

        result = {
            **state,  # conserva mercado, portfolio, etc.
            "orders_for_l1": orders,
            "signals_final": final_signals,
            "sizing": sizings,
        }

        logger.info(f"[L2] Prepared {len(orders)} orders for L1 (all with SL)")
        return result
