# l2_tactic/main_processor.py

from .bus_integration import L2BusAdapter

class L2MainProcessor:
    """
    Orquesta todo el flujo L2:
    señales → composición → sizing → control de riesgo → output final.
    """

    def __init__(self, config: L2Config, bus):
        self.config = config
        self.bus = bus
        self.optimizer = PerformanceOptimizer(config)
        self.generator = L2TacticProcessor(config)
        self.composer = SignalComposer(config)
        self.sizer = PositionSizer(config)
        self.risk = RiskControlManager(config)
        self.metrics = L2Metrics()

        # 🔹 Nuevo: adaptador al bus
        self.bus_adapter = L2BusAdapter(bus, config)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        portfolio = state.get("portfolio", {})
        market_data = state.get("mercado", {})
        features = state.get("features", {})
        correlation_matrix = state.get("correlation_matrix")

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
        final_signals = await self.composer.compose_signals(raw_signals)

        # 3) Pasar señales al bus_integration (con sizing + riesgo + métricas)
        orders: List[Dict] = []
        sizings: List[PositionSize] = []

        for sig in final_signals:
            mf: MarketFeatures = features.get(sig.symbol)
            if mf is None:
                logger.warning(f"No market features for {sig.symbol}. Skipping.")
                continue

            # 👉 delegamos procesamiento a L2BusAdapter
            await self.bus_adapter._process_signal(sig, mf)

            # también guardamos órdenes locales (para retorno)
            ensured = await self.sizer.ensure_stop_and_size(
                signal=sig,
                market_features=mf,
                portfolio_state=portfolio_state,
                corr_matrix=correlation_matrix,
            )
            if ensured is None:
                continue

            sig_ok, ps_ok = ensured
            order = self.sizer.to_l1_order(sig_ok, ps_ok)
            orders.append(order)
            sizings.append(ps_ok)

        result = {
            **state,
            "orders_for_l1": orders,
            "signals_final": final_signals,
            "sizing": sizings,
        }

        # 🔹 Publicamos performance report hacia L4
        await self.bus_adapter.publish_performance_report()

        logger.info(f"[L2] Prepared {len(orders)} orders for L1 (all with SL)")
        return result
