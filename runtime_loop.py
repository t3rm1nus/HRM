#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bucle principal de HRM con ejecución centralizada en ExecutionGateway."""

import asyncio
from typing import Any, Dict

import pandas as pd

from core.state_manager import get_system_state
from core.logging import logger
from core.trading_metrics import get_trading_metrics
from core.portfolio_manager import PortfolioManager
from core.error_handler import ErrorHandler
from l1_operational.data_feed import DataFeed
from l1_operational.order_manager import OrderManager
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l3_strategy.regime_classifier import MarketRegimeClassifier
from l3_strategy.decision_maker import make_decision
from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus
from sentiment.sentiment_manager import update_sentiment_texts


class HRMRuntimeLoop:
    """Coordina los ciclos de análisis y trading de HRM."""

    def __init__(self, portfolio_manager: PortfolioManager, order_manager: OrderManager):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        # HRMBootstrap inyecta el Gateway después de construirlo.
        # No se permite fallback a OrderManager para ejecutar órdenes.
        self.execution_gateway = None
        self.cycle_id = 0
        self.last_sentiment_update = 0
        self.sentiment_texts_cache = []
        self.sentiment_update_interval = 2160
        self.data_feed = DataFeed(config)
        self.l2_config = L2Config()
        self.l2_processor = L2TacticProcessor(
            self.l2_config,
            portfolio_manager=portfolio_manager,
            apagar_l3=APAGAR_L3,
        )
        self.trading_metrics = get_trading_metrics()
        self.regime_classifier = MarketRegimeClassifier()
        self.message_bus = MessageBus()

    async def run(self):
        """Ejecuta el bucle principal hasta su cancelación."""
        logger.info("🚀 Starting HRM Runtime Loop")
        try:
            while True:
                await self._execute_cycle()
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            logger.info("🛑 Runtime loop interrupted by user")
        except asyncio.CancelledError:
            logger.info("🛑 Runtime loop cancelled")
            raise
        except Exception:
            logger.exception("❌ Runtime loop error")
        finally:
            await self._cleanup()

    async def _execute_cycle(self):
        """Ejecuta un ciclo completo del sistema."""
        self.cycle_id += 1
        start_time = pd.Timestamp.utcnow()
        try:
            market_data = await self._update_market_data()
            if not market_data:
                logger.warning("⚠️ No market data available, skipping cycle")
                return
            await self._update_portfolio_state(market_data)
            await self._update_sentiment_analysis()
            l3_decision = await self._execute_l3_processing(market_data)
            l2_signals = await self._execute_l2_processing(market_data, l3_decision)
            orders = await self._execute_l1_processing(l2_signals, market_data)
            processed_orders = await self._execute_orders(orders)
            await self._update_portfolio_from_orders(processed_orders, market_data)
            await self._log_cycle_results(start_time, l2_signals, processed_orders)
        except Exception as error:
            logger.exception("❌ Cycle %s failed", self.cycle_id)
            await ErrorHandler.handle_cycle_error(error, self.cycle_id)

    async def _update_market_data(self) -> Dict[str, Any]:
        """Obtiene los datos de mercado."""
        try:
            market_data = await self.data_feed.get_market_data()
            if market_data:
                logger.info("📊 Market data updated for %s symbols", len(market_data))
            return market_data or {}
        except Exception:
            logger.exception("❌ Failed to update market data")
            return {}

    async def _update_portfolio_state(self, market_data: Dict[str, Any]):
        """Sincroniza la cartera con el exchange PAPER antes de decidir."""
        del market_data
        ejecutor = getattr(self.order_manager, "executors", None)
        exchange = getattr(ejecutor, "simulated_client", None)
        if exchange is None:
            raise RuntimeError("No existe el exchange simulado PAPER")
        sincronizar = getattr(self.portfolio_manager, "sync_from_exchange_async", None)
        if not callable(sincronizar):
            raise RuntimeError("PortfolioManager no implementa sync_from_exchange_async")
        resultado = await sincronizar(exchange)
        if resultado is False:
            raise RuntimeError("Falló la sincronización de cartera con PAPER")

    async def _update_sentiment_analysis(self):
        """Actualiza el sentimiento según el intervalo configurado."""
        if self.cycle_id - self.last_sentiment_update < self.sentiment_update_interval:
            return
        self.sentiment_texts_cache = await update_sentiment_texts()
        self.last_sentiment_update = self.cycle_id
        logger.info("✅ Sentiment analysis updated with %s texts", len(self.sentiment_texts_cache))

    async def _execute_l3_processing(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Obtiene el contexto estratégico."""
        if APAGAR_L3:
            return {"regime": "disabled", "signal": "hold", "confidence": 0.0, "allow_l2_signals": True}
        try:
            if "BTCUSDT" not in market_data:
                return {"regime": "unknown", "signal": "hold", "confidence": 0.0, "allow_l2_signals": True}
            regime = self.regime_classifier.classify_market_regime(market_data["BTCUSDT"], "BTCUSDT")
            decision = make_decision(
                inputs={},
                portfolio_state=self.portfolio_manager.get_portfolio_state(),
                market_data=market_data,
                regime_decision=regime,
                balances_synced=True,
            )
            return decision
        except Exception:
            logger.exception("❌ L3 processing failed")
            return {"regime": "error", "signal": "hold", "confidence": 0.0, "allow_l2_signals": True}

    async def _execute_l2_processing(self, market_data: Dict[str, Any], l3_decision: Dict[str, Any]) -> list:
        """Genera señales tácticas."""
        try:
            signals = self.l2_processor.generate_signals_conservative(market_data=market_data, l3_context=l3_decision)
            logger.info("🎯 Generated %s L2 signals", len(signals))
            return signals
        except Exception:
            logger.exception("❌ L2 processing failed")
            return []

    async def _execute_l1_processing(self, l2_signals: list, market_data: Dict[str, Any]) -> list:
        """Genera órdenes candidatas; la validación pertenece al Gateway."""
        del market_data
        try:
            orders = await self.order_manager.generate_orders(get_system_state(), l2_signals)
            logger.info("✅ Generated %s candidate orders", len(orders))
            return orders
        except Exception:
            logger.exception("❌ L1 processing failed")
            return []

    async def _execute_orders(self, orders: list) -> list:
        """Ejecuta órdenes solo por ExecutionGateway; nunca usa fallback."""
        if not orders:
            return []
        gateway = self.execution_gateway
        if gateway is None or not callable(getattr(gateway, "execute_orders", None)):
            raise RuntimeError("ExecutionGateway no conectado: ejecución bloqueada para no saltarse validación y riesgo")
        resultados = await gateway.execute_orders(orders)
        ejecutadas = sum(1 for orden in resultados if orden.get("status") == "filled")
        logger.info("💰 ExecutionGateway ejecutó %s órdenes", ejecutadas)
        return resultados

    async def _update_portfolio_from_orders(self, processed_orders: list, market_data: Dict[str, Any]):
        """Actualiza NAV y persiste; no reaplica fills ni duplica métricas."""
        del processed_orders
        try:
            precios = self._extraer_precios(market_data)
            actualizar_nav = getattr(self.portfolio_manager, "update_nav_async", None)
            if callable(actualizar_nav):
                await actualizar_nav(market_prices=precios)
            if self.cycle_id % 5 == 0:
                self.portfolio_manager.save_to_json()
        except Exception:
            logger.exception("❌ Failed to refresh or persist portfolio")

    @staticmethod
    def _extraer_precios(market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extrae precios de cierre válidos de los datos de mercado."""
        precios = {}
        for simbolo, datos in market_data.items():
            try:
                if isinstance(datos, dict):
                    precio = datos.get("close")
                elif isinstance(datos, pd.DataFrame) and not datos.empty and "close" in datos.columns:
                    precio = datos["close"].iloc[-1]
                else:
                    continue
                precio = float(precio)
                if precio > 0:
                    precios[simbolo] = precio
            except (TypeError, ValueError, KeyError, IndexError):
                logger.debug("Precio no válido para %s", simbolo)
        return precios

    async def _log_cycle_results(self, start_time: pd.Timestamp, l2_signals: list, processed_orders: list):
        """Registra el resumen del ciclo."""
        try:
            duration = (pd.Timestamp.utcnow() - start_time).total_seconds()
            rechazadas = sum(1 for orden in processed_orders if orden.get("status") == "rejected")
            accionables = sum(1 for signal in l2_signals if getattr(signal, "side", None) not in ("hold", "HOLD"))
            logger.info(
                "📊 Cycle %s | Duration: %.1fs | Signals: %s (actionable: %s) | Orders: %s | Rejected: %s",
                self.cycle_id, duration, len(l2_signals), accionables, len(processed_orders) - rechazadas, rechazadas,
            )
            if self.cycle_id % 10 == 0:
                self.trading_metrics.log_periodic_report()
        except Exception:
            logger.exception("❌ Failed to log cycle results")

    async def _cleanup(self):
        """Guarda la cartera y cierra los componentes del runtime."""
        try:
            self.portfolio_manager.save_to_json()
            for component in (self.data_feed, self.l2_processor):
                cerrar = getattr(component, "close", None)
                if callable(cerrar):
                    resultado = cerrar()
                    if asyncio.iscoroutine(resultado):
                        await resultado
        except Exception:
            logger.exception("❌ Cleanup failed")


async def run_runtime_loop(portfolio_manager: PortfolioManager, order_manager: OrderManager):
    """Construye y ejecuta el bucle de runtime."""
    runtime_loop = HRMRuntimeLoop(portfolio_manager, order_manager)
    await runtime_loop.run()
