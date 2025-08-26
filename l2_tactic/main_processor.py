"""
L2MainProcessor
Capa Tactic – genera señales y prepara órdenes para L1.
Ahora integra l2_tactic.ensemble para combinar señales.
"""

import asyncio
from typing import Dict, Any, List

from core.logging import logger
from .signal_composer import SignalComposer
from .signal_generator import L2TacticProcessor
from .position_sizer import PositionSizerManager
from .technical.multi_timeframe import resample_and_consensus
from .ensemble import VotingEnsemble, BlenderEnsemble
from .metrics import L2Metrics


class L2MainProcessor:
    """
    Responsabilidades:
      1. Recibir state (portfolio + mercado)
      2. Generar señales (AI + técnico)
      3. Combinar señales con ensemble
      4. Transformar señales en órdenes con SL
      5. Devolver órdenes listas para L1
    """

    def __init__(self, config, bus=None) -> None:
        self.config  = config
        self.bus     = bus
        self.generator = L2TacticProcessor(config)
        self.composer  = SignalComposer(config)
        self.sizer     = PositionSizerManager(config)
        self.metrics   = L2Metrics()

        # --- ENSAMBLE ---
        mode = getattr(config, "ensemble_mode", "blender")
        self.use_voting = (mode == "voting")

        if self.use_voting:
            method   = getattr(config, "voting_method",   "hard")
            threshold = getattr(config, "voting_threshold", 0.5)
            self.ensemble = VotingEnsemble(method=method, threshold=threshold)
            logger.info("[L2MainProcessor] Usando VotingEnsemble")
        else:
            weights = getattr(
                config,
                "blender_weights",
                {"ai": 0.6, "technical": 0.3, "risk": 0.1}
            )
            self.ensemble = BlenderEnsemble(weights=weights)
            logger.info("[L2MainProcessor] Usando BlenderEnsemble")

    # ------------------------------------------------------------------ #
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa un ciclo completo de L2.
        Devuelve un state actualizado con la clave 'ordenes'.
        """
        logger.info("[L2] Ejecutando capa Tactic...")

        # 1) Generar señales crudas
        raw_signals = await self._generate_signals(state)
        logger.debug(f"[L2] Señales crudas: {len(raw_signals)}")

        # 2) Combinar con ensemble
        combined = self._combine(raw_signals)
        if not combined:
            logger.warning("[L2] Sin señal tras ensemble")
            state["ordenes"] = []
            return state

        # 3) Convertir señal a órdenes con SL
        orders = self._signal_to_orders(combined, state)
        logger.info(f"[L2] Prepared {len(orders)} orders for L1 (all with SL)")
        state["ordenes"] = orders
        return state

    # ------------------------------------------------------------------ #
    async def _generate_signals(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Llama al generador para que produzca señales.
        Cada señal debe llevar 'symbol', 'side', 'prob', 'source'.
        """
        signals = []

        # -- AI (PPO) --
        ai_signals = await self.generator.ai_signals(state)
        for sig in ai_signals:
            sig["source"] = "ai"
            signals.append(sig)

        # -- Técnico --
        tech_signals = await self.generator.technical_signals(state)
        for sig in tech_signals:
            sig["source"] = "technical"
            signals.append(sig)

        # -- Risk overlay --
        risk_signals = await self.generator.risk_overlay(state)
        for sig in risk_signals:
            sig["source"] = "risk"
            signals.append(sig)

        return signals

    # ------------------------------------------------------------------ #
    def _combine(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Delega en el ensamble elegido.
        """
        if self.use_voting:
            return self.ensemble.vote(signals)
        return self.ensemble.blend(signals)

    # ------------------------------------------------------------------ #
    def _signal_to_orders(self,
                          signal: Dict[str, Any],
                          state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convierte la señal consensuada en lista de órdenes con SL.
        """
        symbol = signal["symbol"]
        side = signal["side"]
        size = self.sizer.calculate_position(signal, state)

        base_order = {
            "symbol": symbol,
            "type": "market",
            "side": side,
            "amount": size,
            "params": {"sl": self._default_sl(symbol, state)}
        }
        return [base_order]

    # ------------------------------------------------------------------ #
    def _default_sl(self, symbol: str, state: Dict[str, Any]) -> float:
        """
        SL fijo o dinámico (ejemplo simple: 2 %).
        """
        return state["mercado"][symbol]["close"] * 0.98