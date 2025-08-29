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
from .models import TacticalSignal, PositionSize

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
        self.config = config
        self.bus = bus
        self.generator = L2TacticProcessor(config)
        self.composer = SignalComposer(config)
        self.sizer = PositionSizerManager(config)
        self.metrics = L2Metrics()

        # --- ENSAMBLE ---
        mode = getattr(config, "ensemble_mode", "blender")
        self.use_voting = (mode == "voting")

        if self.use_voting:
            self.ensemble = VotingEnsemble(
                weights=getattr(config, "voting_weights", {}),
                default=getattr(config, "voting_default_weight", 0.0),
                threshold=getattr(config, "voting_threshold", 0.5)
            )
        else:
            self.ensemble = BlenderEnsemble(
                weights=getattr(config, "blender_weights", {}),
                default=getattr(config, "blender_default_weight", 0.0)
            )

    # ------------------------------------------------------------------ #
    async def process(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info("[L2] Ejecutando capa Tactic...")

        # Generar señales de AI, técnicas y de riesgo de forma independiente
        ai_signals = await self.generator.ai_signals(state)
        tech_signals = await self.generator.technical_signals(state)
        risk_signals = await self.generator.risk_overlay(state['mercado'], state['portfolio'])

        # Combinar señales
        all_signals = [s for s in ai_signals] + [s for s in tech_signals] + [s for s in risk_signals]

        # Componer señales
        composed_signals = self.composer.compose(all_signals)

        # Ensamblar señales
        combined = self.ensemble.blend(composed_signals)

        # Convertir a órdenes
        orders = []
        if combined:
            ps = await self.sizer.calculate_position_size(
                signal=combined,
                portfolio_state=state['portfolio'],
                market_features=state['mercado']
            )

            if ps and ps.size > 0:
                orders.append(self._create_order_dict(ps))

        logger.info(f"[L2] Preparadas {len(orders)} órdenes para L1 (todas con SL)")
        return orders

    # ------------------------------------------------------------------ #
    async def _generate_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Delega la generación de señales a las fuentes (AI, Technical).
        """
        # Generar señales de AI, técnicas y de riesgo de forma independiente.
        ai_signals = await self.generator.ai_signals(state)
        tech_signals = await self.generator.technical_signals(state)
        
        # El ERROR está aquí. risk_overlay necesita mercado y portafolio por separado.
        risk_signals = await self.generator.risk_overlay(state['mercado'], state['portfolio'])
        
        all_signals = [s for s in ai_signals] + [s for s in tech_signals] + [s for s in risk_signals]
        
        return all_signals

    # ------------------------------------------------------------------ #
    def _create_order_dict(self, ps: PositionSize) -> Dict[str, Any]:
        """
        Crea el diccionario de orden a partir de un objeto PositionSize.
        """
        return {
            "symbol": ps.symbol,
            "type": "market",
            "side": ps.side,
            "amount": ps.size,
            "params": {
                "sl": ps.stop_loss,
                "tp": ps.take_profit,
                "notional": ps.notional,
                "leverage": ps.leverage,
                "risk_amount": ps.risk_amount
            }
        }