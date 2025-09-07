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
from .models import TacticalSignal, PositionSize, L2State
from .finrl_integration import FinRLProcessor
# from finrl_final_fix import FinRLPredictor  # REMOVIDO: usar AIModelWrapper
from datetime import datetime

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
        # self.finrl = FinRLProcessor("models/L2/ai_model_data_multiasset.zip")  # REMOVIDO: usar generator.ai_model

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
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la capa Tactic:
        - Genera señales (AI, técnico, riesgo)
        - Combina señales con composer + ensemble
        - Calcula tamaño de posición y convierte a órdenes
        - Guarda señales y órdenes en state['l2']
        - Devuelve dict: {"signals", "orders", "metadata"}
        """
        try:
            logger.info("[L2] Ejecutando capa Tactic...")

            # --- Generar señales ---
            try:
                ai_signals = await self.generator.ai_signals(state) or []
            except Exception as e:
                logger.error(f"[L2] Error en ai_signals: {e}")
                ai_signals = []

            try:
                tech_signals = await self.generator.technical_signals(state) or []
            except Exception as e:
                logger.error(f"[L2] Error en technical_signals: {e}")
                tech_signals = []

            try:
                risk_signals = await self.generator.risk_overlay.generate_risk_signals(
                    state.get('mercado', {}), state.get('portfolio', {})
                ) or []
            except Exception as e:
                logger.error(f"[L2] Error en risk_signals: {e}")
                risk_signals = []

            # --- Combinar señales ---
            all_signals = ai_signals + tech_signals + risk_signals
            composed_signals = self.composer.compose(all_signals) if all_signals else []
            combined = self.ensemble.blend(composed_signals) if composed_signals else []

            # --- Convertir a órdenes ---
            orders = []
            if combined:
                try:
                    # Construir portfolio_state con caja real
                    port = state.get('portfolio', {}) or {}
                    total_cap = float(state.get('total_value', state.get('initial_capital', 0.0)) or 0.0)
                    avail_cap = float(port.get('USDT', 0.0) or 0.0)
                    portfolio_state = {
                        'total_capital': total_cap,
                        'available_capital': avail_cap,
                    }

                    ps = await self.sizer.calculate_position_size(
                        signal=combined,
                        market_features=state.get('mercado', {}),
                        portfolio_state=portfolio_state,
                    )
                    if ps and getattr(ps, "size", 0) > 0:
                        orders.append(self._create_order_dict(ps))
                except Exception as e:
                    logger.error(f"[L2] Error calculando tamaño de posición: {e}")

            # --- Guardar estado L2 ---
            now = datetime.utcnow()
            try:
                l2_state = state.get('l2')
                if l2_state is None:
                    state['l2'] = {"signals": combined, "orders": orders, "last_update": now}
                else:
                    if hasattr(l2_state, "signals"):
                        l2_state.signals = combined
                        l2_state.orders = orders
                        l2_state.last_update = now
                    else:
                        state['l2'] = {"signals": combined, "orders": orders, "last_update": now}
            except Exception as e:
                logger.error(f"[L2] Error sincronizando señales: {e}")
                state['l2'] = {"signals": combined, "orders": orders, "last_update": now}

            logger.info(f"[L2] Preparadas {len(orders)} órdenes para L1, total señales={len(combined)}")

            return {
                "signals": combined,
                "orders": orders,
                "metadata": {
                    "ai_signals": len(ai_signals),
                    "technical_signals": len(tech_signals),
                    "risk_signals": len(risk_signals),
                    "total_signals": len(all_signals)
                }
            }

        except Exception as e:
            logger.error(f"[L2] Error crítico en process(): {e}")
            state['l2'] = {"signals": [], "orders": [], "last_update": datetime.utcnow()}
            return {"signals": [], "orders": [], "metadata": {"error": str(e)}}

    # ------------------------------------------------------------------ #
    async def _generate_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Delega la generación de señales a las fuentes (AI, Technical).
        """
        try:
            # Generar señales con mejor manejo de errores
            ai_signals = await self.generator.ai_signals(state.get('mercado', {}))
            tech_signals = await self.generator.technical_signals(state.get('mercado', {}))
            
            # FIX: Separar mercado y portfolio correctamente
            mercado = state.get('mercado', {})
            portfolio = state.get('portfolio', {})
            
            risk_signals = await self.generator.risk_overlay.generate_risk_signals(mercado, portfolio)
            
            # Combinar todas las señales
            all_signals = ai_signals + tech_signals + risk_signals
            
            # Log detallado para debugging
            logger.info(f"[L2] Señales generadas: AI={len(ai_signals)}, Tech={len(tech_signals)}, Risk={len(risk_signals)}, Total={len(all_signals)}")
            signals = getattr(state.get('l2'), 'signals', []) or []
            logger.info(f"main_processor.py: [L2] Señales generadas: {len(signals)}")
            if not signals:
                logger.info("[L2] No se generaron señales tácticas")
            return all_signals
            
        except Exception as e:
            logger.error(f"[L2] ❌ Error generando señales: {e}")
            return []

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