# app/pipeline.py - Gestores de posición y pipeline de trading

from core.incremental_signal_verifier import get_signal_verifier
from core.logging import logger
from core.position_rotator import PositionRotator, AutoRebalancer
from system.trading_pipeline_manager import TradingPipelineManager

from app.configuracion import leer_paper_mode


def crear_gestores_posicion(portfolio_manager):
    logger.info("🔧 Initializing position management...")
    paper_mode = leer_paper_mode(por_defecto=False)
    rotador = PositionRotator(portfolio_manager=portfolio_manager, paper_mode=paper_mode)
    rebalanceador = AutoRebalancer(portfolio_manager=portfolio_manager, paper_mode=paper_mode)
    logger.info("✅ Position managers ready")
    return rotador, rebalanceador


def crear_trading_pipeline(
    portfolio_manager, order_manager, l2_processor,
    rotador, rebalanceador, state_coordinator, modo: str,
) -> TradingPipelineManager:
    pipeline = TradingPipelineManager(
        portfolio_manager=portfolio_manager,
        order_manager=order_manager,
        l2_processor=l2_processor,
        position_rotator=rotador,
        auto_rebalancer=rebalanceador,
        signal_verifier=get_signal_verifier(),
        state_coordinator=state_coordinator,
        mode=modo,
    )
    logger.info(f"✅ TradingPipelineManager ready (mode: {modo})")
    return pipeline
