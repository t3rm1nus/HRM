# app/autoaprendizaje.py - Integración del sistema de autoaprendizaje

import traceback

from comms.config import config
from core.logging import logger
from core.trading_metrics import get_trading_metrics

from app.modelos import Sistema


async def verificar_anti_overfitting(auto_learning):
    estado = await auto_learning.get_learning_status()
    if estado.get('learning_system', {}).get('anti_overfitting_active', False):
        logger.info("🛡️ Anti-overfitting protection: ACTIVE")
    else:
        logger.warning("⚠️ Anti-overfitting protection not confirmed")


async def integrar_auto_aprendizaje(sistema: Sistema):
    logger.info("🤖 Integrating Auto-Learning System...")
    try:
        from auto_learning.integration import AutoLearningIntegration
        from system.auto_learning_bridge import AutoLearningBridge

        auto_learning = AutoLearningIntegration()
        exito = await auto_learning.initialize_integration(
            state_manager=sistema.state_coordinator,
            order_manager=sistema.order_manager,
            portfolio_manager=sistema.portfolio_manager,
            l2_processor=sistema.l2_processor,
            trading_metrics=get_trading_metrics(),
            config=config,
        )
        if not exito:
            logger.warning("⚠️ Auto-Learning System initialization returned False")
            return

        logger.info("✅ Auto-Learning System integrated successfully")
        sistema.trading_pipeline.auto_learning_bridge = AutoLearningBridge(auto_learning)
        logger.info("✅ Auto-Learning Bridge conectado al Trading Pipeline")
        await verificar_anti_overfitting(auto_learning)
    except Exception as error:
        logger.error(f"❌ Auto-Learning integration failed: {error}")
        logger.error(traceback.format_exc())
