# app/sistema.py - Ensamblado de todos los componentes

from system.shutdown_manager import register_component_for_shutdown
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from core.logging import logger
from system.error_recovery_manager import ErrorRecoveryManager

from app.componentes import (
    crear_market_data_manager,
    crear_order_manager,
    crear_state_coordinator,
    inicializar_bootstrap,
    inyectar_portfolio_en_order_manager,
    obtener_o_crear,
    obtener_portfolio_manager,
)
from app.configuracion import determinar_modo
from app.estado_inicial import obtener_estado_inicial, preparar_salida_l3
from app.modelos import Sistema
from app.pipeline import crear_gestores_posicion, crear_trading_pipeline


async def crear_sistema() -> Sistema:
    modo = determinar_modo()
    state_coordinator = crear_state_coordinator(modo)
    components = inicializar_bootstrap()

    market_data_manager = crear_market_data_manager()
    register_component_for_shutdown('market_data_manager', market_data_manager)

    portfolio_manager = await obtener_portfolio_manager(components, market_data_manager)
    order_manager = obtener_o_crear(
        components, 'order_manager',
        lambda: crear_order_manager(state_coordinator, portfolio_manager, modo),
    )
    l2_processor = obtener_o_crear(components, 'l2_processor', L2TacticProcessor)
    register_component_for_shutdown('portfolio_manager', portfolio_manager)
    register_component_for_shutdown('order_manager', order_manager)

    inyectar_portfolio_en_order_manager(order_manager, portfolio_manager)

    estado = obtener_estado_inicial(state_coordinator)
    preparar_salida_l3(estado, components)

    rotador, rebalanceador = crear_gestores_posicion(portfolio_manager)
    trading_pipeline = crear_trading_pipeline(
        portfolio_manager, order_manager, l2_processor,
        rotador, rebalanceador, state_coordinator, modo,
    )

    logger.info("🔧 Initializing ErrorRecoveryManager...")
    return Sistema(
        state_coordinator=state_coordinator,
        portfolio_manager=portfolio_manager,
        order_manager=order_manager,
        l2_processor=l2_processor,
        market_data_manager=market_data_manager,
        trading_pipeline=trading_pipeline,
        error_recovery=ErrorRecoveryManager(),
        position_rotator=rotador,
        estado=estado,
    )
