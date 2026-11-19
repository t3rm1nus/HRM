# app/componentes.py - Creación de los componentes principales

from system.shutdown_manager import register_component_for_shutdown
from comms.config import config
from core.logging import logger
from core.portfolio_manager import PortfolioManager
from core.state_manager import inject_state_coordinator
from l1_operational.binance_client import BinanceClient
from l1_operational.order_manager import OrderManager
from system.bootstrap import SystemBootstrap, bootstrap_simulated_exchange
from system.market_data_manager import MarketDataManager
from system.state_coordinator import StateCoordinator

from app.constantes import SIMBOLOS


def crear_state_coordinator(modo: str) -> StateCoordinator:
    logger.info("🔧 Initializing StateCoordinator...")
    state_coordinator = StateCoordinator(mode=modo)
    inject_state_coordinator(state_coordinator)
    logger.info("✅ StateCoordinator injected globally")
    return state_coordinator


def inicializar_bootstrap() -> dict:
    """Devuelve los componentes del bootstrap, o {} si este falla."""
    logger.info("🔧 Running SystemBootstrap...")
    try:
        contexto = SystemBootstrap().initialize_system()
        logger.info("✅ SystemBootstrap completed")
        return getattr(contexto, 'components', {})
    except Exception as error:
        logger.error(f"❌ SystemBootstrap failed: {error}")
        logger.warning("⚠️ Using manual fallback initialization")
        return {}


def crear_market_data_manager() -> MarketDataManager:
    logger.info("🔧 Initializing MarketDataManager...")
    market_data_manager = MarketDataManager(symbols=SIMBOLOS, fallback_enabled=True)
    logger.info("✅ MarketDataManager initialized successfully")
    return market_data_manager


async def inicializar_portfolio_async(portfolio_manager: PortfolioManager):
    if not hasattr(portfolio_manager, 'initialize_async'):
        return
    try:
        await portfolio_manager.initialize_async()
        logger.info("✅ PortfolioManager initialized asynchronously")
    except Exception as error:
        logger.warning(f"⚠️ Async initialization failed: {error}")


async def crear_portfolio_manager(market_data_manager: MarketDataManager) -> PortfolioManager:
    cliente_binance = BinanceClient()
    register_component_for_shutdown('binance_client', cliente_binance)
    portfolio_manager = PortfolioManager(
        exchange_client=cliente_binance,
        market_data_manager=market_data_manager,
        mode="simulated",
    )
    await inicializar_portfolio_async(portfolio_manager)
    logger.info("✅ PortfolioManager created with MarketDataManager injected")
    return portfolio_manager


def inyectar_market_data_manager(portfolio_manager, market_data_manager: MarketDataManager):
    if getattr(portfolio_manager, 'market_data_manager', None) is None:
        portfolio_manager.market_data_manager = market_data_manager
        logger.info("✅ MarketDataManager injected into existing PortfolioManager")


async def obtener_portfolio_manager(components: dict, market_data_manager: MarketDataManager):
    portfolio_manager = components.get('portfolio_manager')
    if portfolio_manager is None:
        portfolio_manager = await crear_portfolio_manager(market_data_manager)
        components['portfolio_manager'] = portfolio_manager
    else:
        inyectar_market_data_manager(portfolio_manager, market_data_manager)
        logger.info("✅ PortfolioManager from bootstrap")
    return portfolio_manager


def obtener_o_crear(components: dict, nombre: str, fabrica):
    componente = components.get(nombre)
    if componente is None:
        componente = fabrica()
        components[nombre] = componente
        logger.info(f"✅ {nombre} created manually")
    else:
        logger.info(f"✅ {nombre} from bootstrap")
    return componente


def crear_order_manager(state_coordinator, portfolio_manager, modo: str) -> OrderManager:
    cliente_simulado = bootstrap_simulated_exchange(config)
    return OrderManager(
        state_coordinator,
        portfolio_manager,
        mode=modo,
        simulated_client=cliente_simulado,
    )


def inyectar_portfolio_en_order_manager(order_manager, portfolio_manager):
    try:
        if hasattr(order_manager, 'set_portfolio_manager'):
            order_manager.set_portfolio_manager(portfolio_manager)
        else:
            order_manager.portfolio_manager = portfolio_manager
        logger.info("✅ portfolio_manager injected into order_manager")
    except Exception as error:
        logger.error(f"❌ CRITICAL: Could not inject portfolio_manager into order_manager: {error}")
