#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Bootstrap Module

This module handles system initialization, configuration loading,
and component wiring for the HRM system.

🔥 PRIORIDAD 4: Introducir session_is_fresh - no depender de balances
"""

import os
import sys
import asyncio
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from core.state_manager import initialize_state, validate_state_structure
from core.portfolio_manager import PortfolioManager
from core.logging import logger
from core.config import get_config, set_forced_mode
from core.incremental_signal_verifier import get_signal_verifier, start_signal_verification, stop_signal_verification
from core.trading_metrics import get_trading_metrics

from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l2_tactic.risk_controls.manager import RiskControlManager
from l1_operational.order_manager import OrderManager

from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus

from sentiment.sentiment_manager import update_sentiment_texts

from system.system_cleanup import perform_full_cleanup
from config_loader import get_initial_balances, get_capital_usd


# 🔥 PRIORIDAD 4: SESSION IS FRESH - Flag global para nueva sesión
session_is_fresh = True

def is_session_fresh() -> bool:
    """Check if current session is fresh (not dependent on balances)"""
    global session_is_fresh
    return session_is_fresh

def mark_session_used():
    """Mark session as no longer fresh"""
    global session_is_fresh
    session_is_fresh = False


class HRMBootstrap:
    """HRM system bootstrap and initialization."""
    
    def __init__(self):
        self.components = {}
        self.state = None
        self.portfolio_manager = None
        self.order_manager = None
        self.runtime_loop = None
        self._mode = "paper"  # Default mode, will be set in bootstrap_system
    
    def _get_mode_from_bootstrap(self) -> str:
        """Get the mode from the single source of truth (bootstrap)."""
        return self._mode
        
    async def bootstrap_system(self, mode: str = "paper") -> Tuple[PortfolioManager, OrderManager]:
        """
        Bootstrap the entire HRM system.
        
        🔥 PRIORIDAD 4: session_is_fresh determina si es nueva sesión,
        NO los balances. Esto evita dependencia de estado externo.
        """
        # Store mode as single source of truth for this bootstrap instance
        self._mode = mode
        
        logger.info("🚀 Starting HRM System Bootstrap")
        logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()}")
        logger.info(f"🎯 BOOTSTRAP MODE: {mode}")
        
        try:
            # 1. System Cleanup
            await self._perform_system_cleanup()
            
            # 2. Load Configuration
            env_config = await self._load_configuration(mode)
            
            # 3. Initialize State
            self.state = await self._initialize_state(env_config)
            
            # 4. Initialize Core Components
            await self._initialize_core_components(env_config)
            
            # 5. Initialize L1 Components
            await self._initialize_l1_components(env_config)
            
            # 6. Initialize L2 Components
            await self._initialize_l2_components(env_config)
            
            # 7. Initialize L3 Components (if enabled)
            if not APAGAR_L3:
                await self._initialize_l3_components()
            
            # 8. Initialize Portfolio Manager
            self.portfolio_manager = await self._initialize_portfolio_manager(env_config)
            
            # 9. Initialize Order Manager
            self.order_manager = await self._initialize_order_manager()
            
            # 10. Initialize Runtime Loop
            from runtime_loop import HRMRuntimeLoop
            self.runtime_loop = HRMRuntimeLoop(self.portfolio_manager, self.order_manager)
            
            # 11. Start Background Services
            await self._start_background_services()
            
            # 12. Marcar sesión como usada después de bootstrap exitoso
            mark_session_used()
            logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()} (post-bootstrap)")
            
            logger.info("✅ HRM System Bootstrap Complete")
            return self.portfolio_manager, self.order_manager
            
        except Exception as e:
            logger.error(f"❌ Bootstrap failed: {e}", exc_info=True)
            await self._cleanup_on_failure()
            raise
    
    async def _perform_system_cleanup(self):
        """Perform system cleanup before startup."""
        logger.info("🧹 Running system cleanup...")
        
        cleanup_result = perform_full_cleanup(mode="paper")
        
        if not cleanup_result.get("success", False):
            logger.warning("⚠️ Cleanup completed with warnings")
        else:
            logger.info(f"✅ Cleanup completed: {cleanup_result.get('deleted_files', 0)} files, {cleanup_result.get('deleted_dirs', 0)} directories removed")
        
        # El historial PAPER se gestiona por perfil en preparar_sesion.
        # El bootstrap nunca debe borrar operaciones durante un arranque normal.
    
    async def _load_configuration(self, mode: str) -> Dict[str, Any]:
        """Load system configuration."""
        logger.info(f"⚙️ Loading configuration for mode: {mode}")
        
        # Load environment variables
        load_dotenv()
        
        # 🔥 CRITICAL: Set the forced mode in core.config BEFORE getting config
        # This establishes the single source of truth for the mode
        set_forced_mode(mode)
        logger.info(f"🎯 Mode set as single source of truth: {mode}")
        
        # Get environment configuration
        # 🔥 PRIORIDAD 4: Usar mode explícito, no depender de configuración previa
        env_config = get_config(mode)
        
        # Check Binance operating mode
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        logger.info(f"🏦 BINANCE MODE: {binance_mode}")
        
        return env_config
    
    async def _initialize_state(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize system state."""
        logger.info("🧠 Initializing system state...")
        
        # Initialize state with symbols and initial balance
        # 🔥 PRIORIDAD 4: session_is_fresh determina si es nueva sesión
        symbols = env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        initial_balance = get_capital_usd()  # Cargar desde initial_state.json
        
        state = initialize_state(symbols, initial_balance)
        state = validate_state_structure(state)
        
        logger.info(f"✅ State initialized for symbols: {symbols}")
        logger.info(f"   Initial balance: {initial_balance} USDT")
        logger.info(f"   Session fresh: {is_session_fresh()}")
        return state
    
    async def _initialize_core_components(self, env_config: Dict[str, Any]):
        """Initialize core system components."""
        logger.info("🔧 Initializing core components...")
        
        # Initialize trading metrics
        trading_metrics = get_trading_metrics()
        self.components['trading_metrics'] = trading_metrics
        
        # Initialize signal verifier
        signal_verifier = get_signal_verifier()
        self.components['signal_verifier'] = signal_verifier
        
        # Initialize message bus
        message_bus = MessageBus()
        self.components['message_bus'] = message_bus
        
        # Initialize sentiment manager
        sentiment_manager = await self._initialize_sentiment_manager()
        self.components['sentiment_manager'] = sentiment_manager
        
        logger.info("✅ Core components initialized")
    
    async def _initialize_l1_components(self, env_config: Dict[str, Any]):
        """Initialize L1 operational components."""
        logger.info("🔧 Initializing L1 operational components...")
        
        # Determine mode from the single source of truth
        mode = self._get_mode_from_bootstrap()
        
        # Initialize Binance client with injected mode (single source of truth)
        binance_client = BinanceClient(mode=mode)
        self.components['binance_client'] = binance_client
        
        # Initialize SimulatedExchangeClient (para paper trading)
        from l1_operational.simulated_exchange_client import SimulatedExchangeClient
        
        # 🔥 PRIORIDAD 4: Cargar balances desde initial_state.json
        # Usar config_loader para centralizar la configuración
        initial_balances = get_initial_balances()
        
        logger.info(f"📊 Cargando balances desde initial_state.json: {initial_balances}")
        
        # Limpiar cualquier estado previo y crear nuevo
        SimulatedExchangeClient._instance = None
        SimulatedExchangeClient._initialized = False
        
        # Inicializar con balances conocidos
        simulated_client = SimulatedExchangeClient(initial_balances)
        self.components['simulated_client'] = simulated_client
        
        # Logs requeridos
        logger.info("📊 SIM_INIT: Nueva instancia")
        logger.info(f"🔢 SIM_STATE_ID: {id(simulated_client)}")
        logger.info(f"💰 SIM_BALANCES: {simulated_client.get_balances()}")
        logger.info(f"🔥 SESSION_IS_FRESH: {is_session_fresh()}")
        
        # Verificar que los balances no estén vacíos
        if not simulated_client.get_balances() or all(balance == 0 for balance in simulated_client.get_balances().values()):
            logger.critical("🚨 FATAL: SimulatedExchangeClient initialized with empty or zero balances", exc_info=True)
            raise RuntimeError("SimulatedExchangeClient cannot operate with empty or zero balances")
        
        # Initialize RealTimeDataLoader
        loader = RealTimeDataLoader(config)
        self.components['data_loader'] = loader
        
        # Initialize L1 AI Models
        from l1_operational.trend_ai import models as l1_models
        logger.info(f"✅ Loaded L1 AI Models: {list(l1_models.keys())}")
        
        logger.info("✅ L1 components initialized")
    
    async def _initialize_l2_components(self, env_config: Dict[str, Any]):
        """Initialize L2 tactical components."""
        logger.info("🔧 Initializing L2 tactical components...")
        
        # Quick fix: Disable synchronizer in PAPER mode for better performance
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        if binance_mode != "LIVE":
            logger.info("📝 PAPER/TEST MODE: Disabling BTC/ETH synchronizer")
            os.environ['DISABLE_BTC_ETH_SYNC'] = 'true'
        
        # Initialize L2 Config
        l2_config = L2Config()
        self.components['l2_config'] = l2_config
        
        # Initialize L2 Processor
        l2_processor = L2TacticProcessor(l2_config, portfolio_manager=None, apagar_l3=APAGAR_L3)
        self.components['l2_processor'] = l2_processor
        
        # Initialize Risk Manager
        risk_manager = RiskControlManager(l2_config)
        self.components['risk_manager'] = risk_manager
        
        logger.info("✅ L2 components initialized")
    
    async def _initialize_l3_components(self):
        """Initialize L3 strategic components."""
        logger.info("🔧 Initializing L3 strategic components...")
        
        # Import L3 components
        from l3_strategy.regime_classifier import MarketRegimeClassifier
        from l3_strategy.decision_maker import make_decision
        
        # Initialize L3 Classifier
        regime_classifier = MarketRegimeClassifier()
        self.components['regime_classifier'] = regime_classifier
        
        # Initialize L3 Decision Maker
        self.components['l3_decision_maker'] = make_decision
        
        logger.info("✅ L3 components initialized")
    
    async def _initialize_portfolio_manager(self, env_config: Dict[str, Any]) -> PortfolioManager:
        """Initialize Portfolio Manager."""
        logger.info("💼 Initializing Portfolio Manager...")
        
        # Get environment configuration
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        
        # Setup based on binance_mode
        # 🔥 PRIORIDAD 4: Usar modo explícito, no depender de balances previos
        if binance_mode == "LIVE":
            # Live mode: sync mandatory with exchange
            portfolio_mode = "live"
            initial_balance = 0.0  # Will be synced from exchange
        else:
            # Test mode: use simulated balance
            portfolio_mode = "simulated"
            initial_balance = get_capital_usd()  # Cargar desde initial_state.json
            logger.info(f"🧪 TESTING MODE: Using initial balance of {initial_balance} USDT from initial_state.json")
        
        # Initialize Portfolio Manager
        portfolio_manager = PortfolioManager(
            mode=portfolio_mode,
            initial_balance=initial_balance,
            exchange_client=self.components['simulated_client'] if portfolio_mode == "simulated" else self.components['binance_client'],
            symbols=env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"]),
            enable_commissions=env_config.get("ENABLE_COMMISSIONS", True),
            enable_slippage=env_config.get("ENABLE_SLIPPAGE", True)
        )
        
        # Initialize asynchronously to get balances from simulated client
        if portfolio_mode == "simulated":
            await portfolio_manager.initialize_async()
        
        # 🔥 PRIORIDAD 4: NO sync con exchange en modo paper - confiar en balances iniciales
        # Esto evita dependencia de estado externo
        if portfolio_mode == "simulated":
            logger.info("📊 PAPER MODE: Trusting initial balances, no exchange sync needed")
        else:
            # CRITICAL: Synchronize with exchange for production mode
            try:
                logger.info("🔄 Synchronizing with exchange...")
                sync_success = await portfolio_manager.sync_with_exchange()
                
                if sync_success:
                    logger.info("✅ Portfolio synchronized with exchange")
                else:
                    logger.warning("⚠️ Exchange sync failed, loading local state...")
                    loaded = portfolio_manager.load_from_json()
                    if not loaded:
                        logger.info("📄 No saved portfolio found, starting clean")
                    else:
                        logger.info("📂 Local portfolio loaded")
                        
            except Exception as e:
                logger.error(f"❌ Portfolio synchronization failed: {e}")
                logger.warning("⚠️ Continuing with local state")
        
        logger.info("✅ Portfolio Manager initialized")
        return portfolio_manager
    
    async def _initialize_order_manager(self) -> OrderManager:
        """Initialize Order Manager."""
        logger.info("💰 Initializing Order Manager...")
        
        # Determine mode based on configuration
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        if binance_mode == "LIVE":
            mode = "live"
        else:
            mode = "simulated"
            
        order_manager = OrderManager(
            state_manager=self.state,
            portfolio_manager=self.portfolio_manager,
            mode=mode,
            simulated_client=self.components['simulated_client']
        )
        
        logger.info(f"✅ Order Manager initialized (mode: {mode})")
        return order_manager
    
    async def _initialize_sentiment_manager(self):
        """Initialize sentiment analysis manager."""
        logger.info("🧠 Initializing sentiment analysis...")
        
        # Initialize sentiment cache
        sentiment_texts_cache = []
        last_sentiment_update = 0
        
        # Initial sentiment update
        try:
            sentiment_texts_cache = await update_sentiment_texts()
            last_sentiment_update = 0  # Will be set in runtime loop
            logger.info(f"✅ Sentiment analysis initialized with {len(sentiment_texts_cache)} texts")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment initialization failed: {e}")
        
        return {
            'texts_cache': sentiment_texts_cache,
            'last_update': last_sentiment_update
        }
    
    async def _start_background_services(self):
        """Start background services."""
        logger.info("🔄 Starting background services...")
        
        # Start signal verification
        signal_verifier = self.components['signal_verifier']
        await start_signal_verification()
        logger.info("✅ Signal verification started")
    
    async def _cleanup_on_failure(self):
        """Clean up components if bootstrap fails."""
        logger.info("🧹 Cleaning up after bootstrap failure...")
        
        try:
            # Stop signal verification
            await stop_signal_verification()
            
            # Close components
            for component in self.components.values():
                if hasattr(component, "close"):
                    await component.close()
                    
        except Exception as e:
            logger.error(f"❌ Cleanup after failure failed: {e}")
    
    def get_runtime_loop(self):
        """Get the initialized runtime loop."""
        return self.runtime_loop
    
    def get_state(self) -> Dict[str, Any]:
        """Get the initialized system state."""
        return self.state


async def bootstrap_hrm_system(mode: str = "paper"):
    """
    Bootstrap the HRM system and return core components.
    
    🔥 PRIORIDAD 4: session_is_fresh introducido - nueva sesión independiente de balances
    """
    bootstrap = HRMBootstrap()
    portfolio_manager, order_manager = await bootstrap.bootstrap_system(mode)
    runtime_loop = bootstrap.get_runtime_loop()
    
    return portfolio_manager, order_manager, runtime_loop


def reset_session():
    """
    Reset session to fresh state.
    
    🔥 PRIORIDAD 4: Función para marcar sesión como nueva
    """
    global session_is_fresh
    session_is_fresh = True
    logger.info("🔄 Session reset to fresh state")
    return True


# =============================================================================
# SHUTDOWN LIMPIO GLOBAL
# =============================================================================

# Variable global para almacenar referencias a componentes para shutdown
_shutdown_components = {
    'market_data_manager': None,
    'realtime_loader': None,
    'exchange_client': None,
    'binance_client': None,
    'data_feed': None,
    'portfolio_manager': None,
    'order_manager': None,
    'order_flow_capture': None,
}


def register_component_for_shutdown(name: str, component: Any):
    """
    Registra un componente para ser cerrado durante el shutdown global.
    
    Args:
        name: Nombre identificador del componente
        component: Instancia del componente a registrar
    """
    global _shutdown_components
    _shutdown_components[name] = component
    logger.debug(f"🔧 Componente '{name}' registrado para shutdown")


def unregister_component_for_shutdown(name: str):
    """
    Desregistra un componente del shutdown global.
    
    Args:
        name: Nombre identificador del componente
    """
    global _shutdown_components
    if name in _shutdown_components:
        _shutdown_components[name] = None
        logger.debug(f"🔧 Componente '{name}' desregistrado del shutdown")


async def shutdown():
    """
    Shutdown limpio global - cierra todas las conexiones y sesiones.
    
    Esta función DEBE ser llamada al finalizar el sistema para asegurar:
    - Todas las sesiones aiohttp se cierran correctamente
    - No quedan warnings asyncio al terminar
    - Los recursos se liberan en el orden correcto
    
    Orden de cierre:
    1. MarketDataManager (cierra RealTimeLoader y DataFeed internamente)
    2. RealTimeLoader (WebSocket connections)
    3. ExchangeClient / BinanceClient (sesiones aiohttp)
    4. DataFeed
    5. PortfolioManager (guarda estado)
    6. OrderManager
    7. Cancelar todas las tareas pendientes de asyncio
    8. Cerrar sesiones aiohttp huérfanas
    """
    global _shutdown_components
    
    logger.info("🧹 Iniciando shutdown limpio global...")
    
    # 1. Cerrar MarketDataManager
    if _shutdown_components.get('market_data_manager') is not None:
        try:
            await _shutdown_components['market_data_manager'].close()
            logger.info("✅ MarketDataManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando MarketDataManager: {e}")
    
    # 2. Cerrar RealTimeLoader directamente si existe
    if _shutdown_components.get('realtime_loader') is not None:
        try:
            await _shutdown_components['realtime_loader'].close()
            logger.info("✅ RealTimeLoader cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando RealTimeLoader: {e}")
    
    # 3. Cerrar ExchangeClient / BinanceClient
    for client_name in ['exchange_client', 'binance_client']:
        if _shutdown_components.get(client_name) is not None:
            try:
                client = _shutdown_components[client_name]
                if hasattr(client, 'close'):
                    await client.close()
                elif hasattr(client, 'close_connection'):
                    await client.close_connection()
                logger.info(f"✅ {client_name} cerrado")
            except Exception as e:
                logger.warning(f"⚠️ Error cerrando {client_name}: {e}")
    
    # 4. Cerrar DataFeed
    if _shutdown_components.get('data_feed') is not None:
        try:
            await _shutdown_components['data_feed'].close()
            logger.info("✅ DataFeed cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando DataFeed: {e}")
    
    # 5. Guardar estado del PortfolioManager
    if _shutdown_components.get('portfolio_manager') is not None:
        try:
            pm = _shutdown_components['portfolio_manager']
            if hasattr(pm, 'save_to_json'):
                pm.save_to_json()
                logger.info("✅ PortfolioManager estado guardado")
            if hasattr(pm, 'close'):
                await pm.close()
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando PortfolioManager: {e}")
    
    # 6. Cerrar OrderManager
    if _shutdown_components.get('order_manager') is not None:
        try:
            om = _shutdown_components['order_manager']
            if hasattr(om, 'close'):
                await om.close()
            logger.info("✅ OrderManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando OrderManager: {e}")
    
    # 6.5 Cerrar OrderFlowCapture
    if _shutdown_components.get('order_flow_capture') is not None:
        try:
            ofc = _shutdown_components['order_flow_capture']
            if hasattr(ofc, 'close'):
                await ofc.close()
            logger.info("✅ OrderFlowCapture cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando OrderFlowCapture: {e}")
    
    # 7. Cerrar todas las sesiones aiohttp pendientes
    await _close_aiohttp_sessions()
    
    # 8. Cancelar todas las tareas pendientes de asyncio (excepto la actual)
    await _cancel_pending_tasks()
    
    logger.info("🧹 Shutdown limpio global completado")


async def _close_aiohttp_sessions():
    """
    Cierra todas las sesiones aiohttp abiertas.
    """
    try:
        import aiohttp
        
        # Buscar y cerrar todas las sesiones aiohttp
        closed_count = 0
        for obj in list(globals().values()) + list(_shutdown_components.values()):
            if obj is None:
                continue
            
            # Buscar atributos que sean sesiones aiohttp
            if hasattr(obj, 'session') and hasattr(obj.session, 'closed'):
                try:
                    if not obj.session.closed:
                        await obj.session.close()
                        closed_count += 1
                except Exception:
                    pass
            
            # Buscar atributos que sean ClientSession directamente
            if isinstance(obj, aiohttp.ClientSession):
                try:
                    if not obj.closed:
                        await obj.close()
                        closed_count += 1
                except Exception:
                    pass
        
        if closed_count > 0:
            logger.info(f"✅ {closed_count} sesiones aiohttp cerradas")
    except Exception as e:
        logger.warning(f"⚠️ Error cerrando sesiones aiohttp: {e}")


async def _cancel_pending_tasks():
    """
    Cancela todas las tareas pendientes de asyncio excepto la tarea actual.
    Esto evita warnings de "task was destroyed but it is pending".
    """
    try:
        import asyncio
        
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        
        # Obtener todas las tareas pendientes excepto la actual
        pending_tasks = [
            task for task in asyncio.all_tasks(loop) 
            if task is not current_task and not task.done()
        ]
        
        if pending_tasks:
            logger.info(f"🛑 Cancelando {len(pending_tasks)} tareas pendientes...")
            
            # Cancelar todas las tareas
            for task in pending_tasks:
                task.cancel()
            
            # Esperar a que todas las tareas se completen (o se cancelen)
            results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            
            cancelled_count = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
            logger.info(f"✅ {cancelled_count} tareas canceladas, {len(results) - cancelled_count} completadas")
    except Exception as e:
        logger.warning(f"⚠️ Error cancelando tareas pendientes: {e}")


async def shutdown_with_signal_handling():
    """
    Versión del shutdown que maneja señales del sistema.
    Usar esta función cuando se quiera un shutdown completo con manejo de señales.
    """
    logger.info("🛑 Ejecutando shutdown con manejo de señales...")
    await shutdown()


# Manejadores de señales para shutdown limpio
_shutdown_event = asyncio.Event()

def _signal_handler(signum, frame):
    """
    Manejador de señales que activa el evento de shutdown.
    """
    import signal
    sig_name = signal.Signals(signum).name
    logger.info(f"🛑 Señal {sig_name} recibida, iniciando shutdown...")
    _shutdown_event.set()


def setup_signal_handlers():
    """
    Configura los manejadores de señales para SIGINT y SIGTERM.
    Llama a esta función al inicio de la aplicación para habilitar shutdown limpio.
    """
    import signal
    
    # Configurar manejadores para SIGINT (Ctrl+C) y SIGTERM
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    logger.info("✅ Manejadores de señales configurados (SIGINT, SIGTERM)")


async def wait_for_shutdown_signal():
    """
    Espera hasta que se reciba una señal de shutdown.
    Útil para mantener el programa corriendo hasta que se presione Ctrl+C.
    """
    await _shutdown_event.wait()
    logger.info("🛑 Señal de shutdown recibida, procediendo...")
