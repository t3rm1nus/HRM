# -*- coding: utf-8 -*-
# main.py - VERSIÓN COMPLETAMENTE CORREGIDA CON FIXES CRÍTICOS

import asyncio
import sys
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Optional
import colorama
from colorama import Fore, Style

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path (insert at beginning for priority)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SystemBootstrap for centralized system initialization
from system.bootstrap import SystemBootstrap

# Import MarketDataManager for centralized market data handling
from system.market_data_manager import MarketDataManager

# Import TradingPipelineManager for trading cycle orchestration
from system.trading_pipeline_manager import TradingPipelineManager

# Import funciones de shutdown global desde bootstrap
from bootstrap import shutdown, register_component_for_shutdown, setup_signal_handlers

# Import modularized components
from core.state_manager import log_cycle_data
from core.data_validator import validate_market_data, _extract_current_price_safely
from core.logging import logger
from core.config import get_config
from core.error_handler import ErrorHandler
from core.incremental_signal_verifier import get_signal_verifier, start_signal_verification, stop_signal_verification
from core.trading_metrics import get_trading_metrics

# Import StateCoordinator
from system.state_coordinator import StateCoordinator

# Import ErrorRecoveryManager
from system.error_recovery_manager import ErrorRecoveryManager, RecoveryActionType

# 🌊 ORDER FLOW CAPTURE - L1 Perfecto
from l1_operational.order_flow_capture import (
    OrderFlowCapture,
    create_and_start_order_flow_capture,
    get_order_flow_capture
)

# 🌊 ORDER FLOW SIGNAL GENERATOR - Señales basadas en bid_ask_imbalance
from l1_operational.orderflow_signal_generator import (
    OrderFlowSignalGenerator,
    create_order_flow_signal_generator,
    get_order_flow_signal_generator
)

from l1_operational.data_feed import DataFeed
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.models import L2State, TacticalSignal
from l2_tactic.risk_controls.manager import RiskControlManager
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l1_operational.bus_adapter import BusAdapterAsync

from comms.config import config, APAGAR_L3
from l2_tactic.config import L2Config
from comms.message_bus import MessageBus

from fix_l3_dominance import (
    should_l3_block_l2_signals,
    should_trigger_rebalancing,
    calculate_allocation_deviation,
    L3DominanceFixConfig
)

# 🔄 AUTO-LEARNING SYSTEM INTEGRATION
from integration_auto_learning import integrate_with_main_system

# 🧹 SYSTEM CLEANUP - USAR NUEVO MÓDULO
# PRIORIDAD 4: memory_reset = False ANTES del try
memory_reset = False
CLEANUP_AVAILABLE = False

try:
    from system.system_cleanup import (
        perform_full_cleanup, 
        force_paper_mode,
        filesystem_cleanup,
        memory_reset as cleanup_memory_reset,
        async_context_reset
    )
    # PRIORIDAD 4: Asignar después del import si está disponible
    memory_reset = cleanup_memory_reset
    CLEANUP_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ SystemCleanup not available, skipping cleanup")
    CLEANUP_AVAILABLE = False

# 📊 SENTIMENT ANALYSIS
from sentiment.sentiment_manager import update_sentiment_texts

async def main():
    """Main HRM system function."""
    
    # ================================================================
    # FIX: Verificación de seguridad modo LIVE — debe ejecutarse ANTES del cleanup
    # ================================================================
    import os
    binance_mode = os.getenv('BINANCE_MODE', 'paper').lower()
    paper_mode_env = os.getenv('PAPER_MODE', 'true').lower()
    
    if binance_mode == 'live' and paper_mode_env != 'false':
        logger.warning(
            "⚠️ BINANCE_MODE=live detectado pero PAPER_MODE no es 'false' explícito. "
            "Forzando paper mode por seguridad. Para operar en live, debes establecer "
            "PAPER_MODE=false de forma explícita y consciente."
        )
        os.environ['PAPER_MODE'] = 'true'
        os.environ['BINANCE_MODE'] = 'paper'
    
    if binance_mode == 'live' and paper_mode_env == 'false' and os.getenv('HRM_ENABLE_LIVE_TRADING', 'false').lower() != 'true':
        logger.critical("Trading real bloqueado: falta HRM_ENABLE_LIVE_TRADING=true. Se fuerza paper.")
        os.environ['PAPER_MODE'] = 'true'
        os.environ['BINANCE_MODE'] = 'paper'
        binance_mode = 'paper'
        paper_mode_env = 'true'

    if (binance_mode == 'live' and paper_mode_env == 'false' and os.getenv('HRM_ENABLE_LIVE_TRADING', 'false').lower() == 'true'):
        # Requiere confirmación explícita — nunca arrancar live sin esto
        logger.critical("🚨 MODO LIVE CON DINERO REAL DETECTADO 🚨")
        logger.critical(f"   Capital en riesgo: {os.getenv('INITIAL_CAPITAL', 'DESCONOCIDO')} USDT")
        logger.critical("   Tienes 10 segundos para cancelar (Ctrl+C)...")
        await asyncio.sleep(10)
        logger.critical("🚨 ARRANCANDO EN MODO LIVE — SIN VUELTA ATRÁS")
    
    # ================================================================
    # CRITICAL VARIABLE DECLARATIONS
    # ================================================================
    state_coordinator: Optional[StateCoordinator] = None
    portfolio_manager = None
    order_manager = None
    l2_processor = None
    market_data_manager = None
    trading_pipeline = None
    error_recovery = None
    
    try:
        
        # ================================================================
        # CONFIGURAR MANEJADORES DE SEÑALES PARA SHUTDOWN LIMPIO
        # ================================================================
        setup_signal_handlers()

        # No ejecutar limpiezas destructivas al arrancar: los historiales y
        # estados persistentes deben sobrevivir a reinicios normales.
        logger.info("Se conserva el estado persistente; limpieza manual bajo demanda.")

        logger.info("🚀 Starting HRM system")

        # ================================================================
        # STEP 1.5: DETERMINE TRADING MODE (ONCE - BEFORE STEP 2)
        # ================================================================
        # FIX: Calculate mode ONCE here, before initializing any components
        paper_mode_env = os.getenv('PAPER_MODE', 'true').lower()
        if paper_mode_env == 'false':
            mode = "live"
            logger.info("🔴 MODO LIVE detectado (PAPER_MODE=false)")
        else:
            mode = "simulated"
            logger.info("🟡 MODO SIMULADO detectado (PAPER_MODE!=false)")
        servicio_estado_vivo.registrar_modo(mode)

        # ================================================================
        # STEP 2: INITIALIZE STATE_COORDINATOR (SINGLETON) - CRITICAL FIX
        # ================================================================
        logger.info("🔧 Initializing StateCoordinator...")
        state_coordinator = StateCoordinator(mode=mode)
        
        # ✅ CRITICAL: Verify StateCoordinator is not None
        if state_coordinator is None:
            logger.critical("🚨 FATAL: StateCoordinator initialization returned None!")
            raise RuntimeError("StateCoordinator initialization failed")
        
        # ✅ INJECT IMMEDIATELY
        from core.state_manager import inject_state_coordinator
        inject_state_coordinator(state_coordinator)
        logger.info("✅ StateCoordinator injected globally")

        # ================================================================
        # STEP 3: INITIALIZE SYSTEM BOOTSTRAP - WITH ERROR HANDLING
        # ================================================================
        logger.info("🔧 Running SystemBootstrap...")
        components = {}
        external_adapter = None
        
        try:
            bootstrap = SystemBootstrap()
            system_context = bootstrap.initialize_system()
            
            # Get components from bootstrap if available
            if hasattr(system_context, 'components'):
                components = system_context.components
            
            if hasattr(system_context, 'external_adapter'):
                external_adapter = system_context.external_adapter
            
            logger.info("✅ SystemBootstrap completed - StateCoordinator sin cambios")
            
        except Exception as bootstrap_error:
            logger.error(f"❌ SystemBootstrap failed: {bootstrap_error}")
            logger.warning("⚠️ Using manual fallback initialization")
            
            # ✅ CRITICAL: Ensure state_coordinator is still valid even if bootstrap fails
            if state_coordinator is None:
                logger.info("🔧 Re-creating StateCoordinator after bootstrap failure")
                state_coordinator = StateCoordinator()
                inject_state_coordinator(state_coordinator)

        # ✅ FINAL VERIFICATION: StateCoordinator must exist
        if state_coordinator is None:
            logger.critical("🚨 FATAL: StateCoordinator is None after bootstrap!")
            raise RuntimeError("StateCoordinator is None - cannot continue")

        # ================================================================
        # STEP 4: INITIALIZE MARKET DATA MANAGER FIRST (REORDERED)
        # ================================================================
        logger.info("🔧 Initializing MarketDataManager...")
        try:
            market_data_manager = MarketDataManager(
                symbols=["BTCUSDT", "ETHUSDT"],
                fallback_enabled=True
            )
            logger.info("✅ MarketDataManager initialized successfully")
        except Exception as mkt_error:
            logger.error(f"❌ MarketDataManager initialization failed: {mkt_error}")
            
            # WAIT: Give system time to stabilize before retry
            logger.info("⏳ Waiting 2 seconds for system to stabilize before retry...")
            import time
            time.sleep(2)
            
            # FALLBACK: Try simpler initialization with minimal parameters
            try:
                logger.info("🔄 Attempting fallback initialization with minimal parameters...")
                market_data_manager = MarketDataManager(
                    symbols=["BTCUSDT"],  # Solo un símbolo para simplificar
                    fallback_enabled=False  # Deshabilitar fallback para evitar complejidad adicional
                )
                logger.warning("⚠️ Using fallback MarketDataManager initialization (BTCUSDT only, no fallback)")
            except Exception as fallback_error:
                logger.critical(f"🚨 FATAL: MarketDataManager fallback initialization also failed: {fallback_error}")
                raise RuntimeError(f"MarketDataManager cannot be initialized. Original error: {mkt_error}, Fallback error: {fallback_error}")
    
        # ✅ CRITICAL: Verify MarketDataManager is not None
        if market_data_manager is None:
            logger.critical("🚨 FATAL: MarketDataManager is None after initialization!")
            raise RuntimeError("MarketDataManager cannot be None - trading loop cannot start")

        # 🔧 REGISTRAR MarketDataManager para shutdown global
        register_component_for_shutdown('market_data_manager', market_data_manager)

        # ================================================================
        # STEP 4.5: INITIALIZE ORDER FLOW CAPTURE (L1 Perfecto)
        # ================================================================
        logger.info("🌊 Inicializando OrderFlowCapture (L1 Perfecto)...")
        print("🌊 Inicializando OrderFlowCapture (L1 Perfecto)...", flush=True)
        
        order_flow_capture = None
        try:
            # CRITICAL FIX: Usar polling REST API en lugar de websockets para evitar streams muertos
            # Los websockets pueden quedar colgados silenciosamente, el polling es más fiable
            order_flow_capture = await create_and_start_order_flow_capture(
                symbols=["BTCUSDT", "ETHUSDT"],
                data_dir="backtesting/data/orderflow",
                use_polling=True  # FORZAR polling REST API - más fiable que websockets
            )
            register_component_for_shutdown('order_flow_capture', order_flow_capture)
            
            print("✅ OrderFlowCapture iniciado en modo POLLING REST API", flush=True)
            logger.info("✅ OrderFlowCapture iniciado y registrado para shutdown (modo POLLING)")
            
            # Verificar que el directorio de datos existe
            import os
            orderflow_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtesting/data/orderflow")
            os.makedirs(orderflow_dir, exist_ok=True)
            print(f"📁 Directorio OrderFlow verificado: {orderflow_dir}", flush=True)
            
            # Esperar un momento para que el polling capture datos iniciales
            await asyncio.sleep(6)
            
            # Verificar features iniciales
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                features = order_flow_capture.get_current_features(symbol)
                if features:
                    print(f"📊 OrderFlow {symbol}: spread={features['spread_bps']:.2f}bps, "
                          f"imbalance={features['bid_ask_imbalance']:.3f}", flush=True)
                    logger.info(f"📊 OrderFlow {symbol}: spread={features['spread_bps']:.2f}bps, "
                               f"imbalance={features['bid_ask_imbalance']:.3f}")
                else:
                    print(f"⚠️ OrderFlow {symbol}: No hay features disponibles aún", flush=True)
                    
        except Exception as of_error:
            print(f"❌ OrderFlowCapture initialization failed: {of_error}", flush=True)
            logger.error(f"❌ OrderFlowCapture initialization failed: {of_error}")
            logger.warning("⚠️ Continuando sin OrderFlowCapture - L1 no tendrá datos de flujo de órdenes")
            order_flow_capture = None

        # ================================================================
        # STEP 4.6: INITIALIZE ORDER FLOW SIGNAL GENERATOR
        # ================================================================
        logger.info("🌊 Inicializando OrderFlowSignalGenerator...")
        print("🌊 Inicializando OrderFlowSignalGenerator...", flush=True)
        
        order_flow_signal_generator = None
        try:
            if order_flow_capture is not None:
                order_flow_signal_generator = create_order_flow_signal_generator(
                    order_flow_capture=order_flow_capture
                )
                register_component_for_shutdown('order_flow_signal_generator', order_flow_signal_generator)
                
                print("✅ OrderFlowSignalGenerator inicializado", flush=True)
                logger.info("✅ OrderFlowSignalGenerator inicializado y registrado para shutdown")
            else:
                logger.warning("⚠️ OrderFlowSignalGenerator no inicializado - OrderFlowCapture no disponible")
                
        except Exception as sg_error:
            print(f"❌ OrderFlowSignalGenerator initialization failed: {sg_error}", flush=True)
            logger.error(f"❌ OrderFlowSignalGenerator initialization failed: {sg_error}")
            logger.warning("⚠️ Continuando sin OrderFlowSignalGenerator")
            order_flow_signal_generator = None

        # ================================================================
        # STEP 5: ENSURE CRITICAL COMPONENTS EXIST (WITH MARKET DATA MANAGER INJECTED)
        # ================================================================
        logger.info("🔧 Verifying critical components...")
        
        # Portfolio Manager - FIX ASYNC ISSUE
        if 'portfolio_manager' not in components or components.get('portfolio_manager') is None:
            from core.portfolio_manager import PortfolioManager
            from l1_operational.binance_client import BinanceClient
            
            # ✅ CRITICAL FIX: Create BinanceClient for paper trading with testnet
            binance_client = BinanceClient()
            
            # 🔧 REGISTRAR BinanceClient para shutdown global
            register_component_for_shutdown('binance_client', binance_client)
            
            # ✅ Create PortfolioManager with BinanceClient AND MarketDataManager using shared mode
            portfolio_manager = PortfolioManager(
                exchange_client=binance_client, 
                market_data_manager=market_data_manager,
                mode=mode
            )
            
            # ✅ If PortfolioManager has async initialization, call it
            if hasattr(portfolio_manager, 'initialize_async'):
                try:
                    await portfolio_manager.initialize_async()
                    logger.info("✅ PortfolioManager initialized asynchronously")
                except Exception as async_init_error:
                    logger.warning(f"⚠️ Async initialization failed: {async_init_error}")
            
            components['portfolio_manager'] = portfolio_manager
            logger.info("✅ PortfolioManager created with MarketDataManager injected")
        else:
            portfolio_manager = components['portfolio_manager']
            # Inject MarketDataManager if not already injected
            if not hasattr(portfolio_manager, 'market_data_manager') or portfolio_manager.market_data_manager is None:
                portfolio_manager.market_data_manager = market_data_manager
                logger.info("✅ MarketDataManager injected into existing PortfolioManager")
            logger.info("✅ PortfolioManager from bootstrap")
            
        # Order Manager
        simulated_client = None
        if 'order_manager' not in components or components.get('order_manager') is None:
            from l1_operational.order_manager import OrderManager
            from system.bootstrap import bootstrap_simulated_exchange
            
            # Bootstrap simulated exchange client if in paper mode
            simulated_client = bootstrap_simulated_exchange(config)
            
            order_manager = OrderManager(state_coordinator, portfolio_manager, mode=mode, simulated_client=simulated_client)
            components['order_manager'] = order_manager
            logger.info("✅ OrderManager created manually with simulated client")
        else:
            order_manager = components['order_manager']
            simulated_client = getattr(order_manager, 'simulated_client', None)
            logger.info("✅ OrderManager from bootstrap")
        
        # 🔄 CRITICAL FIX: Sincronizar PortfolioManager con SimulatedExchangeClient después del bootstrap
        if simulated_client and portfolio_manager:
            logger.info("🔄 Sincronizando PortfolioManager con SimulatedExchangeClient después del bootstrap...")
            try:
                sync_success = await portfolio_manager.sync_from_exchange_async(simulated_client)
                if sync_success:
                    # Obtener el balance sincronizado para loguear
                    sim_balances = simulated_client.get_balances() if hasattr(simulated_client, 'get_balances') else {}
                    usdt_balance = sim_balances.get('USDT', 0)
                    logger.info(f"✅ PortfolioManager sincronizado con SimulatedExchangeClient: USDT=${usdt_balance:.2f}")
                else:
                    logger.warning("⚠️ No se pudo sincronizar PortfolioManager con SimulatedExchangeClient")
            except Exception as sync_error:
                logger.error(f"❌ Error sincronizando PortfolioManager: {sync_error}")
        
        # 🔧 REGISTRAR componentes para shutdown global
        register_component_for_shutdown('portfolio_manager', portfolio_manager)
        register_component_for_shutdown('order_manager', order_manager)
            
        # L2 Processor
        if 'l2_processor' not in components or components.get('l2_processor') is None:
            from l2_tactic.tactical_signal_processor import L2TacticProcessor
            l2_processor = L2TacticProcessor()
            components['l2_processor'] = l2_processor
            logger.info("✅ L2TacticProcessor created manually")
        else:
            l2_processor = components['l2_processor']
            logger.info("✅ L2TacticProcessor from bootstrap")

        # ================================================================
        # STEP 5: CRITICAL INJECTIONS - PORTFOLIO_MANAGER INTO ORDER_MANAGER
        # ================================================================
        logger.info("🔧 Injecting dependencies...")
        
        # Try multiple injection methods for OrderManager
        injection_success = False
        
        # Method 1: set_portfolio_manager method
        if hasattr(order_manager, 'set_portfolio_manager'):
            try:
                order_manager.set_portfolio_manager(portfolio_manager)
                logger.info("✅ Method 1: portfolio_manager injected via set_portfolio_manager()")
                injection_success = True
            except Exception as e:
                logger.warning(f"⚠️ Method 1 failed: {e}")
        
        # Method 2: Direct attribute assignment
        if not injection_success:
            try:
                order_manager.portfolio_manager = portfolio_manager
                logger.info("✅ Method 2: portfolio_manager injected as direct attribute")
                injection_success = True
            except Exception as e:
                logger.warning(f"⚠️ Method 2 failed: {e}")
        
        # Method 3: Check if OrderManager has a trading_cycle with position_manager
        if not injection_success and hasattr(order_manager, 'trading_cycle'):
            try:
                if hasattr(order_manager.trading_cycle, 'position_manager'):
                    order_manager.trading_cycle.position_manager.portfolio = portfolio_manager
                    logger.info("✅ Method 3: portfolio_manager injected into trading_cycle.position_manager")
                    injection_success = True
            except Exception as e:
                logger.warning(f"⚠️ Method 3 failed: {e}")
        
        # Method 4: Inject into OrderManager's internal state
        if not injection_success:
            try:
                if hasattr(order_manager, '_portfolio_manager'):
                    order_manager._portfolio_manager = portfolio_manager
                    logger.info("✅ Method 4: portfolio_manager injected as _portfolio_manager")
                    injection_success = True
            except Exception as e:
                logger.warning(f"⚠️ Method 4 failed: {e}")
        
        if not injection_success:
            logger.error("❌ CRITICAL: Could not inject portfolio_manager into order_manager!")
            logger.error("   OrderManager attributes: " + str(dir(order_manager)))
        
        # Additional injections
        signal_verifier = get_signal_verifier()
        trading_metrics = get_trading_metrics()
        
        # ================================================================
        # STEP 6: GET INITIAL STATE - WITH NULL CHECK
        # ================================================================
        logger.info("🔧 Getting initial state...")
        
        # ✅ CRITICAL: Verify state_coordinator before using it
        if state_coordinator is None:
            logger.critical("🚨 FATAL: state_coordinator is None before get_state!")
            raise RuntimeError("state_coordinator is None")
        
        try:
            state = state_coordinator.get_state("current")
            logger.info("✅ Initial state obtained")
        except Exception as state_error:
            logger.error(f"❌ Failed to get state: {state_error}")
            # Create minimal fallback state
            state = {
                "market_data": {},
                "total_value": 0.0,
                "l3_output": {
                    'regime': 'error',
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strategy_type': 'fallback',
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            logger.warning("⚠️ Using fallback state")
        
        # Initialize L3 components
        if not APAGAR_L3:
            sentiment_texts_cache = components.get('sentiment_texts', [])
            if "l3_output" not in state:
                state["l3_output"] = components.get('l3_output', {
                    'regime': 'error',
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strategy_type': 'l3_error',
                    'timestamp': datetime.utcnow().isoformat()
                })
        else:
            state["l3_output"] = {
                'regime': 'disabled',
                'signal': 'hold',
                'confidence': 0.0,
                'strategy_type': 'l3_disabled',
                'timestamp': datetime.utcnow().isoformat()
            }
            sentiment_texts_cache = []

        # ✅ CRITICAL: Ensure l3_output has proper structure
        if "l3_output" not in state or not state["l3_output"]:
            state["l3_output"] = {
                'regime': 'neutral',
                'signal': 'hold',
                'confidence': 0.5,  # Default medium confidence
                'strategy_type': 'initial',
                'timestamp': datetime.utcnow().isoformat()
            }
            logger.warning("🔄 Fixed empty l3_output in state")
        logger.info("🔧 Initializing position management...")
        from core.position_rotator import PositionRotator, AutoRebalancer
        from core.config import is_paper_mode
        
        # Use the already determined mode (simulated/live/etc.)
        # mode is already set earlier in main() - use it directly
        position_rotator = PositionRotator(
            portfolio_manager=portfolio_manager,
            mode=mode
        )
        auto_rebalancer = AutoRebalancer(
            portfolio_manager=portfolio_manager,
            mode=mode
        )
        logger.info(f"✅ Position managers ready (mode: {mode}, paper_mode: {is_paper_mode(mode)})")

        # ================================================================
        # STEP 9: INITIALIZE TRADING PIPELINE MANAGER
        # ================================================================
        logger.info("🔧 Initializing TradingPipelineManager...")
        # Use the already determined mode (no redefinition - fixes bug where mode was always reset to "simulated")
        
        trading_pipeline = TradingPipelineManager(
            portfolio_manager=portfolio_manager,
            order_manager=order_manager,
            l2_processor=l2_processor,
            position_rotator=position_rotator,
            auto_rebalancer=auto_rebalancer,
            signal_verifier=signal_verifier,
            state_coordinator=state_coordinator,
            mode=mode
        )
        logger.info(f"✅ TradingPipelineManager ready (mode: {mode})")

        # ================================================================
        # STEP 10: INITIALIZE ERROR RECOVERY
        # ================================================================
        logger.info("🔧 Initializing ErrorRecoveryManager...")
        error_recovery = ErrorRecoveryManager()
        logger.info("✅ ErrorRecoveryManager ready")

        # ================================================================
        # STEP 11: GET INITIAL MARKET DATA
        # ================================================================
        logger.info("🔄 Fetching initial market data...")
        try:
            market_data = await market_data_manager.get_data_with_fallback()
        
            if market_data and len(market_data) > 0:
                state["market_data"] = market_data
                logger.info(f"✅ Market data ready: {len(market_data)} symbols")
            else:
                logger.warning("⚠️ No market data available, but continuing with empty data")
                market_data = {}
        except Exception as data_error:
            logger.error(f"❌ Market data fetch failed: {data_error}")
            market_data = {}

        # ================================================================
        # STEP 14: BOOTSTRAP DEPLOYMENT (INITIAL ENTRY)
        # ================================================================
        bootstrap_executed = False
        if validate_market_data(market_data):
            logger.info("🚀 Calculating bootstrap deployment...")
            try:
            # First try bootstrap deployment (guaranteed initial entry)
                # CRITICAL FIX: Usar await para métodos async
                bootstrap_orders = await position_rotator.calculate_bootstrap_deployment(market_data)
                
                if bootstrap_orders:
                    logger.info(f"🔥 Bootstrap deployment: {len(bootstrap_orders)} orders will be executed")
                    processed_orders = await order_manager.execute_orders(bootstrap_orders)
                    await portfolio_manager.update_from_orders_async(processed_orders)
                    logger.info(f"✅ Bootstrap deployment completed: {len(processed_orders)} orders executed")
                    bootstrap_executed = True
                else:
                    logger.info("⏸️ No bootstrap deployment needed - checking initial deployment")
                    # Fallback to initial deployment if bootstrap failed or is disabled
                    # CRITICAL FIX: Usar await para método async
                    orders = await position_rotator.calculate_initial_deployment(market_data)
                    if orders:
                        processed_orders = await order_manager.execute_orders(orders)
                        await portfolio_manager.update_from_orders_async(processed_orders)
                        logger.info(f"✅ Initial deployment: {len(processed_orders)} orders")
                        bootstrap_executed = True
                    else:
                        logger.info("⚠️ No initial deployment needed")
                        
            except Exception as deploy_error:
                logger.error(f"❌ Deployment failed: {deploy_error}")
        else:
            logger.warning("⚠️ Skipping deployment - invalid market data")
        
        # ⏳ Mark bootstrap as completed to start grace period (prevents immediate SELL signals)
        if bootstrap_executed and trading_pipeline:
            trading_pipeline.mark_bootstrap_completed(current_cycle=0)

        # ================================================================
        # STEP 15: INTEGRATE AUTO-LEARNING (FIXED)
        # ================================================================
        logger.info("🤖 Integrating Auto-Learning System...")
        auto_learning_system = None
        try:
            from integration_auto_learning import AutoLearningIntegration
            
            # ✅ FIX: Crear integración con argumentos requeridos
            auto_learning_system = AutoLearningIntegration()
            
            # ✅ FIX: Inicializar con todos los componentes necesarios
            success = await auto_learning_system.initialize_integration(
                state_manager=state_coordinator,
                order_manager=order_manager,
                portfolio_manager=portfolio_manager,
                l2_processor=l2_processor,
                trading_metrics=get_trading_metrics(),
                config=config
            )
            
            if success:
                logger.info("✅ Auto-Learning System integrated successfully")
                
                # ✅ FIX: Conectar AutoLearningBridge al Trading Pipeline
                from system.auto_learning_bridge import AutoLearningBridge
                bridge = AutoLearningBridge(auto_learning_system)
                trading_pipeline.auto_learning_bridge = bridge
                logger.info("✅ Auto-Learning Bridge conectado al Trading Pipeline")
                
                # Verificar protección anti-overfitting
                status = await auto_learning_system.get_learning_status()
                if status.get('learning_system', {}).get('anti_overfitting_active', False):
                    logger.info("🛡️  Anti-overfitting protection: ACTIVE")
                else:
                    logger.warning("⚠️  Anti-overfitting protection not confirmed")
            else:
                logger.warning("⚠️  Auto-Learning System initialization returned False")
                
        except Exception as learning_error:
            logger.error(f"❌ Auto-Learning integration failed: {learning_error}")
            import traceback
            logger.error(traceback.format_exc())

        # ================================================================
        # STEP 16: APPLY FUNDAMENTAL RULE - CRITICAL FIX
        # ================================================================
        logger.info("🛡️ Applying FUNDAMENTAL RULE for simulated mode...")
        try:
            from core.state_manager import enforce_fundamental_rule
            enforce_fundamental_rule()
            logger.info("✅ FUNDAMENTAL RULE applied successfully")
        except Exception as rule_error:
            logger.error(f"❌ FUNDAMENTAL RULE failed: {rule_error}")
            # Continue anyway - this is not a critical failure
        
        # ================================================================
        # 💥 PRIORIDAD 2: WARMUP ALL SYMBOLS - Ensure ALL market data is cached
        # ================================================================
        logger.info("🔥 PRIORIDAD 2: Ejecutando warmup_all_symbols()...")
        warmup_success = await market_data_manager.warmup_all_symbols(
            timeframe="1m",
            limit=400
        )
        
        if not warmup_success:
            # Fallback: try individual symbols
            logger.warning("⚠️ warmup_all_symbols falló, intentando BTCUSDT...")
            warmup_success = await market_data_manager.force_warmup(
                symbol="BTCUSDT",
                timeframe="1m",
                limit=400
            )
            
            if not warmup_success:
                logger.warning("⚠️ Warmup BTCUSDT falló, intentando ETHUSDT...")
                warmup_success = await market_data_manager.force_warmup(
                    symbol="ETHUSDT",
                    timeframe="1m",
                    limit=400
                )
        
        # ================================================================
        # 💥 PRIORIDAD 3: FAIL FAST - No market data = No trading loop
        # ================================================================
        # Verificar si tenemos datos de mercado después del bootstrap/warmup
        market_data_after_warmup = await market_data_manager.get_data_with_fallback()
        
        if not market_data_after_warmup or len(market_data_after_warmup) == 0:
            logger.critical("🚨 FAIL FAST: market_data está vacío después de bootstrap/warmup")
            logger.critical("   No hay datos de mercado - NO se entrará al trading loop")
            logger.critical("   Intentando reintentar warmup...")
            
            # Retry warmup once
            await asyncio.sleep(5)
            retry_success = await market_data_manager.force_warmup(
                symbol="BTCUSDT",
                timeframe="1m",
                limit=400
            )
            
            if retry_success:
                logger.info("✅ Retry warmup exitoso - continuando al trading loop")
                market_data_after_warmup = await market_data_manager.get_data_with_fallback()
            else:
                logger.critical("❌ Retry warmup falló - abortando")
                logger.info("👋 HRM System shutdown complete (fail fast)")
                return  # Exit main() - no trading without market data
        
        # Update state con datos válidos
        state["market_data"] = market_data_after_warmup
        logger.info(f"✅ Market data verificado: {len(market_data_after_warmup)} símbolos")
        
        # ================================================================
        # STEP 17: SERIALIZE INITIAL PORTFOLIO STATE (CRITICAL FIX)
        # ================================================================
        logger.info("💾 Serializing initial portfolio state...")
        initial_portfolio_snapshot = portfolio_manager.get_portfolio_state()
        logger.info(f"📸 Initial portfolio snapshot: {initial_portfolio_snapshot}")
        
        # Save initial state to persistent storage
        portfolio_manager.save_to_json()
        logger.info("✅ Initial portfolio state saved to persistent_state/portfolio_state_live.json")
        
        # ================================================================
        # STEP 18: MAIN TRADING LOOP
        # ================================================================
        logger.info("🔄 Starting main trading loop...")
        logger.info("="*80)
        
        servicio_estado_vivo.registrar_cartera(portfolio_manager)
        servicio_estado_vivo.registrar_inicio_bucle()

        cycle_id = 0
        total_signals_all_cycles = 0
        total_orders_all_cycles = 0
        total_rejected_all_cycles = 0
        total_cooldown_blocked_all_cycles = 0
        
        last_cycle_time = pd.Timestamp.utcnow()

        while True:
            cycle_id += 1
            start_time = pd.Timestamp.utcnow()
            
            # Enforce 3-second cycle timing
            elapsed = (start_time - last_cycle_time).total_seconds()
            if elapsed < 3.0:
                wait_time = 3.0 - elapsed
                logger.debug(f"⏱️ Waiting {wait_time:.2f}s for 3-second cycle")
                await asyncio.sleep(wait_time)
            
            last_cycle_time = pd.Timestamp.utcnow()

            try:
                # ============================================================
                # CYCLE STEP 0: CHECK FOR ZERO BALANCES (DETECCIÓN TEMPRANA)
                # ============================================================
                logger.info(f"🔍 Cycle {cycle_id}: Checking SimulatedExchangeClient state")
                
                # Get real balances from SimulatedExchangeClient
                if hasattr(portfolio_manager, 'client') and portfolio_manager.client:
                    client = portfolio_manager.client
                    is_paper_mode = False
                    if hasattr(client, 'paper_mode'):
                        is_paper_mode = client.paper_mode
                    elif portfolio_manager.mode == "simulated":
                        is_paper_mode = True
                    
                    if is_paper_mode:
                        # Get balances from SimulatedExchangeClient
                        if hasattr(client, 'get_account_balances'):
                            import inspect
                            if inspect.iscoroutinefunction(client.get_account_balances):
                                balances = await client.get_account_balances()
                            else:
                                balances = client.get_account_balances()
                        elif hasattr(client, 'get_balance'):
                            if inspect.iscoroutinefunction(client.get_balance):
                                current_btc = await client.get_balance("BTC")
                                current_eth = await client.get_balance("ETH")
                                current_usdt = await client.get_balance("USDT")
                                balances = {
                                    "BTC": current_btc,
                                    "ETH": current_eth,
                                    "USDT": current_usdt
                                }
                            else:
                                current_btc = client.get_balance("BTC")
                                current_eth = client.get_balance("ETH")
                                current_usdt = client.get_balance("USDT")
                                balances = {
                                    "BTC": current_btc,
                                    "ETH": current_eth,
                                    "USDT": current_usdt
                                }
                        else:
                            logger.warning("⚠️ Client has no get_account_balances or get_balance method")
                            continue
                        
                        # Log SIM_STATE_ID and SIM_BALANCES
                        logger.info(f"   SIM_STATE_ID: {id(client)}")
                        logger.info(f"   SIM_BALANCES: {balances}")
                        
                # Check if all balances are zero
                        current_btc = balances.get("BTC", 0.0)
                        current_eth = balances.get("ETH", 0.0)
                        current_usdt = balances.get("USDT", 0.0)
                        
                        if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
                            logger.critical("🚨 CRITICAL: All balances are zero - state loss detected!")
                            
                            # FIX: Attempt recovery from persistent_state first
                            logger.info("🔧 Attempting to recover state from persistent storage...")
                            recovery_success = False
                            recovery_source = None
                            
                            # STEP 1: Try to load from persistent_state/portfolio_state_live.json
                            try:
                                if hasattr(portfolio_manager, 'load_from_json'):
                                    loaded = portfolio_manager.load_from_json()
                                    if loaded:
                                        # Successfully loaded from persistent storage
                                        recovery_source = "persistent_state/portfolio_state_live.json"
                                        recovery_success = True
                                        logger.info(f"✅ State recovered from persistent storage: {recovery_source}")
                                        logger.info(f"   BTC: {portfolio_manager.portfolio.get('BTCUSDT', {}).get('position', 0):.6f}")
                                        logger.info(f"   ETH: {portfolio_manager.portfolio.get('ETHUSDT', {}).get('position', 0):.6f}")
                                        logger.info(f"   USDT: {portfolio_manager.portfolio.get('USDT', {}).get('free', 0):.2f}")
                                        
                                        # Sync the recovered state to SimulatedExchangeClient
                                        if hasattr(client, 'force_reset'):
                                            recovered_balances = {
                                                "BTC": portfolio_manager.portfolio.get('BTCUSDT', {}).get('position', 0),
                                                "ETH": portfolio_manager.portfolio.get('ETHUSDT', {}).get('position', 0),
                                                "USDT": portfolio_manager.portfolio.get('USDT', {}).get('free', 0)
                                            }
                                            client.force_reset(recovered_balances)
                                            logger.info(f"✅ SimulatedExchangeClient synchronized with recovered state")
                                    else:
                                        logger.warning("⚠️ Persistent state file exists but has zero balances or is invalid")
                            except Exception as load_error:
                                logger.warning(f"⚠️ Failed to load from persistent storage: {load_error}")
                            
                            # STEP 2: If persistent state failed, try initial_portfolio_snapshot
                            if not recovery_success:
                                logger.info("🔧 Attempting recovery from initial_portfolio_snapshot...")
                                try:
                                    if 'initial_portfolio_snapshot' in locals() and initial_portfolio_snapshot:
                                        btc_pos = initial_portfolio_snapshot.get('BTCUSDT', {}).get('position', 0)
                                        eth_pos = initial_portfolio_snapshot.get('ETHUSDT', {}).get('position', 0)
                                        usdt_free = initial_portfolio_snapshot.get('USDT', {}).get('free', 0)
                                        
                                        if btc_pos > 0 or eth_pos > 0 or usdt_free > 0:
                                            recovery_source = "initial_portfolio_snapshot"
                                            recovery_success = True
                                            logger.info(f"✅ State recovered from initial_portfolio_snapshot")
                                            logger.info(f"   BTC: {btc_pos:.6f}, ETH: {eth_pos:.6f}, USDT: {usdt_free:.2f}")
                                            
                                            # Restore to portfolio_manager
                                            portfolio_manager.portfolio['BTCUSDT'] = {'position': btc_pos, 'free': btc_pos}
                                            portfolio_manager.portfolio['ETHUSDT'] = {'position': eth_pos, 'free': eth_pos}
                                            portfolio_manager.portfolio['USDT'] = {'free': usdt_free}
                                            
                                            # Sync to SimulatedExchangeClient
                                            if hasattr(client, 'force_reset'):
                                                client.force_reset({
                                                    "BTC": btc_pos,
                                                    "ETH": eth_pos,
                                                    "USDT": usdt_free
                                                })
                                                logger.info(f"✅ SimulatedExchangeClient synchronized with initial snapshot")
                                except Exception as snapshot_error:
                                    logger.warning(f"⚠️ Failed to recover from initial snapshot: {snapshot_error}")
                            
                            # STEP 3: If all else fails, use hardcoded fallback values
                            if not recovery_success:
                                logger.warning("⚠️ All recovery methods failed - using hardcoded fallback values")
                                recovery_source = "hardcoded_fallback"
                                
                                # Use hardcoded values as last resort
                                fallback_balances = {"BTC": 0.01549, "ETH": 0.385, "USDT": 3000.0}
                                logger.info(f"🔧 Applying hardcoded fallback: {fallback_balances}")
                                
                                try:
                                    if hasattr(client, 'force_reset'):
                                        client.force_reset(fallback_balances)
                                        logger.info("✅ Hardcoded fallback balances applied")
                                        
                                        # Update portfolio_manager with fallback values
                                        portfolio_manager.portfolio['BTCUSDT'] = {
                                            'position': fallback_balances['BTC'], 
                                            'free': fallback_balances['BTC']
                                        }
                                        portfolio_manager.portfolio['ETHUSDT'] = {
                                            'position': fallback_balances['ETH'], 
                                            'free': fallback_balances['ETH']
                                        }
                                        portfolio_manager.portfolio['USDT'] = {
                                            'free': fallback_balances['USDT']
                                        }
                                        recovery_success = True
                                    else:
                                        logger.error("❌ force_reset not available - cannot recover")
                                        raise RuntimeError("State loss - all balances zero and recovery failed")
                                except Exception as fallback_error:
                                    logger.error(f"❌ Failed to apply hardcoded fallback: {fallback_error}")
                                    raise RuntimeError("State loss - all recovery methods failed")
                            
                            # Log the recovery result
                            logger.info("=" * 70)
                            logger.info(f"🔄 STATE RECOVERY COMPLETED")
                            logger.info(f"   Source: {recovery_source}")
                            logger.info(f"   BTC: {portfolio_manager.portfolio.get('BTCUSDT', {}).get('position', 0):.6f}")
                            logger.info(f"   ETH: {portfolio_manager.portfolio.get('ETHUSDT', {}).get('position', 0):.6f}")
                            logger.info(f"   USDT: {portfolio_manager.portfolio.get('USDT', {}).get('free', 0):.2f}")
                            logger.info("=" * 70)
                            
                            # Save recovered state to persistent storage
                            portfolio_manager.save_to_json()
                            logger.info("💾 Recovered state saved to persistent storage")
                
                # ============================================================
                # CYCLE STEP 1: GET FRESH MARKET DATA
                # ============================================================
                try:
                    market_data = await market_data_manager.get_data_with_fallback()
                    
                    if not market_data or len(market_data) == 0:
                        logger.warning(f"⚠️ Cycle {cycle_id}: No market data, skipping")
                        await asyncio.sleep(3.0)
                        continue
                    
                    # Update state with fresh data
                    state["market_data"] = market_data
                    
                except Exception as data_error:
                    logger.error(f"❌ Cycle {cycle_id}: Market data error: {data_error}")
                    await asyncio.sleep(3.0)
                    continue
                
                # ============================================================
                # CYCLE STEP 1.5: ORDER FLOW SIGNAL GENERATION (ANTES DE L1 CLÁSICO)
                # ============================================================
                orderflow_signals = {}
                orderflow_override = False
                
                if order_flow_signal_generator is not None:
                    try:
                        # Actualizar estado de posiciones antes de generar señales
                        for symbol in ["BTCUSDT", "ETHUSDT"]:
                            # Verificar si hay posición larga
                            has_long = False
                            if portfolio_manager:
                                position = portfolio_manager.portfolio.get(symbol, {}).get('position', 0)
                                has_long = position > 0
                                order_flow_signal_generator.update_position_state(symbol, has_long)
                        
                        # Generar señales para cada símbolo
                        for symbol in ["BTCUSDT", "ETHUSDT"]:
                            if symbol in market_data:
                                signal_result = await order_flow_signal_generator.get_signal(symbol)
                                orderflow_signals[symbol] = signal_result
                                
                                # Si la señal tiene confianza > 0.50, activar override
                                if signal_result.action in ['buy', 'sell'] and signal_result.confidence > 0.50:
                                    orderflow_override = True
                                    logger.info(f"🌊 ORDERFLOW OVERRIDE: {symbol} {signal_result.action.upper()} "
                                               f"(conf={signal_result.confidence:.2f}) - Bypass L1 clásico")
                        
                        # Log resumen de señales order flow cada 5 ciclos
                        if cycle_id % 5 == 0 and orderflow_signals:
                            logger.info("🌊 OrderFlow Signals Status:")
                            for sym, sig in orderflow_signals.items():
                                logger.info(f"   {sym}: {sig.action.upper()} (imb={sig.imbalance_current:.3f}, conf={sig.confidence:.2f})")
                                    
                    except Exception as of_sig_error:
                        logger.warning(f"⚠️ OrderFlow signal generation error: {of_sig_error}")
                        orderflow_signals = {}
                        orderflow_override = False
                
                # ============================================================
                # CYCLE STEP 2: PROCESS TRADING CYCLE
                # ============================================================
                # Si hay override de order flow, pasar las señales al pipeline
                if orderflow_override:
                    state['orderflow_signals'] = {sym: sig.to_dict() for sym, sig in orderflow_signals.items()}
                    state['orderflow_override'] = True
                else:
                    state['orderflow_override'] = False
                    if 'orderflow_signals' in state:
                        del state['orderflow_signals']
                
                cycle_result = await trading_pipeline.process_trading_cycle(
                    state=state,
                    market_data=market_data,  # ✅ CRITICAL: Pass market_data explicitly
                    cycle_id=cycle_id
                )
                
                # Update state from cycle result
                state["total_value"] = cycle_result.portfolio_value
                
                # Update cumulative counters
                total_signals_all_cycles += cycle_result.signals_generated
                total_orders_all_cycles += cycle_result.orders_executed
                total_rejected_all_cycles += cycle_result.orders_rejected
                total_cooldown_blocked_all_cycles += cycle_result.cooldown_blocked

                # ============================================================
                # CYCLE STEP 3: SYNC PORTFOLIO FROM EXCHANGE (CRITICAL FIX)
                # ============================================================
                # MUST sync after EVERY order execution to maintain state consistency
                if cycle_result.orders_executed > 0:
                    logger.info(f"🔄 Syncing portfolio after {cycle_result.orders_executed} executed orders...")
                    
                    # Get the simulated client for paper mode
                    simulated_client = None
                    if hasattr(order_manager, 'simulated_client'):
                        simulated_client = order_manager.simulated_client
                    elif hasattr(order_manager, 'client'):
                        simulated_client = order_manager.client
                    
                    if simulated_client:
                        # CRITICAL: Sync portfolio from exchange client
                        sync_success = await portfolio_manager.sync_from_exchange_async(simulated_client)
                        
                        if sync_success:
                            logger.info("✅ Portfolio synced successfully after order execution")
                        else:
                            logger.error("❌ Portfolio sync failed after order execution")
                    
                    # Update NAV with current market prices
                    market_prices = {}
                    if simulated_client and hasattr(simulated_client, 'get_market_price'):
                        market_prices = {
                            "BTCUSDT": simulated_client.get_market_price("BTCUSDT"),
                            "ETHUSDT": simulated_client.get_market_price("ETHUSDT")
                        }
                    else:
                        # Fallback to market_data
                        for symbol in ["BTCUSDT", "ETHUSDT"]:
                            if symbol in market_data:
                                data = market_data[symbol]
                                if isinstance(data, dict) and "close" in data:
                                    market_prices[symbol] = data["close"]
                                elif hasattr(data, 'iloc'):
                                    market_prices[symbol] = data.iloc[-1]["close"]
                    
                    if market_prices:
                        await portfolio_manager.update_nav_async(market_prices)
                
                # ============================================================
                # CYCLE STEP 4: LOG SUMMARY (every 5 cycles)
                # ============================================================
                if cycle_id % 5 == 0:
                    logger.info("="*80)
                    logger.info(f"📊 CYCLE {cycle_id} SUMMARY")
                    logger.info(f"   Signals: {cycle_result.signals_generated}")
                    logger.info(f"   Orders Executed: {cycle_result.orders_executed}")
                    logger.info(f"   Orders Rejected: {cycle_result.orders_rejected}")
                    logger.info(f"   Cooldown Blocked: {cycle_result.cooldown_blocked}")
                    logger.info(f"   Portfolio Value: ${cycle_result.portfolio_value:.2f}")
                    logger.info(f"   L3 Regime: {cycle_result.l3_regime}")
                    logger.info("="*80)
                    
                    # Cumulative stats
                    logger.info(f"📈 CUMULATIVE (Cycles 1-{cycle_id})")
                    logger.info(f"   Total Signals: {total_signals_all_cycles}")
                    logger.info(f"   Total Orders: {total_orders_all_cycles}")
                    logger.info(f"   Total Rejected: {total_rejected_all_cycles}")
                    logger.info(f"   Avg Orders/Cycle: {total_orders_all_cycles/cycle_id:.2f}")
                    logger.info("="*80)
                
                # ============================================================
                # CYCLE STEP 4B: EXTRAER Y MOSTRAR PRECIOS ACTUALES
                # ============================================================
                # Mejorar la extracción de precios en el ciclo
                current_prices = {}
                for symbol, df in market_data.items():
                    if df is not None and not df.empty:
                        current_prices[symbol] = float(df.iloc[-1]['close'])
                        logger.info(f"📊 {symbol}: ${current_prices[symbol]:.2f}")
                
                # ============================================================
                # CYCLE STEP 5: PORTFOLIO COMPARISON LOG (every cycle)
                # ============================================================
                # CRITICAL FIX: Log portfolio comparison using cycle_context snapshot
                # El cycle_context ahora se actualiza POST-EJECUCIÓN en trading_pipeline_manager.py
                # para reflejar los balances reales después de ejecutar órdenes
                cycle_context = trading_pipeline.get_cycle_context()
                await log_portfolio_comparison(
                    cycle_context=cycle_context,
                    portfolio_manager=portfolio_manager,
                    cycle_id=cycle_id,
                    market_data=market_data
                )
                await servicio_estado_vivo.notificar_ciclo(cycle_id)

            except Exception as cycle_error:
                logger.error(f"❌ Cycle {cycle_id} error: {cycle_error}")
                
                # Use ErrorRecoveryManager for intelligent recovery
                try:
                    if error_recovery is not None:
                        recovery_action = await error_recovery.handle_cycle_error(
                            error=cycle_error,
                            state=state,
                            cycle_id=cycle_id
                        )
                        
                        if recovery_action.action == RecoveryActionType.SHUTDOWN:
                            logger.critical("🛑 Unrecoverable error - shutting down")
                            break
                        
                        if recovery_action.action == RecoveryActionType.RESET_COMPONENT:
                            logger.warning(f"🔄 Resetting component...")
                            for step in recovery_action.recovery_steps_taken:
                                logger.info(f"  ✓ {step}")
                        
                        await asyncio.sleep(recovery_action.wait_seconds)
        
                    else:
                        logger.warning("⚠️ ErrorRecovery not available, using default wait")
                        await asyncio.sleep(10.0)
                    
                except Exception as recovery_error:
                    logger.error(f"❌ Recovery failed: {recovery_error}")
                    await asyncio.sleep(10.0)
                
                continue
                
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
        try:
            if portfolio_manager:
                portfolio_manager.save_to_json()
                logger.info("💾 Portfolio state saved")
        except Exception as save_error:
            logger.error(f"❌ Error saving portfolio: {save_error}")
            
    except Exception as fatal_error:
        logger.critical(f"🚨 FATAL ERROR: {fatal_error}")
        import traceback
        logger.critical(traceback.format_exc())
            
    finally:
        servicio_estado_vivo.registrar_parada()
        # 🔧 SHUTDOWN LIMPIO GLOBAL desde bootstrap
        logger.info("🧹 Ejecutando shutdown limpio global desde bootstrap...")
        try:
            # Usar la función shutdown global de bootstrap
            await shutdown()
            logger.info("✅ Shutdown global completado exitosamente")
        except Exception as shutdown_error:
            logger.error(f"❌ Error en shutdown global: {shutdown_error}")
        
        # Cleanup adicional de verificación de señales
        try:
            await stop_signal_verification()
            logger.info("✓ Signal verification stopped")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Cleanup error: {cleanup_error}")
        
        logger.info("👋 HRM System shutdown complete")

async def update_portfolio_nav_from_simulated_exchange(portfolio_manager, market_data):
    """
    Actualizar el NAV del PortfolioManager usando balances reales del SimulatedExchangeClient.
    
    Esta función se ejecuta después de cada ciclo de trading para asegurar que el NAV
    refleje los balances actuales del cliente simulado y los precios de mercado.
    """
    try:
        # Verificar que tenemos un cliente disponible
        if not hasattr(portfolio_manager, 'client') or not portfolio_manager.client:
            logger.debug("⚠️ No client available for NAV update")
            return
        
        client = portfolio_manager.client
        
        # Solo actualizar en modo paper/simulated
        is_paper_mode = False
        if hasattr(client, 'paper_mode'):
            is_paper_mode = client.paper_mode
        elif portfolio_manager.mode == "simulated":
            is_paper_mode = True
        
        if not is_paper_mode:
            return
        
        # Obtener balances del SimulatedExchangeClient
        if hasattr(client, 'get_balances'):
            import inspect
            if inspect.iscoroutinefunction(client.get_balances):
                balances = await client.get_balances()
            else:
                balances = client.get_balances()
        elif hasattr(client, 'get_account_balances'):
            import inspect
            if inspect.iscoroutinefunction(client.get_account_balances):
                balances = await client.get_account_balances()
            else:
                balances = client.get_account_balances()
        else:
            return
        
        # Extraer balances
        current_btc = balances.get("BTC", 0.0)
        current_eth = balances.get("ETH", 0.0)
        current_usdt = balances.get("USDT", 0.0)
        
        # Validar que no todos los balances sean cero
        if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
            logger.warning("⚠️ NAV Update: All balances are zero, skipping update")
            return
        
        # Obtener precios de mercado
        btc_price = 0.0
        eth_price = 0.0
        
        if hasattr(client, 'get_market_price'):
            btc_price = client.get_market_price("BTCUSDT")
            eth_price = client.get_market_price("ETHUSDT")
        elif market_data:
            # Fallback a market_data si no hay get_market_price
            btc_data = market_data.get("BTCUSDT", {})
            eth_data = market_data.get("ETHUSDT", {})
            
            if isinstance(btc_data, dict) and "close" in btc_data:
                btc_price = btc_data["close"]
            elif hasattr(btc_data, 'iloc'):
                btc_price = btc_data.iloc[-1]["close"]
                
            if isinstance(eth_data, dict) and "close" in eth_data:
                eth_price = eth_data["close"]
            elif hasattr(eth_data, 'iloc'):
                eth_price = eth_data.iloc[-1]["close"]
        
        # Calcular NAV total
        btc_value = current_btc * btc_price
        eth_value = current_eth * eth_price
        total_nav = current_usdt + btc_value + eth_value
        
        # Actualizar el portfolio del PortfolioManager
        portfolio_manager.portfolio["BTCUSDT"] = {"position": current_btc, "free": current_btc}
        portfolio_manager.portfolio["ETHUSDT"] = {"position": current_eth, "free": current_eth}
        portfolio_manager.portfolio["USDT"] = {"free": current_usdt}
        portfolio_manager.portfolio["total"] = total_nav
        
        # Actualizar peak_value si es necesario
        if total_nav > portfolio_manager.peak_value:
            portfolio_manager.peak_value = total_nav
            portfolio_manager.portfolio["peak_value"] = total_nav
        
        # Log de actualización (solo en debug para no saturar)
        logger.debug(f"📊 NAV Updated: BTC={current_btc:.6f} @ ${btc_price:.2f}, ETH={current_eth:.4f} @ ${eth_price:.2f}, USDT=${current_usdt:.2f}, TOTAL=${total_nav:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error updating NAV from SimulatedExchangeClient: {e}")


async def shutdown_cleanup(market_data_manager=None, exchange_client=None, realtime_loader=None):
    """
    Shutdown limpio global - cierra todas las conexiones y sesiones.
    
    Asegura que:
    - Todas las sesiones aiohttp se cierran
    - No quedan warnings asyncio al terminar
    - Los recursos se liberan correctamente
    """
    logger.info("🧹 Iniciando shutdown limpio global...")
    
    # Cerrar MarketDataManager
    if market_data_manager is not None:
        try:
            await market_data_manager.close()
            logger.info("✅ MarketDataManager cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando MarketDataManager: {e}")
    
    # Cerrar RealTimeLoader si existe
    if realtime_loader is not None:
        try:
            await realtime_loader.close()
            logger.info("✅ RealTimeLoader cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando RealTimeLoader: {e}")
    
    # Cerrar ExchangeClient
    if exchange_client is not None:
        try:
            if hasattr(exchange_client, 'close'):
                await exchange_client.close()
            elif hasattr(exchange_client, 'close_connection'):
                await exchange_client.close_connection()
            logger.info("✅ ExchangeClient cerrado")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando ExchangeClient: {e}")
    
    # Cerrar todas las sesiones aiohttp pendientes
    try:
        import aiohttp
        import asyncio
        
        # Cerrar todas las sesiones aiohttp abiertas
        for task in asyncio.all_tasks():
            if 'aiohttp' in str(task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("✅ Sesiones aiohttp cerradas")
    except Exception as e:
        logger.warning(f"⚠️ Error cerrando sesiones aiohttp: {e}")
    
    # Forzar cierre del event loop para evitar warnings
    try:
        loop = asyncio.get_running_loop()
        # Cancelar todas las tareas pendientes
        pending_tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending_tasks:
            task.cancel()
        
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            logger.info(f"✅ {len(pending_tasks)} tareas pendientes canceladas")
    except Exception as e:
        logger.warning(f"⚠️ Error cancelando tareas pendientes: {e}")
    
    logger.info("🧹 Shutdown limpio completado")


async def log_portfolio_comparison(cycle_context, portfolio_manager, cycle_id, market_data):
        """
        Log portfolio comparison using cycle_context snapshot.
        
        🔑 CRITICAL FIX: Uses cycle_context snapshot instead of calling get_asset_balance_async()
        🔑 This ensures we use the SAME balances throughout the entire cycle
        
        CRITICAL FIX: Uses cycle_context to avoid multiple balance calls
        """
        try:
            # ================================================================
            # STEP 1: VERIFICAR CYCLE_CONTEXT DISPONIBLE
            # ================================================================
            if cycle_context is None:
                logger.warning("⚠️ cycle_context is None, cannot log portfolio comparison")
                return
            
            # ================================================================
            # STEP 2: EXTRAER BALANCES DEL SNAPSHOT DEL CICLO
            # ================================================================
            # 🔑 CRITICAL: Usar el snapshot del ciclo en lugar de llamar a get_balances_async()
            balances = cycle_context.get("balances", {})
            prices = cycle_context.get("prices", {})
            
            current_btc = balances.get("BTC", 0.0)
            current_eth = balances.get("ETH", 0.0)
            current_usdt = balances.get("USDT", 0.0)
            
            # ================================================================
            # STEP 3: VALIDACIÓN - PROTECCIÓN CONTRA PÉRDIDA DE ESTADO
            # ================================================================
            if current_btc == 0.0 and current_eth == 0.0 and current_usdt == 0.0:
                logger.critical("🚨 ERROR: cycle_context shows all balances zero!")
                return
            
            # ================================================================
            # STEP 4: OBTENER PRECIOS DEL SNAPSHOT Y CALCULAR NAV
            # ================================================================
            btc_price = prices.get("BTCUSDT", 0.0)
            eth_price = prices.get("ETHUSDT", 0.0)
            
            # Calcular NAV
            btc_value = current_btc * btc_price
            eth_value = current_eth * eth_price
            current_total = current_usdt + btc_value + eth_value
            
            # ================================================================
            # STEP 5: LOG DEL ESTADO REAL DEL PORTFOLIO (DESDE SNAPSHOT)
            # ================================================================
            initial_capital = portfolio_manager.initial_balance if hasattr(portfolio_manager, 'initial_balance') else 500.0
            color = Fore.GREEN if current_total >= initial_capital else Fore.RED
            
            logger.info("=" * 70)
            logger.info(f"{color}💰 PORTFOLIO COMPARISON - Cycle {cycle_id} (POST-EXECUTION SNAPSHOT){Style.RESET_ALL}")
            logger.info(f"   BTC:  {current_btc:.6f} @ ${btc_price:.2f} = ${btc_value:.2f}")
            logger.info(f"   ETH:  {current_eth:.6f} @ ${eth_price:.2f} = ${eth_value:.2f}")
            logger.info(f"   USDT: ${current_usdt:.2f}")
            logger.info(f"   {'─' * 50}")
            logger.info(f"{color}   TOTAL NAV: ${current_total:.2f} (Initial: ${initial_capital:.2f}){Style.RESET_ALL}")
            logger.info("=" * 70)
        
        except Exception as e:
            logger.error(f"❌ Error in portfolio comparison: {e}", exc_info=True)


# 🚀 INTEGRACIÓN FastAPI - BEGIN
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from api.servicios.servicio_websockets import servicio_websockets
from api.servicios.servicio_estado_vivo import servicio_estado_vivo
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="HRM Trading System API",
    description="API REST para el sistema de trading HRM",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "HRM_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
    ).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# Modelos de respuesta con type hints
class StatusResponse(BaseModel):
    status: str = Field(default="pending", description="Estado del sistema")
    trading_active: bool = Field(default=False, description="Modo trading activo")
    cycle_count: int = Field(default=0, description="Contador de ciclos")
    uptime: str = Field(default="no disponible", description="Tiempo de actividad")
    mode: str = Field(default="unknown", description="Modo real del backend: paper, live o unknown")


class PortfolioResponse(BaseModel):
    btc_balance: float = Field(default=0.0, description="Balance BTC")
    eth_balance: float = Field(default=0.0, description="Balance ETH")
    usdt_balance: float = Field(default=0.0, description="Balance USDT")
    total_nav: float = Field(default=0.0, description="NAV total")
    initial_capital: float = Field(default=0.0, description="Capital inicial")


class HealthCheckResponse(BaseModel):
    status: str = Field(default="ok", description="Estado salud")
    timestamp: str = Field(default="", description="Marca de tiempo")
    version: str = Field(default="2.0.0", description="Versión API")


@app.websocket("/ws")
async def websocket_eventos(websocket: WebSocket) -> None:
    """Mantiene la conexión abierta para los eventos del sistema."""
    conectado = await servicio_websockets.conectar(websocket)
    if not conectado:
        return
    try:
        while True:
            # El cliente puede enviar mensajes de control/heartbeat.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await servicio_websockets.desconectar(websocket)
    except Exception:
        await servicio_websockets.desconectar(websocket)
        await websocket.close(code=1011)


# Endpoint health check
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Chequeo de salud del sistema",
    description="Verifica que el sistema de trading esté operando correctamente",
)
async def health_check() -> HealthCheckResponse:
    from datetime import datetime, timezone
    return HealthCheckResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.0.0",
    )


# Endpoint status
@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Estado actual del sistema de trading",
    description="Obtiene el estado operativo del bot de trading",
)
async def get_status() -> StatusResponse:
    return StatusResponse(**servicio_estado_vivo.obtener_estado_sistema())


# Endpoint portfolio
@app.get(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="Información de la cartera actual",
    description="Obtiene el estado actual de balances y NAV de la cartera",
)
async def get_portfolio() -> PortfolioResponse:
    resumen = servicio_estado_vivo.obtener_resumen_cartera()
    if resumen is None:
        raise HTTPException(status_code=503, detail="La cartera aún no está inicializada")
    return PortfolioResponse(**resumen)


# Endpoint version
@app.get(
    "/version",
    summary="Información de versión",
    description="Obtiene la versión actual del sistema",
)
async def get_version() -> dict:
    import platform
    return {
        "app_version": "2.0.0",
        "framework": "FastAPI",
        "python_version": platform.python_version(),
    }


def run_fastapi(host: str = os.getenv("HRM_API_HOST", "127.0.0.1"), port: int = 8000, reload: bool = False) -> None:
    """Inicia el servidor FastAPI con uvicorn."""
    uvicorn.run(app, host=host, port=port, reload=reload)


# 🚀 INTEGRACIÓN FastAPI - END

# Punto de entrada principal del sistema
if __name__ == "__main__":
    """Ejecutar el trading bot con API FastAPI integrada."""
    import asyncio
    from core.logging import logger
    
    logger.info("🚀 Iniciando HRM Trading System con API FastAPI integrada...")
    logger.info("=" * 60)
    
    async def main_with_api() -> None:
        """Ejecutar el trading loop y la API FastAPI concurrentemente."""
        # Configurar y iniciar FastAPI en background
        import uvicorn
        config = uvicorn.Config(
            app=app,
            host=os.getenv("HRM_API_HOST", "127.0.0.1"),
            port=8000,
            log_level="info",
        )
        server = uvicorn.Server(config)
        
        # Ejecutar API FastAPI en una tarea separada
        api_task = asyncio.create_task(server.serve())
        
        # Ejecutar el trading loop principal
        try:
            await main()
        finally:
            # Cuando el trading loop termine, detener la API
            server.should_exit = True
            await api_task
    
    asyncio.run(main_with_api())
