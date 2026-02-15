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
        
        # ================================================================
        # STEP 1: SYSTEM CLEANUP - USAR perform_full_cleanup()
        # ================================================================
        logger.info("🧹 Running system cleanup...")
        if CLEANUP_AVAILABLE:
            try:
                # Usar perform_full_cleanup que resetea singletons y fuerza paper mode
                cleanup_result = perform_full_cleanup(mode="paper")
                
                if cleanup_result.get("success", False):
                    logger.info(f"✅ Cleanup completo exitoso")
                    logger.info(f"   📁 Archivos eliminados: {cleanup_result.get('filesystem', {}).get('deleted_files', 0)}")
                    logger.info(f"   🧠 Singletons reseteados")
                    logger.info(f"   🎯 Modo forzado: paper")
                else:
                    logger.warning("⚠️ Cleanup completed with warnings")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Cleanup failed: {cleanup_error}")
                # Still try to force paper mode as fallback
                try:
                    force_paper_mode()
                except:
                    pass
        else:
            logger.info("⚠️ SystemCleanup not available, skipping")

        # Paper trades cleanup
        try:
            from storage.paper_trade_logger import get_paper_logger
            get_paper_logger(clear_on_init=True)
            logger.info("✅ Paper trades cleared")
        except Exception as e:
            logger.warning(f"⚠️ Paper trades cleanup failed: {e}")

        logger.info("🚀 Starting HRM system")

        # ================================================================
        # STEP 2: INITIALIZE STATE_COORDINATOR (SINGLETON) - CRITICAL FIX
        # ================================================================
        logger.info("🔧 Initializing StateCoordinator...")
        # Determine mode based on configuration
        mode = "simulated"  # Default to simulated mode
        try:
            live_config = get_config("live")
            if not getattr(live_config, 'PAPER_MODE', True):
                mode = "live"
        except Exception:
            mode = "simulated"
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
            # CRITICAL: Create a minimal fallback MarketDataManager
            market_data_manager = MarketDataManager(
                symbols=["BTCUSDT", "ETHUSDT"],
                fallback_enabled=True
            )
            logger.warning("⚠️ Using fallback MarketDataManager initialization")
    
        # ✅ CRITICAL: Verify MarketDataManager is not None
        if market_data_manager is None:
            logger.critical("🚨 FATAL: MarketDataManager is None after initialization!")
            raise RuntimeError("MarketDataManager cannot be None - trading loop cannot start")

        # 🔧 REGISTRAR MarketDataManager para shutdown global
        register_component_for_shutdown('market_data_manager', market_data_manager)

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
            
            # ✅ Create PortfolioManager with BinanceClient AND MarketDataManager in simulated mode
            portfolio_manager = PortfolioManager(
                exchange_client=binance_client, 
                market_data_manager=market_data_manager,
                mode="simulated"
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
            logger.info("✅ OrderManager from bootstrap")
        
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
        
        # Get paper_mode from config and inject into PositionRotator
        try:
            live_config = get_config("live")
            paper_mode = getattr(live_config, 'PAPER_MODE', False)
        except Exception:
            paper_mode = False
        
        position_rotator = PositionRotator(
            portfolio_manager=portfolio_manager,
            paper_mode=paper_mode
        )
        auto_rebalancer = AutoRebalancer(
            portfolio_manager=portfolio_manager,
            paper_mode=paper_mode
        )
        logger.info("✅ Position managers ready")

        # ================================================================
        # STEP 9: INITIALIZE TRADING PIPELINE MANAGER
        # ================================================================
        logger.info("🔧 Initializing TradingPipelineManager...")
        # Determine mode based on configuration
        mode = "simulated"  # Default to simulated mode
        try:
            live_config = get_config("live")
            if not getattr(live_config, 'PAPER_MODE', True):
                mode = "live"
        except Exception:
            mode = "simulated"
        
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
                else:
                    logger.info("⏸️ No bootstrap deployment needed - checking initial deployment")
                    # Fallback to initial deployment if bootstrap failed or is disabled
                    # CRITICAL FIX: Usar await para método async
                    orders = await position_rotator.calculate_initial_deployment(market_data)
                    if orders:
                        processed_orders = await order_manager.execute_orders(orders)
                        await portfolio_manager.update_from_orders_async(processed_orders)
                        logger.info(f"✅ Initial deployment: {len(processed_orders)} orders")
                    else:
                        logger.info("⚠️ No initial deployment needed")
                        
            except Exception as deploy_error:
                logger.error(f"❌ Deployment failed: {deploy_error}")
        else:
            logger.warning("⚠️ Skipping deployment - invalid market data")

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
            limit=100
        )
        
        if not warmup_success:
            # Fallback: try individual symbols
            logger.warning("⚠️ warmup_all_symbols falló, intentando BTCUSDT...")
            warmup_success = await market_data_manager.force_warmup(
                symbol="BTCUSDT",
                timeframe="1m",
                limit=100
            )
            
            if not warmup_success:
                logger.warning("⚠️ Warmup BTCUSDT falló, intentando ETHUSDT...")
                warmup_success = await market_data_manager.force_warmup(
                    symbol="ETHUSDT",
                    timeframe="1m",
                    limit=100
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
                limit=100
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
        # STEP 17: MAIN TRADING LOOP
        # ================================================================
        logger.info("🔄 Starting main trading loop...")
        logger.info("="*80)
        
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
                            logger.critical("ERROR: Pérdida de estado - todos los balances son cero")
                            # Intentar restaurar los balances iniciales
                            logger.info("🔧 Intentando restaurar balances iniciales...")
                            try:
                                # Force reset del SimulatedExchangeClient para restaurar balances
                                if hasattr(client, 'force_reset'):
                                    client.force_reset({
                                        "BTC": 0.01549,
                                        "ETH": 0.385,
                                        "USDT": 3000.0
                                    })
                                    logger.info("✅ Balances iniciales restaurados")
                                    # Volver a obtener los balances después del reset
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
                                            balances = {
                                                "BTC": client.get_balance("BTC"),
                                                "ETH": client.get_balance("ETH"),
                                                "USDT": client.get_balance("USDT")
                                            }
                                    logger.info(f"✅ Nuevos balances: {balances}")
                                else:
                                    logger.error("❌ No se puede restaurar balances - force_reset no disponible")
                                    raise RuntimeError("Pérdida de estado - todos los balances son cero y no se puede restaurar")
                            except Exception as reset_error:
                                logger.error(f"❌ Error al restaurar balances: {reset_error}")
                                raise RuntimeError("Pérdida de estado - todos los balances son cero")
                
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
                # CYCLE STEP 2: PROCESS TRADING CYCLE
                # ============================================================
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
                # Esto evita llamadas adicionales a get_balances_async()
                cycle_context = trading_pipeline.get_cycle_context()
                await log_portfolio_comparison(
                    cycle_context=cycle_context,
                    portfolio_manager=portfolio_manager,
                    cycle_id=cycle_id,
                    market_data=market_data
                )

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
            logger.info(f"{color}💰 PORTFOLIO COMPARISON - Cycle {cycle_id} (CYCLE SNAPSHOT){Style.RESET_ALL}")
            logger.info(f"   BTC:  {current_btc:.6f} @ ${btc_price:.2f} = ${btc_value:.2f}")
            logger.info(f"   ETH:  {current_eth:.6f} @ ${eth_price:.2f} = ${eth_value:.2f}")
            logger.info(f"   USDT: ${current_usdt:.2f}")
            logger.info(f"   {'─' * 50}")
            logger.info(f"{color}   TOTAL NAV: ${current_total:.2f} (Initial: ${initial_capital:.2f}){Style.RESET_ALL}")
            logger.info("=" * 70)
        
        except Exception as e:
            logger.error(f"❌ Error in portfolio comparison: {e}", exc_info=True)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
