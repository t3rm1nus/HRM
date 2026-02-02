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

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import SystemBootstrap for centralized system initialization
from system.bootstrap import SystemBootstrap

# Import MarketDataManager for centralized market data handling
from system.market_data_manager import MarketDataManager

# Import TradingPipelineManager for trading cycle orchestration
from system.trading_pipeline_manager import TradingPipelineManager

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

# 🧹 SYSTEM CLEANUP
try:
    from system_cleanup import SystemCleanup
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
        # STEP 1: SYSTEM CLEANUP
        # ================================================================
        logger.info("🧹 Running system cleanup...")
        if CLEANUP_AVAILABLE:
            try:
                cleanup = SystemCleanup()
                cleanup_result = cleanup.perform_full_cleanup()
                
                if cleanup_result.get("success", False):
                    logger.info(f"✅ Cleanup: {cleanup_result.get('deleted_files', 0)} files removed")
                else:
                    logger.warning("⚠️ Cleanup completed with warnings")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Cleanup failed: {cleanup_error}")
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
        state_coordinator = StateCoordinator()
        
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
            
            # IMPORTANT: Re-inject StateCoordinator from bootstrap if different
            if hasattr(system_context, 'state_coordinator') and system_context.state_coordinator is not None:
                state_coordinator = system_context.state_coordinator
                inject_state_coordinator(state_coordinator)
                logger.info("✅ StateCoordinator from bootstrap re-injected")
            
            logger.info("✅ SystemBootstrap completed")
            
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
        # STEP 4: ENSURE CRITICAL COMPONENTS EXIST
        # ================================================================
        logger.info("🔧 Verifying critical components...")
        
        # Portfolio Manager - FIX ASYNC ISSUE
        if 'portfolio_manager' not in components or components.get('portfolio_manager') is None:
            from core.portfolio_manager import PortfolioManager
            from l1_operational.binance_client import BinanceClient
            
            # ✅ CRITICAL FIX: Create BinanceClient for paper trading with testnet
            binance_client = BinanceClient()
            
            # ✅ Create PortfolioManager with BinanceClient in simulated mode
            portfolio_manager = PortfolioManager(client=binance_client, mode="simulated")
            
            # ✅ If PortfolioManager has async initialization, call it
            if hasattr(portfolio_manager, 'initialize_async'):
                try:
                    await portfolio_manager.initialize_async()
                    logger.info("✅ PortfolioManager initialized asynchronously")
                except Exception as async_init_error:
                    logger.warning(f"⚠️ Async initialization failed: {async_init_error}")
            
            components['portfolio_manager'] = portfolio_manager
            logger.info("✅ PortfolioManager created with PaperExchangeAdapter")
        else:
            portfolio_manager = components['portfolio_manager']
            logger.info("✅ PortfolioManager from bootstrap")
            
        # Order Manager
        if 'order_manager' not in components or components.get('order_manager') is None:
            from l1_operational.order_manager import OrderManager
            order_manager = OrderManager(state_coordinator, portfolio_manager, config)
            components['order_manager'] = order_manager
            logger.info("✅ OrderManager created manually")
        else:
            order_manager = components['order_manager']
            logger.info("✅ OrderManager from bootstrap")
            
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
        
        position_rotator = PositionRotator(portfolio_manager)
        auto_rebalancer = AutoRebalancer(portfolio_manager)
        logger.info("✅ Position managers ready")

        # ================================================================
        # STEP 9: INITIALIZE TRADING PIPELINE MANAGER
        # ================================================================
        logger.info("🔧 Initializing TradingPipelineManager...")
        trading_pipeline = TradingPipelineManager(
            portfolio_manager=portfolio_manager,
            order_manager=order_manager,
            l2_processor=l2_processor,
            position_rotator=position_rotator,
            auto_rebalancer=auto_rebalancer,
            signal_verifier=signal_verifier,
            state_coordinator=state_coordinator,
            config=config
        )
        logger.info("✅ TradingPipelineManager ready")

        # ================================================================
        # STEP 10: INITIALIZE ERROR RECOVERY
        # ================================================================
        logger.info("🔧 Initializing ErrorRecoveryManager...")
        error_recovery = ErrorRecoveryManager()
        logger.info("✅ ErrorRecoveryManager ready")

        # ================================================================
        # STEP 11: INITIALIZE MARKET DATA MANAGER - CRITICAL FIX
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
    
        # ================================================================
        # STEP 12: VERIFY INITIAL PORTFOLIO STATE
        # ================================================================
        logger.info("🔍 INITIAL PORTFOLIO STATE:")
        try:
            btc_bal = portfolio_manager.get_balance('BTCUSDT')
            eth_bal = portfolio_manager.get_balance('ETHUSDT')
            usdt_bal = portfolio_manager.get_balance('USDT')
            total_val = portfolio_manager.get_total_value()
        
            logger.info(f"   BTC: {btc_bal:.6f}")
            logger.info(f"   ETH: {eth_bal:.3f}")
            logger.info(f"   USDT: ${usdt_bal:.2f}")
            logger.info(f"   Total: ${total_val:.2f}")
        
            # Store initial portfolio values for comparison
            initial_portfolio = {
                'btc_balance': btc_bal,
                'eth_balance': eth_bal,
                'usdt_balance': usdt_bal,
                'total_value': total_val
            }
            logger.info("✅ Initial portfolio values stored for comparison")
        except Exception as portfolio_error:
            logger.error(f"❌ Error reading portfolio: {portfolio_error}")
            initial_portfolio = None

        # ================================================================
        # STEP 13: GET INITIAL MARKET DATA
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
        # STEP 14: INITIAL DEPLOYMENT
        # ================================================================
        if validate_market_data(market_data):
            logger.info("🔄 Calculating initial deployment...")
            try:
                orders = position_rotator.calculate_initial_deployment(market_data)
                
                if orders:
                    processed_orders = await order_manager.execute_orders(orders)
                    await portfolio_manager.update_from_orders_async(processed_orders, market_data)
                    logger.info(f"✅ Initial deployment: {len(processed_orders)} orders")
                else:
                    logger.info("⚠️ No initial deployment needed")
            except Exception as deploy_error:
                logger.error(f"❌ Initial deployment failed: {deploy_error}")
        else:
            logger.warning("⚠️ Skipping initial deployment - invalid market data")

        # ================================================================
        # STEP 15: INTEGRATE AUTO-LEARNING
        # ================================================================
        logger.info("🤖 Integrating Auto-Learning System...")
        try:
            auto_learning_system = integrate_with_main_system()
            logger.info("✅ Auto-Learning System integrated")
        except Exception as learning_error:
            logger.error(f"❌ Auto-Learning integration failed: {learning_error}")

        # ================================================================
        # STEP 16: MAIN TRADING LOOP
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
                    market_data=market_data  # ✅ CRITICAL: Pass market_data explicitly
                )
                
                # Update state from cycle result
                state["total_value"] = cycle_result.portfolio_value
                
                # Update cumulative counters
                total_signals_all_cycles += cycle_result.signals_generated
                total_orders_all_cycles += cycle_result.orders_executed
                total_rejected_all_cycles += cycle_result.orders_rejected
                total_cooldown_blocked_all_cycles += cycle_result.cooldown_blocked
                
                # ============================================================
                # CYCLE STEP 3: LOG SUMMARY (every 5 cycles)
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
                # CYCLE STEP 4: PORTFOLIO COMPARISON LOG (every cycle)
                # ============================================================
                await log_portfolio_comparison(
                    portfolio_manager=portfolio_manager,
                    initial_portfolio=initial_portfolio,
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

        # ================================================================
        # STEP 13: INITIAL DEPLOYMENT
        # ================================================================
        if validate_market_data(market_data):
            logger.info("🔄 Calculating initial deployment...")
            try:
                orders = position_rotator.calculate_initial_deployment(market_data)
                
                if orders:
                    processed_orders = await order_manager.execute_orders(orders)
                    await portfolio_manager.update_from_orders_async(processed_orders, market_data)
                    logger.info(f"✅ Initial deployment: {len(processed_orders)} orders")
                else:
                    logger.info("⚠️ No initial deployment needed")
            except Exception as deploy_error:
                logger.error(f"❌ Initial deployment failed: {deploy_error}")
        else:
            logger.warning("⚠️ Skipping initial deployment - invalid market data")

        # ================================================================
        # STEP 14: INTEGRATE AUTO-LEARNING
        # ================================================================
        logger.info("🤖 Integrating Auto-Learning System...")
        try:
            auto_learning_system = integrate_with_main_system()
            logger.info("✅ Auto-Learning System integrated")
        except Exception as learning_error:
            logger.error(f"❌ Auto-Learning integration failed: {learning_error}")

        # ================================================================
        # STEP 15: APPLY FUNDAMENTAL RULE - CRITICAL FIX
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
        # STEP 16: MAIN TRADING LOOP
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
                    market_data=market_data  # ✅ CRITICAL: Pass market_data explicitly
                )
                
                # Update state from cycle result
                state["total_value"] = cycle_result.portfolio_value
                
                # Update cumulative counters
                total_signals_all_cycles += cycle_result.signals_generated
                total_orders_all_cycles += cycle_result.orders_executed
                total_rejected_all_cycles += cycle_result.orders_rejected
                total_cooldown_blocked_all_cycles += cycle_result.cooldown_blocked
                
                # ============================================================
                # CYCLE STEP 3: LOG SUMMARY (every 5 cycles)
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
                # CYCLE STEP 4: PORTFOLIO COMPARISON LOG (every cycle)
                # ============================================================
                await log_portfolio_comparison(
                    portfolio_manager=portfolio_manager,
                    initial_portfolio=initial_portfolio,
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
        # Cleanup
        logger.info("🧹 Cleaning up...")
        try:
            await stop_signal_verification()
            logger.info("✓ Signal verification stopped")
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Cleanup error: {cleanup_error}")
        
        logger.info("👋 HRM System shutdown complete")

async def log_portfolio_comparison(portfolio_manager, initial_portfolio, cycle_id, market_data):
    """Log portfolio comparison with colored output every cycle"""
    try:
        if initial_portfolio is None:
            logger.warning("⚠️ Initial portfolio not available for comparison")
            return
        
        # Get current portfolio values
        current_btc = portfolio_manager.get_balance('BTCUSDT')
        current_eth = portfolio_manager.get_balance('ETHUSDT')
        current_usdt = portfolio_manager.get_balance('USDT')
        current_total = portfolio_manager.get_total_value(market_data)
        
        # Calculate differences
        btc_diff = current_btc - initial_portfolio['btc_balance']
        eth_diff = current_eth - initial_portfolio['eth_balance']
        usdt_diff = current_usdt - initial_portfolio['usdt_balance']
        total_diff = current_total - initial_portfolio['total_value']
        
        # Determine color based on total portfolio performance
        if total_diff >= 0:
            # Portfolio value increased or stayed the same - GREEN
            color_start = Fore.GREEN
            color_end = Style.RESET_ALL
            status = "PROFIT"
        else:
            # Portfolio value decreased - RED
            color_start = Fore.RED
            color_end = Style.RESET_ALL
            status = "LOSS"
        
        # Create 80-character colored border
        border = color_start + "=" * 80 + color_end
        
        # Log the comparison with colored borders
        print(border)  # Print border above
        logger.info(f"{color_start}💰 PORTFOLIO COMPARISON - Cycle {cycle_id} ({status}){color_end}")
        logger.info(f"{color_start}   BTC: {current_btc:.6f} ({'+' if btc_diff >= 0 else ''}{btc_diff:.6f}){color_end}")
        logger.info(f"{color_start}   ETH: {current_eth:.3f} ({'+' if eth_diff >= 0 else ''}{eth_diff:.3f}){color_end}")
        logger.info(f"{color_start}   USDT: ${current_usdt:.2f} ({'+' if usdt_diff >= 0 else ''}${usdt_diff:.2f}){color_end}")
        logger.info(f"{color_start}   TOTAL: ${current_total:.2f} ({'+' if total_diff >= 0 else ''}${total_diff:.2f}){color_end}")
        print(border)  # Print border below
        
    except Exception as e:
        logger.error(f"❌ Error in portfolio comparison: {e}")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
