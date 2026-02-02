#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Bootstrap Module

This module handles system initialization, configuration loading,
and component wiring for the HRM system.
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
from core.config import get_config
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
from system.orchestrator import HRMOrchestrator

from system_cleanup import SystemCleanup
from storage.paper_trade_logger import get_paper_logger


class HRMBootstrap:
    """HRM system bootstrap and initialization."""
    
    def __init__(self):
        self.components = {}
        self.state = None
        self.portfolio_manager = None
        self.order_manager = None
        self.runtime_loop = None
        
    async def bootstrap_system(self, mode: str = "live") -> Tuple[PortfolioManager, OrderManager]:
        """Bootstrap the entire HRM system."""
        logger.info("🚀 Starting HRM System Bootstrap")
        
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
            
            logger.info("✅ HRM System Bootstrap Complete")
            return self.portfolio_manager, self.order_manager
            
        except Exception as e:
            logger.error(f"❌ Bootstrap failed: {e}", exc_info=True)
            await self._cleanup_on_failure()
            raise
    
    async def _perform_system_cleanup(self):
        """Perform system cleanup before startup."""
        logger.info("🧹 Running system cleanup...")
        
        cleanup = SystemCleanup()
        cleanup_result = cleanup.perform_full_cleanup()
        
        if not cleanup_result.get("success", False):
            logger.warning("⚠️ Cleanup completed with warnings")
        else:
            logger.info(f"✅ Cleanup completed: {cleanup_result.get('deleted_files', 0)} files, {cleanup_result.get('deleted_dirs', 0)} directories removed")
        
        # Clean paper trades for independent testing
        try:
            logger.info("🧹 Cleaning paper trades history...")
            get_paper_logger(clear_on_init=True)
            logger.info("✅ Paper trades history cleaned")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning paper trades: {e}")
    
    async def _load_configuration(self, mode: str) -> Dict[str, Any]:
        """Load system configuration."""
        logger.info(f"⚙️ Loading configuration for mode: {mode}")
        
        # Load environment variables
        load_dotenv()
        
        # Get environment configuration
        env_config = get_config(mode)
        
        # Check Binance operating mode
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        logger.info(f"🏦 BINANCE MODE: {binance_mode}")
        
        return env_config
    
    async def _initialize_state(self, env_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize system state."""
        logger.info("🧠 Initializing system state...")
        
        # Initialize state with symbols and initial balance
        symbols = env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])
        initial_balance = 3000.0
        
        state = initialize_state(symbols, initial_balance)
        state = validate_state_structure(state)
        
        logger.info(f"✅ State initialized for symbols: {symbols}")
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
        
        # Initialize Binance client
        binance_client = BinanceClient()
        self.components['binance_client'] = binance_client
        
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
        if binance_mode == "LIVE":
            # Live mode: sync mandatory with exchange
            portfolio_mode = "live"
            initial_balance = 0.0  # Will be synced from exchange
        else:
            # Test mode: use simulated balance
            portfolio_mode = "simulated"
            initial_balance = 3000.0
            logger.info(f"🧪 TESTING MODE: Using initial balance of {initial_balance} USDT")
        
        # Initialize Portfolio Manager
        portfolio_manager = PortfolioManager(
            mode=portfolio_mode,
            initial_balance=initial_balance,
            client=self.components['binance_client'],
            symbols=env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"]),
            enable_commissions=env_config.get("ENABLE_COMMISSIONS", True),
            enable_slippage=env_config.get("ENABLE_SLIPPAGE", True)
        )
        
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
        
        order_manager = OrderManager(
            binance_client=self.components['binance_client'],
            market_data=self.state.get("market_data", {}),
            portfolio_manager=self.portfolio_manager
        )
        
        # Clean up stale orders
        logger.info("🧹 Cleaning up stale orders...")
        try:
            current_positions = {}
            for symbol in config["SYMBOLS"]:
                if symbol != "USDT":
                    current_positions[symbol] = self.portfolio_manager.get_balance(symbol)
            
            cleanup_stats = order_manager.cleanup_stale_orders(current_positions)
            logger.info(f"🧹 Cleanup completed: {cleanup_stats}")
        except Exception as e:
            logger.error(f"❌ Error during order cleanup: {e}")
        
        logger.info("✅ Order Manager initialized")
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


async def bootstrap_hrm_system(mode: str = "live") -> Tuple[PortfolioManager, OrderManager, HRMRuntimeLoop]:
    """Bootstrap the HRM system and return core components."""
    bootstrap = HRMBootstrap()
    portfolio_manager, order_manager = await bootstrap.bootstrap_system(mode)
    runtime_loop = bootstrap.get_runtime_loop()
    
    return portfolio_manager, order_manager, runtime_loop