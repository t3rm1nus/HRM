# -*- coding: utf-8 -*-
# main.py

import asyncio
import sys
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict

# Suppress TensorFlow warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.state_manager import initialize_state, validate_state_structure, log_cycle_data
from core.portfolio_manager import update_portfolio_from_orders, save_portfolio_to_csv, PortfolioManager
from core.technical_indicators import calculate_technical_indicators
from core.feature_engineering import integrate_features_with_l2, debug_l2_features
from core.logging import logger
from core.config import get_config
from core.unified_validation import UnifiedValidator, validate_market_data_structure, validate_and_fix_market_data
from core.error_handler import ErrorHandler, async_with_fallback
from core.incremental_signal_verifier import get_signal_verifier, start_signal_verification, stop_signal_verification
from core.trading_metrics import get_trading_metrics
from core.position_rotator import PositionRotator

from l1_operational.data_feed import DataFeed
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.models import L2State, TacticalSignal
from l2_tactic.risk_controls.manager import RiskControlManager
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l3_strategy.regime_classifier import ejecutar_estrategia_por_regimen
from l3_strategy.decision_maker import make_decision
from l3_strategy.sentiment_inference import download_reddit, download_news, infer_sentiment
from l1_operational.bus_adapter import BusAdapterAsync

from comms.config import config, APAGAR_L3
from l2_tactic.config import L2Config
from comms.message_bus import MessageBus

def _extract_current_price_safely(symbol: str, market_data: Dict) -> float:
    """
    Safely extract current price from market data for validation and deployment.

    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        market_data: Market data dictionary

    Returns:
        Current price as float, or 0.0 if extraction fails
    """
    try:
        if not market_data or symbol not in market_data:
            return 0.0

        data = market_data[symbol]

        if isinstance(data, dict):
            if 'close' in data:
                return float(data.get('close', 0.0))
            elif 'price' in data:
                return float(data.get('price', 0.0))
        elif isinstance(data, pd.DataFrame):
            if 'close' in data.columns and len(data) > 0:
                return float(data['close'].iloc[-1])
        elif isinstance(data, pd.Series) and len(data) > 0:
            return float(data.iloc[-1])

        return 0.0
    except Exception as e:
        logger.error(f"❌ Error extracting price for {symbol}: {e}")
        return 0.0

def validate_market_data(market_data: Dict) -> bool:
    """Validate market data structure before deployment."""
    required_symbols = ['BTCUSDT', 'ETHUSDT']

    for symbol in required_symbols:
        if symbol not in market_data:
            logger.error(f"❌ Missing {symbol} in market_data")
            return False

        df = market_data[symbol]
        if isinstance(df, pd.DataFrame):
            if df.empty or 'close' not in df.columns:
                logger.error(f"❌ Invalid DataFrame for {symbol}")
                return False
            price = df['close'].iloc[-1]
        elif isinstance(df, dict):
            if 'close' not in df:
                logger.error(f"❌ Missing 'close' field in {symbol} dict")
                return False
            price = df['close']
        else:
            logger.error(f"❌ Unsupported data type for {symbol}: {type(df)}")
            return False

        if pd.isna(price) or price <= 0:
            logger.error(f"❌ Invalid price for {symbol}: {price}")
            return False

    return True

# 🔄 AUTO-LEARNING SYSTEM INTEGRATION
from integration_auto_learning import integrate_with_main_system

def should_execute_with_l3_dominance(l2_signal, l3_info):
    """
    Decide if L2 signal should execute based on L3 dominance logic.

    Args:
        l2_signal: Dict with L2 signal info (action, symbol, confidence, etc.)
        l3_info: Dict with L3 regime info (regime, signal, confidence, allow_l2)

    Returns:
        tuple: (should_execute: bool, reason: str)
    """
    # Extraer datos clave
    l3_regime = l3_info.get('regime', 'unknown')
    l3_signal = l3_info.get('signal', 'hold')
    l3_confidence = l3_info.get('confidence', 0.0)
    l3_allow_l2 = l3_info.get('allow_l2', True)

    l2_action = l2_signal.get('action', 'hold')
    l2_confidence = l2_signal.get('confidence', 0.5)

    # LOG DIAGNÓSTICO TEMPORAL
    logger.info("="*80)
    logger.info("SIGNAL EXECUTION DECISION")
    logger.info(f"L3 Info: {l3_info}")
    logger.info(f"L2 Signals: {l2_signal}")
    logger.info("="*80)

    # L3 tiene data válida?
    if l3_regime == 'ERROR' or l3_confidence <= 0.0:
        logger.warning(f"⚠️ L3 data invalid (regime={l3_regime}, confidence={l3_confidence}), allowing L2")
        return True, "L3 data invalid, permissive mode"

    # L3 permite explícitamente L2?
    if l3_allow_l2:
        return True, f"L3 allows L2 signals (allow_l2={l3_allow_l2})"

    # L3 con confianza alta en HOLD bloquea todo
    if l3_signal.lower() == 'hold' and l3_confidence > 0.50:
        reason = f"L3 HOLD with {l3_confidence:.2f} confidence blocks L2 {l2_action}"
        logger.warning(f"🚫 {reason}")
        return False, reason

    # L3 tiene señal diferente pero con confianza media, permitir con restricción
    if l3_confidence < 0.70:
        return True, f"L3 low confidence ({l3_confidence:.2f}), allowing L2"

    # Por defecto permitir si L3 no tiene confianza suficiente para dominar
    return True, "Default permissive - L3 confidence insufficient for dominance"

async def main():
    """Main HRM system function."""
    try:
        logger.info("🚀 Starting HRM system")

        # Check Binance operating mode
        binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
        logger.info(f"🏦 BINANCE MODE: {binance_mode}")

        state = initialize_state(config["SYMBOLS"], 3000.0)
        state = validate_state_structure(state)

        # Initialize components
        loader = RealTimeDataLoader(config)
        data_feed = DataFeed(config)
        bus_adapter = BusAdapterAsync(config, state)
        binance_client = BinanceClient()

        # Get environment configuration
        env_config = get_config("live")  # Use "live" mode for now

        # Setup based on binance_mode
        if binance_mode == "LIVE":
            # Live mode: sync mandatory with exchange
            portfolio_mode = "live"
            initial_balance = 0.0  # Will be synced from exchange
            test_initial_balance = 0.0
        else:
            # Test mode: use simulated balance
            portfolio_mode = "simulated"
            test_initial_balance = 3000.0
            initial_balance = test_initial_balance
            logger.info(f"🧪 TESTING MODE: Using initial balance of {test_initial_balance} USDT (overriding config)")

        # Initialize Portfolio Manager for persistence with new config system
        portfolio_manager = PortfolioManager(
            mode=portfolio_mode,
            initial_balance=initial_balance,
            client=binance_client,
            symbols=env_config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"]),
            enable_commissions=env_config.get("ENABLE_COMMISSIONS", True),
            enable_slippage=env_config.get("ENABLE_SLIPPAGE", True)
        )

        # Initialize L1 Models
        from l1_operational.trend_ai import models as l1_models
        logger.info(f"✅ Loaded L1 AI Models: {list(l1_models.keys())}")

        # Initialize L2
        try:
            # Quick fix: Disable synchronizer in PAPER mode for better performance
            binance_mode = os.getenv("BINANCE_MODE", "TEST").upper()
            if binance_mode != "LIVE":
                logger.info("📝 PAPER/TEST MODE: Disabling BTC/ETH synchronizer")
                os.environ['DISABLE_BTC_ETH_SYNC'] = 'true'

            l2_config = L2Config()
            l2_processor = L2TacticProcessor(l2_config, portfolio_manager=portfolio_manager, apagar_l3=APAGAR_L3)
            risk_manager = RiskControlManager(l2_config)
            logger.info("✅ L2 components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing L2 components: {e}", exc_info=True)
            raise

        # Initialize incremental signal verifier (don't start loop yet)
        signal_verifier = get_signal_verifier()

        # Initialize trading metrics for performance monitoring
        trading_metrics = get_trading_metrics()

    # CRÍTICO: SINCRONIZACIÓN CON EXCHANGE PARA MODO PRODUCCIÓN
        logger.info("🔄 Sincronizando con estado real de Binance...")
        sync_success = await portfolio_manager.sync_with_exchange()

        if sync_success:
            logger.info("✅ Portfolio sincronizado con exchange real")
            logger.info(f"   Estado actual: BTC={portfolio_manager.get_balance('BTCUSDT'):.6f}, ETH={portfolio_manager.get_balance('ETHUSDT'):.3f}, USDT={portfolio_manager.get_balance('USDT'):.2f}")
        else:
            # Fallback: cargar desde JSON local
            logger.warning("⚠️ Falló sincronización con exchange, cargando estado local...")
            loaded = portfolio_manager.load_from_json()

            if not loaded:
                logger.info("📄 No saved portfolio found, starting with clean state")
            else:
                logger.info(f"📂 Portfolio local cargado: BTC={portfolio_manager.get_balance('BTCUSDT'):.6f}, ETH={portfolio_manager.get_balance('ETHUSDT'):.3f}, USDT={portfolio_manager.get_balance('USDT'):.2f}")

                # FOR TESTING: Force clean reset to start with initial balance
                logger.info("🧪 TESTING MODE: Forcing clean portfolio reset for validation")
                portfolio_manager.force_clean_reset()
                logger.info("✅ Portfolio reset to clean state for testing")

        # ========================================================================================
        # L3 REGIME CLASSIFICATION WITH SETUP DETECTION
        # ========================================================================================
        from l3_strategy.regime_classifier import MarketRegimeClassifier

        classifier = MarketRegimeClassifier()
        regime_result = classifier.classify_market_regime(state.get("market_data", {}).get("BTCUSDT", pd.DataFrame()), "BTCUSDT")

        logger.info(f"🧠 L3 Regime: {regime_result['primary_regime']} | Subtype: {regime_result['subtype']} | Confidence: {regime_result['confidence']:.2f}")

        # Check for setup detection
        setup_detected = regime_result['subtype'] in ['OVERSOLD_SETUP', 'OVERBOUGHT_SETUP']
        setup_type = regime_result['subtype'].lower() if setup_detected else None

        if setup_detected:
            logger.info(f"🎯 TRADING SETUP DETECTED: {regime_result['subtype']}")

        # ========================================================================================
        # GENERATE L3 STRATEGY DECISION
        # ========================================================================================
        from l3_strategy.regime_classifier import _generate_strategy_from_regime

        regime_strategy = _generate_strategy_from_regime(regime_result)

        # DEBUG: Log regime strategy content
        logger.info("-------------------------------------------------------------------------------------------------------------")
        logger.info("🎯 REGIME STRATEGY DETAILS:")
        logger.info(f"   Regime: {regime_strategy.get('regime')}")
        logger.info(f"   Subtype: {regime_strategy.get('subtype')}")
        logger.info(f"   Setup Type: {regime_strategy.get('setup_type')}")
        logger.info(f"   Signal: {regime_strategy.get('signal')}")
        logger.info(f"   Allow L2: {regime_strategy.get('allow_l2_signal')}")
        logger.info("-------------------------------------------------------------------------------------------------------------")

        # ========================================================================================
        # L3 DECISION MAKER WITH SETUP HANDLING
        # ========================================================================================
        from l3_strategy.decision_maker import make_decision

        l3_decision = make_decision(
            inputs={},
            portfolio_state=portfolio_manager.get_portfolio_state(),
            market_data=state.get("market_data", {}),
            regime_decision=regime_strategy
        )

        # 🔒 SINGLE SOURCE OF TRUTH: PortfolioManager is the authoritative source
        # Remove duplicate sync calls - let PortfolioManager manage its own state

        # 🧹 CLEANUP STALE ORDERS AFTER PORTFOLIO SYNC
        logger.info("🧹 Performing startup cleanup of stale stop-loss and profit-taking orders...")
        try:
            # Get current portfolio positions for cleanup
            current_positions = {}
            for symbol in config["SYMBOLS"]:
                if symbol != "USDT":
                    current_positions[symbol] = portfolio_manager.get_balance(symbol)

            # Initialize order manager first if not already done
            order_manager = OrderManager(binance_client=binance_client, market_data=state.get("market_data", {}))

            # Perform cleanup
            cleanup_stats = order_manager.cleanup_stale_orders(current_positions)
            logger.info(f"🧹 Startup cleanup completed: {cleanup_stats}")

        except Exception as cleanup_error:
            logger.error(f"❌ Error during startup cleanup: {cleanup_error}")
            # Continue with system startup even if cleanup fails
            order_manager = OrderManager(binance_client=binance_client, market_data=state.get("market_data", {}))

        # Initialize L3 (will be called after market data is loaded)

        order_manager = OrderManager(binance_client=binance_client, market_data=state.get("market_data", {}))

        # Get initial data with centralized error handling and fallback
        logger.info("🔄 Loading initial market data...")
        initial_data, load_success = await ErrorHandler.load_market_data_with_fallback(
            loader, data_feed, "initial_market_data_loading"
        )

        if not load_success or not initial_data:
            raise RuntimeError("Could not get initial market data after retries and fallback attempts")

        state["market_data"] = initial_data

        
        # Validate required data is present
        required_symbols = config["SYMBOLS"]
        missing_symbols = [sym for sym in required_symbols if sym not in initial_data]
        if missing_symbols:
            logger.warning(f"⚠️ Missing data for symbols: {missing_symbols}")

        # Sentiment analysis state and function
        sentiment_texts_cache = []
        last_sentiment_update = 0
        SENTIMENT_UPDATE_INTERVAL = 2160  # Update sentiment every 2160 cycles (~6 hours, aligned with BERT cache expiration)

        async def update_sentiment_texts():
            """Update sentiment texts from Reddit and News API"""
            try:
                logger.info("🔄 SENTIMENT: Iniciando descarga de datos de sentimiento...")
                # Download Reddit data
                df_reddit = await download_reddit()
                logger.info(f"✅ SENTIMENT: Descargados {len(df_reddit)} posts de Reddit")

                # Download news data
                df_news = download_news()
                logger.info(f"✅ SENTIMENT: Descargados {len(df_news)} artículos de noticias")

                # Combine all texts
                df_all = pd.concat([df_reddit, df_news], ignore_index=True)
                df_all.dropna(subset=['text'], inplace=True)

                if df_all.empty:
                    logger.warning("⚠️ SENTIMENT: No se obtuvieron textos válidos")
                    return []

                texts_list = df_all['text'].tolist()
                logger.info(f"📊 SENTIMENT: {len(texts_list)} textos recolectados para análisis")

                # Perform sentiment inference on the texts
                sentiment_results = infer_sentiment(texts_list)
                logger.info(f"🧠 SENTIMENT: Análisis completado para {len(sentiment_results)} textos")

                # Return the texts that were analyzed
                return texts_list

            except Exception as e:
                logger.error(f"❌ SENTIMENT: Error actualizando datos de sentimiento: {e}")
                return []

        # Initialize L3 now that we have market data
        if APAGAR_L3:
            # L3 is disabled - skip initialization and set default state
            logger.info("-------------------------------------------------------------------------------------------------------------")
            logger.info("\x1b[31m🔴 L3 MODULE DISABLED - SKIPPING L3 INITIALIZATION - ONLY L1+L2 WILL OPERATE\x1b[0m")
            logger.info("-------------------------------------------------------------------------------------------------------------")
            sentiment_texts_cache = []
            state["l3_output"] = {
                'regime': 'disabled',
                'signal': 'hold',
                'confidence': 0.0,
                'strategy_type': 'l3_disabled',
                'timestamp': pd.Timestamp.utcnow().isoformat()
            }
        else:
            # L3 is enabled - run normal initialization
            logger.info("-------------------------------------------------------------------------------------------------------------")
            logger.info("\x1b[32m🟢 L3 MODULE ENABLED - FULL SYSTEM OPERATING WITH L1+L2+L3\x1b[0m")
            logger.info("-------------------------------------------------------------------------------------------------------------")
            try:
                # Import required function (moved from top-level for import consistency)
                from l3_strategy.l3_processor import generate_l3_output

                # Get initial sentiment data for L3
                initial_sentiment_texts = await update_sentiment_texts()
                sentiment_texts_cache = initial_sentiment_texts

                l3_output = generate_l3_output(state, texts_for_sentiment=initial_sentiment_texts)  # Generate initial L3 output with market data and sentiment
                state["l3_output"] = l3_output  # Store L3 output in state for L2 access

                logger.debug("✅ L3 initialized successfully")

                logger.info("✅ L3 initialized successfully with market data and sentiment analysis")
            except Exception as e:
                logger.error(f"❌ Error initializing L3: {e}", exc_info=True)

        # CRITICAL FIX: Ensure portfolio is properly initialized and logged
        logger.info("🔍 INITIAL PORTFOLIO STATE:")
        logger.info(f"   BTC Position: {portfolio_manager.get_balance('BTCUSDT'):.6f}")
        logger.info(f"   ETH Position: {portfolio_manager.get_balance('ETHUSDT'):.3f}")
        logger.info(f"   USDT Balance: {portfolio_manager.get_balance('USDT'):.2f}")
        logger.info(f"   Total Value: {portfolio_manager.get_total_value():.2f}")

        # 🔄 SOLUTION 1: Corregir Timing del Initial Deployment - Cargar datos de mercado PRIMERO
        logger.info("🔄 Loading market data for PositionRotator initialization...")
        market_data = state.get("market_data", {})
        if market_data:
            logger.info(f"✅ Market data loaded: BTC=${market_data.get('BTCUSDT', {}).get('close', 'N/A') if isinstance(market_data.get('BTCUSDT'), dict) else _extract_current_price_safely('BTCUSDT', market_data)}")
        else:
            logger.error("❌ Failed to load market data for PositionRotator")
            market_data = {}

        # 🔄 SOLUTION 3: Validación de Estructura de Datos - antes del deployment
        if validate_market_data(market_data):
            # 2. LUEGO calcular deployment con datos disponibles
            position_rotator = PositionRotator(portfolio_manager)
            orders = position_rotator.calculate_initial_deployment(market_data)

            # 3. Ejecutar órdenes
            if orders:
                processed_orders = await order_manager.execute_orders(orders)
                # Update portfolio after initial deployment
                await portfolio_manager.update_from_orders_async(processed_orders, market_data)
                logger.info(f"✅ Initial deployment executed with {len(processed_orders)} orders")
            else:
                logger.info("⚠️ No initial deployment orders generated")
                position_rotator = PositionRotator(portfolio_manager)  # Initialize for later use
        else:
            logger.error("🚨 Cannot deploy - invalid market data")
            position_rotator = PositionRotator(portfolio_manager)  # Initialize without deployment

        # 🔄 INTEGRATE AUTO-LEARNING SYSTEM
        logger.info("🤖 Integrating Auto-Learning System...")
        auto_learning_system = integrate_with_main_system()
        logger.info("✅ Auto-Learning System integrated - Models will improve automatically!")

        # Main loop
        cycle_id = 0

        # Cumulative counters for all cycles
        total_signals_all_cycles = 0
        total_orders_all_cycles = 0
        total_rejected_all_cycles = 0
        total_cooldown_blocked_all_cycles = 0

        while True:
            cycle_id += 1
            start_time = pd.Timestamp.utcnow()

            # Initialize total_value to prevent reference before assignment errors
            try:
                total_value = portfolio_manager.get_total_value(state.get("market_data", {}))
            except Exception as tv_error:
                logger.warning(f"Failed to get total_value, using default: {tv_error}")
                total_value = portfolio_manager.get_total_value({})  # Default calculation with empty market data

            try:
                # Reset cooldown counter at start of each cycle
                order_manager.cooldown_blocked_count = 0

                # 1. Update market data with centralized validation
                logger.info("🔄 Attempting to get realtime market data...")
                market_data = await loader.get_realtime_data()
                logger.info(f"📊 Realtime data result: type={type(market_data)}, keys={list(market_data.keys()) if isinstance(market_data, dict) else 'N/A'}")

                is_valid, validation_msg = UnifiedValidator.validate_market_data_structure(market_data)

                if not is_valid:
                    logger.warning(f"⚠️ Failed to get realtime data: {validation_msg}, falling back to data feed")
                    logger.info("🔄 Falling back to data_feed.get_market_data()...")
                    market_data = await data_feed.get_market_data()
                    logger.info(f"📊 Data feed result: type={type(market_data)}, keys={list(market_data.keys()) if isinstance(market_data, dict) else 'N/A'}")

                    is_valid, validation_msg = UnifiedValidator.validate_market_data_structure(market_data)
                    market_data = {} if not is_valid else market_data
                
                if is_valid:
                    # Further validate required symbols with thorough type checking
                    if not isinstance(market_data, dict):
                        logger.error(f"❌ Invalid market_data type during validation: {type(market_data)}")
                        raise ValueError(f"Invalid market_data type: {type(market_data)}")
                        
                    missing_symbols = [sym for sym in config["SYMBOLS"] if sym not in market_data]
                    valid_data = {}
                    
                    for sym, data in market_data.items():
                        if sym not in config["SYMBOLS"]:
                            continue
                            
                        if not isinstance(data, (pd.DataFrame, dict)):
                            logger.warning(f"⚠️ Invalid data type for {sym}: {type(data)}")
                            continue
                            
                        valid_data[sym] = data
                    
                    if missing_symbols:
                        logger.error(f"❌ Missing required symbols: {missing_symbols}")
                        
                    if valid_data:
                        # Update state with validated data (single source)
                        state["market_data"] = valid_data
                        logger.info(f"✅ Market data updated for symbols: {list(valid_data.keys())}")
                        
                        if missing_symbols:
                            logger.warning("⚠️ Some symbols missing, continuing with partial data")
                    else:
                        raise ValueError("No valid market data for required symbols")
                else:
                    raise RuntimeError(f"❌ Failed to get valid market data: {validation_msg}")
                
                # 🛡️ 2. MONITOR STOP-LOSS ORDERS ACTIVOS WITH VALIDATION
                try:
                    # Get current portfolio positions for stop-loss validation
                    current_positions = {}
                    for symbol in config["SYMBOLS"]:
                        if symbol != "USDT":
                            current_positions[symbol] = portfolio_manager.get_balance(symbol)

                    # Monitorear y ejecutar stop-loss que se activen con validación de portfolio
                    executed_stop_losses = await order_manager.monitor_and_execute_stop_losses_with_validation(
                        state.get("market_data", {}), current_positions
                    )

                    if executed_stop_losses:
                        # Actualizar portfolio con stop-loss ejecutados
                        await portfolio_manager.update_from_orders_async(executed_stop_losses, state.get("market_data", {}))
                        logger.info(f"🛡️ Ejecutados {len(executed_stop_losses)} stop-loss automáticos con validación")

                except Exception as e:
                    logger.error(f"❌ Error monitoreando stop-loss con validación: {e}")

                # 3. Update L3 state and process signals
                # ========================================================================================
                # L3 REGIME INFO RETRIEVAL - ALWAYS FRESH, NO CACHE
                # ========================================================================================

                # 🔧 SOLUCIÓN INMEDIATA: Completamente sin cache - siempre calcular fresco
                try:
                    from l3_strategy.decision_maker import get_regime_info

                    # Siempre obtener info fresca - NUNCA usar cache
                    l3_info_fresh = get_regime_info(state.get("market_data", {}))

                    # Preparar contexto para L2 con datos completamente frescos
                    l3_regime_info = {
                        'regime': l3_info_fresh.get('regime', 'unknown'),
                        'subtype': l3_info_fresh.get('subtype', 'unknown'),
                        'confidence': l3_info_fresh.get('confidence', 0.5),
                        'signal': l3_info_fresh.get('signal', 'hold'),
                        'allow_l2': l3_info_fresh.get('allow_l2_signal', True),
                        'l3_output': l3_info_fresh
                    }

                except Exception as e:
                    logger.warning(f"❌ Error obteniendo L3 regime info fresco: {e}")
                    l3_regime_info = {
                        'regime': 'unknown',
                        'subtype': 'unknown',
                        'confidence': 0.0,
                        'signal': 'hold',
                        'allow_l2': True
                    }

                # LOG VALIDATION
                logger.info(f"✅ L3 Regime Info obtained: {l3_regime_info.get('regime', 'unknown')} (allow_l2: {l3_regime_info.get('allow_l2', True)})")

                # VALIDAR que tiene datos (ahora siempre tiene datos válidos ya que usamos fallback)
                # No es necesario validación adicional aquí ya que siempre retornamos un dict válido

                # 🚨 SIGNAL HIERARCHY LOGIC: L3 HOLD signals with confidence > 40% block L2 BUY/SELL signals
                l3_signal_blocks_l2 = False
                l3_blocking_info = ""

                if APAGAR_L3:
                    # L3 is disabled - bypass all L3 processing
                    logger.info("-------------------------------------------------------------------------------------------------------------")
                    logger.info("\x1b[31m🔴 L3 MODULE DISABLED - ONLY L1+L2 WILL OPERATE\x1b[0m")
                    logger.info("-------------------------------------------------------------------------------------------------------------")
                    # Ensure L3 output is set to a default state for L2 compatibility
                    if 'l3_context_cache' not in state:
                        state['l3_context_cache'] = {}
                    state['l3_context_cache']['last_output'] = {
                        'regime': 'disabled',
                        'signal': 'hold',
                        'confidence': 0.0,
                        'strategy_type': 'l3_disabled',
                        'timestamp': pd.Timestamp.utcnow().isoformat()
                    }
                else:
                    # L3 is enabled - run normal L3 processing
                    logger.info("-------------------------------------------------------------------------------------------------------------")
                    logger.info("\x1b[32m🟢 L3 MODULE ENABLED - FULL SYSTEM OPERATING WITH L1+L2+L3\x1b[0m")
                    logger.info("-------------------------------------------------------------------------------------------------------------")
                    try:
                        # Update sentiment data periodically (every 50 cycles ~8-9 minutes)
                        if cycle_id - last_sentiment_update >= SENTIMENT_UPDATE_INTERVAL:
                            logger.info(f"🔄 SENTIMENT: Actualización periódica iniciada (ciclo {cycle_id}, cada {SENTIMENT_UPDATE_INTERVAL} ciclos)")
                            sentiment_texts_cache = await update_sentiment_texts()
                            last_sentiment_update = cycle_id
                            logger.info(f"💬 SENTIMENT: Cache actualizado con {len(sentiment_texts_cache)} textos para análisis L3")

                        # Use cached sentiment texts for L3 processing
                        current_sentiment_texts = sentiment_texts_cache if sentiment_texts_cache else []

                        # 🚀 INTEGRACIÓN DE ESTRATEGIAS POR RÉGIMEN DE MERCADO
                        # Get regime-specific decision as the foundation
                        regimen_resultado = ejecutar_estrategia_por_regimen(state.get("market_data", {}))

                        # Always get fresh sentiment for L3 output consistency
                        from l3_strategy.sentiment_inference import get_cached_sentiment_score
                        # Get fresh sentiment score (reduced cache time for better sync)
                        sentiment_score = get_cached_sentiment_score(max_age_hours=0.5)  # 30 minutes max age
                        if sentiment_score is None:
                            # Fallback to infer if cache miss
                            from l3_strategy.l3_processor import predict_sentiment, load_sentiment_model
                            tokenizer, sentiment_model = load_sentiment_model()
                            if current_sentiment_texts:
                                sentiment_score = predict_sentiment(current_sentiment_texts, tokenizer, sentiment_model)
                            else:
                                sentiment_score = 0.5  # Neutral default
                        logger.info(f"🧠 L3 Sentiment Score: {sentiment_score:.4f}")

                        # 🎯 PRIORITY DECISION MAKING: Regime-specific models take precedence
                        if regimen_resultado and 'regime' in regimen_resultado:
                            # Use DecisionMaker with regime-specific priority
                            portfolio_state = state.get("portfolio", {})

                            # Call DecisionMaker with regime_decision to prioritize regime-specific logic
                            strategic_decision = make_decision(
                                inputs={},  # Not using fallback L3 inputs anymore
                                portfolio_state=portfolio_state,
                                market_data=state.get("market_data", {}),
                                regime_decision=regimen_resultado  # PRIORITY: Regime decision drives the output
                            )

                            # Create L3 output from strategic decision with regime-specific details
                            l3_output = {
                                'regime': regimen_resultado['regime'],
                                'signal': regimen_resultado.get('signal', 'hold'),
                                'confidence': regimen_resultado.get('confidence', 0.5),
                                'strategy_type': 'regime_adaptive_priority',
                                'sentiment_score': sentiment_score,
                                'asset_allocation': strategic_decision.get('asset_allocation', {}),
                                'risk_appetite': strategic_decision.get('risk_appetite', 'moderate'),
                                'loss_prevention_filters': strategic_decision.get('loss_prevention_filters', {}),
                                'winning_trade_rules': strategic_decision.get('winning_trade_rules', {}),
                                'exposure_decisions': strategic_decision.get('exposure_decisions', {}),
                                'strategic_guidelines': strategic_decision.get('strategic_guidelines', {}),
                                'market_data_hash': hash(str(state.get("market_data", {}))),
                                'timestamp': pd.Timestamp.utcnow().isoformat()
                            }

                            # Añadir datos específicos del régimen if available
                            if regimen_resultado['regime'] == 'range':
                                l3_output.update({
                                    'profit_target': regimen_resultado.get('profit_target', 0.008),
                                    'stop_loss': regimen_resultado.get('stop_loss', 0.015),
                                    'max_position_time': regimen_resultado.get('max_position_time', 6)
                                })
                            elif regimen_resultado['regime'] in ['bull', 'bear']:
                                l3_output.update({
                                    'profit_target': regimen_resultado.get('profit_target', 0.025),
                                    'stop_loss': regimen_resultado.get('stop_loss', 0.012),
                                    'max_position_time': regimen_resultado.get('max_position_time', 12)
                                })

                            state["l3_output"] = l3_output

                            logger.info(f"🎯 REGIME PRIORITY L3: {regimen_resultado['regime'].upper()} regime driving portfolio allocation with {regimen_resultado.get('confidence', 0.5):.2f} confidence")
                        else:
                            # Fallback to original L3 if regime detection fails
                            logger.warning("⚠️ Regime detection failed, falling back to original L3")
                            l3_output = generate_l3_output(state, texts_for_sentiment=current_sentiment_texts)
                            # Ensure fallback L3 output has fresh sentiment
                            if l3_output and 'sentiment_score' not in l3_output:
                                l3_output['sentiment_score'] = sentiment_score
                            state["l3_output"] = l3_output

                    except Exception as e:
                        logger.error(f"❌ L3 Error: {e}", exc_info=True)

                # 🚨 SIGNAL HIERARCHY: L3 HOLD signals block L2 BUY/SELL when confidence > 40%
                if not APAGAR_L3 and state.get("l3_output"):
                    l3_output = state["l3_output"]
                    l3_signal = l3_output.get('signal', 'hold').lower()
                    l3_confidence = l3_output.get('confidence', 0.0)

                    # ADD DIAGNOSTIC LOGGING - Trace where confidence is lost
                    logger.info("="*80)
                    logger.info("SIGNAL EXECUTION DECISION")
                    logger.info(f"L3 Info: {l3_output}")
                    logger.info(f"Initial L3 Decision: {l3_decision}")
                    logger.info(f"L3 Signal: {l3_signal}, Confidence: {l3_confidence}")
                    logger.info("="*80)

                    if l3_signal == 'hold' and l3_confidence > 0.50:
                        l3_signal_blocks_l2 = True
                        l3_blocking_info = f"L3 HOLD signal with {l3_confidence:.2f} confidence blocks all L2 BUY/SELL signals"
                        logger.warning(f"🚫 L3 DOMINANCE: {l3_blocking_info}")
                    else:
                        l3_signal_blocks_l2 = False
                        l3_blocking_info = f"L3 signal: {l3_signal.upper()} ({l3_confidence:.2f} confidence) - L2 signals allowed"

                # ========================================================================================
                # SIGNAL PROCESSING WITH SETUP OVERRIDE
                # ========================================================================================
                setup_type = l3_decision.get('setup_type')
                allow_l2_signals = l3_decision.get('allow_l2_signals', False)

                if setup_type:
                    logger.info(f"🎯 SETUP STRATEGY: {setup_type} setup allows controlled L2 signals")
                    logger.info(f"📊 Max allocation for setup trades: {l3_decision['strategic_guidelines']['max_single_asset_exposure']:.1%}")

                # ========================================================================================
                # L2 SIGNAL GENERATION WITH L3 CONTEXT (SOLUCIÓN COMPLETA)
                # ========================================================================================

                # Pasar a L2
                l2_signals = l2_processor.generate_signals(
                    market_data=state.get("market_data", {}),
                    l3_context=l3_regime_info  # Con validación
                )

                # Process L2 signals
                try:
                    l2_signals_before_filtering = 0
                    # Use the l2_signals we just generated
                    valid_signals = l2_signals if l2_signals else []
                    l2_signals_before_filtering = len(valid_signals)

                    # 🚨 APPLY L3 DOMINANCE: Filter L2 signals based on L3 regime context (SOLUCIÓN COMPLETA)
                    if valid_signals:
                        blocked_signals = 0
                        filtered_signals = []

                        for signal in valid_signals:
                            # Convert TacticalSignal object to dict for should_execute_with_l3_dominance
                            signal_dict = {
                                'action': getattr(signal, 'side', 'hold'),  # side might be 'buy', 'sell', 'hold'
                                'symbol': getattr(signal, 'symbol', 'UNKNOWN'),
                                'confidence': getattr(signal, 'confidence', 0.5)
                            }

                            # Decide ejecuta based on L3 dominance
                            should_execute, reason = should_execute_with_l3_dominance(signal_dict, l3_regime_info)

                            if should_execute:
                                filtered_signals.append(signal)  # Keep the original TacticalSignal object
                            else:
                                logger.warning(f"🚫 L3 DOMINANCE: {getattr(signal, 'symbol', 'Unknown')} {getattr(signal, 'side', 'hold').upper()} signal blocked - {reason}")
                                blocked_signals += 1

                        valid_signals = filtered_signals

                        if blocked_signals > 0:
                            logger.info(f"🚫 L3 DOMINANCE: Blocked {blocked_signals} L2 signals per regime analysis")

                    # Submit signals for incremental verification
                    for signal in valid_signals:
                        try:
                            # Create verification copy from TacticalSignal object
                            verification_signal = TacticalSignal(
                                symbol=getattr(signal, 'symbol', 'UNKNOWN'),
                                strength=getattr(signal, 'strength', 0.5),
                                confidence=getattr(signal, 'confidence', 0.5),
                                side=getattr(signal, 'side', 'hold'),  # Changed from 'direction' to 'side'
                                signal_type='setup_aware',
                                source='l2_processor',
                                timestamp=getattr(signal, 'timestamp', pd.Timestamp.now()),
                                features=getattr(signal, 'features', {}),
                                metadata=getattr(signal, 'metadata', {})
                            )

                            await signal_verifier.submit_signal_for_verification(
                                verification_signal, state.get("market_data", {})
                            )
                        except Exception as verify_error:
                            logger.warning(f"⚠️ Failed to submit signal for verification: {verify_error}")

                except Exception as e:
                    logger.error(f"❌ L2 Error: {e}", exc_info=True)
                    valid_signals = []

                # 3. Generate and execute orders with validation

                # PRE-GENERATION CHECK: Filter signals if USDT balance too low for buy orders
                min_order_usdt_balance = 500.0  # Minimum USDT to keep for trading
                current_usdt_balance = portfolio_manager.get_balance("USDT")

                if current_usdt_balance < min_order_usdt_balance:
                    logger.warning(f"💰 USDT BALANCE TOO LOW: {current_usdt_balance:.2f} < {min_order_usdt_balance:.2f}, skipping buy orders")

                    # Filter out buy signals to prevent "fondos insuficientes" errors
                    buy_signals_to_skip = [s for s in valid_signals if getattr(s, 'side', None) == 'buy' or (hasattr(s, 'signal_type') and 'buy' in getattr(s, 'signal_type', '').lower())]
                    if buy_signals_to_skip:
                        logger.info(f"🚫 Skipping {len(buy_signals_to_skip)} buy signals due to low USDT balance")
                        valid_signals = [s for s in valid_signals if s not in buy_signals_to_skip]

                # ========================================================================================
                # ORDER EXECUTION WITH SETUP-AWARE SIZING
                # ========================================================================================
                orders = await order_manager.generate_orders(state, valid_signals)

                # Apply setup-aware sizing to trades
                for order in orders:
                    # Find corresponding signal for this order
                    symbol = order.get('symbol', '')
                    signal = None

                    for sig in valid_signals:
                        if isinstance(sig, dict) and sig.get('symbol') == symbol:
                            signal = sig
                            break

                    if signal and signal.get('setup_trade', False):
                        # Apply setup-specific risk management
                        order['stop_loss_pct'] = 0.008  # Tighter stop for setup trades
                        order['profit_target_pct'] = 0.015  # Tighter target for mean reversion
                        order['max_hold_time'] = 4  # Max 4 hours for setup trades

                        # Apply size multiplier if available
                        size_multiplier = signal.get('size_multiplier', 0.50)
                        if 'quantity' in order:
                            order['quantity'] *= size_multiplier
                            order['original_quantity'] = order['quantity'] / size_multiplier  # For logging

                        logger.info(f"🎯 SETUP TRADE PARAMS for {symbol}: "
                                   f"Stop: {order['stop_loss_pct']:.1%} | "
                                   f"Target: {order['profit_target_pct']:.1%} | "
                                   f"Max Hold: {order['max_hold_time']}h | "
                                   f"Size: {size_multiplier:.0%}")

                # Validate orders before execution using new validation method
                validated_orders = []
                for order in orders:
                    if order.get("status") == "pending":  # Only validate pending orders
                        symbol = order.get("symbol")
                        quantity = order.get("quantity", 0.0)
                        price = order.get("price", 0.0)
                        portfolio = state.get("portfolio", {})

                        # Additional check: Prevent buy orders if USDT balance insufficient for this specific order
                        if order.get("side") == "buy":
                            required_usdt = quantity * price
                            available_usdt = portfolio_manager.get_balance("USDT")
                            min_keep_usdt = 100.0  # Always keep minimum USDT

                            if available_usdt <= min_keep_usdt:
                                logger.warning(f"❌ BUY ORDER BLOCKED: Insufficient USDT balance ({available_usdt:.2f} <= {min_keep_usdt:.2f} minimum keep)")
                                order["status"] = "rejected"
                                order["validation_error"] = f"Insufficient USDT balance for buy order (balance: {available_usdt:.2f})"
                                validated_orders.append(order)
                                continue
                            elif required_usdt > available_usdt - min_keep_usdt:
                                logger.warning(f"❌ BUY ORDER BLOCKED: Required USDT ({required_usdt:.2f}) exceeds available ({available_usdt:.2f} - {min_keep_usdt:.2f} reserve)")
                                order["status"] = "rejected"
                                order["validation_error"] = f"Required USDT for buy order exceeds available balance"
                                validated_orders.append(order)
                                continue

                        # Use new validation method
                        validation_result = order_manager.validate_order_size(symbol, quantity, price, portfolio)

                        if validation_result["valid"]:
                            validated_orders.append(order)
                            logger.info(f"✅ Order validated: {symbol} {order.get('side')} {quantity:.4f} @ ${price:.2f}")
                        else:
                            logger.warning(f"❌ Order rejected: {validation_result['reason']}")
                            order["status"] = "rejected"
                            order["validation_error"] = validation_result["reason"]
                            validated_orders.append(order)  # Still add to track rejections
                    else:
                        validated_orders.append(order)  # Add non-pending orders as-is

                processed_orders = await order_manager.execute_orders(validated_orders)

                # 4. Update portfolio using PortfolioManager as single source of truth
                await portfolio_manager.update_from_orders_async(processed_orders, state.get("market_data", {}))

                # 🔒 SINGLE SOURCE OF TRUTH: PortfolioManager manages its own state
                # Values are calculated on-demand from PortfolioManager to avoid sync issues

                # Calculate total_value before using in trading metrics
                total_value = portfolio_manager.get_total_value(state.get("market_data", {}))

                # Update trading metrics with executed orders and portfolio value
                trading_metrics.update_from_orders(processed_orders, total_value)

                # =================================================================
                # POSITION ROTATION - RE-ENABLED with enhanced price validation
                # =================================================================
                if position_rotator and True:  # RE-ENABLED after fixing price logging
                    try:
                        logger.info("🔄 Checking position rotation...")
                        rotation_orders = await position_rotator.check_and_rotate_positions(state, market_data)
                        if rotation_orders:
                            # Execute rotation orders
                            processed_rotations = await order_manager.execute_orders(rotation_orders)

                            # Update portfolio with rotation results
                            await portfolio_manager.update_from_orders_async(processed_rotations, market_data)
                            rotation_count = len([o for o in processed_rotations if o.get('status') == 'filled'])
                            logger.info(f"🔄 Executed {rotation_count} position rotations")

                            if rotation_count > 0:
                                portfolio_manager.save_state()  # Add this line

                                # Refresh portfolio values for display
                                btc_value = state["btc_balance"]
                                eth_value = state["eth_balance"]
                                total_value = portfolio_manager.get_total_value(market_data)

                                logger.info(f"\n")
                                logger.info("*"*100)
                                logger.info(f"💰 Portfolio actualizado: Total={total_value:.2f} USDT, "
                                           f"BTC={btc_value:.5f}, "
                                           f"ETH={eth_value:.3f}, "
                                           f"USDT={state['usdt_balance']:.2f}")
                                logger.info("*"*100)

                            # Force rebalance if stop-loss triggered after rotations
                            if any(r.get('reason') == 'stop_loss_protection' for r in processed_rotations):
                                logger.info("🔄 STOP-LOSS TRIGGERED - Forcing portfolio rebalance...")
                                # TODO: Add forced rebalancing logic here

                    except Exception as e:
                        logger.error(f"❌ Position rotation failed: {e}")

                # =================================================================
                # FORCED REBALANCE AFTER STOP-LOSS ALIGNMENT
                # =================================================================
                # If we just had a stop-loss, force L3 alignment and rebalance
                recent_stop_losses = [o for o in processed_orders if o.get('reward_type') == 'stop_loss_protection']
                if recent_stop_losses:
                    logger.info(f"🔄 FORCED REBALANCE: {len(recent_stop_losses)} stop-losses detected, aligning with L3 target...")

                    # Get L3 target allocation
                    l3_output = state.get("l3_output", {})
                    target_alloc = l3_output.get("portfolio_allocation", {})

                    if target_alloc and any(target_alloc.values()):
                        logger.info(f"🎯 L3 Target Allocation: {target_alloc}")

                        # Calculate current allocations and force rebalance if skewed
                        total_value = portfolio_manager.get_total_value(market_data)
                        current_btc_pct = (portfolio_manager.get_balance("BTCUSDT") * state.get("btc_price", 60000)) / total_value if total_value > 0 else 0
                        current_eth_pct = (portfolio_manager.get_balance("ETHUSDT") * state.get("eth_price", 4000)) / total_value if total_value > 0 else 0
                        current_usdt_pct = portfolio_manager.get_balance("USDT") / total_value if total_value > 0 else 0

                        # Check if portfolio is imbalanced (>10% deviation from target)
                        target_btc = target_alloc.get("BTC", 40) / 100
                        target_eth = target_alloc.get("ETH", 30) / 100
                        target_usdt = target_alloc.get("USDT", 30) / 100

                        btc_deviation = abs(current_btc_pct - target_btc)
                        eth_deviation = abs(current_eth_pct - target_eth)

                        if btc_deviation > 0.10 or eth_deviation > 0.10:  # 10% deviation threshold
                            logger.info(f"⚠️ PORTFOLIO IMBALANCED - Current: BTC={current_btc_pct*100:.1f}%, ETH={current_eth_pct*100:.1f}%, USDT={current_usdt_pct*100:.1f}%")
                            logger.info(f"🎯 Target: BTC={target_btc*100:.1f}%, ETH={target_eth*100:.1f}%, USDT={target_usdt*100:.1f}%")

                            # Generate rebalancing orders
                            rebalance_orders = []
                            # TODO: Implement rebalancing logic to bring portfolio back to target allocations

                            # For now, ensure we have minimum USDT liquidity
                            min_usdt_target = 800.0  # Minimum USDT to keep
                            if portfolio_manager.get_balance("USDT") < min_usdt_target and total_value > min_usdt_target:
                                # Sell portion of largest holding to reach minimum USDT
                                if current_btc_pct > current_eth_pct:
                                    sell_symbol = "BTCUSDT"
                                    sell_quantity = min(0.01, portfolio_manager.get_balance("BTCUSDT") * 0.10)  # Max 10% or 0.01 BTC
                                else:
                                    sell_symbol = "ETHUSDT"
                                    sell_quantity = min(0.1, portfolio_manager.get_balance("ETHUSDT") * 0.10)  # Max 10% or 0.1 ETH

                                if sell_quantity > 0:
                                    sell_price = state.get("btc_price", 60000) if sell_symbol == "BTCUSDT" else state.get("eth_price", 4000)
                                    rebalance_order = {
                                        "symbol": sell_symbol,
                                        "side": "sell",
                                        "type": "MARKET",
                                        "quantity": sell_quantity,
                                        "price": sell_price,
                                        "reason": "forced_rebalance_after_sl",
                                        "status": "pending"
                                    }
                                    rebalance_orders.append(rebalance_order)
                                    logger.info(f"💰 FORCED REBALANCE: Selling {sell_quantity:.4f} {sell_symbol.replace('USDT', '')} for liquidity")

                            if rebalance_orders:
                                processed_rebalance = await order_manager.execute_orders(rebalance_orders)
                                await portfolio_manager.update_from_orders_async(processed_rebalance, market_data)
                                logger.info(f"✅ Executed {len(processed_rebalance)} rebalancing orders after stop-loss")
                # try:
                #     rotation_orders = await position_rotator.check_and_rotate_positions(state, state.get("market_data", {}))
                #     if rotation_orders:
                #         # Ejecutar órdenes de rotación
                #         executed_rotations = await order_manager.execute_orders(rotation_orders)
                #         # Actualizar portfolio con rotaciones ejecutadas
                #         await portfolio_manager.update_from_orders_async(executed_rotations, state.get("market_data", {}))
                #         # Re-sync state después de rotaciones
                #         state["portfolio"] = portfolio_manager.get_portfolio_state()
                #         state["total_value"] = portfolio_manager.get_total_value(state.get("market_data", {}))
                #         state["usdt_balance"] = portfolio_manager.get_balance("USDT")
                #         logger.info(f"🔄 Ejecutadas {len(executed_rotations)} órdenes de rotación automática")
                # except Exception as rotation_error:
                #     logger.error(f"❌ Error en position rotation: {rotation_error}")

                # Save portfolio state periodically (every 5 cycles or when significant changes)
                if cycle_id % 5 == 0:
                    portfolio_manager.save_to_json()

                # CRITICAL FIX: Refresh portfolio values directly from portfolio_manager to ensure up-to-date display
                btc_balance = portfolio_manager.get_balance("BTCUSDT")
                eth_balance = portfolio_manager.get_balance("ETHUSDT")
                usdt_balance = portfolio_manager.get_balance("USDT")
                total_value = portfolio_manager.get_total_value(state.get("market_data", {}))

                # Sync state with current values
                state["btc_balance"] = btc_balance
                state["eth_balance"] = eth_balance
                state["usdt_balance"] = usdt_balance
                state["total_value"] = total_value

                # Enhanced visual logging with color coding and borders
                if total_value > 3000.0:
                    # Green for profit
                    logger.info(f"")
                    logger.info(f"********************************************************************************************")
                    logger.info(f"\x1b[32m\x1b[1m\x1b[2m💰 Portfolio actualizado: Total={total_value:.2f} USDT, BTC={btc_balance:.5f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}\x1b[0m")
                    logger.info(f"********************************************************************************************")
                    logger.info(f"")
                elif total_value < 3000.0:
                    # Red for loss
                    logger.info(f"")
                    logger.info(f"********************************************************************************************")
                    logger.info(f"\x1b[31m\x1b[1m\x1b[2m💰 Portfolio actualizado: Total={total_value:.2f} USDT, BTC={btc_balance:.5f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}\x1b[0m")
                    logger.info(f"********************************************************************************************")
                    logger.info(f"")
                else:
                    # Blue for equal
                    logger.info(f"")
                    logger.info(f"********************************************************************************************")
                    logger.info(f"\x1b[34m\x1b[1m\x1b[2m💰 Portfolio actualizado: Total={total_value:.2f} USDT, BTC={btc_balance:.5f}, ETH={eth_balance:.3f}, USDT={usdt_balance:.2f}\x1b[0m")
                    logger.info(f"********************************************************************************************")
                    logger.info(f"")

                # Log periodic trading metrics report
                if cycle_id % 10 == 0:  # Every 10 cycles
                    trading_metrics.log_periodic_report()

                # ========================================================================================
                # MONITORING: Log setup status every cycle
                # ========================================================================================
                if setup_detected:
                    logger.info(f"📊 SETUP STATUS: {regime_result['subtype']} active")
                    logger.info(f"📊 Setup Metrics: RSI={regime_result['metrics']['rsi']:.1f}, "
                               f"ADX={regime_result['metrics']['adx']:.1f}, "
                               f"BB_width={regime_result['metrics']['bb_width']:.2%}")
                else:
                    logger.info(f"📊 REGIME STATUS: {regime_result['primary_regime']} | "
                               f"Subtype: {regime_result['subtype']} | "
                               f"Confidence: {regime_result['confidence']:.2f}")

                # Log cycle stats
                valid_orders = [o for o in processed_orders if o.get("status") != "rejected"]
                await log_cycle_data(state, cycle_id, start_time)

                # Update cumulative counters
                total_signals_all_cycles += len(valid_signals)
                total_orders_all_cycles += len(valid_orders)
                total_rejected_all_cycles += len(processed_orders) - len(valid_orders)
                total_cooldown_blocked_all_cycles += order_manager.cooldown_blocked_count

                # Include L3 blocking info in cycle logging
                l3_block_display = ""
                if l3_signal_blocks_l2 and 'l2_signals_before_filtering' in locals() and l2_signals_before_filtering > len(valid_signals):
                    blocked_count = l2_signals_before_filtering - len(valid_signals)
                    l3_block_display = f" | L3 Blocked: {blocked_count}"

                logger.info(
                    f"📊 Cycle {cycle_id} | "
                    f"Time: {(pd.Timestamp.utcnow() - start_time).total_seconds():.1f}s | "
                    f"Signals: {len(valid_signals)} | "
                    f"Orders: {len(valid_orders)} | "
                    f"Rejected: {len(processed_orders) - len(valid_orders)} | "
                    f"Cooldown: {order_manager.cooldown_blocked_count}{l3_block_display}"
                )

                # Log cumulative totals every 5 cycles
                if cycle_id % 5 == 0:
                    logger.info(
                        f"📈 CUMULATIVE TOTALS (Cycles 1-{cycle_id}) | "
                        f"Total Signals: {total_signals_all_cycles} | "
                        f"Total Orders: {total_orders_all_cycles} | "
                        f"Total Rejected: {total_rejected_all_cycles} | "
                        f"Total Cooldown: {total_cooldown_blocked_all_cycles} | "
                        f"Avg Signals/Cycle: {total_signals_all_cycles/cycle_id:.1f} | "
                        f"Avg Orders/Cycle: {total_orders_all_cycles/cycle_id:.1f}"
                    )

                await asyncio.sleep(10)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Cycle error: {error_msg}")
                # Don't use exc_info=True to avoid the AttributeError in logging
                
                # Handle specific error types using centralized validation
                if "No hay market_data en el estado" in error_msg or "truth value of a DataFrame is ambiguous" in error_msg:
                    cycle_key = f"validation_attempts_{cycle_id}"
                    validation_attempts = state.get(cycle_key, 0)

                    if validation_attempts > 2:  # Allow max 3 attempts per cycle
                        logger.warning("⚠️ Maximum validation attempts reached for cycle")
                        try:
                            # Attempt full data refresh
                            fresh_data = await loader.get_realtime_data()
                            valid_data, validation_msg = validate_market_data_structure(fresh_data)

                            if valid_data:
                                state["market_data"] = fresh_data
                                logger.info("✅ Successfully refreshed market data")
                            else:
                                logger.error(f"❌ Failed to refresh data: {validation_msg}")
                                state["market_data"] = {}  # Reset for next cycle
                        except Exception as refresh_error:
                            logger.error(f"Failed to refresh data: {refresh_error}")

                        await asyncio.sleep(10)
                        state[cycle_key] = 0  # Reset counter
                        continue

                    state[cycle_key] = validation_attempts + 1
                    valid_market_data, validation_msg = UnifiedValidator.validate_and_fix_market_data(state, config)

                    if valid_market_data:
                        # Update state with validated data (single source)
                        state["market_data"] = valid_market_data
                        logger.info(f"✅ Validated market data: {validation_msg}")

                        # Clear validation counter on success
                        state[cycle_key] = 0
                    else:
                        logger.error(f"❌ No valid market data after validation: {validation_msg}")
                        state["market_data"] = {}  # Force refresh next cycle

                    await asyncio.sleep(5)

                elif isinstance(e, (ValueError, RuntimeError)):
                    # Data quality or availability issues
                    await asyncio.sleep(5)

                    # Handle data quality errors with thorough validation
                    try:
                        fresh_data = await loader.get_realtime_data()
                        valid_market_data, validation_msg = UnifiedValidator.validate_and_fix_market_data(state, config)

                        if valid_market_data:
                            # Ensure consistent DataFrame format
                            normalized_data = {}
                            for symbol, data in valid_market_data.items():
                                if isinstance(data, dict):
                                    try:
                                        df = pd.DataFrame(data)
                                        if not df.empty and df.shape[1] >= 5:
                                            normalized_data[symbol] = df
                                    except Exception as conv_error:
                                        logger.warning(f"Failed to convert {symbol} data: {conv_error}")
                                elif isinstance(data, pd.DataFrame) and not data.empty:
                                    normalized_data[symbol] = data

                            if normalized_data:
                                state["market_data"] = normalized_data
                                logger.info(f"✅ Market data refreshed: {validation_msg}")
                            else:
                                logger.error("❌ No valid data after normalization")
                        else:
                            logger.error(f"❌ No valid market data from refresh: {validation_msg}")
                    except Exception as refresh_error:
                        logger.error(f"Failed to refresh market data: {refresh_error}")

                elif ("tf referenced before assignment" in error_msg
                      or "tensorflow" in error_msg.lower()
                      or "truth value of a DataFrame is ambiguous" in error_msg):
                    # ML framework errors (TensorFlow, PyTorch, pandas)
                    await asyncio.sleep(10)
                    try:
                        # Clean up existing resources
                        cleanup_models()
                        import gc
                        gc.collect()

                        # Initialize frameworks with proper settings in isolated scope
                        tf = None  # Declare in outer scope
                        try:
                            def init_tensorflow():
                                """Initialize TensorFlow with proper GPU settings"""
                                try:
                                    import tensorflow as tf

                                    # Clear any existing sessions
                                    tf.keras.backend.clear_session()

                                    # Configure GPU settings
                                    gpus = tf.config.list_physical_devices('GPU')
                                    if gpus:
                                        for gpu in gpus:
                                            try:
                                                tf.config.experimental.set_memory_growth(gpu, True)
                                                logger.info(f"Enabled memory growth for GPU: {gpu}")
                                            except RuntimeError as e:
                                                logger.warning(f"Error configuring GPU {gpu}: {e}")

                                    # Verify TensorFlow is working
                                    tf.constant([1.0])
                                    return tf
                                except ImportError:
                                    logger.warning("TensorFlow not available")
                                    return None
                                except Exception as e:
                                    logger.error(f"Error initializing TensorFlow: {e}")
                                    return None

                            # Initialize TensorFlow with error handling
                            try:
                                tf = init_tensorflow()
                                if tf is not None:
                                    logger.info("✅ TensorFlow initialized with memory growth enabled")
                                else:
                                    logger.warning("⚠️ TensorFlow initialization skipped")
                            except Exception as tf_error:
                                logger.error(f"Failed to initialize TensorFlow: {tf_error}")
                                tf = None

                            # PyTorch settings for FinRL (independent of TF)
                            import torch
                            if torch.cuda.is_available():
                                torch.backends.cuda.matmul.allow_tf32 = True
                                torch.backends.cudnn.allow_tf32 = True
                                logger.info("✅ PyTorch CUDA optimizations enabled")
                            else:
                                logger.info("PyTorch running in CPU mode")

                        except Exception as framework_error:
                            logger.error(f"Failed to initialize ML frameworks: {framework_error}")

                        # Get fresh market data with validation
                        try:
                            fresh_data = await loader.get_realtime_data()
                            if fresh_data is None or not isinstance(fresh_data, dict):
                                logger.error(f"❌ Invalid fresh data type: {type(fresh_data)}")
                                fresh_data = {}

                            valid_market_data, validation_msg = UnifiedValidator.validate_and_fix_market_data(state, config)

                            if valid_market_data:
                                # Ensure DataFrame conversions are complete
                                for symbol, data in valid_market_data.items():
                                    if isinstance(data, dict):
                                        valid_market_data[symbol] = pd.DataFrame(data)

                                state["market_data"] = valid_market_data
                                logger.info("✅ Fresh market data loaded and validated")
                        except Exception as data_error:
                            logger.error(f"Failed to refresh market data: {data_error}")

                        # Reinitialize with clean state and proper framework settings
                        from l3_strategy.l3_processor import generate_l3_output
                        l3_output = generate_l3_output(state)
                        state["l3_output"] = l3_output  # Store L3 output in state
                        logger.info("🔄 L3 components reinitialized successfully")

                    except Exception as reinit_error:
                        logger.error(f"Failed to reinitialize L3: {reinit_error}")
                        if any(x in str(reinit_error).lower() for x in ["tensorflow", "torch", "cuda"]):
                            logger.warning("⚠️ ML framework initialization failed, will retry next cycle")
                            # Force cleanup and reset
                            cleanup_models()
                            gc.collect()
                            await asyncio.sleep(30)  # Longer wait after failed initialization
                
                else:
                    # Other errors - wait longer
                    await asyncio.sleep(30)
                
                # Always validate and repair state with thorough type checking
                try:
                    if not isinstance(state, dict):
                        logger.error(f"❌ Invalid state type: {type(state)}")
                        state = initialize_state(config["SYMBOLS"], 3000.0)
                    else:
                        state = validate_state_structure(state)
                    
                    # Validate and repair market data
                    from comms.data_validation import fix_market_data
                    
                    if not isinstance(state, dict):
                        logger.error("❌ State is not a dictionary after validation")
                        state = {"market_data": {}}
                    elif "market_data" not in state:
                        logger.warning("Missing market_data in state, initializing empty")
                        state["market_data"] = {}
                    elif not isinstance(state["market_data"], dict):
                        logger.warning(f"Invalid market_data type: {type(state['market_data'])}, fixing")
                        fixed_data = fix_market_data(state["market_data"])
                        if fixed_data:
                            state["market_data"] = fixed_data
                        else:
                            state["market_data"] = {}

                    # Validate market data is properly structured (single source)
                    data = state["market_data"]
                    for symbol, symbol_data in list(data.items()):
                        if not isinstance(symbol_data, (dict, pd.DataFrame)):
                            logger.warning(f"Invalid data type for {symbol} in market_data: {type(symbol_data)}")
                            del data[symbol]
                        
                except Exception as state_error:
                    logger.error(f"Failed to validate state: {state_error}")
                    state = initialize_state(config["SYMBOLS"], 3000.0)  # Reset to initial state
                    logger.info("State reset to initial values")
                
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        # Save final portfolio state before shutdown
        try:
            portfolio_manager.save_to_json()
            logger.info("💾 Final portfolio state saved")
        except Exception as save_error:
            logger.error(f"❌ Error saving final portfolio state: {save_error}")
    finally:
        # Cleanup
        try:
            await stop_signal_verification()
            logger.info("🛑 Signal verification stopped")
        except Exception as verify_cleanup_error:
            logger.warning(f"⚠️ Error stopping signal verification: {verify_cleanup_error}")

        for component in [loader, data_feed, bus_adapter, binance_client]:
            if hasattr(component, "close"):
                await component.close()

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
