# -*- coding: utf-8 -*-
# main.py

import asyncio
import sys
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

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
from core.incremental_signal_verifier import get_signal_verifier, start_signal_verification, stop_signal_verification
from core.trading_metrics import get_trading_metrics

from l1_operational.data_feed import DataFeed
from l2_tactic.signal_generator import L2TacticProcessor
from l2_tactic.models import L2State, TacticalSignal
from l2_tactic.risk_controls.manager import RiskControlManager
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l3_strategy.l3_processor import generate_l3_output, cleanup_models
from l3_strategy.sentiment_inference import download_reddit, download_news, infer_sentiment
from l1_operational.bus_adapter import BusAdapterAsync

from comms.config import config
from l2_tactic.config import L2Config
from comms.message_bus import MessageBus

# 🔄 AUTO-LEARNING SYSTEM INTEGRATION
from integration_auto_learning import integrate_with_main_system

async def main():
    """Main HRM system function."""
    try:
        logger.info("🚀 Starting HRM system")
        state = initialize_state(config["SYMBOLS"], 3000.0)
        state = validate_state_structure(state)
        
        # Initialize components
        loader = RealTimeDataLoader(config)
        data_feed = DataFeed(config)
        bus_adapter = BusAdapterAsync(config, state)
        binance_client = BinanceClient()
        
        # Get environment configuration
        env_config = get_config("live")  # Use "live" mode for now

        # FOR TESTING: Override initial balance to 3000.0 for validation
        test_initial_balance = 3000.0
        logger.info(f"🧪 TESTING MODE: Using initial balance of {test_initial_balance} USDT (overriding config)")

        # Initialize Portfolio Manager for persistence with new config system
        portfolio_manager = PortfolioManager(
            mode="simulated",  # Use simulated mode for testing with 3000 USDT
            initial_balance=test_initial_balance,  # Use test balance instead of config
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
            l2_config = L2Config()
            l2_processor = L2TacticProcessor(l2_config, portfolio_manager=portfolio_manager)
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

        state["portfolio"] = portfolio_manager.get_portfolio_state()
        state["peak_value"] = portfolio_manager.peak_value
        state["total_fees"] = portfolio_manager.total_fees

        # Initialize L3 (will be called after market data is loaded)

        order_manager = OrderManager(binance_client=binance_client, market_data=state.get("market_data", {}))
        
        # Get initial data with retries
        max_retries = 3
        retry_count = 0
        initial_data = None
        
        while retry_count < max_retries:
            try:
                initial_data = await loader.get_realtime_data()
                if initial_data is None or (isinstance(initial_data, dict) and len(initial_data) == 0):
                    initial_data = await data_feed.get_market_data()
                
                if isinstance(initial_data, dict) and any(initial_data):
                    logger.info("✅ Successfully loaded initial market data")
                    break
                else:
                    logger.warning("⚠️ Got empty market data, retrying...")
            except Exception as e:
                logger.error(f"Failed to get market data (attempt {retry_count + 1}/{max_retries}): {e}")
            
            retry_count += 1
            await asyncio.sleep(5)
            
        if not isinstance(initial_data, dict) or not any(initial_data):
            raise RuntimeError("Could not get initial market data after multiple retries")
            
        state["market_data"] = initial_data
        state["market_data_full"] = initial_data.copy()
        
        # Validate required data is present
        required_symbols = config["SYMBOLS"]
        missing_symbols = [sym for sym in required_symbols if sym not in initial_data]
        if missing_symbols:
            logger.warning(f"⚠️ Missing data for symbols: {missing_symbols}")

        # Sentiment analysis state and function
        sentiment_texts_cache = []
        last_sentiment_update = 0
        SENTIMENT_UPDATE_INTERVAL = 50  # Update sentiment every 50 cycles (~8-9 minutes)

        async def update_sentiment_texts():
            """Update sentiment texts from Reddit and News API"""
            try:
                logger.info("🔄 SENTIMENT: Iniciando actualización de datos de sentimiento...")

                # Download Reddit data
                logger.info("📱 SENTIMENT: Descargando datos de Reddit...")
                reddit_df = await download_reddit(limit=500)  # Reduced limit for performance
                reddit_texts = []
                if not reddit_df.empty:
                    reddit_texts = reddit_df['text'].dropna().tolist()[:100]  # Limit to 100 texts
                    logger.info(f"📱 SENTIMENT: Reddit - {len(reddit_texts)} posts descargados y procesados")
                else:
                    logger.warning("⚠️ SENTIMENT: No se obtuvieron datos de Reddit")

                # Download News data
                logger.info("📰 SENTIMENT: Descargando datos de noticias...")
                news_df = download_news()
                news_texts = []
                if not news_df.empty:
                    news_texts = news_df['text'].dropna().tolist()[:50]  # Limit to 50 texts
                    logger.info(f"📰 SENTIMENT: News - {len(news_texts)} artículos descargados y procesados")
                else:
                    logger.warning("⚠️ SENTIMENT: No se obtuvieron datos de noticias")

                # Combine and limit total texts
                all_texts = reddit_texts + news_texts
                original_count = len(all_texts)

                if len(all_texts) > 100:  # Limit total to 100 texts for performance
                    all_texts = all_texts[:100]
                    logger.info(f"✂️ SENTIMENT: Limitado de {original_count} a {len(all_texts)} textos para rendimiento")

                # Filtrar textos vacíos
                valid_texts = [t for t in all_texts if t and str(t).strip()]
                if len(valid_texts) != len(all_texts):
                    logger.info(f"🧹 SENTIMENT: Filtrados {len(all_texts) - len(valid_texts)} textos vacíos")

                logger.info(f"💬 SENTIMENT: Análisis de sentimiento listo con {len(valid_texts)} textos válidos")
                logger.info(f"   📊 Distribución: Reddit={len([t for t in valid_texts if t in reddit_texts[:100]])} | News={len([t for t in valid_texts if t in news_texts[:50]])}")

                return valid_texts

            except Exception as e:
                logger.error(f"❌ SENTIMENT: Error actualizando datos de sentimiento: {e}")
                return []

        # Initialize L3 now that we have market data
        try:
            # Get initial sentiment data for L3
            initial_sentiment_texts = await update_sentiment_texts()
            sentiment_texts_cache = initial_sentiment_texts

            l3_output = generate_l3_output(state, texts_for_sentiment=initial_sentiment_texts)  # Generate initial L3 output with market data and sentiment
            state["l3_output"] = l3_output  # Store L3 output in state for L2 access

            # CRITICAL FIX: Initial sync of L3 output with L3 context cache
            if 'l3_context_cache' not in state:
                state['l3_context_cache'] = {}
            if l3_output:
                state['l3_context_cache']['last_output'] = l3_output.copy()
                from l3_strategy.l3_processor import _calculate_market_data_hash
                state['l3_context_cache']['market_data_hash'] = _calculate_market_data_hash(state.get("market_data", {}))
                logger.debug("✅ Initial L3 context cache synced")

            logger.info("✅ L3 initialized successfully with market data and sentiment analysis")
        except Exception as e:
            logger.error(f"❌ Error initializing L3: {e}", exc_info=True)

        # CRITICAL FIX: Ensure portfolio is properly initialized and logged
        logger.info("🔍 INITIAL PORTFOLIO STATE:")
        logger.info(f"   BTC Position: {portfolio_manager.get_balance('BTCUSDT'):.6f}")
        logger.info(f"   ETH Position: {portfolio_manager.get_balance('ETHUSDT'):.3f}")
        logger.info(f"   USDT Balance: {portfolio_manager.get_balance('USDT'):.2f}")
        logger.info(f"   Total Value: {portfolio_manager.get_total_value():.2f}")

        # 🔄 INTEGRATE AUTO-LEARNING SYSTEM
        logger.info("🤖 Integrating Auto-Learning System...")
        auto_learning_system = integrate_with_main_system()
        logger.info("✅ Auto-Learning System integrated - Models will improve automatically!")

        # Main loop
        cycle_id = 0
        while True:
            cycle_id += 1
            start_time = pd.Timestamp.utcnow()
            
            try:
                # 1. Update market data with validation
                def validate_market_data_structure(data):
                    """Safely validate market data structure"""
                    if data is None:
                        return False, "Data is None"

                    if not isinstance(data, dict):
                        return False, f"Not a dictionary (type: {type(data)})"

                    if not data or len(data) == 0:
                        return False, "Empty data dictionary"

                    valid_symbols = []
                    errors = []

                    try:
                        for symbol, v in data.items():
                            if isinstance(v, pd.DataFrame):
                                if v.shape[0] > 0 and v.shape[1] >= 5:
                                    valid_symbols.append(symbol)
                                else:
                                    errors.append(f"{symbol}: Invalid DataFrame shape {v.shape}")
                            elif isinstance(v, dict) and len(v) >= 5:
                                valid_symbols.append(symbol)
                            else:
                                errors.append(f"{symbol}: Invalid data type {type(v)}")
                    except AttributeError as e:
                        return False, f"Data structure error: {e}"

                    if valid_symbols:
                        return True, f"Valid symbols: {valid_symbols}"
                    else:
                        return False, f"No valid data. Errors: {errors}"
                
                # Get and validate realtime data with type checking
                logger.info("🔄 Attempting to get realtime market data...")
                market_data = await loader.get_realtime_data()
                logger.info(f"📊 Realtime data result: type={type(market_data)}, keys={list(market_data.keys()) if isinstance(market_data, dict) else 'N/A'}")

                if not isinstance(market_data, dict):
                    logger.error(f"❌ Invalid market_data type from realtime_data: {type(market_data)}")
                    market_data = {}

                is_valid, validation_msg = validate_market_data_structure(market_data)

                if not is_valid:
                    logger.warning(f"⚠️ Failed to get realtime data: {validation_msg}, falling back to data feed")
                    logger.info("🔄 Falling back to data_feed.get_market_data()...")
                    market_data = await data_feed.get_market_data()
                    logger.info(f"📊 Data feed result: type={type(market_data)}, keys={list(market_data.keys()) if isinstance(market_data, dict) else 'N/A'}")

                    if not isinstance(market_data, dict):
                        logger.error(f"❌ Invalid market_data type from data_feed: {type(market_data)}")
                        market_data = {}

                    is_valid, validation_msg = validate_market_data_structure(market_data)
                
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
                        # Update state with validated data
                        state["market_data"] = valid_data
                        state["market_data_full"].update(valid_data)
                        logger.info(f"✅ Market data updated for symbols: {list(valid_data.keys())}")
                        
                        if missing_symbols:
                            logger.warning("⚠️ Some symbols missing, continuing with partial data")
                    else:
                        raise ValueError("No valid market data for required symbols")
                else:
                    raise RuntimeError(f"❌ Failed to get valid market data: {validation_msg}")
                
                # 2. Update L3 state and process signals
                try:
                    # Update sentiment data periodically (every 50 cycles ~8-9 minutes)
                    if cycle_id - last_sentiment_update >= SENTIMENT_UPDATE_INTERVAL:
                        logger.info(f"🔄 SENTIMENT: Actualización periódica iniciada (ciclo {cycle_id}, cada {SENTIMENT_UPDATE_INTERVAL} ciclos)")
                        sentiment_texts_cache = await update_sentiment_texts()
                        last_sentiment_update = cycle_id
                        logger.info(f"💬 SENTIMENT: Cache actualizado con {len(sentiment_texts_cache)} textos para análisis L3")

                    # Use cached sentiment texts for L3 processing
                    current_sentiment_texts = sentiment_texts_cache if sentiment_texts_cache else []

                    l3_output = generate_l3_output(state, texts_for_sentiment=current_sentiment_texts)  # Update L3 output with sentiment
                    state["l3_output"] = l3_output  # Store updated L3 output in state

                    # CRITICAL FIX: Sync L3 output with L3 context cache for L2 processor
                    if 'l3_context_cache' not in state:
                        state['l3_context_cache'] = {}

                    # Always sync the latest L3 output with the cache that L2 reads
                    if l3_output:
                        state['l3_context_cache']['last_output'] = l3_output.copy()
                        # Ensure cache has all required fields for L2 freshness check
                        if 'market_data_hash' not in state['l3_context_cache']:
                            from l3_strategy.l3_processor import _calculate_market_data_hash
                            state['l3_context_cache']['market_data_hash'] = _calculate_market_data_hash(state.get("market_data", {}))
                        logger.debug("✅ L3 context cache synced with latest L3 output")

                except Exception as e:
                    logger.error(f"❌ L3 Error: {e}", exc_info=True)
                    
                # Process L2 signals
                try:
                    signals = await l2_processor.process_signals(state)
                    valid_signals = [s for s in signals if isinstance(s, TacticalSignal)]

                    # Submit signals for incremental verification
                    for signal in valid_signals:
                        try:
                            await signal_verifier.submit_signal_for_verification(
                                signal, state.get("market_data", {})
                            )
                        except Exception as verify_error:
                            logger.warning(f"⚠️ Failed to submit signal for verification: {verify_error}")

                except Exception as e:
                    logger.error(f"❌ L2 Error: {e}", exc_info=True)
                    valid_signals = []

                # 3. Generate and execute orders
                orders = await order_manager.generate_orders(state, valid_signals)
                processed_orders = await order_manager.execute_orders(orders)
                
                # 4. Update portfolio using PortfolioManager as single source of truth
                await portfolio_manager.update_from_orders_async(processed_orders, state.get("market_data", {}))

                # Sync main state with PortfolioManager state
                state["portfolio"] = portfolio_manager.get_portfolio_state()
                state["peak_value"] = portfolio_manager.peak_value
                state["total_fees"] = portfolio_manager.total_fees

                # Calculate and update total value in state
                total_value = portfolio_manager.get_total_value(state.get("market_data", {}))
                state["total_value"] = total_value
                state["btc_balance"] = portfolio_manager.get_balance("BTCUSDT")
                state["eth_balance"] = portfolio_manager.get_balance("ETHUSDT")
                state["usdt_balance"] = portfolio_manager.get_balance("USDT")

                # Calculate BTC and ETH values safely
                btc_market_data = state.get("market_data", {}).get("BTCUSDT")
                if btc_market_data is not None:
                    if isinstance(btc_market_data, dict) and 'close' in btc_market_data:
                        btc_price = btc_market_data['close']
                    elif hasattr(btc_market_data, 'iloc') and len(btc_market_data) > 0:
                        btc_price = btc_market_data['close'].iloc[-1] if 'close' in btc_market_data.columns else 50000.0
                    else:
                        btc_price = 50000.0
                    state["btc_value"] = state["btc_balance"] * btc_price
                else:
                    state["btc_value"] = 0.0

                eth_market_data = state.get("market_data", {}).get("ETHUSDT")
                if eth_market_data is not None:
                    if isinstance(eth_market_data, dict) and 'close' in eth_market_data:
                        eth_price = eth_market_data['close']
                    elif hasattr(eth_market_data, 'iloc') and len(eth_market_data) > 0:
                        eth_price = eth_market_data['close'].iloc[-1] if 'close' in eth_market_data.columns else 3000.0
                    else:
                        eth_price = 3000.0
                    state["eth_value"] = state["eth_balance"] * eth_price
                else:
                    state["eth_value"] = 0.0

                # Update trading metrics with executed orders and portfolio value
                trading_metrics.update_from_orders(processed_orders, total_value)

                # Save portfolio state periodically (every 5 cycles or when significant changes)
                if cycle_id % 5 == 0:
                    portfolio_manager.save_to_json()

                # CRITICAL DEBUG: Log portfolio state after update with enhanced formatting
                portfolio_after = state.get("portfolio", {})
                btc_balance = portfolio_after.get('BTCUSDT', {}).get('position', 0.0)
                eth_balance = portfolio_after.get('ETHUSDT', {}).get('position', 0.0)
                usdt_balance = portfolio_after.get('USDT', {}).get('free', 0.0)
                total_value = state.get('total_value', 0.0)

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

                # Log cycle stats
                valid_orders = [o for o in processed_orders if o.get("status") != "rejected"]
                await log_cycle_data(state, cycle_id, start_time)
                
                logger.info(
                    f"📊 Cycle {cycle_id} | "
                    f"Time: {(pd.Timestamp.utcnow() - start_time).total_seconds():.1f}s | "
                    f"Signals: {len(valid_signals)} | " 
                    f"Orders: {len(valid_orders)} | "
                    f"Rejected: {len(processed_orders) - len(valid_orders)}"
                )
                
                await asyncio.sleep(10)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Cycle error: {error_msg}")
                # Don't use exc_info=True to avoid the AttributeError in logging
                
                def validate_ohlcv_data(data):
                    """Validate OHLCV data structure."""
                    if not isinstance(data, (pd.DataFrame, dict)):
                        return None, "Invalid data type"
                        
                    try:
                        # Convert dict to DataFrame if needed
                        if isinstance(data, dict):
                            data = pd.DataFrame(data)
                            
                        # Basic structure validation using explicit checks
                        if len(data.index) == 0 or data.size == 0:
                            return None, "Empty DataFrame"
                            
                        # Validate OHLCV columns with explicit membership test
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        missing_cols = [col for col in required_cols if col not in data.columns]
                        if missing_cols:
                            return None, f"Missing required columns: {missing_cols}. Has: {list(data.columns)}"
                            
                        # Ensure numeric data with explicit type checking
                        for col in required_cols:
                            # First check if the column contains any non-numeric values
                            if data[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
                                try:
                                    data[col] = pd.to_numeric(data[col], errors='raise')
                                except (ValueError, TypeError):
                                    return None, f"Non-numeric data in {col} and cannot convert"
                                
                        return data, None
                    except Exception as e:
                        return None, str(e)
                
                def validate_and_fix_market_data():
                    """Validate and fix market data structure, returns (valid_data, needs_refresh)"""
                    valid_market_data = {}
                    needs_refresh = False
                    validation_errors = {}
                    
                    # Get market data with type validation
                    market_data = state.get("market_data", None)
                    if not isinstance(market_data, dict):
                        logger.error(f"❌ Invalid market_data type: {type(market_data)}")
                        needs_refresh = True
                        return {}, True
                        
                    if not market_data or len(market_data) == 0:
                        logger.warning("⚠️ Empty market_data dictionary")
                        needs_refresh = True
                        return {}, True
                    
                    for symbol in config["SYMBOLS"]:
                        data = market_data.get(symbol)
                        
                        if data is None:
                            validation_errors[symbol] = "Missing data"
                            needs_refresh = True
                            continue
                            
                        validated_data, error = validate_ohlcv_data(data)
                        if validated_data is not None:
                            valid_market_data[symbol] = validated_data
                        else:
                            validation_errors[symbol] = error
                            needs_refresh = True
                    
                    # Log validation results
                    if validation_errors:
                        for symbol, error in validation_errors.items():
                            logger.warning(f"Validation failed for {symbol}: {error}")
                    
                    return valid_market_data, needs_refresh
                
                # Handle specific error types
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
                                state["market_data_full"] = fresh_data.copy()
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
                    valid_market_data, needs_refresh = validate_and_fix_market_data()
                    
                    if valid_market_data:
                        # Update state with validated data
                        state["market_data"] = valid_market_data
                        state["market_data_full"].update(valid_market_data)
                        logger.info(f"✅ Validated market data for symbols: {list(valid_market_data.keys())}")
                        
                        # Clear validation counter on success
                        state[cycle_key] = 0
                        
                        if needs_refresh:
                            logger.warning("⚠️ Partial data validated, will refresh missing symbols next cycle")
                    else:
                        logger.error("❌ No valid market data after validation")
                        state["market_data"] = {}  # Force refresh next cycle
                        
                    await asyncio.sleep(5)
                
                elif isinstance(e, (ValueError, RuntimeError)):
                    # Data quality or availability issues
                    await asyncio.sleep(5)
                    
                    # Handle data quality errors with thorough validation
                    try:
                        fresh_data = await loader.get_realtime_data()
                        valid_market_data, _ = validate_and_fix_market_data()
                        
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
                                state["market_data_full"].update(normalized_data)
                                logger.info(f"✅ Market data refreshed for symbols: {list(normalized_data.keys())}")
                            else:
                                logger.error("❌ No valid data after normalization")
                        else:
                            logger.error("❌ No valid market data from refresh")
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
                                
                            valid_market_data, _ = validate_and_fix_market_data()
                            
                            if valid_market_data:
                                # Ensure DataFrame conversions are complete
                                for symbol, data in valid_market_data.items():
                                    if isinstance(data, dict):
                                        valid_market_data[symbol] = pd.DataFrame(data)
                                
                                state["market_data"] = valid_market_data
                                state["market_data_full"].update(valid_market_data)
                                logger.info("✅ Fresh market data loaded and validated")
                        except Exception as data_error:
                            logger.error(f"Failed to refresh market data: {data_error}")
                        
                        # Reinitialize with clean state and proper framework settings
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
                        state = {"market_data": {}, "market_data_full": {}}
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
                            
                    # Ensure market_data_full exists and is valid
                    if "market_data_full" not in state:
                        state["market_data_full"] = state["market_data"].copy()
                    elif not isinstance(state["market_data_full"], dict):
                        logger.warning("Invalid market_data_full, resetting from market_data")
                        state["market_data_full"] = state["market_data"].copy()
                        
                    # Validate all market data is properly structured
                    for key in ["market_data", "market_data_full"]:
                        data = state[key]
                        for symbol, symbol_data in list(data.items()):
                            if not isinstance(symbol_data, (dict, pd.DataFrame)):
                                logger.warning(f"Invalid data type for {symbol} in {key}: {type(symbol_data)}")
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
