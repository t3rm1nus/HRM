#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Runtime Loop Module

This module contains the main runtime loop logic for the HRM system.
It handles the execution cycle, timing, and system heartbeat.
"""

import asyncio
import time
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime

from core.state_manager import get_system_state
from core.logging import logger
from core.trading_metrics import get_trading_metrics
from core.portfolio_manager import PortfolioManager
from core.error_handler import ErrorHandler

from l1_operational.data_feed import DataFeed
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l3_strategy.regime_classifier import MarketRegimeClassifier
from l3_strategy.decision_maker import make_decision
from l3_strategy.sentiment_inference import download_reddit, download_news, infer_sentiment

from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus
from sentiment.sentiment_manager import update_sentiment_texts


class HRMRuntimeLoop:
    """Main runtime loop for HRM system execution."""
    
    def __init__(self, portfolio_manager: PortfolioManager, order_manager: OrderManager):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.cycle_id = 0
        self.last_sentiment_update = 0
        self.sentiment_texts_cache = []
        self.sentiment_update_interval = 2160  # 6 hours in cycles
        
        # Initialize components
        self.data_feed = DataFeed(config)
        self.l2_config = L2Config()
        self.l2_processor = L2TacticProcessor(self.l2_config, portfolio_manager=portfolio_manager, apagar_l3=APAGAR_L3)
        self.trading_metrics = get_trading_metrics()
        
        # L3 components
        self.regime_classifier = MarketRegimeClassifier()
        self.message_bus = MessageBus()
        
    async def run(self):
        """Execute the main runtime loop."""
        logger.info("🚀 Starting HRM Runtime Loop")
        
        try:
            while True:
                await self._execute_cycle()
                await asyncio.sleep(10)  # 10-second cycle interval
                
        except KeyboardInterrupt:
            logger.info("🛑 Runtime loop interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runtime loop error: {e}", exc_info=True)
        finally:
            await self._cleanup()
    
    async def _execute_cycle(self):
        """Execute a single cycle of the HRM system."""
        self.cycle_id += 1
        start_time = pd.Timestamp.utcnow()
        
        try:
            # 1. Update market data
            market_data = await self._update_market_data()
            if not market_data:
                logger.warning("⚠️ No market data available, skipping cycle")
                return
            
            # 2. Update portfolio state
            await self._update_portfolio_state(market_data)
            
            # 3. Update sentiment analysis
            await self._update_sentiment_analysis()
            
            # 4. Execute L3 processing
            l3_decision = await self._execute_l3_processing(market_data)
            
            # 5. Execute L2 processing
            l2_signals = await self._execute_l2_processing(market_data, l3_decision)
            
            # 6. Execute L1 processing
            orders = await self._execute_l1_processing(l2_signals, market_data)
            
            # 7. Execute orders
            processed_orders = await self._execute_orders(orders)
            
            # 8. Update portfolio with results
            await self._update_portfolio_from_orders(processed_orders, market_data)
            
            # 9. Log cycle results
            await self._log_cycle_results(start_time, l2_signals, processed_orders)
            
        except Exception as e:
            logger.error(f"❌ Cycle {self.cycle_id} failed: {e}", exc_info=True)
            await ErrorHandler.handle_cycle_error(e, self.cycle_id)
    
    async def _update_market_data(self) -> Dict[str, Any]:
        """Update market data from data feed."""
        try:
            market_data = await self.data_feed.get_market_data()
            if market_data:
                logger.info(f"📊 Market data updated for {len(market_data)} symbols")
            return market_data
        except Exception as e:
            logger.error(f"❌ Failed to update market data: {e}")
            return {}
    
    async def _update_portfolio_state(self, market_data: Dict[str, Any]):
        """Update portfolio state with current market data."""
        try:
            # Sync portfolio with exchange
            sync_success = await self.portfolio_manager.sync_with_exchange()
            if sync_success:
                logger.info("✅ Portfolio synchronized with exchange")
            else:
                logger.warning("⚠️ Portfolio sync failed, using local state")
                
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio state: {e}")
    
    async def _update_sentiment_analysis(self):
        """Update sentiment analysis if needed."""
        cycles_since_update = max(0, self.cycle_id - self.last_sentiment_update)
        
        if cycles_since_update >= self.sentiment_update_interval:
            logger.info("🔄 Updating sentiment analysis...")
            self.sentiment_texts_cache = await update_sentiment_texts()
            self.last_sentiment_update = self.cycle_id
            logger.info(f"✅ Sentiment analysis updated with {len(self.sentiment_texts_cache)} texts")
    
    async def _execute_l3_processing(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute L3 strategic processing."""
        if APAGAR_L3:
            logger.info("🔴 L3 processing disabled")
            return {
                'regime': 'disabled',
                'signal': 'hold',
                'confidence': 0.0,
                'allow_l2_signals': True
            }
        
        try:
            # Classify market regime
            if 'BTCUSDT' in market_data:
                regime_result = self.regime_classifier.classify_market_regime(
                    market_data.get("BTCUSDT", pd.DataFrame()), "BTCUSDT"
                )
                
                # Generate L3 decision
                l3_decision = make_decision(
                    inputs={},
                    portfolio_state=self.portfolio_manager.get_portfolio_state(),
                    market_data=market_data,
                    regime_decision=regime_result,
                    balances_synced=True
                )
                
                logger.info(f"🧠 L3 decision: {l3_decision.get('regime')} regime with {l3_decision.get('confidence', 0):.2f} confidence")
                return l3_decision
            
            logger.warning("⚠️ No BTCUSDT data for L3 processing")
            return {
                'regime': 'unknown',
                'signal': 'hold',
                'confidence': 0.0,
                'allow_l2_signals': True
            }
            
        except Exception as e:
            logger.error(f"❌ L3 processing failed: {e}")
            return {
                'regime': 'error',
                'signal': 'hold',
                'confidence': 0.0,
                'allow_l2_signals': True
            }
    
    async def _execute_l2_processing(self, market_data: Dict[str, Any], l3_decision: Dict[str, Any]) -> list:
        """Execute L2 tactical processing."""
        try:
            # Generate L2 signals with L3 context
            l2_signals = self.l2_processor.generate_signals_conservative(
                market_data=market_data,
                l3_context=l3_decision
            )
            
            logger.info(f"🎯 Generated {len(l2_signals)} L2 signals")
            return l2_signals
            
        except Exception as e:
            logger.error(f"❌ L2 processing failed: {e}")
            return []
    
    async def _execute_l1_processing(self, l2_signals: list, market_data: Dict[str, Any]) -> list:
        """Execute L1 operational processing."""
        try:
            # Generate orders from L2 signals
            orders = await self.order_manager.generate_orders(get_system_state(), l2_signals)
            
            # Validate orders
            validated_orders = []
            for order in orders:
                if order.get("status") == "pending":
                    validation_result = self.order_manager.validate_order_size(
                        order.get("symbol"),
                        order.get("quantity", 0.0),
                        order.get("price", 0.0),
                        get_system_state().get("portfolio", {})
                    )
                    
                    if validation_result["valid"]:
                        validated_orders.append(order)
                    else:
                        logger.warning(f"❌ Order rejected: {validation_result['reason']}")
                        order["status"] = "rejected"
                        order["validation_error"] = validation_result["reason"]
                        validated_orders.append(order)
                else:
                    validated_orders.append(order)
            
            logger.info(f"✅ Generated {len(validated_orders)} validated orders")
            return validated_orders
            
        except Exception as e:
            logger.error(f"❌ L1 processing failed: {e}")
            return []
    
    async def _execute_orders(self, orders: list) -> list:
        """Execute orders through the order manager."""
        try:
            processed_orders = await self.order_manager.execute_orders(orders)
            
            # Count successful executions
            successful_orders = [o for o in processed_orders if o.get("status") == "filled"]
            logger.info(f"💰 Executed {len(successful_orders)} orders successfully")
            
            return processed_orders
            
        except Exception as e:
            logger.error(f"❌ Order execution failed: {e}")
            return []
    
    async def _update_portfolio_from_orders(self, processed_orders: list, market_data: Dict[str, Any]):
        """Update portfolio state from processed orders."""
        try:
            await self.portfolio_manager.update_from_orders_async(processed_orders, market_data)
            
            # Update trading metrics - CRITICAL FIX: Use async method (no market_data needed - PortfolioManager uses injected MarketDataManager)
            total_value = await self.portfolio_manager.get_total_value_async()
            self.trading_metrics.update_from_orders(processed_orders, total_value)
            
            # Save portfolio state periodically
            if self.cycle_id % 5 == 0:
                self.portfolio_manager.save_to_json()
                
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio from orders: {e}")
    
    async def _log_cycle_results(self, start_time: pd.Timestamp, l2_signals: list, processed_orders: list):
        """Log the results of the current cycle."""
        try:
            # Calculate cycle duration
            cycle_duration = (pd.Timestamp.utcnow() - start_time).total_seconds()
            
            # Count different types of orders
            valid_orders = [o for o in processed_orders if o.get("status") != "rejected"]
            rejected_orders = [o for o in processed_orders if o.get("status") == "rejected"]
            
            # Calculate actionable signals (excluding HOLD)
            actionable_signals = len([s for s in l2_signals if getattr(s, 'side', None) not in ['hold', 'HOLD']])
            
            # Log cycle summary
            logger.info(
                f"📊 Cycle {self.cycle_id} | "
                f"Duration: {cycle_duration:.1f}s | "
                f"Signals: {len(l2_signals)} (actionable: {actionable_signals}) | "
                f"Orders: {len(valid_orders)} | "
                f"Rejected: {len(rejected_orders)}"
            )
            
            # Log periodic trading metrics
            if self.cycle_id % 10 == 0:
                self.trading_metrics.log_periodic_report()
                
        except Exception as e:
            logger.error(f"❌ Failed to log cycle results: {e}")
    
    async def _cleanup(self):
        """Clean up resources."""
        try:
            # Save final portfolio state
            self.portfolio_manager.save_to_json()
            logger.info("💾 Final portfolio state saved")
            
            # Close components
            for component in [self.data_feed, self.l2_processor]:
                if hasattr(component, "close"):
                    await component.close()
                    
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


async def run_runtime_loop(portfolio_manager: PortfolioManager, order_manager: OrderManager):
    """Run the HRM runtime loop."""
    runtime_loop = HRMRuntimeLoop(portfolio_manager, order_manager)
    await runtime_loop.run()