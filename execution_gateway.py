#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Execution Gateway Module

This module provides the single point of execution for the HRM system.
It handles order validation, execution, and risk management.
"""

import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from core.logging import logger
from core.portfolio_manager import PortfolioManager
from core.error_handler import ErrorHandler
from core.trading_metrics import TradingMetrics

from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l1_operational.order_validators import OrderValidator
from l1_operational.risk_guard import RiskGuard

from l2_tactic.tactical_signal_processor import TacticalSignal
from l2_tactic.config import L2Config

from comms.config import config


class ExecutionGateway:
    """Single point of execution for HRM system orders."""
    
    def __init__(self, portfolio_manager: PortfolioManager, order_manager: OrderManager):
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.binance_client = order_manager.binance_client
        
        # Initialize execution components
        self.validator = OrderValidator(portfolio_manager)
        self.risk_guard = RiskGuard(portfolio_manager)
        self.trading_metrics = TradingMetrics()
        
        # Execution state
        self.execution_stats = {
            'total_orders': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'rejected_orders': 0,
            'risk_blocks': 0
        }
    
    async def execute_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a list of orders through the gateway."""
        if not orders:
            return []
        
        logger.info(f"💰 Gateway: Processing {len(orders)} orders")
        
        # Update execution stats
        self.execution_stats['total_orders'] += len(orders)
        
        # 1. Validate all orders
        validated_orders = await self._validate_orders(orders)
        
        # 2. Apply risk management
        risk_checked_orders = await self._apply_risk_management(validated_orders)
        
        # 3. Execute orders
        executed_orders = await self._execute_validated_orders(risk_checked_orders)
        
        # 4. Update portfolio and metrics
        await self._update_portfolio_and_metrics(executed_orders)
        
        # 5. Log execution results
        await self._log_execution_results(orders, validated_orders, risk_checked_orders, executed_orders)
        
        return executed_orders
    
    async def _validate_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate orders before execution."""
        validated_orders = []
        
        for order in orders:
            try:
                # Basic order validation
                validation_result = self.validator.validate_order(order)
                
                if validation_result['valid']:
                    # Portfolio-specific validation
                    portfolio_validation = self.validator.validate_portfolio_constraints(order)
                    
                    if portfolio_validation['valid']:
                        validated_orders.append(order)
                        logger.debug(f"✅ Order validated: {order.get('symbol')} {order.get('side')} {order.get('quantity')}")
                    else:
                        logger.warning(f"❌ Portfolio validation failed: {portfolio_validation['reason']}")
                        order['status'] = 'rejected'
                        order['validation_error'] = portfolio_validation['reason']
                        self.execution_stats['rejected_orders'] += 1
                        validated_orders.append(order)
                else:
                    logger.warning(f"❌ Order validation failed: {validation_result['reason']}")
                    order['status'] = 'rejected'
                    order['validation_error'] = validation_result['reason']
                    self.execution_stats['rejected_orders'] += 1
                    validated_orders.append(order)
                    
            except Exception as e:
                logger.error(f"❌ Order validation error: {e}")
                order['status'] = 'rejected'
                order['validation_error'] = str(e)
                self.execution_stats['rejected_orders'] += 1
                validated_orders.append(order)
        
        return validated_orders
    
    async def _apply_risk_management(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply risk management rules to orders."""
        risk_checked_orders = []
        
        for order in orders:
            try:
                # Check if order should be blocked by risk management
                risk_check = await self.risk_guard.check_order_risk(order)
                
                if risk_check['allowed']:
                    risk_checked_orders.append(order)
                    logger.debug(f"🛡️ Risk check passed: {order.get('symbol')} {order.get('side')}")
                else:
                    logger.warning(f"🛡️ Risk check blocked: {risk_check['reason']}")
                    order['status'] = 'rejected'
                    order['risk_error'] = risk_check['reason']
                    self.execution_stats['risk_blocks'] += 1
                    risk_checked_orders.append(order)
                    
            except Exception as e:
                logger.error(f"❌ Risk management error: {e}")
                order['status'] = 'rejected'
                order['risk_error'] = str(e)
                self.execution_stats['risk_blocks'] += 1
                risk_checked_orders.append(order)
        
        return risk_checked_orders
    
    async def _execute_validated_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute validated and risk-checked orders."""
        executed_orders = []
        
        for order in orders:
            if order.get('status') == 'rejected':
                executed_orders.append(order)
                continue
            
            try:
                # Execute order through order manager
                execution_result = await self.order_manager.execute_single_order(order)
                
                if execution_result.get('status') == 'filled':
                    self.execution_stats['successful_executions'] += 1
                    logger.info(f"✅ Order executed: {order.get('symbol')} {order.get('side')} {order.get('quantity')} @ {execution_result.get('fill_price')}")
                else:
                    self.execution_stats['failed_executions'] += 1
                    logger.warning(f"⚠️ Order failed: {order.get('symbol')} {order.get('side')} - {execution_result.get('status')}")
                
                executed_orders.append(execution_result)
                
            except Exception as e:
                logger.error(f"❌ Order execution failed: {e}")
                order['status'] = 'failed'
                order['execution_error'] = str(e)
                self.execution_stats['failed_executions'] += 1
                executed_orders.append(order)
        
        return executed_orders
    
    async def _update_portfolio_and_metrics(self, executed_orders: List[Dict[str, Any]]):
        """Update portfolio state and trading metrics."""
        try:
            # Update portfolio from executed orders - VERSIÓN PURA: no requiere market_data
            await self.portfolio_manager.update_from_orders_async(executed_orders)
            
            # Update trading metrics - CRITICAL FIX: Use async method (no market_data needed)
            total_value = await self.portfolio_manager.get_total_value_async()
            self.trading_metrics.update_from_orders(executed_orders, total_value)
            
            # Save portfolio state periodically
            if self.execution_stats['total_orders'] % 5 == 0:
                self.portfolio_manager.save_to_json()
                
        except Exception as e:
            logger.error(f"❌ Failed to update portfolio and metrics: {e}")
    
    async def _log_execution_results(self, original_orders: List[Dict[str, Any]], 
                                   validated_orders: List[Dict[str, Any]],
                                   risk_checked_orders: List[Dict[str, Any]],
                                   executed_orders: List[Dict[str, Any]]):
        """Log execution results and statistics."""
        try:
            # Count different types of orders
            successful = [o for o in executed_orders if o.get('status') == 'filled']
            failed = [o for o in executed_orders if o.get('status') == 'failed']
            rejected = [o for o in executed_orders if o.get('status') == 'rejected']
            
            # Log summary
            logger.info(
                f"💰 Gateway Results: "
                f"Original: {len(original_orders)} | "
                f"Validated: {len(validated_orders)} | "
                f"Risk-checked: {len(risk_checked_orders)} | "
                f"Executed: {len(executed_orders)} | "
                f"Successful: {len(successful)} | "
                f"Failed: {len(failed)} | "
                f"Rejected: {len(rejected)}"
            )
            
            # Log detailed stats periodically
            if self.execution_stats['total_orders'] % 10 == 0:
                logger.info(
                    f"📊 Gateway Stats (Total: {self.execution_stats['total_orders']}): "
                    f"Successful: {self.execution_stats['successful_executions']} | "
                    f"Failed: {self.execution_stats['failed_executions']} | "
                    f"Rejected: {self.execution_stats['rejected_orders']} | "
                    f"Risk Blocks: {self.execution_stats['risk_blocks']}"
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to log execution results: {e}")
    
    async def execute_signal(self, signal: TacticalSignal, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a single tactical signal."""
        try:
            # Convert signal to orders
            orders = await self._convert_signal_to_orders(signal, market_data)
            
            # Execute orders through gateway
            executed_orders = await self.execute_orders(orders)
            
            return executed_orders
            
        except Exception as e:
            logger.error(f"❌ Signal execution failed: {e}")
            return []
    
    async def _convert_signal_to_orders(self, signal: TacticalSignal, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert a tactical signal to executable orders."""
        try:
            symbol = getattr(signal, 'symbol', '')
            side = getattr(signal, 'side', 'hold')
            confidence = getattr(signal, 'confidence', 0.5)
            
            if side == 'hold':
                return []
            
            # Get current price for the symbol
            current_price = self._get_current_price(symbol, market_data)
            if not current_price:
                logger.warning(f"⚠️ No price data for {symbol}")
                return []
            
            # Calculate order size based on signal strength and portfolio allocation
            order_size = await self._calculate_order_size(signal, market_data, current_price)
            
            if order_size <= 0:
                logger.warning(f"⚠️ Invalid order size for {symbol}")
                return []
            
            # Create order
            order = {
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': order_size,
                'price': current_price,
                'confidence': confidence,
                'signal_source': 'tactical',
                'status': 'pending',
                'timestamp': pd.Timestamp.utcnow().isoformat()
            }
            
            logger.info(f"🎯 Signal converted to order: {symbol} {side} {order_size} @ {current_price}")
            return [order]
            
        except Exception as e:
            logger.error(f"❌ Signal conversion failed: {e}")
            return []
    
    def _get_current_price(self, symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            symbol_data = market_data.get(symbol, {})
            if isinstance(symbol_data, dict) and 'close' in symbol_data:
                return float(symbol_data['close'])
            elif isinstance(symbol_data, pd.DataFrame) and not symbol_data.empty:
                return float(symbol_data['close'].iloc[-1])
            return None
        except Exception:
            return None
    
    async def _calculate_order_size(self, signal: TacticalSignal, market_data: Dict[str, Any], current_price: float) -> float:
        """Calculate order size based on signal and portfolio constraints."""
        try:
            symbol = getattr(signal, 'symbol', '')
            confidence = getattr(signal, 'confidence', 0.5)
            
            # Get portfolio state - CRITICAL FIX: Use async method (no market_data needed)
            portfolio_state = self.portfolio_manager.get_portfolio_state()
            total_value = await self.portfolio_manager.get_total_value_async()
            
            # Calculate position size based on confidence and risk limits
            base_allocation = total_value * 0.10  # 10% base allocation
            risk_adjusted_allocation = base_allocation * confidence
            
            # Calculate quantity
            quantity = risk_adjusted_allocation / current_price
            
            # Apply minimum order size constraints
            min_notional = 10.0  # Minimum $10 order
            if quantity * current_price < min_notional:
                quantity = min_notional / current_price
            
            # Apply portfolio constraints
            max_position_size = total_value * 0.30  # Max 30% in single position
            max_quantity = max_position_size / current_price
            
            quantity = min(quantity, max_quantity)
            
            return max(0.0, quantity)
            
        except Exception as e:
            logger.error(f"❌ Order size calculation failed: {e}")
            return 0.0
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()
    
    async def cleanup_stale_orders(self, current_positions: Dict[str, float]):
        """Clean up stale stop-loss and profit-taking orders."""
        try:
            cleanup_stats = self.order_manager.cleanup_stale_orders(current_positions)
            logger.info(f"🧹 Gateway cleanup: {cleanup_stats}")
            return cleanup_stats
        except Exception as e:
            logger.error(f"❌ Gateway cleanup failed: {e}")
            return {}


async def create_execution_gateway(portfolio_manager: PortfolioManager, 
                                 order_manager: OrderManager) -> ExecutionGateway:
    """Create and initialize an execution gateway."""
    gateway = ExecutionGateway(portfolio_manager, order_manager)
    logger.info("✅ Execution Gateway initialized")
    return gateway