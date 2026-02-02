#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Telemetry Metrics Module

This module provides metrics collection and reporting functionality for the HRM system.
It tracks performance, trading metrics, and system health.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from core.logging import logger


@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    rejected_trades: int = 0
    total_profit_loss: float = 0.0
    average_profit_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    def update_from_orders(self, orders: List[Dict[str, Any]], portfolio_value: float):
        """Update trading metrics from executed orders."""
        for order in orders:
            if order.get('status') == 'filled':
                self.total_trades += 1
                self.successful_trades += 1
                self.total_profit_loss += order.get('profit_loss', 0.0)
            elif order.get('status') == 'failed':
                self.total_trades += 1
                self.failed_trades += 1
            elif order.get('status') == 'rejected':
                self.rejected_trades += 1
        
        if self.successful_trades > 0:
            self.average_profit_loss = self.total_profit_loss / self.successful_trades
        
        if self.total_trades > 0:
            self.win_rate = self.successful_trades / self.total_trades
        
        # Calculate profit factor (gains/losses)
        if self.total_profit_loss != 0:
            self.profit_factor = abs(self.total_profit_loss) / max(1, self.failed_trades)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get trading metrics summary."""
        return {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'rejected_trades': self.rejected_trades,
            'total_profit_loss': self.total_profit_loss,
            'average_profit_loss': self.average_profit_loss,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""
    uptime_seconds: float = 0.0
    cycles_completed: int = 0
    average_cycle_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    error_count: int = 0
    warning_count: int = 0
    
    def update_cycle_metrics(self, cycle_duration: float):
        """Update cycle-related metrics."""
        self.cycles_completed += 1
        self.uptime_seconds += cycle_duration
        
        # Calculate rolling average cycle time
        if self.cycles_completed == 1:
            self.average_cycle_time = cycle_duration
        else:
            self.average_cycle_time = (
                (self.average_cycle_time * (self.cycles_completed - 1)) + cycle_duration
            ) / self.cycles_completed
    
    def get_summary(self) -> Dict[str, Any]:
        """Get system metrics summary."""
        return {
            'uptime_seconds': self.uptime_seconds,
            'cycles_completed': self.cycles_completed,
            'average_cycle_time': self.average_cycle_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'active_connections': self.active_connections,
            'error_count': self.error_count,
            'warning_count': self.warning_count
        }


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_value: float = 0.0
    initial_value: float = 0.0
    current_roi: float = 0.0
    max_portfolio_value: float = 0.0
    min_portfolio_value: float = float('inf')
    asset_distribution: Dict[str, float] = None
    rebalance_count: int = 0
    last_rebalance_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.asset_distribution is None:
            self.asset_distribution = {}
    
    def update_portfolio_value(self, value: float, timestamp: datetime):
        """Update portfolio value and calculate metrics."""
        self.total_value = value
        
        if self.max_portfolio_value == 0 or value > self.max_portfolio_value:
            self.max_portfolio_value = value
        
        if value < self.min_portfolio_value:
            self.min_portfolio_value = value
        
        if self.initial_value == 0:
            self.initial_value = value
        
        if self.initial_value > 0:
            self.current_roi = (value - self.initial_value) / self.initial_value
    
    def update_asset_distribution(self, distribution: Dict[str, float]):
        """Update asset distribution."""
        self.asset_distribution = distribution.copy()
    
    def record_rebalance(self):
        """Record a portfolio rebalance."""
        self.rebalance_count += 1
        self.last_rebalance_time = datetime.utcnow()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio metrics summary."""
        return {
            'total_value': self.total_value,
            'initial_value': self.initial_value,
            'current_roi': self.current_roi,
            'max_portfolio_value': self.max_portfolio_value,
            'min_portfolio_value': self.min_portfolio_value if self.min_portfolio_value != float('inf') else 0.0,
            'asset_distribution': self.asset_distribution,
            'rebalance_count': self.rebalance_count,
            'last_rebalance_time': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
        }


class MetricsCollector:
    """Central metrics collector for HRM system."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.trading_metrics = TradingMetrics()
        self.system_metrics = SystemMetrics()
        self.portfolio_metrics = PortfolioMetrics()
        
        # Historical data
        self.cycle_times = []
        self.portfolio_values = []
        self.error_history = []
        
        # Metrics persistence
        self.metrics_dir = Path("logs/metrics")
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Start background metrics collection
        self._metrics_task = None
    
    async def start(self):
        """Start background metrics collection."""
        if self._metrics_task is None:
            self._metrics_task = asyncio.create_task(self._collect_system_metrics())
            logger.info("📊 Metrics collection started")
    
    async def stop(self):
        """Stop background metrics collection."""
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
            self._metrics_task = None
            logger.info("📊 Metrics collection stopped")
    
    async def _collect_system_metrics(self):
        """Background task to collect system metrics."""
        while True:
            try:
                # Collect system metrics
                await self._update_system_metrics()
                
                # Save metrics periodically
                if len(self.cycle_times) % 10 == 0:
                    await self.save_metrics()
                
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_metrics.memory_usage_mb = memory.used / (1024 * 1024)
            
            # CPU usage
            self.system_metrics.cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            # Uptime
            current_uptime = (datetime.utcnow() - self.start_time).total_seconds()
            self.system_metrics.uptime_seconds = current_uptime
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.warning(f"⚠️ System metrics collection failed: {e}")
    
    def record_cycle_completion(self, cycle_duration: float):
        """Record completion of a system cycle."""
        self.system_metrics.update_cycle_metrics(cycle_duration)
        self.cycle_times.append({
            'timestamp': datetime.utcnow().isoformat(),
            'duration': cycle_duration
        })
        
        # Keep only last 1000 cycle times
        if len(self.cycle_times) > 1000:
            self.cycle_times = self.cycle_times[-1000:]
    
    def record_trading_activity(self, orders: List[Dict[str, Any]], portfolio_value: float):
        """Record trading activity and update metrics."""
        self.trading_metrics.update_from_orders(orders, portfolio_value)
        
        # Record portfolio value history
        self.portfolio_values.append({
            'timestamp': datetime.utcnow().isoformat(),
            'value': portfolio_value
        })
        
        # Keep only last 1000 portfolio values
        if len(self.portfolio_values) > 1000:
            self.portfolio_values = self.portfolio_values[-1000:]
    
    def record_portfolio_update(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]):
        """Record portfolio state update."""
        # Calculate total portfolio value
        total_value = 0.0
        distribution = {}
        
        for symbol, balance in portfolio_state.items():
            if symbol == 'total_value':
                total_value = balance
            else:
                # Get current price for the symbol
                price = self._get_symbol_price(symbol, market_data)
                asset_value = balance * price
                total_value += asset_value
                distribution[symbol] = asset_value
        
        # Normalize distribution to percentages
        if total_value > 0:
            distribution = {k: v / total_value for k, v in distribution.items()}
        
        self.portfolio_metrics.update_portfolio_value(total_value, datetime.utcnow())
        self.portfolio_metrics.update_asset_distribution(distribution)
    
    def record_rebalance(self):
        """Record a portfolio rebalance."""
        self.portfolio_metrics.record_rebalance()
    
    def record_error(self, error_type: str, error_message: str):
        """Record an error occurrence."""
        self.system_metrics.error_count += 1
        self.error_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': error_type,
            'message': error_message
        })
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def record_warning(self, warning_type: str, warning_message: str):
        """Record a warning occurrence."""
        self.system_metrics.warning_count += 1
    
    def _get_symbol_price(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Get current price for a symbol."""
        try:
            symbol_data = market_data.get(symbol, {})
            if isinstance(symbol_data, dict) and 'close' in symbol_data:
                return float(symbol_data['close'])
            elif isinstance(symbol_data, pd.DataFrame) and not symbol_data.empty:
                return float(symbol_data['close'].iloc[-1])
            return 0.0
        except Exception:
            return 0.0
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'trading': self.trading_metrics.get_summary(),
            'system': self.system_metrics.get_summary(),
            'portfolio': self.portfolio_metrics.get_summary(),
            'historical': {
                'cycle_count': len(self.cycle_times),
                'portfolio_value_history_count': len(self.portfolio_values),
                'error_history_count': len(self.error_history)
            }
        }
    
    def log_periodic_report(self):
        """Log periodic metrics report."""
        summary = self.get_metrics_summary()
        
        # Log trading metrics
        trading = summary['trading']
        logger.info(
            f"📊 Trading Metrics | "
            f"Total: {trading['total_trades']} | "
            f"Success: {trading['successful_trades']} | "
            f"Win Rate: {trading['win_rate']:.1%} | "
            f"Profit/Loss: ${trading['total_profit_loss']:.2f}"
        )
        
        # Log system metrics
        system = summary['system']
        logger.info(
            f"💻 System Metrics | "
            f"Uptime: {system['uptime_seconds']:.0f}s | "
            f"Cycles: {system['cycles_completed']} | "
            f"Avg Cycle: {system['average_cycle_time']:.2f}s | "
            f"Memory: {system['memory_usage_mb']:.1f}MB"
        )
        
        # Log portfolio metrics
        portfolio = summary['portfolio']
        logger.info(
            f"💼 Portfolio Metrics | "
            f"Total Value: ${portfolio['total_value']:.2f} | "
            f"ROI: {portfolio['current_roi']:.1%} | "
            f"Rebalances: {portfolio['rebalance_count']}"
        )
    
    async def save_metrics(self):
        """Save metrics to persistent storage."""
        try:
            # Save current metrics
            metrics_file = self.metrics_dir / "current_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.get_metrics_summary(), f, indent=2, ensure_ascii=False)
            
            # Save historical data
            history_files = {
                'cycle_times': self.cycle_times,
                'portfolio_values': self.portfolio_values,
                'error_history': self.error_history
            }
            
            for filename, data in history_files.items():
                file_path = self.metrics_dir / f"{filename}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug("💾 Metrics saved to persistent storage")
            
        except Exception as e:
            logger.error(f"❌ Failed to save metrics: {e}")
    
    async def load_metrics(self):
        """Load metrics from persistent storage."""
        try:
            metrics_file = self.metrics_dir / "current_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Restore trading metrics
                    if 'trading' in data:
                        self.trading_metrics = TradingMetrics(**data['trading'])
                    
                    # Restore system metrics
                    if 'system' in data:
                        self.system_metrics = SystemMetrics(**data['system'])
                    
                    # Restore portfolio metrics
                    if 'portfolio' in data:
                        portfolio_data = data['portfolio']
                        if 'asset_distribution' in portfolio_data:
                            portfolio_data['asset_distribution'] = portfolio_data['asset_distribution'] or {}
                        if 'last_rebalance_time' in portfolio_data and portfolio_data['last_rebalance_time']:
                            portfolio_data['last_rebalance_time'] = datetime.fromisoformat(portfolio_data['last_rebalance_time'])
                        else:
                            portfolio_data['last_rebalance_time'] = None
                        self.portfolio_metrics = PortfolioMetrics(**portfolio_data)
            
            # Load historical data
            history_files = {
                'cycle_times': 'cycle_times',
                'portfolio_values': 'portfolio_values',
                'error_history': 'error_history'
            }
            
            for attr_name, filename in history_files.items():
                file_path = self.metrics_dir / f"{filename}.json"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        setattr(self, attr_name, json.load(f))
            
            logger.info("📂 Metrics loaded from persistent storage")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load metrics: {e}")
    
    def cleanup_old_metrics(self, days_to_keep: int = 7):
        """Clean up old metrics files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up metrics directory
            for metrics_file in self.metrics_dir.glob("*.json"):
                if metrics_file.stat().st_mtime < cutoff_date.timestamp():
                    metrics_file.unlink()
                    logger.info(f"🧹 Cleaned up old metrics file: {metrics_file}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector