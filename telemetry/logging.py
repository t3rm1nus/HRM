#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HRM Telemetry Logging Module

This module provides centralized logging functionality for the HRM system.
It handles structured logging, log rotation, and different log levels.
"""

import logging
import logging.handlers
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from pathlib import Path

from core.config import get_config


class HRMLogger:
    """Centralized logging system for HRM."""
    
    def __init__(self, name: str = "HRM", log_level: str = "INFO"):
        self.name = name
        self.log_level = log_level.upper()
        
        # Setup logging configuration
        self._setup_logging()
        
        # Get logger instance
        self.logger = logging.getLogger(name)
        
        # Performance tracking
        self.performance_metrics = {}
        
    def _setup_logging(self):
        """Setup logging configuration with rotation and formatting."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        console_handler.setFormatter(simple_formatter)
        
        # File handlers with rotation
        # Main log file
        main_log_file = logs_dir / "hrm_main.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(detailed_formatter)
        
        # Error log file
        error_log_file = logs_dir / "hrm_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Performance log file
        perf_log_file = logs_dir / "hrm_performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(main_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(perf_handler)
        
        # Suppress verbose logs from external libraries
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        
        # Suppress TensorFlow warnings if present
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
    def log_system_startup(self, components: Dict[str, Any]):
        """Log system startup information."""
        startup_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'system_startup',
            'components': components,
            'python_version': sys.version,
            'platform': os.name
        }
        
        self.logger.info(f"🚀 HRM System Startup | Components: {list(components.keys())}")
        self._log_structured('startup', startup_info)
    
    def log_system_shutdown(self, shutdown_reason: str = "normal"):
        """Log system shutdown information."""
        shutdown_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'system_shutdown',
            'reason': shutdown_reason
        }
        
        self.logger.info(f"🛑 HRM System Shutdown | Reason: {shutdown_reason}")
        self._log_structured('shutdown', shutdown_info)
    
    def log_cycle_start(self, cycle_id: int, market_data: Dict[str, Any]):
        """Log cycle start information."""
        cycle_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'cycle_start',
            'cycle_id': cycle_id,
            'market_symbols': list(market_data.keys()) if market_data else [],
            'portfolio_value': self._get_portfolio_value(market_data)
        }
        
        self.logger.info(f"🔄 Cycle {cycle_id} Started | Symbols: {len(market_data) if market_data else 0}")
        self._log_structured('cycle', cycle_info)
    
    def log_cycle_end(self, cycle_id: int, execution_stats: Dict[str, Any], duration: float):
        """Log cycle end information."""
        cycle_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'cycle_end',
            'cycle_id': cycle_id,
            'duration_seconds': duration,
            'execution_stats': execution_stats,
            'success_rate': self._calculate_success_rate(execution_stats)
        }
        
        self.logger.info(f"📊 Cycle {cycle_id} Completed | Duration: {duration:.2f}s | Success Rate: {cycle_info['success_rate']:.1%}")
        self._log_structured('cycle', cycle_info)
    
    def log_signal_processing(self, signal_type: str, signals: list, l3_context: Dict[str, Any]):
        """Log signal processing information."""
        signal_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'signal_processing',
            'signal_type': signal_type,
            'signal_count': len(signals),
            'l3_regime': l3_context.get('regime', 'unknown'),
            'l3_confidence': l3_context.get('confidence', 0.0),
            'signals': [self._serialize_signal(s) for s in signals]
        }
        
        self.logger.info(f"🎯 Signal Processing | Type: {signal_type} | Count: {len(signals)} | L3: {l3_context.get('regime', 'unknown')}")
        self._log_structured('signals', signal_info)
    
    def log_order_execution(self, orders: list, execution_results: list):
        """Log order execution information."""
        execution_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'order_execution',
            'order_count': len(orders),
            'successful_orders': len([r for r in execution_results if r.get('status') == 'filled']),
            'failed_orders': len([r for r in execution_results if r.get('status') == 'failed']),
            'rejected_orders': len([r for r in execution_results if r.get('status') == 'rejected']),
            'orders': orders,
            'results': execution_results
        }
        
        success_rate = self._calculate_order_success_rate(execution_results)
        self.logger.info(f"💰 Order Execution | Total: {len(orders)} | Success Rate: {success_rate:.1%}")
        self._log_structured('execution', execution_info)
    
    def log_portfolio_update(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]):
        """Log portfolio update information."""
        portfolio_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'portfolio_update',
            'portfolio_state': portfolio_state,
            'total_value': self._get_portfolio_value(market_data),
            'asset_distribution': self._calculate_asset_distribution(portfolio_state, market_data)
        }
        
        self.logger.info(f"💼 Portfolio Update | Total Value: ${portfolio_info['total_value']:.2f}")
        self._log_structured('portfolio', portfolio_info)
    
    def log_performance_metric(self, metric_name: str, value: Union[float, int, str], context: Dict[str, Any] = None):
        """Log performance metrics."""
        metric_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'performance_metric',
            'metric_name': metric_name,
            'value': value,
            'context': context or {}
        }
        
        self.logger.info(f"📈 Performance Metric | {metric_name}: {value}")
        self._log_structured('performance', metric_info)
        
        # Store in performance metrics for reporting
        self.performance_metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.utcnow(),
            'context': context
        }
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None, exc_info: bool = False):
        """Log error information."""
        error_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {}
        }
        
        self.logger.error(f"❌ Error | Type: {error_type} | Message: {error_message}")
        self._log_structured('errors', error_info, level='ERROR')
    
    def log_warning(self, warning_type: str, warning_message: str, context: Dict[str, Any] = None):
        """Log warning information."""
        warning_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'warning',
            'warning_type': warning_type,
            'warning_message': warning_message,
            'context': context or {}
        }
        
        self.logger.warning(f"⚠️ Warning | Type: {warning_type} | Message: {warning_message}")
        self._log_structured('warnings', warning_info, level='WARNING')
    
    def log_debug(self, debug_message: str, context: Dict[str, Any] = None):
        """Log debug information."""
        debug_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'debug',
            'message': debug_message,
            'context': context or {}
        }
        
        self.logger.debug(f"🔍 Debug | {debug_message}")
        self._log_structured('debug', debug_info, level='DEBUG')
    
    def _log_structured(self, log_type: str, data: Dict[str, Any], level: str = 'INFO'):
        """Log structured data to JSON log files."""
        try:
            # Create structured log directory
            structured_dir = Path("logs/structured")
            structured_dir.mkdir(exist_ok=True)
            
            # Create log file for this type
            log_file = structured_dir / f"{log_type}.jsonl"
            
            # Add log level to data
            data['log_level'] = level
            
            # Write to JSONL file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to write structured log: {e}")
    
    def _serialize_signal(self, signal) -> Dict[str, Any]:
        """Serialize a signal object to dictionary."""
        try:
            if hasattr(signal, '__dict__'):
                return signal.__dict__
            elif isinstance(signal, dict):
                return signal
            else:
                return {'signal_data': str(signal)}
        except Exception:
            return {'serialization_error': True}
    
    def _get_portfolio_value(self, market_data: Dict[str, Any]) -> float:
        """Calculate portfolio value from market data."""
        try:
            # This would typically come from PortfolioManager
            # For now, return a placeholder
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_success_rate(self, execution_stats: Dict[str, Any]) -> float:
        """Calculate order execution success rate."""
        try:
            total = execution_stats.get('total_orders', 0)
            successful = execution_stats.get('successful_executions', 0)
            return successful / total if total > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_order_success_rate(self, execution_results: list) -> float:
        """Calculate order execution success rate from results."""
        try:
            total = len(execution_results)
            successful = len([r for r in execution_results if r.get('status') == 'filled'])
            return successful / total if total > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_asset_distribution(self, portfolio_state: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate asset distribution percentages."""
        try:
            distribution = {}
            total_value = self._get_portfolio_value(market_data)
            
            if total_value > 0:
                for symbol, balance in portfolio_state.items():
                    if symbol != 'total_value':
                        asset_value = balance * self._get_symbol_price(symbol, market_data)
                        distribution[symbol] = asset_value / total_value
            
            return distribution
        except Exception:
            return {}
    
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
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance metrics report."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': self.performance_metrics,
            'summary': {
                'total_metrics': len(self.performance_metrics),
                'latest_update': max([m['timestamp'] for m in self.performance_metrics.values()]) if self.performance_metrics else None
            }
        }
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        try:
            logs_dir = Path("logs")
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            for log_file in logs_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.logger.info(f"🧹 Cleaned up old log file: {log_file}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")


# Global logger instance
logger = HRMLogger()


def get_logger(name: str = None) -> HRMLogger:
    """Get logger instance."""
    if name:
        return HRMLogger(name)
    return logger