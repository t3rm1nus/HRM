"""
HRM Core Module - Central services and utilities.

This module provides essential services for the HRM trading system including:
- Error handling and exception hierarchy
- Configuration management
- State management
- Logging system
- Portfolio management
- Model factory for unified model creation
- Utility functions
"""

from .exceptions import (
    # Base exception
    HRMException,
    # Error categories
    ConfigurationError, EnvironmentError, ValidationError,
    TradingError, SignalError, OrderError, RiskError, PositionError,
    ModelError, AILError, InferenceError, FactoryError,
    DataError, ExchangeError, APIError, ConnectivityError, RateLimitError,
    SystemError, InitializationError, PersistenceError, LoggingError,
    LearningError, OverfittingError, TrainingError,
    # Helper functions
    safe_execute, with_error_handling, create_error_response, log_and_raise,
    # Error factories
    signal_validation_error, order_execution_error, model_inference_error,
    connectivity_error, configuration_missing_error
)
from .model_factory import TradingModelFactory, model_factory, get_model_factory
from .configuration_manager import HRMConfigurationManager, get_config_manager, get_config_value, set_config_value, HRMConfig
from .async_processor import HRMAsyncProcessor, get_async_processor, ProcessingResult
from .memory_manager import HRMMemoryManager, get_memory_manager, cache_model, get_cached_model, MemoryStats
from .configuration_manager import HRMConfig as LegacyHRMConfig  # Keep legacy compatibility
from .logging import setup_logger, logger
from .portfolio_manager import PortfolioManager

__all__ = [
    # Error Handling & Exceptions - NEW
    'HRMException',
    'ConfigurationError', 'EnvironmentError', 'ValidationError',
    'TradingError', 'SignalError', 'OrderError', 'RiskError', 'PositionError',
    'ModelError', 'AILError', 'InferenceError', 'FactoryError',
    'DataError', 'ExchangeError', 'APIError', 'ConnectivityError', 'RateLimitError',
    'SystemError', 'InitializationError', 'PersistenceError', 'LoggingError',
    'LearningError', 'OverfittingError', 'TrainingError',
    'safe_execute', 'with_error_handling', 'create_error_response', 'log_and_raise',
    'signal_validation_error', 'order_execution_error', 'model_inference_error',
    'connectivity_error', 'configuration_missing_error',

    # Configuration Management
    'HRMConfigurationManager',
    'get_config_manager',
    'get_config_value',
    'set_config_value',
    'HRMConfig',

    # Async Processing - NEW
    'HRMAsyncProcessor',
    'get_async_processor',
    'ProcessingResult',

    # Model factory
    'TradingModelFactory',
    'model_factory',
    'get_model_factory',

    # Memory Management - NEW
    'HRMMemoryManager',
    'get_memory_manager',
    'cache_model',
    'get_cached_model',
    'MemoryStats',

    # Existing services
    'LegacyHRMConfig',
    'setup_logger',
    'logger',
    'PortfolioManager'
]


# Example usage:
#
# # Error Handling - Consistent exception hierarchy:
# from core import SignalError, safe_execute, with_error_handling, signal_validation_error
#
# # Standardized error raising
# raise signal_validation_error('sig_123', 'insufficient_confidence', {'confidence': 0.3})
#
# # Safe execution with fallback
# result = safe_execute(lambda: risky_operation(), fallback=[])
#
# # Error handling decorator
# @with_error_handling("model_inference", InferenceError)
# def infer_signal(self, data):
#     # ... implementation
#
# # Configuration Management:
# from core import get_config_manager, get_config_value, set_config_value
#
# # Load environment-specific config
# config_mgr = get_config_manager('live')  # live, testnet, backtest, dev
# balance = config_mgr.get('trading.initial_balance')
#
# # Get values by dotted path
# symbols = get_config_value('trading.symbols', ['BTCUSDT'])
# risk_pct = get_config_value('trading.risk_per_trade_percent')
#
# # Set runtime values
# set_config_value('trading.hrm_path_mode', 'PATH3')
#
# # Model Factory:
# from core import model_factory
#
# # Create all L2 models
# l2_models = model_factory.create_models_for_level('L2')
#
# # Create specific model
# trend_analyzer = model_factory.create_model('TechnicalAnalyzer', 'L2')
#
# # Instantiate with custom config
# config = {'threshold': 0.8}
# lr_model = model_factory.create_model('LogisticRegression', 'L1', config)

# # Async Processing - Concurrent L1/L2 operations:
# from core import get_async_processor, HRMAsyncProcessor
# import asyncio
#
# async def run_optimized_cycle():
#     processor = get_async_processor(max_workers=4)
#     await processor.initialize()
#
#     # Execute full async trading cycle (6-8s vs 8-10s sync)
#     result = await processor.execute_trading_cycle_async(
#         market_data={'BTC': 50000, 'ETH': 3000},
#         portfolio_state={'balance': 1000, 'max_position_size_usdt': 600}
#     )
#
#     print(f"Cycle completed in {result['execution_time']:.2f}s")
#     return result
#
# # Usage: asyncio.run(run_optimized_cycle())

# # Memory Management - Model caching and cleanup:
# from core import get_memory_manager, cache_model, get_cached_model
#
# # Get memory manager
# mem_mgr = get_memory_manager()
#
# # Cache trained model with automatic cleanup
# model = train_expensive_model()
# cache_model('l1_lr_model', model, {'features': ['rsi', 'macd']})
#
# # Retrieve from cache (fast, avoids retraining)
# cached_model = get_cached_model('l1_lr_model')
# if cached_model:
#     prediction = cached_model.predict(new_data)
#
# # Memory statistics and cleanup
# stats = mem_mgr.get_memory_stats()
# print(f"Cache using: {stats.cache_memory_usage_mb:.1f}MB")
#
# # Manual cleanup for maintenance
# mem_mgr.manual_cleanup(force_gc=True)</content>
