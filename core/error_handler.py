# -*- coding: utf-8 -*-
"""
Centralized Error Handling System - HRM Trading System

Standardizes error handling patterns across the entire codebase to eliminate
code duplication and ensure consistent behavior.
"""

import asyncio
import time
from typing import Any, Optional, Callable, Dict, Tuple, TypeVar
from contextlib import contextmanager
from core.logging import logger

T = TypeVar('T')


class ErrorHandler:
    """
    Centralized error handling system for HRM trading system.
    Provides standardized patterns for async operations, retries, and fallbacks.
    """

    @staticmethod
    async def async_with_fallback(
        primary_func: Callable[[], Any],
        fallback_func: Optional[Callable[[], Any]] = None,
        max_retries: int = 1,
        retry_delay: float = 1.0,
        operation_name: str = "operation",
        log_success: bool = True
    ) -> Tuple[Any, bool]:
        """
        Execute async function with automatic fallback and retry logic.

        Args:
            primary_func: Primary async function to execute
            fallback_func: Optional fallback function if primary fails
            max_retries: Maximum retry attempts for primary function
            retry_delay: Delay between retries in seconds
            operation_name: Name for logging purposes
            log_success: Whether to log successful operations

        Returns:
            Tuple of (result, success: bool)
        """
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    await asyncio.sleep(retry_delay)

                result = await primary_func()

                if log_success:
                    status_msg = f" (attempt {attempt + 1})" if attempt > 0 else ""
                    logger.info(f"✅ {operation_name} successful{status_msg}")

                return result, True

            except Exception as e:
                error_msg = f"❌ {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"

                if attempt == max_retries:
                    # Last attempt - log full error and try fallback
                    logger.error(error_msg)

                    if fallback_func:
                        try:
                            logger.info(f"🔄 Falling back to secondary {operation_name}...")
                            fallback_result = await fallback_func()

                            if log_success:
                                logger.info(f"✅ Fallback {operation_name} successful")

                            return fallback_result, True

                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback {operation_name} also failed: {fallback_error}")

                    return None, False
                else:
                    logger.warning(error_msg)

        return None, False

    @staticmethod
    async def load_market_data_with_fallback(
        loader,
        data_feed,
        operation_name: str = "market_data_loading"
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Specialized method for loading market data with common HRM pattern.

        Follows the pattern:
        1. Try realtime loader
        2. Fall back to data feed
        3. Validate result structure

        Args:
            loader: RealTimeDataLoader instance with get_realtime_data() method
            data_feed: DataFeed instance with get_market_data() method
            operation_name: Name for logging

        Returns:
            Tuple of (market_data, success: bool)
        """

        async def primary_load():
            data = await loader.get_realtime_data()

            # Validate received data
            if data is None or (isinstance(data, dict) and len(data) == 0):
                raise ValueError("Empty or None market data from realtime loader")

            if not isinstance(data, dict):
                logger.warning(f"Unexpected data type from loader: {type(data)}")
                raise ValueError(f"Invalid data type: expected dict, got {type(data)}")

            return data

        async def fallback_load():
            data = await data_feed.get_market_data()

            if data is None or (isinstance(data, dict) and len(data) == 0):
                raise ValueError("Empty or None market data from data feed")

            if not isinstance(data, dict):
                logger.warning(f"Unexpected data type from data feed: {type(data)}")
                raise ValueError(f"Invalid data type: expected dict, got {type(data)}")

            return data

        return await ErrorHandler.async_with_fallback(
            primary_func=primary_load,
            fallback_func=fallback_load,
            max_retries=1,  # One retry for primary
            retry_delay=5.0,
            operation_name=operation_name,
            log_success=False  # We handle logging here
        )

    @staticmethod
    def sync_with_fallback(
        primary_func: Callable[[], T],
        fallback_func: Optional[Callable[[], T]] = None,
        max_retries: int = 1,
        retry_delay: float = 0.1,
        operation_name: str = "sync_operation",
        log_success: bool = True
    ) -> Tuple[Optional[T], bool]:
        """
        Execute synchronous function with fallback and retry logic.

        Args:
            primary_func: Primary synchronous function to execute
            fallback_func: Optional fallback function if primary fails
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
            operation_name: Name for logging
            log_success: Whether to log successful operations

        Returns:
            Tuple of (result, success: bool)
        """
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    time.sleep(retry_delay)

                result = primary_func()

                if log_success:
                    status_msg = f" (attempt {attempt + 1})" if attempt > 0 else ""
                    logger.info(f"✅ {operation_name} successful{status_msg}")

                return result, True

            except Exception as e:
                error_msg = f"❌ {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"

                if attempt == max_retries:
                    logger.error(error_msg)

                    if fallback_func:
                        try:
                            logger.info(f"🔄 Falling back to secondary {operation_name}...")
                            fallback_result = fallback_func()

                            if log_success:
                                logger.info(f"✅ Fallback {operation_name} successful")

                            return fallback_result, True

                        except Exception as fallback_error:
                            logger.error(f"❌ Fallback {operation_name} also failed: {fallback_error}")

                    return None, False
                else:
                    logger.warning(error_msg)

        return None, False

    @staticmethod
    @contextmanager
    def safe_operation(operation_name: str = "operation"):
        """
        Context manager for safe operations with standardized error logging.

        Usage:
            with ErrorHandler.safe_operation("database_update"):
                # Your code here
                pass
        """
        try:
            yield
            logger.debug(f"✅ {operation_name} completed successfully")
        except Exception as e:
            logger.error(f"❌ {operation_name} failed: {e}")
            raise

    @staticmethod
    def log_and_continue(operation_name: str):
        """
        Decorator to log errors but continue execution.

        Usage:
            @ErrorHandler.log_and_continue("signal_processing")
            def process_signals(self, signals):
                # Your code that might fail
                pass
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"❌ {operation_name} error (continuing): {e}")
                    return None
            return wrapper
        return decorator

    @staticmethod
    def retry_on_failure(
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        exceptions: Tuple = (Exception,)
    ):
        """
        Decorator to retry operations on failure.

        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier for delay
            exceptions: Tuple of exception types to catch
        """
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                current_delay = delay
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} in {current_delay:.1f}s: {e}")
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff

                logger.error(f"❌ Operation failed after {max_retries + 1} attempts: {last_exception}")
                raise last_exception

            def sync_wrapper(*args, **kwargs):
                current_delay = delay
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(f"⚠️ Retry {attempt + 1}/{max_retries} in {current_delay:.1f}s: {e}")
                            time.sleep(current_delay)
                            current_delay *= backoff

                logger.error(f"❌ Operation failed after {max_retries + 1} attempts: {last_exception}")
                raise last_exception

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator


# Convenience functions for backward compatibility
async def async_with_fallback(
    primary_func: Callable[[], Any],
    fallback_func: Optional[Callable[[], Any]] = None,
    max_retries: int = 1,
    retry_delay: float = 1.0,
    operation_name: str = "operation",
    log_success: bool = True
) -> Tuple[Any, bool]:
    """Backward compatibility wrapper."""
    return await ErrorHandler.async_with_fallback(
        primary_func, fallback_func, max_retries, retry_delay, operation_name, log_success
    )

def sync_with_fallback(
    primary_func: Callable[[], Any],
    fallback_func: Optional[Callable[[], Any]] = None,
    max_retries: int = 1,
    retry_delay: float = 0.1,
    operation_name: str = "sync_operation",
    log_success: bool = True
) -> Tuple[Optional[Any], bool]:
    """Backward compatibility wrapper."""
    return ErrorHandler.sync_with_fallback(
        primary_func, fallback_func, max_retries, retry_delay, operation_name, log_success
    )
