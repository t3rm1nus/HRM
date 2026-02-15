"""
Async Balance Helper - Utility for detecting async context and enforcing async balance access

This module provides utilities to:
1. Detect if code is running in an async context
2. Enforce async balance access patterns
3. Provide logging for sync/async path usage
"""

import asyncio
import inspect
from functools import wraps
from typing import Optional, Any, Callable
from core.logging import logger


class AsyncContextDetector:
    """Detects if code is running in async context"""
    
    @staticmethod
    def is_in_async_context() -> bool:
        """
        Check if the current code is running in an async context.
        
        Returns:
            True if in async context, False otherwise
        """
        try:
            loop = asyncio.get_running_loop()
            return loop.is_running()
        except RuntimeError:
            return False
    
    @staticmethod
    def get_caller_info() -> str:
        """Get information about the calling function"""
        try:
            stack = inspect.stack()
            # Skip this function and the caller
            if len(stack) > 2:
                caller = stack[2]
                return f"{caller.filename}:{caller.lineno} in {caller.function}"
            return "unknown"
        except Exception:
            return "unknown"


class BalanceAccessLogger:
    """Logs balance access with sync/async path information"""
    
    @staticmethod
    def log_balance_access(asset: str, path_type: str, value: float, source: str = ""):
        """
        Log balance access with path type (SYNC/ASYNC).
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')
            path_type: 'SYNC' or 'ASYNC'
            value: Balance value
            source: Source of the balance (e.g., 'exchange', 'cache', 'fallback')
        """
        caller_info = AsyncContextDetector.get_caller_info()
        logger.info(
            f"[BALANCE_ACCESS] {path_type} | Asset: {asset} | "
            f"Value: {value:.6f} | Source: {source} | From: {caller_info}"
        )
    
    @staticmethod
    def log_sync_in_async_error(method_name: str):
        """
        Log error when sync method is called in async context.
        
        Args:
            method_name: Name of the sync method that was called
        """
        caller_info = AsyncContextDetector.get_caller_info()
        logger.error(
            f"[ASYNC_VIOLATION] Sync method '{method_name}' called in async context! "
            f"Use the async version instead. From: {caller_info}"
        )


class AsyncBalanceRequiredError(Exception):
    """Exception raised when sync balance access is attempted in async context"""
    pass


def enforce_async_balance_access(sync_method: Callable) -> Callable:
    """
    Decorator that enforces async balance access in async contexts.
    
    If called in an async context, raises AsyncBalanceRequiredError.
    If called in sync context, proceeds normally.
    
    Usage:
        @enforce_async_balance_access
        def get_balance_sync(self, asset: str) -> float:
            ...
    """
    @wraps(sync_method)
    def wrapper(*args, **kwargs):
        if AsyncContextDetector.is_in_async_context():
            method_name = sync_method.__qualname__
            BalanceAccessLogger.log_sync_in_async_error(method_name)
            raise AsyncBalanceRequiredError(
                f"Cannot call sync method '{method_name}' in async context. "
                f"Use the async version (e.g., '{method_name}_async') instead."
            )
        return sync_method(*args, **kwargs)
    return wrapper


def async_aware_balance_access(async_method: Callable, sync_method: Callable) -> Callable:
    """
    Creates a method that automatically chooses between async and sync versions
    based on the current context.
    
    This is a helper for backward compatibility during migration.
    
    Args:
        async_method: The async version of the method
        sync_method: The sync version of the method
        
    Returns:
        A method that routes to the appropriate version
    """
    @wraps(async_method)
    async def async_wrapper(*args, **kwargs):
        return await async_method(*args, **kwargs)
    
    @wraps(sync_method)
    def sync_wrapper(*args, **kwargs):
        return sync_method(*args, **kwargs)
    
    def wrapper(*args, **kwargs):
        if AsyncContextDetector.is_in_async_context():
            # We're in async context - must use async method
            # Note: This returns a coroutine that must be awaited
            return async_wrapper(*args, **kwargs)
        else:
            # We're in sync context - can use sync method
            return sync_wrapper(*args, **kwargs)
    
    # Attach both versions for explicit access
    wrapper.async_version = async_wrapper
    wrapper.sync_version = sync_wrapper
    
    # Mark wrapper to help with type checking
    wrapper._is_async_aware = True
    
    return wrapper


class BalanceVerificationStatus:
    """Tracks whether balances were obtained from verified async sync"""
    
    def __init__(self):
        self.last_sync_source: Optional[str] = None
        self.last_sync_timestamp: Optional[float] = None
        self.was_fallback_used: bool = False
        self.sync_errors: list = []
    
    def mark_synced(self, source: str):
        """Mark balances as properly synced from exchange"""
        import time
        self.last_sync_source = source
        self.last_sync_timestamp = time.time()
        self.was_fallback_used = False
        logger.info(f"[BALANCE_VERIFICATION] Balances marked as synced from: {source}")
    
    def mark_fallback(self, reason: str):
        """Mark that fallback balances were used"""
        self.was_fallback_used = True
        self.sync_errors.append(reason)
        logger.warning(f"[BALANCE_VERIFICATION] Fallback balances used: {reason}")
    
    def is_verified(self) -> bool:
        """Check if current balances are from verified async sync"""
        return self.last_sync_source is not None and not self.was_fallback_used
    
    def get_status(self) -> dict:
        """Get current verification status"""
        import time
        return {
            'is_verified': self.is_verified(),
            'last_sync_source': self.last_sync_source,
            'seconds_since_sync': time.time() - self.last_sync_timestamp if self.last_sync_timestamp else None,
            'was_fallback_used': self.was_fallback_used,
            'sync_errors': self.sync_errors[-5:]  # Last 5 errors
        }