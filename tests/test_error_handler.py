# -*- coding: utf-8 -*-
"""
Test suite for unified error handling system - HRM Trading System

Comprehensive validation of centralized error handling patterns to eliminate
code duplication and ensure consistent behavior across the system.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.error_handler import ErrorHandler, async_with_fallback, sync_with_fallback


class TestAsyncWithFallback:
    """Test async_with_fallback functionality."""

    @pytest.mark.asyncio
    async def test_successful_primary_operation(self):
        """Test successful primary operation without fallback."""
        async def primary_func():
            return "success"

        result, success = await async_with_fallback(primary_func)
        assert result == "success"
        assert success is True

    @pytest.mark.asyncio
    async def test_fallback_after_primary_failure(self):
        """Test fallback execution after primary operation fails."""
        async def primary_func():
            raise ValueError("Primary failed")

        async def fallback_func():
            return "fallback_success"

        result, success = await async_with_fallback(
            primary_func=primary_func,
            fallback_func=fallback_func,
            max_retries=1,
            operation_name="test_operation"
        )

        assert result == "fallback_success"
        assert success is True

    @pytest.mark.asyncio
    async def test_both_operations_fail(self):
        """Test when both primary and fallback operations fail."""
        async def primary_func():
            raise ValueError("Primary failed")

        async def fallback_func():
            raise RuntimeError("Fallback also failed")

        result, success = await async_with_fallback(
            primary_func=primary_func,
            fallback_func=fallback_func,
            max_retries=1,
            operation_name="test_operation"
        )

        assert result is None
        assert success is False

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic with gradual success."""
        call_count = 0

        async def primary_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        result, success = await async_with_fallback(
            primary_func=primary_func,
            max_retries=3,
            operation_name="retry_test"
        )

        assert result == "success"
        assert success is True
        assert call_count == 3


class TestSyncWithFallback:
    """Test sync_with_fallback functionality."""

    def test_successful_primary_operation(self):
        """Test successful synchronous primary operation."""
        def primary_func():
            return "sync_success"

        result, success = sync_with_fallback(primary_func)
        assert result == "sync_success"
        assert success is True

    def test_fallback_after_sync_failure(self):
        """Test fallback execution after synchronous primary failure."""
        def primary_func():
            raise ValueError("Sync primary failed")

        def fallback_func():
            return "sync_fallback_success"

        result, success = sync_with_fallback(
            primary_func=primary_func,
            fallback_func=fallback_func,
            operation_name="sync_test"
        )

        assert result == "sync_fallback_success"
        assert success is True


class TestLoadMarketDataWithFallback:
    """Test specialized market data loading with fallback."""

    @pytest.mark.asyncio
    @patch('core.logging.logger')
    async def test_successful_realtime_loader(self, mock_logger):
        """Test successful loading from realtime loader."""
        # Create mock loader and data_feed
        mock_loader = MagicMock()
        mock_loader.get_realtime_data = AsyncMock(return_value={
            'BTCUSDT': {'close': 50000, 'volume': 100},
            'ETHUSDT': {'close': 3000, 'volume': 200}
        })

        mock_data_feed = MagicMock()

        result, success = await ErrorHandler.load_market_data_with_fallback(
            mock_loader, mock_data_feed, "test_loading"
        )

        assert success is True
        assert 'BTCUSDT' in result
        assert 'ETHUSDT' in result
        # Verify realtime loader was called but data feed was not
        mock_loader.get_realtime_data.assert_called_once()
        mock_data_feed.get_market_data.assert_not_called()

    @pytest.mark.asyncio
    @patch('core.logging.logger')
    async def test_fallback_to_data_feed(self, mock_logger):
        """Test fallback from realtime loader to data feed."""
        # Mock loader fails with empty data
        mock_loader = MagicMock()
        mock_loader.get_realtime_data = AsyncMock(return_value={})

        # Mock data feed succeeds
        mock_data_feed = MagicMock()
        mock_data_feed.get_market_data = AsyncMock(return_value={
            'BTCUSDT': {'close': 50000},
            'ETHUSDT': {'close': 3000}
        })

        result, success = await ErrorHandler.load_market_data_with_fallback(
            mock_loader, mock_data_feed, "test_fallback"
        )

        assert success is True
        assert 'BTCUSDT' in result
        assert 'ETHUSDT' in result
        # Verify both were called (loader tried twice - 1 initial + 1 retry)
        assert mock_loader.get_realtime_data.call_count == 2  # 1 initial + 1 retry
        mock_data_feed.get_market_data.assert_called_once()

    @pytest.mark.asyncio
    @patch('core.logging.logger')
    async def test_both_loaders_fail(self, mock_logger):
        """Test when both realtime loader and data feed fail."""
        # Both fail with empty results
        mock_loader = MagicMock()
        mock_loader.get_realtime_data = AsyncMock(return_value={})

        mock_data_feed = MagicMock()
        mock_data_feed.get_market_data = AsyncMock(return_value={})

        result, success = await ErrorHandler.load_market_data_with_fallback(
            mock_loader, mock_data_feed, "test_failure"
        )

        assert success is False
        assert result is None

    @pytest.mark.asyncio
    @patch('core.logging.logger')
    async def test_invalid_data_types(self, mock_logger):
        """Test handling of invalid data types."""
        # Mock loader returns invalid data types
        mock_loader = MagicMock()
        mock_loader.get_realtime_data = AsyncMock(return_value="invalid_string")

        mock_data_feed = MagicMock()
        mock_data_feed.get_market_data = AsyncMock(return_value=None)

        result, success = await ErrorHandler.load_market_data_with_fallback(
            mock_loader, mock_data_feed, "test_invalid"
        )

        assert success is False
        assert result is None


class TestContextManager:
    """Test context manager functionality."""

    @patch('core.logging.logger')
    def test_successful_operation(self, mock_logger):
        """Test successful context manager operation."""
        try:
            with ErrorHandler.safe_operation("test_operation"):
                x = 1 + 1  # Some operation
            assert x == 2
        except Exception:
            pytest.fail("Context manager raised unexpected exception")

    @patch('core.logging.logger')
    def test_exception_in_context(self, mock_logger):
        """Test exception handling in context manager."""
        with pytest.raises(ValueError):
            with ErrorHandler.safe_operation("failing_operation"):
                raise ValueError("Test exception")


class TestDecorator:
    """Test decorator functionality."""

    def test_log_and_continue_decorator(self):
        """Test log_and_continue decorator."""
        call_count = 0

        @ErrorHandler.log_and_continue("test_function")
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Expected failure")
            return "should not reach here"

        # Function should not raise exception but return None
        result = failing_function()
        assert result is None
        assert call_count == 1

    def test_retry_on_failure_decorator_async(self):
        """Test async retry on failure decorator."""
        call_count = 0

        @ErrorHandler.retry_on_failure(max_retries=2, delay=0.1)
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return f"Success on attempt {call_count}"

        async def run_test():
            result = await async_failing_function()
            return result

        # Should succeed on 3rd attempt
        result = asyncio.run(run_test())
        assert "Success on attempt 3" in result
        assert call_count == 3

    def test_retry_on_failure_decorator_sync(self):
        """Test synchronous retry on failure decorator."""
        call_count = 0

        @ErrorHandler.retry_on_failure(max_retries=1, delay=0.01)
        def sync_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError(f"Sync attempt {call_count} failed")
            return f"Sync success on attempt {call_count}"

        # Should succeed on 2nd attempt
        result = sync_failing_function()
        assert "Sync success on attempt 2" in result
        assert call_count == 2


class TestIntegrationPatterns:
    """Test integration with HRM system patterns."""

    @pytest.mark.asyncio
    @patch('core.logging.logger')
    async def test_data_refresh_pattern(self, mock_logger):
        """Test the common data refresh pattern from main.py."""
        # Simulate the pattern from main.py where data refresh failures are handled
        attempts = []

        async def try_refresh_data():
            # Simulate occasional failures
            nonlocal attempts
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("Simulated connection failure")
            return {"BTCUSDT": {"close": 50000}, "ETHUSDT": {"close": 3000}}

        # Test using ErrorHandler - should retry and succeed
        result, success = await ErrorHandler.async_with_fallback(
            primary_func=try_refresh_data,
            max_retries=3,
            retry_delay=0.01,  # Small delay for test
            operation_name="data_refresh"
        )

        assert success is True
        assert len(attempts) == 3  # Failed twice, succeeded on third
        assert "BTCUSDT" in result

    def test_config_loading_pattern(self):
        """Test the common configuration loading pattern."""
        load_attempts = []

        def try_load_config():
            nonlocal load_attempts
            load_attempts.append(1)
            if len(load_attempts) < 2:
                raise FileNotFoundError("Config file not found")
            return {"api_key": "test_key", "timeout": 30}

        # Test using ErrorHandler
        result, success = ErrorHandler.sync_with_fallback(
            primary_func=try_load_config,
            max_retries=2,
            retry_delay=0.01,
            operation_name="config_loading"
        )

        assert success is True
        assert len(load_attempts) == 2
        assert result["api_key"] == "test_key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
