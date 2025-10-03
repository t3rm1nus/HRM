# -*- coding: utf-8 -*-
"""
Unified Validation System - HRM Trading System

Centralized validation utilities to eliminate code duplication across the system.
Provides comprehensive data validation for market data, OHLCV data, and trading parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple
from core.logging import logger


class UnifiedValidator:
    """
    Centralized validation system for HRM trading system.
    Replaces scattered validation functions throughout the codebase.
    """

    @staticmethod
    def validate_market_data_structure(data: Any) -> Tuple[bool, str]:
        """
        Validate market data structure at the top level.

        Args:
            data: Market data to validate (dict, DataFrame, etc.)

        Returns:
            Tuple of (is_valid: bool, message: str)
        """
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
                    if v.shape[0] > 0:
                        valid_symbols.append(symbol)
                    else:
                        errors.append(f"{symbol}: Empty DataFrame")
                elif isinstance(v, dict) and len(v) > 0:
                    valid_symbols.append(symbol)
                else:
                    errors.append(f"{symbol}: Invalid data type {type(v)}")
        except AttributeError as e:
            return False, f"Data structure error: {e}"

        if valid_symbols:
            return True, f"Valid symbols: {valid_symbols}"
        else:
            return False, f"No valid data. Errors: {errors}"

    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame,
                           required_cols: Optional[list] = None) -> Tuple[pd.DataFrame, str]:
        """
        Comprehensive OHLCV data validation and cleaning.

        Args:
            df: DataFrame with OHLCV data
            required_cols: List of required columns (default: ['open', 'high', 'low', 'close', 'volume'])

        Returns:
            Tuple of (validated_df: pd.DataFrame or None, error_message: str)
        """
        if required_cols is None:
            required_cols = ['open', 'high', 'low', 'close', 'volume']

        if df is None:
            return None, "DataFrame is None"

        if not isinstance(df, pd.DataFrame):
            return None, f"Expected DataFrame, got {type(df)}"

        if df.empty:
            return df, "DataFrame is empty"

        # Check for required columns
        existing_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return None, f"Missing required columns: {missing_cols}"

        # Convert to numeric and clean data
        df_clean = df.copy()

        for col in existing_cols:
            try:
                # Convert to string first to handle mixed types
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str)

                # Convert to numeric with error handling
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

                # Remove NaN values
                df_clean = df_clean.dropna(subset=[col])

                # Remove non-positive values for price columns
                if col in ['open', 'high', 'low', 'close']:
                    df_clean = df_clean[df_clean[col] > 0]
                elif col == 'volume':
                    df_clean = df_clean[df_clean[col] >= 0]

            except Exception as e:
                return None, f"Error processing column '{col}': {e}"

        # Ensure OHLC consistency
        if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
            # Clean any inconsistencies
            df_clean['high'] = np.maximum(df_clean['high'],
                                         df_clean[['open', 'close']].max(axis=1))
            df_clean['low'] = np.minimum(df_clean['low'],
                                        df_clean[['open', 'close']].min(axis=1))

        if df_clean.empty:
            return df_clean, "DataFrame became empty after validation"

        return df_clean, f"Successfully validated {len(df_clean)} rows with columns {list(df_clean.columns)}"

    @staticmethod
    def validate_symbol_data_required(symbols: list,
                                    market_data: Dict[str, Any],
                                    l2_config: Optional[Any] = None) -> Tuple[Dict[str, pd.DataFrame], str]:
        """
        Validate that required symbols have valid market data.

        Args:
            symbols: List of required symbols
            market_data: Market data dictionary
            l2_config: Optional L2 config for symbol validation

        Returns:
            Tuple of (valid_data: dict, error_message: str)
        """
        if not symbols:
            return {}, "No symbols specified"

        if not isinstance(market_data, dict):
            return {}, f"Market data must be dict, got {type(market_data)}"

        valid_data = {}
        missing_symbols = []
        invalid_symbols = []

        for symbol in symbols:
            if symbol not in market_data:
                missing_symbols.append(symbol)
                continue

            data = market_data[symbol]
            if isinstance(data, pd.DataFrame):
                validated_df, error_msg = UnifiedValidator.validate_ohlcv_data(data)
                if validated_df is not None and not validated_df.empty:
                    valid_data[symbol] = validated_df
                else:
                    invalid_symbols.append(f"{symbol}: {error_msg}")
            elif isinstance(data, dict):
                try:
                    df = pd.DataFrame([data])  # Single tick
                    validated_df, error_msg = UnifiedValidator.validate_ohlcv_data(df)
                    if validated_df is not None and not validated_df.empty:
                        valid_data[symbol] = validated_df
                    else:
                        invalid_symbols.append(f"{symbol}: {error_msg}")
                except Exception as e:
                    invalid_symbols.append(f"{symbol}: Failed to convert dict to DataFrame: {e}")
            else:
                invalid_symbols.append(f"{symbol}: Unsupported data type {type(data)}")

        error_messages = []
        if missing_symbols:
            error_messages.append(f"Missing symbols: {missing_symbols}")
        if invalid_symbols:
            error_messages.append(f"Invalid symbols: {invalid_symbols}")

        error_msg = "; ".join(error_messages) if error_messages else ""

        if valid_data:
            return valid_data, f"Validated {len(valid_data)}/{len(symbols)} symbols" + (f" - Errors: {error_msg}" if error_msg else "")
        else:
            return {}, f"No valid symbols found. {error_msg}"

    @staticmethod
    def validate_and_fix_market_data(state: dict,
                                   config: Dict[str, Any],
                                   force_refresh: bool = False) -> Tuple[Dict[str, Any], str]:
        """
        Comprehensive market data validation and fix for main loop.

        Args:
            state: Current system state
            config: System configuration
            force_refresh: Force refresh even if data exists

        Returns:
            Tuple of (fixed_market_data: dict, status_message: str)
        """
        try:
            # Get current market data
            market_data = state.get("market_data", {})

            if not isinstance(market_data, dict):
                logger.warning(f"Invalid market_data type: {type(market_data)}, resetting")
                market_data = {}

            if not market_data or force_refresh:
                return {}, "Market data needs refresh"

            # Get required symbols from config
            symbols = config.get("SYMBOLS", ["BTCUSDT", "ETHUSDT"])

            # Validate all symbol data
            valid_data, validation_msg = UnifiedValidator.validate_symbol_data_required(
                symbols, market_data
            )

            if valid_data and len(valid_data) >= len(symbols) * 0.5:  # At least 50% valid
                logger.info(f"✅ Market data validated: {validation_msg}")
                return valid_data, validation_msg
            else:
                logger.warning(f"⚠️ Insufficient valid market data: {validation_msg}")
                return {}, f"Insufficient valid data: {validation_msg}"

        except Exception as e:
            logger.error(f"Error in validate_and_fix_market_data: {e}")
            return {}, f"Validation error: {e}"

    @staticmethod
    def validate_trading_parameters(symbol: str,
                                  quantity: float,
                                  price: float,
                                  side: str) -> Tuple[bool, str]:
        """
        Validate basic trading parameters.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            side: 'buy' or 'sell'

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        errors = []

        if not symbol or not isinstance(symbol, str):
            errors.append("Invalid symbol")

        if not isinstance(quantity, (int, float)) or quantity <= 0:
            errors.append(f"Invalid quantity: {quantity}")

        if not isinstance(price, (int, float)) or price <= 0:
            errors.append(f"Invalid price: {price}")

        if side not in ['buy', 'sell']:
            errors.append(f"Invalid side: {side}")

        if errors:
            return False, "; ".join(errors)

        return True, "Valid trading parameters"

    @staticmethod
    def sanitize_numeric_value(value: Any,
                             default: float = 0.0,
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None) -> float:
        """
        Safely convert value to float within bounds.

        Args:
            value: Value to convert
            default: Default value if conversion fails
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Sanitized float value
        """
        try:
            if value is None:
                return default

            if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                return default

            result = float(value)

            if min_val is not None and result < min_val:
                return default

            if max_val is not None and result > max_val:
                return default

            return result

        except (ValueError, TypeError):
            return default

    @staticmethod
    def clean_portfolio_data(portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Clean and validate portfolio data.

        Args:
            portfolio_data: Raw portfolio data

        Returns:
            Cleaned portfolio data with float values
        """
        if not isinstance(portfolio_data, dict):
            return {}

        cleaned = {}
        for key, value in portfolio_data.items():
            if isinstance(key, str):
                cleaned[key] = UnifiedValidator.sanitize_numeric_value(value, default=0.0)

        return cleaned


# Convenience functions for backward compatibility
def validate_market_data_structure(data: Any) -> Tuple[bool, str]:
    """Backward compatibility wrapper."""
    return UnifiedValidator.validate_market_data_structure(data)

def validate_ohlcv_data(df: pd.DataFrame,
                       required_cols: Optional[list] = None) -> Tuple[pd.DataFrame, str]:
    """Backward compatibility wrapper."""
    return UnifiedValidator.validate_ohlcv_data(df, required_cols)

def validate_and_fix_market_data(state: dict,
                               config: Dict[str, Any],
                               force_refresh: bool = False) -> Tuple[Dict[str, Any], str]:
    """Backward compatibility wrapper."""
    return UnifiedValidator.validate_and_fix_market_data(state, config, force_refresh)
