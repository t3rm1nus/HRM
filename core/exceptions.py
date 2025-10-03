"""
HRM Exception Hierarchy - Standardized Error Handling

Provides a common exception hierarchy and consistent error handling patterns
across all HRM components (L1, L2, L3, core services).
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass
from core.logging import logger


# Base HRM Exception
class HRMException(Exception):
    """
    Base exception class for all HRM-related errors.

    All HRM exceptions should inherit from this class to ensure consistent
    error handling, logging, and user-facing messages.
    """

    def __init__(self, message: str, error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

        # Auto-log critical errors
        self._log_error()

    def _log_error(self):
        """Automatically log error with structured information."""
        log_data = {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }
        logger.error(f"HRM Exception: {self.error_code} - {self.message}", extra=log_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


# Configuration Errors
class ConfigurationError(HRMException):
    """Errors related to system configuration."""
    pass

class EnvironmentError(ConfigurationError):
    """Errors with environment setup or detection."""
    pass

class ValidationError(ConfigurationError):
    """Configuration or parameter validation errors."""
    pass


# Trading Logic Errors
class TradingError(HRMException):
    """Base class for trading-related errors."""
    pass

class SignalError(TradingError):
    """Errors in signal generation or validation."""
    pass

class OrderError(TradingError):
    """Errors related to order placement or execution."""
    pass

class RiskError(TradingError):
    """Risk management and validation errors."""
    pass

class PositionError(TradingError):
    """Position management errors."""
    pass


# Model & AI Errors
class ModelError(HRMException):
    """Model-related errors (loading, inference, etc.)."""
    pass

class AILError(ModelError):
    """Artificial intelligence and machine learning errors."""
    pass

class InferenceError(AILError):
    """Model inference and prediction errors."""
    pass

class FactoryError(ModelError):
    """Model factory creation and registration errors."""
    pass


# Data & Exchange Errors
class DataError(HRMException):
    """Data acquisition, processing, or validation errors."""
    pass

class ExchangeError(DataError):
    """Exchange API and connectivity errors."""
    pass

class APIError(ExchangeError):
    """Exchange API-specific errors."""
    pass

class ConnectivityError(ExchangeError):
    """Network and connectivity issues."""
    pass

class RateLimitError(APIError):
    """Exchange rate limiting errors."""
    pass


# System & Infrastructure Errors
class SystemError(HRMException):
    """System-level errors."""
    pass

class InitializationError(SystemError):
    """Component initialization failures."""
    pass

class PersistenceError(SystemError):
    """Data persistence and storage errors."""
    pass

class LoggingError(SystemError):
    """Logging system errors."""
    pass


# Learning & Auto-Improvement Errors
class LearningError(HRMException):
    """Auto-learning and model improvement errors."""
    pass

class OverfittingError(LearningError):
    """Overfitting detection and prevention errors."""
    pass

class TrainingError(LearningError):
    """Model training and retraining errors."""
    pass


# Helper Functions for Error Handling
def safe_execute(func, *args, fallback=None, **kwargs):
    """
    Execute a function safely with consistent error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for function
        fallback: Value to return on error
        **kwargs: Keyword arguments for function

    Returns:
        Function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Safe execution failed: {e}")
        return fallback


def with_error_handling(operation_name: str, error_class=HRMException):
    """
    Decorator for consistent error handling and logging.

    Usage:
        @with_error_handling("model_inference", InferenceError)
        def infer_signal(self, data):
            # ... implementation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_class as e:
                logger.error(f"Operation '{operation_name}' failed: {e}")
                raise
            except Exception as e:
                # Wrap unexpected errors
                logger.error(f"Unexpected error in '{operation_name}': {e}")
                raise error_class(f"Unexpected error in {operation_name}: {str(e)}") from e
        return wrapper
    return decorator


def create_error_response(error: Exception) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.

    Args:
        error: The exception that occurred

    Returns:
        Standardized error response
    """
    if isinstance(error, HRMException):
        return error.to_dict()
    else:
        return {
            'error_type': error.__class__.__name__,
            'error_code': 'UnknownError',
            'message': str(error),
            'details': {}
        }


def log_and_raise(error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an error with context and re-raise it.

    Args:
        error: The exception to log and raise
        context: Additional context information
    """
    context = context or {}

    if isinstance(error, HRMException):
        if context:
            error.details.update(context)
    else:
        # Create HRM-wrapped error
        hrm_error = HRMException(str(error),
                                error_code='WrappedError',
                                details=context)
        raise hrm_error

    raise error


# Specific Error Factories
def signal_validation_error(signal_id: str, reason: str, details: Dict[str, Any] = None) -> SignalError:
    """Create a standardized signal validation error."""
    details = details or {}
    details.update({'signal_id': signal_id, 'reason': reason})
    return SignalError(f"Signal validation failed: {reason}",
                      error_code='SignalValidationError',
                      details=details)

def order_execution_error(order_id: str, reason: str, exchange_response: Dict[str, Any] = None) -> OrderError:
    """Create a standardized order execution error."""
    details = {'order_id': order_id, 'reason': reason}
    if exchange_response:
        details['exchange_response'] = exchange_response
    return OrderError(f"Order execution failed: {reason}",
                     error_code='OrderExecutionError',
                     details=details)

def model_inference_error(model_name: str, input_shape: tuple = None, details: Dict[str, Any] = None) -> InferenceError:
    """Create a standardized model inference error."""
    details = details or {}
    details.update({'model_name': model_name})
    if input_shape:
        details['input_shape'] = input_shape
    return InferenceError(f"Model inference failed for {model_name}",
                         error_code='ModelInferenceError',
                         details=details)

def connectivity_error(endpoint: str, operation: str, retry_count: int = 0) -> ConnectivityError:
    """Create a standardized connectivity error."""
    return ConnectivityError(f"Connectivity failed for {operation} on {endpoint}",
                           error_code='ConnectivityError',
                           details={
                               'endpoint': endpoint,
                               'operation': operation,
                               'retry_count': retry_count
                           })

def configuration_missing_error(parameter: str, section: str) -> ConfigurationError:
    """Create a standardized missing configuration error."""
    return ConfigurationError(f"Required configuration parameter '{parameter}' missing in section '{section}'",
                            error_code='ConfigurationMissingError',
                            details={
                                'parameter': parameter,
                                'section': section
                            })


# Example Usage:
#
# # Consistent error raising
# raise signal_validation_error('sig_123', 'insufficient_confidence', {'confidence': 0.3})
#
# # Safe execution wrapper
# result = safe_execute(lambda: risky_operation(), fallback=[])
#
# # Decorator for error handling
# @with_error_handling("model_loading", ModelError)
# def load_model(path):
#     # ... implementation
#
# # Error response for APIs
# try:
#     # ... some operation
# except Exception as e:
#     response = create_error_response(e)
#     return {'success': False, 'error': response}
