# l2_tactic/__init__.py
"""
L2 Tactic Module - Refactored FinRL Integration

This module provides a clean, modular interface for FinRL model integration.
The monolithic finrl_integration.py has been refactored into logical components:

- feature_extractors.py: Custom feature extractors for different models
- observation_builders.py: Methods for building observations for different model types
- model_loaders.py: Unified model loading utilities
- signal_generators.py: Signal generation and conversion utilities
- finrl_processor.py: Main FinRL processor class
- finrl_wrapper.py: Wrapper for handling different model types

For backward compatibility, the original classes are still available through this module.
"""

# Import main classes for easy access
from .finrl_processor import FinRLProcessor
from .finrl_wrapper import FinRLProcessorWrapper
from .feature_extractors import RiskAwareExtractor
from .model_loaders import ModelLoaders
from .observation_builders import ObservationBuilders
from .signal_generators import SignalGenerators

# Backward compatibility - import the old monolithic classes
# This allows existing code to continue working without changes
try:
    from .finrl_integration import FinRLProcessor as LegacyFinRLProcessor
    from .finrl_integration import FinRLProcessorWrapper as LegacyFinRLProcessorWrapper
    from .finrl_integration import RiskAwareExtractor as LegacyRiskAwareExtractor
except ImportError:
    # If the old file doesn't exist, just use the new ones
    LegacyFinRLProcessor = FinRLProcessor
    LegacyFinRLProcessorWrapper = FinRLProcessorWrapper
    LegacyRiskAwareExtractor = RiskAwareExtractor

# Export the main classes
__all__ = [
    'FinRLProcessor',
    'FinRLProcessorWrapper',
    'RiskAwareExtractor',
    'ModelLoaders',
    'ObservationBuilders',
    'SignalGenerators',
    # Backward compatibility
    'LegacyFinRLProcessor',
    'LegacyFinRLProcessorWrapper',
    'LegacyRiskAwareExtractor'
]
