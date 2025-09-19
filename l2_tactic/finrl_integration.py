# l2_tactic/finrl_integration.py
"""
BACKWARD COMPATIBILITY MODULE

This file maintains backward compatibility with the original monolithic finrl_integration.py
while using the new modular architecture internally.

For new code, please use the modular components directly:
- from l2_tactic.finrl_processor import FinRLProcessor
- from l2_tactic.finrl_wrapper import FinRLProcessorWrapper
- etc.

This file will be deprecated in future versions.
"""

# Import the new modular components
from .finrl_processor import FinRLProcessor
from .finrl_wrapper import FinRLProcessorWrapper
from .feature_extractors import RiskAwareExtractor
from .model_loaders import ModelLoaders
from .observation_builders import ObservationBuilders
from .signal_generators import SignalGenerators

# Re-export for backward compatibility
__all__ = [
    'FinRLProcessor',
    'FinRLProcessorWrapper',
    'RiskAwareExtractor',
    'ModelLoaders',
    'ObservationBuilders',
    'SignalGenerators'
]
