"""
Trading Model Factory - Centralized model creation and management.
Implements Factory Pattern for consistent model instantiation across HRM levels.
"""

from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import importlib
import traceback

logger = logging.getLogger(__name__)

class TradingModelFactory:
    """
    Factory class for creating and managing trading models across HRM levels.
    Provides centralized model instantiation with consistent configuration.
    """

    def __init__(self):
        """Initialize factory with model registry."""
        self.model_registry = {
            'L1': {
                'LogisticRegression': 'models.L1.logistic_regression_model.LogisticRegressionModel',
                'RandomForest': 'models.L1.random_forest_model.RandomForestModel',
                'LightGBM': 'models.L1.lightgbm_model.LightGBMModel',
                'Ensemble': 'models.L1.ensemble_model.EnsembleModel'
            },
            'L2': {
                'FinRLProcessor': 'l2_tactic.finrl_processor.FinRLProcessor',
                'SignalComposer': 'l2_tactic.signal_composer.SignalComposer',
                'TechnicalAnalyzer': 'l2_tactic.generators.technical_analyzer.TechnicalAnalyzer',
                'Ensemble': 'l2_tactic.ensemble.blender.BlenderEnsemble'
            },
            'L3': {
                'RegimeClassifier': 'l3_strategy.regime_classifier.RegimeClassifier',
                'SentimentAnalyzer': 'l3_strategy.sentiment_analyzer.SentimentAnalyzer',
                'PortfolioOptimizer': 'l3_strategy.portfolio_optimizer.PortfolioOptimizer'
            }
        }
        self.model_cache = {}

    def create_models_for_level(self, level: str, config: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Create all models for a given HRM level.

        Args:
            level: HRM level ('L1', 'L2', 'L3')
            config: Optional configuration overrides

        Returns:
            List of instantiated model objects
        """
        if level not in self.model_registry:
            raise ValueError(f"Unknown HRM level: {level}")

        config = config or {}
        models = []

        for model_name, module_path in self.model_registry[level].items():
            try:
                model_instance = self._create_model_instance(module_path, config)
                if model_instance:
                    models.append(model_instance)
                    logger.info(f"✓ Created {model_name} for {level}")
            except Exception as e:
                logger.error(f"✗ Failed to create {model_name} for {level}: {e}")
                continue

        logger.info(f"Created {len(models)} models for HRM {level}")
        return models

    def create_model(self, model_type: str, level: str = None,
                    config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Create a single model instance.

        Args:
            model_type: Type of model to create
            level: HRM level (optional, required for level-specific models)
            config: Model configuration

        Returns:
            Model instance or None if creation failed
        """
        config = config or {}

        # Find model in registry
        module_path = None

        if level:
            module_path = self.model_registry.get(level, {}).get(model_type)
        else:
            # Search across all levels
            for level_models in self.model_registry.values():
                if model_type in level_models:
                    module_path = level_models[model_type]
                    break

        if not module_path:
            logger.error(f"Model type '{model_type}' not found in registry")
            return None

        try:
            return self._create_model_instance(module_path, config)
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            return None

    def _create_model_instance(self, module_path: str, config: Dict[str, Any]) -> Any:
        """
        Internal method to create model instance from module path.

        Args:
            module_path: Full module path (e.g., 'models.L1.some_model.SomeModel')
            config: Configuration dictionary

        Returns:
            Model instance
        """
        try:
            # Split module path
            module_name, class_name = module_path.rsplit('.', 1)

            # Import module
            module = importlib.import_module(module_name)

            # Get class
            model_class = getattr(module, class_name)

            # Create instance with config
            return model_class(**config)

        except (ImportError, AttributeError) as e:
            logger.error(f"Import error for {module_path}: {e}")
            raise FactoryError(f"Failed to import model class {module_path}",
                              details={'module_path': module_path, 'original_error': str(e)})
        except Exception as e:
            logger.error(f"Unexpected error creating model {module_path}: {e}")
            raise FactoryError(f"Model instantiation failed for {module_path}",
                              details={'module_path': module_path, 'config': config, 'original_error': str(e)})

    def get_available_models(self, level: Optional[str] = None) -> List[str]:
        """Get list of available model types for a level or all levels."""
        if level:
            return list(self.model_registry.get(level, {}))
        else:
            all_models = []
            for level_models in self.model_registry.values():
                all_models.extend(level_models.keys())
            return all_models

    def register_model(self, level: str, model_name: str, module_path: str):
        """
        Register a new model type in the factory.

        Args:
            level: HRM level
            model_name: Model name
            module_path: Full module path
        """
        if level not in self.model_registry:
            self.model_registry[level] = {}

        self.model_registry[level][model_name] = module_path
        logger.info(f"Registered model: {model_name} for {level}")

    def clear_cache(self):
        """Clear model cache."""
        self.model_cache.clear()
        logger.info("Model cache cleared")

# Global factory instance
model_factory = TradingModelFactory()

def get_model_factory() -> TradingModelFactory:
    """Get the global model factory instance."""
    return model_factory
