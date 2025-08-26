# ai_model_integration.py
"""
L2 Tactical — AI Model Integration
==================================

Wrapper para integrar modelos de IA (.zip) con el ecosistema L2.
Usa PerformanceOptimizer para caching, batching y optimización de predicciones.

Soporta modelos entrenados con Stable-Baselines (PPO.load).
"""

import os
import zipfile
import tempfile
import logging
from typing import Any, Dict, Optional

from stable_baselines3 import PPO

from l2_tactic.performance_optimizer import PerformanceOptimizer, PerfConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Excepciones específicas
# ---------------------------------------------------------------------------

class ModelLoadError(Exception):
    """Error al cargar el modelo IA."""


# ---------------------------------------------------------------------------
# Wrapper principal
# ---------------------------------------------------------------------------

class AIModelWrapper:
    def __init__(
        self,
        config,
        optimizer: Optional[PerformanceOptimizer] = None,
        model_id: str = "ppo_multiasset",
    ):
        self.config = config
        self.model_path = getattr(config, "model_path", "models/ai_model_data_multiasset.zip")
        self.optimizer = optimizer or PerformanceOptimizer(PerfConfig())
        self.model_id = model_id
        self.model = self._load_model()
        self.opt_model = self.optimizer.wrap_model(self.model, self.model_id)

    # -----------------------------------------------------------------------
    # Carga del modelo desde .zip
    # -----------------------------------------------------------------------
    def _load_model(self):
        """
        Carga PPO desde un .zip directo o desde una carpeta ya extraída.
        """
        if not os.path.exists(self.model_path):
            raise ModelLoadError(f"Modelo no encontrado en {self.model_path}")

        try:
            model = PPO.load(self.model_path)
            logger.info(f"Modelo PPO cargado correctamente desde {self.model_path}")
            return model
        except Exception as e:
            raise ModelLoadError(f"Error al cargar PPO desde {self.model_path}: {e}") from e
    # -----------------------------------------------------------------------
    # Predicciones
    # -----------------------------------------------------------------------

    def predict(self, features: Any, deterministic: bool = True) -> Any:
        """
        Predicción síncrona (fallback).
        """
        try:
            return self.model.predict(features, deterministic=deterministic)[0]
        except Exception as e:
            logger.exception("Error en predict()")
            raise

    async def predict_async(
        self,
        *,
        symbol: str,
        horizon: str,
        features: Any,
    ) -> Any:
        """
        Predicción asíncrona con optimizer (cache + batch).
        """
        return await self.opt_model.predict_async(
            symbol=symbol, horizon=horizon, features=features
        )

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------

    @property
    def optimized(self):
        """Acceso directo al modelo optimizado (OptimizedModel)."""
        return self.opt_model

AIModelIntegration = AIModelWrapper