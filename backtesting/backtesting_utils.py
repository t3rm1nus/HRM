# C:\proyectos\HRM\backtesting\backtesting_utils.py

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

# Definiciones movidas de hrm_tester.py
class TestMode(Enum):
    LIVE_TRADES = "live_trades"
    HISTORICAL = "historical"
    SIMULATION = "simulation"

class TestLevel(Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"

@dataclass
class TestResult:
    test_id: str
    level: TestLevel
    name: str
    description: str
    success: bool
    message: str
    duration_ms: float
    details: Optional[Dict] = None

class L1Model:
    def __init__(self, predictions_path: Optional[str] = None):
        self.is_real = False
        self.models = {}
        self.predictions_path = predictions_path or os.path.join(Path(__file__).parent, 'L1_predictions.json')
        self._load_predictions()

    def _load_predictions(self):
        """Carga las predicciones desde el archivo JSON si existe."""
        if os.path.exists(self.predictions_path):
            try:
                with open(self.predictions_path, 'r') as f:
                    self.models = json.load(f)
                    self.is_real = True
                    logging.info(f"✅ Predicciones de L1 cargadas con éxito desde '{self.predictions_path}'.")
            except Exception as e:
                logging.warning(f"Error al cargar el archivo de predicciones: {e}. Usando implementación mock.")
                self._load_mock_predictions()
        else:
            logging.warning("Archivo de predicciones de L1 no encontrado. Usando implementación mock.")
            self._load_mock_predictions()

    def _load_mock_predictions(self):
        """Crea un diccionario con métricas simuladas."""
        self.models = {
            'LogisticRegression': {
                'accuracy': 0.85,
                'precision': 0.80,
                'f1_score': 0.82,
                'profit_contribution': 1000,
                'latency_ms': 50
            },
            'RandomForest': {
                'accuracy': 0.82,
                'precision': 0.78,
                'f1_score': 0.8,
                'profit_contribution': 800,
                'latency_ms': 45
            },
            'LightGBM': {
                'accuracy': 0.87,
                'precision': 0.83,
                'f1_score': 0.85,
                'profit_contribution': 1200,
                'latency_ms': 60
            }
        }
    
    def predict(self, data: Any) -> Dict:
        """
        Devuelve las predicciones cargadas o simuladas.
        """
        return self.models