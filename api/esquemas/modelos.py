# Modelos - Modelos ML y metadatos

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class MetricasModelo(BaseModel):
    """Métricas de un modelo de ML."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: Optional[float] = None
    last_evaluation: datetime = datetime.utcnow
    test_samples: int = 0


class InfoModelo(BaseModel):
    """Información de un modelo registrado."""
    id: str
    nombre: str
    version: str
    path_archivo: str
    estado: str = "pending"
    metrics: MetricasModelo = MetricasModelo()
    fecha_creacion: datetime = datetime.utcnow
    fecha_ultima_actualizacion: datetime = datetime.utcnow
    veces_entrenadas: int = 0


class RegistroModelo(BaseModel):
    """Registro de modelo en el sistema."""
    modelo_id: str
    tipo: Literal["regime_detector", "sentiment", "volatility", "lstm", "ppo", "sacred"]
    simbolos: List[str] = []
    hiperparametros: Dict[str, Any] = {}
    metrics_before: MetricasModelo = MetricasModelo()
    metrics_after: MetricasModelo = MetricasModelo()
    triggered_by: Literal["manual", "automatic", "schedule"] = "manual"
    timestamp: datetime = datetime.utcnow


class ComparacionModelos(BaseModel):
    """Comparación entre múltiples modelos."""
    modelo_a: str
    modelo_b: str
    metricas_comparadas: List[str] = []
    ganador: Optional[str] = None
    diferencia: Dict[str, float] = {}
    evaluado_en: datetime = datetime.utcnow