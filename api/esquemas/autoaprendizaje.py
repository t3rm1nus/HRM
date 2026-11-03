# Auto-aprendizaje - Configuración y estado del auto-aprendizaje

from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel


class ConfigAutoAprendizaje(BaseModel):
    """Configuración de auto-aprendizaje."""
    modo: Literal["fix", "adaptive", "production", "disabled"] = "fix"
    enabled: bool = True
    adaptive_mode: bool = True
    retrain_interval_hours: int = 168  # 7 dias
    min_trades_for_retrain: int = 100
    win_rate_threshold: float = 0.52
    max_drawdown_threshold: float = 0.12


class EstadoAutoAprendizaje(BaseModel):
    """Estado actual del auto-aprendizaje."""
    modo_actual: str = "fix"
    enabled: bool = True
    ultima_actualizacion: datetime = datetime.utcnow
    next_retrain: Optional[datetime] = None
    total_retrains: int = 0
    mejor_winrate: float = 0.0
    ultimo_resultado: Optional[Dict[str, Any]] = None


class ModeloInfo(BaseModel):
    """Informacion de un modelo de auto-aprendizaje."""
    nombre: str
    tipo: str
    path: str
    version: str
    accuracy: float
    trained_at: datetime
    metrics: Optional[Dict[str, float]] = None


class HistorialEntrenamiento(BaseModel):
    """Historial de entrenamientos."""
    id: str
    fecha: datetime
    winrate: float
    total_trades: int
    mejora: float
    status: Literal["success", "failed", "skipped"]
    metadata: Optional[Dict[str, Any]] = None