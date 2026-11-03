# Señales - Señales de trading y filtros

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class FilterCondition(BaseModel):
    """Condición de filtro para señales."""
    campo: str
    operador: Literal[">", "<", ">=", "<=", "==", "!="]
    valor: float
    description: Optional[str] = None


class SeñalFilter(BaseModel):
    """Filtro para filtrar señales."""
    min_confianza: float = 0.0
    max_confianza: float = 1.0
    direcciones: Optional[List[Literal["buy", "sell", "hold"]]] = None
    fuentes: Optional[List[str]] = None
    activos: Optional[List[str]] = None
    regimen_mercado: Optional[str] = None


class SeñalStats(BaseModel):
    """Estadísticas de señales."""
    total_generadas: int = 0
    total_executed: int = 0
    accuracy: float = 0.0
    avg_confianza: float = 0.0
    por_fuente: Dict[str, int] = {}
    por_dirección: Dict[str, int] = {}


class SeñaLActiva(BaseModel):
    """Señal activa en el sistema."""
    id: str
    simbolo: str
    tipo: Literal["entry", "exit", "take_profit", "stop_loss"]
    confianza: float
    precio_objetivo: Optional[float] = None
    precio_stop_loss: Optional[float] = None
    tiempo_vida: float
    creado_en: datetime = datetime.utcnow
    ultima_actualizacion: datetime = datetime.utcnow