# Estado del sistema - Estados y modo del sistema

from datetime import datetime
from typing import Literal, Optional, Dict, Any

from pydantic import BaseModel


class ModoOperacion(BaseModel):
    """Modo de operación del sistema."""
    modo: Literal["paper", "live", "simulated", "testnet", "development", "backtest"]
    timestamp: datetime = datetime.utcnow()


class EstadoSistema(BaseModel):
    """Estado general del sistema HRM."""
    id: str
    modo: ModoOperacion
    inicializado: bool = False
    timestamp_ultima_actualizacion: datetime = datetime.utcnow()
    version: str = "1.0.0"


class SingletonStatus(BaseModel):
    """Estado de los singletons del sistema."""
    configuracion_cargada: bool = False
    gestion_riesgo_inicializada: bool = False
    generador_señales_inicializado: bool = False
    ejecutor_ordenes_inicializado: bool = False
    coordinador_estado_inicializado: bool = False