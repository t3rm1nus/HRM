# Respuesta genérica API - Envoltorio para todas las respuestas de la API

from typing import Generic, TypeVar, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from datetime import datetime

T = TypeVar("T", bound=BaseModel)


class Respuesta(BaseModel, Generic[T]):
    """Envoltorio genérico para todas las respuestas de la API HRM."""
    
    datos: T  # Datos de la respuesta (el modelo typeado)
    origen: str = "hrm-api"  # Origen de los datos (servicio, módulo, etc.)
    generado_en: datetime = Field(default_factory=lambda: datetime.utcnow().replace(tzinfo=None))  # ISO 8601 UTC
    es_sintetico: bool = False  # Indica si los datos fueron sintetizados/calculados
    es_respaldo: bool = False  # Indica si los datos son de un backup/respaldo
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z" if v.tzinfo else v.isoformat() + "Z"
        }
    
    def model_dump_with_prefix(self) -> Dict[str, Any]:
        """Exportar con prefijo de nombres de campos."""
        return {
            "data": self.datos.model_dump() if self.datos else None,
            "origin": self.origen,
            "generated_at": self.generado_en.isoformat() + "Z",
            "is_synthetic": self.es_sintetico,
            "is_backup": self.es_respaldo,
        }