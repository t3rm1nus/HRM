# Salud - Health check y monitoreo

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class HealthCheck(BaseModel):
    """Resultado de un health check del sistema."""
    status: Literal["healthy", "degraded", "unhealthy", "unknown"] = "healthy"
    timestamp: datetime = datetime.utcnow
    componentes: Dict[str, "HealthCheckItem"] = {}


class HealthCheckItem(BaseModel):
    """Estado de un componente individual."""
    status: Literal["healthy", "degraded", "unhealthy", "unknown"] = "healthy"
    message: str = ""
    latency_ms: Optional[float] = None
    detalles: Dict[str, Any] = {}


class MetricasSistema(BaseModel):
    """Métricas generales del sistema."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage: float = 0.0
    process_uptime: float = 0.0
    last_restart: Optional[datetime] = None


class HealthSummary(BaseModel):
    """Resumen de health check para API response."""
    overall_status: Literal["healthy", "degraded", "unhealthy", "unknown"]
    timestamp: datetime
    uptime_human: str
    components_ok: int = 0
    components_total: int = 0
    critical_issues: List[str] = []