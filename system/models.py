"""
Modelos de datos para el sistema de bootstrap y gestión de errores.

Define las estructuras de datos utilizadas por el ErrorRecoveryManager
y otros componentes del sistema para encapsular el estado y resultados.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd


class HealthStatus(Enum):
    """Estado de salud del sistema."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ErrorType(Enum):
    """Tipos de errores del sistema."""
    DATA_QUALITY = "data_quality"
    ML_FRAMEWORK = "ml_framework"
    STATE_CORRUPTION = "state_corruption"
    NETWORK = "network"
    UNKNOWN = "unknown"


class RecoveryActionType(Enum):
    """Tipos de acciones de recovery."""
    RETRY = "retry"
    SKIP_CYCLE = "skip_cycle"
    RESET_COMPONENT = "reset_component"
    SHUTDOWN = "shutdown"


@dataclass
class RecoveryAction:
    """Acción de recovery a tomar."""
    action: RecoveryActionType
    wait_seconds: int
    recovery_steps_taken: List[str] = field(default_factory=list)
    success: bool = False


@dataclass
class CleanupResult:
    """Resultado de la limpieza del sistema."""
    success: bool
    cleaned_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class TradingCycleResult:
    """Resultado de un ciclo de trading."""
    signals_generated: int
    orders_executed: int
    orders_rejected: int
    cooldown_blocked: int
    l3_regime: str
    portfolio_value: float
    execution_time: float
    
    def __str__(self) -> str:
        return (
            f"Signals: {self.signals_generated} | "
            f"Executed: {self.orders_executed} | "
            f"Rejected: {self.orders_rejected} | "
            f"Regime: {self.l3_regime} | "
            f"Portfolio: ${self.portfolio_value:.2f} | "
            f"Time: {self.execution_time:.2f}s"
        )


@dataclass
class ComponentRegistry:
    """Registro de componentes del sistema."""
    components: Dict[str, Any] = field(default_factory=dict)
    registered_count: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class SystemContext:
    """Contexto del sistema después del bootstrap."""
    state_coordinator: Any = None
    components: Dict[str, Any] = field(default_factory=dict)
    external_adapter: Any = None
    health_status: HealthStatus = HealthStatus.UNHEALTHY
    initialization_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def is_ready(self) -> bool:
        """Indica si el sistema está listo para operar."""
        return (
            self.state_coordinator is not None and
            self.external_adapter is not None and
            self.health_status == HealthStatus.HEALTHY
        )


@dataclass
class ErrorRecoveryResult:
    """Resultado de una operación de recovery de error."""
    action_taken: RecoveryAction
    error_handled: bool
    state_recovered: bool
    data_recovered: bool
    ml_framework_recovered: bool
    recovery_time_ms: float = 0.0
    error_details: Dict[str, Any] = field(default_factory=dict)
