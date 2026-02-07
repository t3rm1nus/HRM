"""
Modelos de datos del sistema HRM

Definiciones de dataclasses para estructuras de datos del sistema.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class HealthStatus(Enum):
    """Estado de salud del sistema"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class RecoveryAction:
    """Acción de recuperación de errores"""
    def __init__(self, action, wait_seconds, recovery_steps_taken, success):
        self.action = action
        self.wait_seconds = wait_seconds
        self.recovery_steps_taken = recovery_steps_taken
        self.success = success

class ErrorType(Enum):
    """Tipos de errores del sistema"""
    DATA_QUALITY = "data_quality"
    ML_FRAMEWORK = "ml_framework"
    STATE_CORRUPTION = "state_corruption"
    NETWORK = "network"
    UNKNOWN = "unknown"

class RecoveryActionType(Enum):
    """Tipos de acciones de recuperación"""
    RETRY = "retry"
    SKIP_CYCLE = "skip_cycle"
    RESET_COMPONENT = "reset_component"
    SHUTDOWN = "shutdown"

@dataclass
class ErrorRecoveryResult:
    """Resultado de una operación de recuperación de errores"""
    success: bool
    action_taken: str
    wait_time: int
    recovery_steps: List[str]
    error_type: str

@dataclass
class TradingCycleResult:
    """Resultado de un ciclo completo de trading"""
    signals_generated: int = 0
    orders_executed: int = 0
    orders_rejected: int = 0
    cooldown_blocked: int = 0
    l3_regime: str = "unknown"
    portfolio_value: float = 0.0
    execution_time: float = 0.0

@dataclass
class CleanupResult:
    """Resultado de la operación de limpieza"""
    success: bool = True
    cleaned_files: List[str] = None
    errors: List[str] = None
    duration_ms: float = 0.0
    
    def __post_init__(self):
        if self.cleaned_files is None:
            self.cleaned_files = []
        if self.errors is None:
            self.errors = []

@dataclass
class ComponentRegistry:
    """Registro de componentes del sistema"""
    components: Dict[str, Any] = None
    registered_count: int = 0
    success: bool = True
    errors: List[str] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = {}
        if self.errors is None:
            self.errors = []

@dataclass
class SystemContext:
    """Contexto completo del sistema"""
    state_coordinator: Any = None
    components: Dict[str, Any] = None
    external_adapter: Any = None
    health_status: HealthStatus = HealthStatus.UNKNOWN
    initialization_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = {}
        if self.errors is None:
            self.errors = []
