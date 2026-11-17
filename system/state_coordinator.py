"""Fuente única de verdad del estado operativo de HRM."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ESTADOS_RUNTIME = {"STOPPED", "STARTING", "RUNNING", "DEGRADED", "STOPPING", "FAILED"}
TRANSICIONES_VALIDAS = {
    "STOPPED": {"STARTING"},
    "STARTING": {"RUNNING", "DEGRADED", "STOPPING", "FAILED"},
    "RUNNING": {"DEGRADED", "STOPPING", "FAILED"},
    "DEGRADED": {"RUNNING", "STOPPING", "FAILED"},
    "STOPPING": {"STOPPED", "FAILED"},
    "FAILED": {"STARTING"},
}


def _ahora_iso() -> str:
    """Devuelve fecha UTC con zona horaria."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SystemState:
    """Estado general compatible con los consumidores legacy."""
    is_ready: bool = False
    health_status: str = "unknown"
    components: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: Optional[float] = None


class StateCoordinator:
    """Coordina ciclo de vida, componentes y datos compartidos."""
    _instance: Optional["StateCoordinator"] = None
    _instance_lock = threading.RLock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "StateCoordinator":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    @classmethod
    def get_instance(cls) -> "StateCoordinator":
        """Obtiene la única instancia compartida."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Restablece el singleton para aislar pruebas."""
        with cls._instance_lock:
            cls._instance = None

    def __init__(self, mode: str = "simulated") -> None:
        with self._instance_lock:
            if getattr(self, "_inicializado", False):
                if mode:
                    self.mode = mode
                return
            self.logger = logger
            self.mode = mode
            self._lock = threading.RLock()
            self.system_state = SystemState(health_status="initializing")
            self._market_data: Dict[str, Any] = {}
            self._portfolio_state: Dict[str, Any] = {}
            self._total_value = 0.0
            self._l3_output: Dict[str, Any] = {
                "regime": "neutral", "signal": "hold", "confidence": 0.5,
                "strategy_type": "initial", "timestamp": 0.0,
            }
            self._current_prices: Dict[str, float] = {}
            self._runtime = {
                "state": "STOPPED", "trading_active": False, "mode": mode,
                "started_at": None, "started_monotonic": None,
                "updated_at": _ahora_iso(),
                "reason": "El runtime todavía no ha arrancado",
                "component": "StateCoordinator", "cycle_count": 0,
                "cycle_in_progress": False, "last_cycle_at": None,
                "last_cycle_monotonic": None,
                "last_error": None, "history": [],
            }
            self._inicializado = True

    # Ciclo de vida -----------------------------------------------------
    def iniciar_arranque(self, motivo: str = "Solicitud de arranque", componente: str = "runtime") -> bool:
        """Adquiere el runtime de forma atómica; rechaza dobles arranques."""
        with self._lock:
            if self._runtime["state"] not in {"STOPPED", "FAILED"}:
                self.logger.warning("Arranque rechazado: runtime en %s", self._runtime["state"])
                return False
            self._runtime.update(
                cycle_count=0,
                cycle_in_progress=False,
                last_cycle_at=None,
                last_cycle_monotonic=None,
                last_error=None,
                started_at=_ahora_iso(),
                started_monotonic=time.monotonic(),
            )
            self._transicionar_locked("STARTING", motivo, componente)
            return True

    def iniciar_ciclo(self, numero_ciclo: int, componente: str = "main") -> None:
        """Marca RUNNING al comenzar un ciclo real, nunca al abrir un WebSocket."""
        with self._lock:
            estado = self._runtime["state"]
            if estado not in {"STARTING", "RUNNING", "DEGRADED"}:
                raise RuntimeError(f"No se puede iniciar un ciclo con runtime en {estado}")
            if estado != "RUNNING":
                self._transicionar_locked("RUNNING", f"Comienza el ciclo operativo {numero_ciclo}", componente)
            else:
                self._registrar_evento_locked(estado, f"Comienza el ciclo operativo {numero_ciclo}", componente)
            self._runtime["cycle_count"] = max(self._runtime["cycle_count"], int(numero_ciclo) - 1)
            self._runtime["cycle_in_progress"] = True

    def registrar_ciclo_completado(self, numero_ciclo: int, componente: str = "main") -> None:
        """Registra el cierre correcto de un ciclo."""
        with self._lock:
            if self._runtime["state"] not in {"RUNNING", "DEGRADED"}:
                raise RuntimeError("No se puede completar un ciclo con runtime inactivo")
            self._runtime.update(
                cycle_count=int(numero_ciclo),
                cycle_in_progress=False,
                last_cycle_at=_ahora_iso(),
                last_cycle_monotonic=time.monotonic(),
            )
            if self._runtime["state"] == "DEGRADED":
                self._transicionar_locked("RUNNING", f"Ciclo {numero_ciclo} completado; recuperación confirmada", componente)
            else:
                self._registrar_evento_locked("RUNNING", f"Ciclo {numero_ciclo} completado", componente)

    def registrar_error_ciclo(self, error: BaseException, numero_ciclo: int, componente: str = "main") -> None:
        """Marca DEGRADED ante un fallo recuperable de ciclo."""
        with self._lock:
            self._runtime.update(cycle_in_progress=False, last_error=str(error))
            if self._runtime["state"] in {"STARTING", "RUNNING"}:
                self._transicionar_locked("DEGRADED", f"Error en ciclo {numero_ciclo}: {error}", componente)
            else:
                self._registrar_evento_locked(self._runtime["state"], f"Error en ciclo {numero_ciclo}: {error}", componente)

    def iniciar_parada(self, motivo: str = "Parada solicitada", componente: str = "runtime") -> None:
        """Bloquea nuevos ciclos y comienza el apagado ordenado."""
        with self._lock:
            estado = self._runtime["state"]
            if estado in {"STOPPED", "STOPPING", "FAILED"}:
                return
            self._runtime["cycle_in_progress"] = False
            self._transicionar_locked("STOPPING", motivo, componente)

    def finalizar_parada(self, motivo: str = "Apagado completado", componente: str = "runtime") -> None:
        """Confirma STOPPED solo después de limpiar componentes."""
        with self._lock:
            estado = self._runtime["state"]
            if estado in {"STOPPED", "FAILED"}:
                return
            if estado != "STOPPING":
                self._transicionar_locked("STOPPING", "Iniciando apagado ordenado", componente)
            self._transicionar_locked("STOPPED", motivo, componente)
            self._runtime.update(
                cycle_in_progress=False,
                started_at=None,
                started_monotonic=None,
                last_cycle_monotonic=None,
            )

    def registrar_fallo(self, error: BaseException, componente: str = "runtime") -> None:
        """Publica FAILED ante fallos fatales."""
        with self._lock:
            self._runtime.update(last_error=str(error), cycle_in_progress=False)
            if self._runtime["state"] != "FAILED":
                self._transicionar_locked("FAILED", f"Fallo fatal: {error}", componente)

    def obtener_estado_runtime(self) -> Dict[str, Any]:
        """Devuelve una instantánea consistente del ciclo de vida."""
        with self._lock:
            estado = dict(self._runtime)
            estado["history"] = [dict(item) for item in self._runtime["history"]]
            estado["trading_active"] = estado["state"] == "RUNNING"
            estado["mode"] = self.mode
            return estado

    def _transicionar_locked(self, nuevo: str, motivo: str, componente: str) -> None:
        actual = self._runtime["state"]
        if nuevo not in ESTADOS_RUNTIME:
            raise ValueError(f"Estado runtime desconocido: {nuevo}")
        if nuevo != actual and nuevo not in TRANSICIONES_VALIDAS[actual]:
            raise RuntimeError(f"Transición no permitida: {actual} -> {nuevo}")
        self._runtime["state"] = nuevo
        self._runtime["trading_active"] = nuevo == "RUNNING"
        self.system_state.health_status = {
            "STOPPED": "stopped", "STARTING": "initializing", "RUNNING": "healthy",
            "DEGRADED": "degraded", "STOPPING": "stopping", "FAILED": "unhealthy",
        }[nuevo]
        self.system_state.is_ready = nuevo == "RUNNING"
        self.system_state.timestamp = datetime.now(timezone.utc).timestamp()
        self._registrar_evento_locked(nuevo, motivo, componente)

    def _registrar_evento_locked(self, estado: str, motivo: str, componente: str) -> None:
        evento = {"state": estado, "timestamp": _ahora_iso(), "reason": str(motivo), "component": str(componente)}
        self._runtime.update(updated_at=evento["timestamp"], reason=evento["reason"], component=evento["component"])
        self._runtime["history"].append(evento)
        self._runtime["history"] = self._runtime["history"][-200:]
        self.logger.info("Runtime state=%s component=%s reason=%s", estado, componente, motivo)

    # Interfaz legacy ---------------------------------------------------
    def update_state(self, component_name: Any, status: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> None:
        """Acepta tanto update_state(dict) como update_state(nombre, estado, datos)."""
        with self._lock:
            if isinstance(component_name, dict) and status is None:
                for clave, valor in component_name.items():
                    setattr(self, f"_{clave}", valor)
                self.system_state.timestamp = datetime.now(timezone.utc).timestamp()
                return
            if not isinstance(component_name, str) or not status:
                raise ValueError("Se requiere componente y estado")
            self.system_state.components[component_name] = {
                "status": status, "data": data,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            }

    def get_system_state(self) -> SystemState:
        """Devuelve una copia del estado general."""
        with self._lock:
            return SystemState(self.system_state.is_ready, self.system_state.health_status,
                               dict(self.system_state.components), list(self.system_state.errors),
                               self.system_state.timestamp)

    def get_state(self, state_name: str) -> Any:
        """Mantiene la interfaz current consumida por L3 y portfolio."""
        if state_name != "current":
            self.logger.warning("Estado desconocido: %s", state_name)
            return None
        with self._lock:
            estado = {"market_data": self._market_data, "total_value": self._total_value,
                      "l3_output": self._l3_output}
            if self._portfolio_state:
                estado["portfolio"] = self._portfolio_state
            for clave in ("system_state_type", "system_state_reason", "system_state_metadata", "system_state_timestamp"):
                if hasattr(self, f"_{clave}"):
                    estado[clave] = getattr(self, f"_{clave}")
            return estado

    def update_market_data(self, market_data: Dict[str, Any]) -> None:
        if isinstance(market_data, dict) and market_data:
            with self._lock:
                self._market_data = market_data

    def update_total_value(self, total_value: float) -> None:
        if isinstance(total_value, (int, float)) and total_value >= 0:
            with self._lock:
                self._total_value = float(total_value)

    def update_portfolio_state(self, portfolio_state: Dict[str, Any]) -> None:
        if isinstance(portfolio_state, dict):
            with self._lock:
                self._portfolio_state = portfolio_state

    def update_l3_output(self, l3_output: Dict[str, Any]) -> None:
        if isinstance(l3_output, dict) and {"regime", "signal"} <= l3_output.keys():
            with self._lock:
                self._l3_output = l3_output

    def is_healthy(self) -> bool:
        return self.system_state.health_status == "healthy"

    def mark_ready(self) -> None:
        """Compatibilidad legacy sin modificar el ciclo de vida del runtime."""
        self.system_state.is_ready = True
        self.system_state.health_status = "healthy"
        self.system_state.timestamp = datetime.now(timezone.utc).timestamp()

    def record_error(self, error_msg: str) -> None:
        with self._lock:
            self.system_state.errors.append({"message": error_msg, "timestamp": datetime.now(timezone.utc).timestamp()})

    def update_current_prices(self, current_prices: Dict[str, float]) -> None:
        if isinstance(current_prices, dict):
            with self._lock:
                self._current_prices = dict(current_prices)
