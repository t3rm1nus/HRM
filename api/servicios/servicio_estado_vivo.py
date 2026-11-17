"""Fachada de solo lectura para el estado global del runtime."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Protocol

from api.servicios.servicio_websockets import EventoTipo, ServicioWebSockets, servicio_websockets
from system.state_coordinator import StateCoordinator

registro = logging.getLogger(__name__)
SEGUNDOS_MAXIMOS_SIN_CICLO = 30.0
ACTIVO_BTC = "BTCUSDT"
ACTIVO_ETH = "ETHUSDT"
ACTIVO_USDT = "USDT"


class ProveedorCartera(Protocol):
    initial_balance: float
    portfolio: dict[str, Any]


def normalizar_modo(modo_interno: Optional[str]) -> str:
    """Traduce el modo interno al vocabulario público."""
    return {"simulated": "paper", "paper": "paper", "live": "live"}.get((modo_interno or "").lower(), "unknown")


def formatear_duracion(segundos: float) -> str:
    """Formatea segundos como duración legible."""
    segundos_enteros = max(0, int(segundos))
    horas, resto = divmod(segundos_enteros, 3600)
    minutos, segundos_restantes = divmod(resto, 60)
    return f"{horas}h {minutos:02d}m {segundos_restantes:02d}s"


def leer_numero(valores: dict[str, Any], clave: str, campo: str) -> float:
    """Lee un valor numérico de la cartera."""
    entrada = valores.get(clave) or {}
    return float(entrada.get(campo, 0.0) or 0.0)


class ServicioEstadoVivo:
    """Conecta la API y los eventos con el StateCoordinator único."""
    def __init__(self, servicio_eventos: ServicioWebSockets,
                 reloj: Callable[[], float] = time.monotonic,
                 coordinador: Optional[StateCoordinator] = None) -> None:
        self._servicio_eventos = servicio_eventos
        self._reloj = reloj
        self._coordinador = coordinador
        self._cartera: Optional[ProveedorCartera] = None

    def registrar_coordinador(self, coordinador: Optional[StateCoordinator]) -> None:
        """Inyecta la misma instancia que utiliza main.py."""
        self._coordinador = coordinador

    def _obtener_coordinador(self) -> StateCoordinator:
        if self._coordinador is None:
            self._coordinador = StateCoordinator.get_instance()
        return self._coordinador

    def registrar_cartera(self, cartera: ProveedorCartera) -> None:
        self._cartera = cartera

    def registrar_modo(self, modo_interno: str) -> None:
        self._obtener_coordinador().mode = modo_interno

    def registrar_inicio_bucle(self) -> None:
        """Registra el fin del bootstrap, pero mantiene STARTING."""
        # Compatibilidad: la hora de arranque ya pertenece al coordinador.
        return None

    def registrar_inicio_ciclo(self, numero_ciclo: int) -> None:
        """Marca RUNNING solo al comenzar un ciclo real."""
        self._obtener_coordinador().iniciar_ciclo(numero_ciclo, "main")

    def registrar_parada(self) -> None:
        """Publica STOPPING antes de cerrar recursos."""
        self._obtener_coordinador().iniciar_parada("Parada ordenada del runtime", "main")

    def confirmar_parada(self) -> None:
        """Publica STOPPED después de terminar el cleanup."""
        self._obtener_coordinador().finalizar_parada("Shutdown completado", "main")

    def registrar_fallo(self, error: BaseException) -> None:
        self._obtener_coordinador().registrar_fallo(error, "main")

    def registrar_error_ciclo(self, numero_ciclo: int, error: BaseException) -> None:
        self._obtener_coordinador().registrar_error_ciclo(error, numero_ciclo, "main")

    def registrar_ciclo(self, numero_ciclo: int) -> None:
        self._obtener_coordinador().registrar_ciclo_completado(numero_ciclo, "main")

    def bucle_activo(self) -> bool:
        estado = self._obtener_coordinador().obtener_estado_runtime()
        if estado["state"] != "RUNNING":
            return False
        if estado["cycle_in_progress"]:
            return True
        ultimo_ciclo = estado.get("last_cycle_monotonic")
        if ultimo_ciclo is None:
            return False
        return self._reloj() - ultimo_ciclo <= SEGUNDOS_MAXIMOS_SIN_CICLO

    def obtener_estado_sistema(self) -> dict[str, Any]:
        estado = self._obtener_coordinador().obtener_estado_runtime()
        activo = self.bucle_activo()
        publico = {"STOPPED": "paused", "STARTING": "starting", "RUNNING": "running" if activo else "paused",
                   "DEGRADED": "degraded", "STOPPING": "stopping", "FAILED": "failed"}.get(estado["state"], "paused")
        return {"status": publico, "trading_active": activo, "cycle_count": estado["cycle_count"],
                "uptime": self._calcular_uptime(), "mode": normalizar_modo(estado["mode"]),
                "runtime_state": estado["state"], "reason": estado["reason"],
                "component": estado["component"], "updated_at": estado["updated_at"],
                "last_error": estado["last_error"]}

    def obtener_resumen_cartera(self) -> Optional[dict[str, float]]:
        if self._cartera is None:
            return None
        valores = self._cartera.portfolio
        return {"btc_balance": leer_numero(valores, ACTIVO_BTC, "position"),
                "eth_balance": leer_numero(valores, ACTIVO_ETH, "position"),
                "usdt_balance": leer_numero(valores, ACTIVO_USDT, "free"),
                "total_nav": float(valores.get("total", 0.0) or 0.0),
                "initial_capital": float(self._cartera.initial_balance)}

    async def notificar_ciclo(self, numero_ciclo: int) -> None:
        self.registrar_ciclo(numero_ciclo)
        try:
            await self._servicio_eventos.publicar(EventoTipo.ESTADO, self.obtener_estado_sistema())
            resumen = self.obtener_resumen_cartera()
            if resumen is not None:
                await self._servicio_eventos.publicar(EventoTipo.CIERRE, resumen)
        except Exception:
            registro.exception("No se pudo notificar el ciclo %s por WebSocket", numero_ciclo)

    def _calcular_uptime(self) -> str:
        inicio = self._obtener_coordinador().obtener_estado_runtime().get("started_monotonic")
        if inicio is None:
            return "no disponible"
        return formatear_duracion(self._reloj() - inicio)


servicio_estado_vivo = ServicioEstadoVivo(servicio_websockets)
