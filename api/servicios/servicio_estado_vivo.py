"""Estado en vivo del sistema para la API (solo lectura).

El bucle de trading registra aquí su actividad y su cartera; los endpoints
de la API consultan este servicio. El servicio no ejecuta órdenes ni modifica
la cartera: únicamente lee y notifica.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Protocol

from api.servicios.servicio_websockets import EventoTipo, ServicioWebSockets, servicio_websockets

registro = logging.getLogger(__name__)

SEGUNDOS_MAXIMOS_SIN_CICLO = 30.0
ACTIVO_BTC = "BTCUSDT"
ACTIVO_ETH = "ETHUSDT"
ACTIVO_USDT = "USDT"


class ProveedorCartera(Protocol):
    """Contrato mínimo que debe cumplir la cartera registrada."""

    initial_balance: float
    portfolio: dict[str, Any]


def normalizar_modo(modo_interno: Optional[str]) -> str:
    """Traduce el modo interno de main.py al vocabulario público de la API."""
    traduccion = {"simulated": "paper", "paper": "paper", "live": "live"}
    return traduccion.get((modo_interno or "").lower(), "unknown")


def formatear_duracion(segundos: float) -> str:
    segundos_enteros = max(0, int(segundos))
    horas, resto = divmod(segundos_enteros, 3600)
    minutos, segundos_restantes = divmod(resto, 60)
    return f"{horas}h {minutos:02d}m {segundos_restantes:02d}s"


def leer_numero(valores: dict[str, Any], clave: str, campo: str) -> float:
    entrada = valores.get(clave) or {}
    return float(entrada.get(campo, 0.0) or 0.0)


class ServicioEstadoVivo:
    """Guarda la actividad del bucle y expone estado y cartera."""

    def __init__(
        self,
        servicio_eventos: ServicioWebSockets,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        self._servicio_eventos = servicio_eventos
        self._reloj = reloj
        self._cartera: Optional[ProveedorCartera] = None
        self._modo: Optional[str] = None
        self._inicio_bucle: Optional[float] = None
        self._ultimo_ciclo_en: Optional[float] = None
        self._ciclos_completados = 0

    def registrar_cartera(self, cartera: ProveedorCartera) -> None:
        self._cartera = cartera

    def registrar_modo(self, modo_interno: str) -> None:
        self._modo = modo_interno

    def registrar_inicio_bucle(self) -> None:
        self._inicio_bucle = self._reloj()
        self._ultimo_ciclo_en = None
        self._ciclos_completados = 0

    def registrar_parada(self) -> None:
        self._inicio_bucle = None

    def registrar_ciclo(self, numero_ciclo: int) -> None:
        self._ciclos_completados = numero_ciclo
        self._ultimo_ciclo_en = self._reloj()

    def bucle_activo(self) -> bool:
        if self._inicio_bucle is None:
            return False
        referencia = self._ultimo_ciclo_en or self._inicio_bucle
        return (self._reloj() - referencia) <= SEGUNDOS_MAXIMOS_SIN_CICLO

    def obtener_estado_sistema(self) -> dict[str, Any]:
        activo = self.bucle_activo()
        return {
            "status": "running" if activo else "paused",
            "trading_active": activo,
            "cycle_count": self._ciclos_completados,
            "uptime": self._calcular_uptime(),
            "mode": normalizar_modo(self._modo),
        }

    def obtener_resumen_cartera(self) -> Optional[dict[str, float]]:
        if self._cartera is None:
            return None
        valores = self._cartera.portfolio
        return {
            "btc_balance": leer_numero(valores, ACTIVO_BTC, "position"),
            "eth_balance": leer_numero(valores, ACTIVO_ETH, "position"),
            "usdt_balance": leer_numero(valores, ACTIVO_USDT, "free"),
            "total_nav": float(valores.get("total", 0.0) or 0.0),
            "initial_capital": float(self._cartera.initial_balance),
        }

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
        if self._inicio_bucle is None:
            return "no disponible"
        return formatear_duracion(self._reloj() - self._inicio_bucle)


servicio_estado_vivo = ServicioEstadoVivo(servicio_websockets)
