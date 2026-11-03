"""Gestión asíncrona de conexiones WebSocket y difusión de eventos.

El servicio no conoce la lógica de trading: recibe conexiones, valida mensajes
entrantes y distribuye eventos serializables a los clientes conectados.
"""
from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Protocol


class ConexionWebSocket(Protocol):
    async def accept(self) -> None: ...
    async def send_text(self, data: str) -> None: ...
    async def close(self, code: int = 1000) -> None: ...


class EventoTipo(str, Enum):
    CIERRE = "cierre"
    ESTADO = "estado"
    MERCADO = "mercado"
    OPERACION = "operacion"
    SENAL = "senal"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ServicioWebSockets:
    """Mantiene clientes activos y difunde eventos sin bloquear por cliente."""

    def __init__(self, max_clientes: int = 100) -> None:
        if max_clientes < 1:
            raise ValueError("max_clientes debe ser mayor que cero")
        self._max_clientes = max_clientes
        self._clientes: set[ConexionWebSocket] = set()
        self._bloqueo = asyncio.Lock()

    @property
    def clientes_conectados(self) -> int:
        return len(self._clientes)

    async def conectar(self, websocket: ConexionWebSocket) -> bool:
        async with self._bloqueo:
            if len(self._clientes) >= self._max_clientes:
                await websocket.close(code=1013)
                return False
            await websocket.accept()
            self._clientes.add(websocket)
            return True

    async def desconectar(self, websocket: ConexionWebSocket) -> None:
        async with self._bloqueo:
            self._clientes.discard(websocket)

    async def publicar(self, tipo: EventoTipo | str, datos: Any) -> int:
        mensaje = json.dumps(
            {"tipo": str(tipo.value if isinstance(tipo, EventoTipo) else tipo),
             "datos": datos},
            ensure_ascii=False,
            default=str,
        )
        clientes = tuple(self._clientes)
        resultados = await asyncio.gather(
            *(cliente.send_text(mensaje) for cliente in clientes),
            return_exceptions=True,
        )
        fallidos = {
            cliente
            for cliente, resultado in zip(clientes, resultados)
            if isinstance(resultado, Exception)
        }
        if fallidos:
            async with self._bloqueo:
                self._clientes.difference_update(fallidos)
        return len(clientes) - len(fallidos)


servicio_websockets = ServicioWebSockets()
