# -*- coding: utf-8 -*-
"""Aplicación HTTP de HRM y sus endpoints."""

import os
from datetime import datetime, timezone
import platform

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from api.servicios.servicio_websockets import servicio_websockets
from api.servicios.servicio_estado_vivo import servicio_estado_vivo
from api.rutas.rutas_operaciones import router_operaciones
from api.rutas.rutas_modo_sesion import router_modo_sesion
from api.rutas.rutas_mercado import router_mercado


app = FastAPI(
    title="HRM Trading System API",
    description="API REST para el sistema de trading HRM",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "HRM_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
    ).split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

app.include_router(router_operaciones)

app.include_router(router_modo_sesion)

app.include_router(router_mercado)

class StatusResponse(BaseModel):
    status: str = Field(default="pending", description="Estado del sistema")
    trading_active: bool = Field(default=False, description="Modo trading activo")
    cycle_count: int = Field(default=0, description="Contador de ciclos")
    uptime: str = Field(default="no disponible", description="Tiempo de actividad")
    mode: str = Field(default="unknown", description="Modo real del backend: paper, live o unknown")
    runtime_state: str = Field(default="STOPPED", description="Estado interno del ciclo de vida")
    reason: str = Field(default="", description="Motivo del último cambio de estado")
    component: str = Field(default="", description="Componente que registró el cambio")
    updated_at: str = Field(default="", description="Fecha UTC del último cambio")
    last_error: str | None = Field(default=None, description="Último error del runtime")

class PortfolioResponse(BaseModel):
    btc_balance: float = Field(default=0.0, description="Balance BTC")
    eth_balance: float = Field(default=0.0, description="Balance ETH")
    usdt_balance: float = Field(default=0.0, description="Balance USDT")
    total_nav: float = Field(default=0.0, description="NAV total")
    initial_capital: float = Field(default=0.0, description="Capital inicial")

class HealthCheckResponse(BaseModel):
    status: str = Field(default="ok", description="Estado salud")
    timestamp: str = Field(default="", description="Marca de tiempo")
    version: str = Field(default="2.0.0", description="Versión API")

@app.websocket("/ws")
async def websocket_eventos(websocket: WebSocket) -> None:
    """Mantiene la conexión abierta para los eventos del sistema."""
    conectado = await servicio_websockets.conectar(websocket)
    if not conectado:
        return
    try:
        while True:
            # El cliente puede enviar mensajes de control/heartbeat.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await servicio_websockets.desconectar(websocket)
    except Exception:
        await servicio_websockets.desconectar(websocket)
        await websocket.close(code=1011)

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Chequeo de salud del sistema",
    description="Verifica que el sistema de trading esté operando correctamente",
)
async def health_check() -> HealthCheckResponse:
    from datetime import datetime, timezone
    return HealthCheckResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="2.0.0",
    )

@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Estado actual del sistema de trading",
    description="Obtiene el estado operativo del bot de trading",
)
async def get_status() -> StatusResponse:
    return StatusResponse(**servicio_estado_vivo.obtener_estado_sistema())

@app.get(
    "/portfolio",
    response_model=PortfolioResponse,
    summary="Información de la cartera actual",
    description="Obtiene el estado actual de balances y NAV de la cartera",
)
async def get_portfolio() -> PortfolioResponse:
    resumen = servicio_estado_vivo.obtener_resumen_cartera()
    if resumen is None:
        raise HTTPException(status_code=503, detail="La cartera aún no está inicializada")
    return PortfolioResponse(**resumen)

@app.get(
    "/version",
    summary="Información de versión",
    description="Obtiene la versión actual del sistema",
)
async def get_version() -> dict:
    import platform
    return {
        "app_version": "2.0.0",
        "framework": "FastAPI",
        "python_version": platform.python_version(),
    }

def run_fastapi(host: str = os.getenv("HRM_API_HOST", "127.0.0.1"), port: int = 8000, reload: bool = False) -> None:
    """Inicia el servidor FastAPI con uvicorn."""
    uvicorn.run(app, host=host, port=port, reload=reload)
