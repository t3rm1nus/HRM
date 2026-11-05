# Rutas del modo de sesión - Solo recibe parámetros y llama al servicio
from fastapi import APIRouter, HTTPException

from api.esquemas.modo_sesion import EstadoModoSesion, SolicitudModoSesion
from api.servicios.servicio_modo_sesion import guardar_modo_configurado, obtener_estado_modo

router_modo_sesion = APIRouter(prefix="/sesion", tags=["sesion"])


@router_modo_sesion.get(
    "/modo",
    response_model=EstadoModoSesion,
    summary="Modo de sesión",
    description="Modo con el que arrancó el proceso y modo configurado para el próximo arranque.",
)
def consultar_modo() -> EstadoModoSesion:
    return obtener_estado_modo()


@router_modo_sesion.post(
    "/modo",
    response_model=EstadoModoSesion,
    summary="Cambiar modo de sesión",
    description="Se guarda ahora y se aplica en el próximo arranque.",
)
def cambiar_modo(solicitud: SolicitudModoSesion) -> EstadoModoSesion:
    try:
        guardar_modo_configurado(solicitud.modo)
    except OSError as error:
        raise HTTPException(status_code=503, detail="No se pudo guardar el modo de sesión") from error
    return obtener_estado_modo()