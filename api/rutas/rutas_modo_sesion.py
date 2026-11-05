# Rutas del modo de sesión - Solo recibe parámetros y llama al servicio
from fastapi import APIRouter, HTTPException, status

from api.esquemas.modo_sesion import EstadoModoSesion, ModoSesion, SolicitudModoSesion, SolicitudReinicioDatosPaper
from api.servicios import servicio_modo_sesion
from api.servicios.servicio_modo_sesion import guardar_modo_configurado, obtener_estado_modo
from api.servicios.servicio_estado_vivo import servicio_estado_vivo
from storage.perfil_datos import solicitar_reset_persistente

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

@router_modo_sesion.post(
    "/reiniciar-datos",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Programar reinicio de datos PAPER persistentes",
    description="Programa el borrado del perfil PAPER persistente para el próximo arranque. No borra datos durante la petición.",
)
def programar_reinicio_datos_paper(solicitud: SolicitudReinicioDatosPaper) -> dict[str, str]:
    """Programa un reinicio diferido y protegido del perfil persistente PAPER."""
    modo_operativo = servicio_estado_vivo.obtener_estado_sistema().get("mode", "unknown")
    if modo_operativo != "paper":
        raise HTTPException(status_code=403, detail="El reinicio de datos solo está permitido en modo PAPER")

    modo_activo = servicio_modo_sesion.registro_modo_activo.obtener()
    modo_configurado = obtener_estado_modo().modo_configurado
    if modo_activo is not ModoSesion.PERSISTENTE or modo_configurado is not ModoSesion.PERSISTENTE:
        raise HTTPException(status_code=409, detail="El reinicio solo está permitido con el perfil persistente activo y configurado")

    try:
        solicitar_reset_persistente()
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail="Ya existe un reinicio PAPER pendiente") from error
    except OSError as error:
        raise HTTPException(status_code=503, detail="No se pudo programar el reinicio PAPER") from error

    return {
        "estado": "pendiente",
        "mensaje": "Los datos PAPER persistentes se reiniciarán en el próximo arranque. No se ha borrado nada ahora.",
    }
