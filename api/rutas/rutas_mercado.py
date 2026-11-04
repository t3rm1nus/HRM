# Rutas de Mercado - Solo recibe parámetros, llama al servicio y traduce errores a HTTP
from fastapi import APIRouter, HTTPException, Query

from api.esquemas.mercado_velas import EstadoMercado, ListaSimbolos, LoteVelas
from api.servicios.servicio_mercado import (
    DEFAULT_VELAS_LIMIT,
    MAX_VELAS_LIMIT,
    MercadoNoInicializado,
    SimboloDesconocido,
    servicio_mercado,
)

router_mercado = APIRouter(prefix="/mercado", tags=["mercado"])


@router_mercado.get("/estado", response_model=EstadoMercado, summary="Estado de la fuente de mercado")
def estado_mercado() -> EstadoMercado:
    return servicio_mercado.obtener_estado()


@router_mercado.get("/simbolos", response_model=ListaSimbolos, summary="Símbolos con datos")
def listar_simbolos() -> ListaSimbolos:
    return ListaSimbolos(simbolos=servicio_mercado.listar_simbolos())


@router_mercado.get("/velas/{simbolo}", response_model=LoteVelas, summary="Velas OHLCV recientes")
def obtener_velas(
    simbolo: str,
    limite: int = Query(DEFAULT_VELAS_LIMIT, ge=1, le=MAX_VELAS_LIMIT),
) -> LoteVelas:
    try:
        return servicio_mercado.obtener_velas(simbolo, limite)
    except MercadoNoInicializado as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except SimboloDesconocido as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
