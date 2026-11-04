# Rutas de Operaciones - Solo recibe parámetros, llama al servicio y traduce errores a HTTP
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.esquemas.operaciones import LoteOrdenes
from api.servicios.servicio_operaciones import (
    TAMANO_MAXIMO,
    TAMANO_POR_DEFECTO,
    ErrorLecturaOperaciones,
    Lado,
    cargar_ordenes,
)

router_operaciones = APIRouter(prefix="/operaciones", tags=["operaciones"])


@router_operaciones.get(
    "",
    response_model=LoteOrdenes,
    summary="Órdenes ejecutadas",
    description="Lista paginada de órdenes ejecutadas, de la más reciente a la más antigua.",
)
def listar_operaciones(
    pagina: int = Query(1, ge=1),
    tamano: int = Query(TAMANO_POR_DEFECTO, ge=1, le=TAMANO_MAXIMO),
    simbolo: Optional[str] = Query(None, description="No distingue mayúsculas"),
    lado: Optional[Lado] = Query(None),
) -> LoteOrdenes:
    try:
        return cargar_ordenes(pagina=pagina, tamano=tamano, simbolo=simbolo, lado=lado)
    except ErrorLecturaOperaciones as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
