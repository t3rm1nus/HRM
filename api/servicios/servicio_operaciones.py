# Servicio de Operaciones - Órdenes ejecutadas por el PaperTradeLogger
#
# Lee directamente el JSON que escribe storage/paper_trade_logger.py, sin
# importar el logger, para no arrastrar el resto del sistema.
#
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import ValidationError

from api.esquemas.operaciones import LoteOrdenes, OrderRecord
from storage.perfil_datos import obtener_directorio_operaciones

registro = logging.getLogger(__name__)

# Constantes
VARIABLE_ENTORNO_RUTA = "HRM_RUTA_OPERACIONES"
RUTA_OPERACIONES_POR_DEFECTO = "paper_trades.json"
TAMANO_POR_DEFECTO = 50
TAMANO_MAXIMO = 200

Lado = Literal["buy", "sell"]


class ErrorLecturaOperaciones(Exception):
    """El archivo de operaciones existe pero no se puede leer o interpretar."""


def obtener_ruta_operaciones() -> Path:
    """Resuelve la ruta del historial del perfil PAPER activo."""
    ruta_configurada = os.getenv(VARIABLE_ENTORNO_RUTA)
    if ruta_configurada:
        return Path(ruta_configurada)
    return obtener_directorio_operaciones() / RUTA_OPERACIONES_POR_DEFECTO


def leer_contenido_json(ruta: Path) -> Optional[Any]:
    """Devuelve el JSON del archivo, o None si todavía no existe."""
    if not ruta.exists():
        return None
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ErrorLecturaOperaciones(f"No se pudo leer {ruta}: {error}") from error


def extraer_registros(contenido: Optional[Any]) -> List[Any]:
    """Obtiene la lista cruda de trades del contenido del archivo."""
    if contenido is None:
        return []
    if not isinstance(contenido, dict) or not isinstance(contenido.get("trades"), list):
        raise ErrorLecturaOperaciones("El archivo no contiene una lista 'trades'")
    return contenido["trades"]


def construir_metadata(registro_crudo: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ciclo": registro_crudo.get("cycle_id"),
        "motivo": registro_crudo.get("reason"),
        "condiciones_mercado": registro_crudo.get("market_conditions") or {},
    }


def normalizar_identificador(identificador: Any) -> Optional[str]:
    """El logger escribe order_id como número; la API lo expone siempre como texto."""
    return None if identificador is None else str(identificador)


def convertir_a_orden(registro_crudo: Dict[str, Any]) -> OrderRecord:
    """Traduce un trade del logger a OrderRecord. Lanza ValidationError si es inválido."""
    precio = registro_crudo.get("filled_price") or registro_crudo.get("price")
    return OrderRecord(
        order_id=normalizar_identificador(registro_crudo.get("order_id")),
        simbolo=registro_crudo.get("symbol"),
        lado=str(registro_crudo.get("side", "")).lower(),
        cantidad=registro_crudo.get("quantity"),
        precio_llenado=precio,
        comision=registro_crudo.get("commission", 0.0),
        status=registro_crudo.get("status"),
        timestamp=registro_crudo.get("timestamp"),
        estrategia=registro_crudo.get("strategy"),
        metadata=construir_metadata(registro_crudo),
    )


def convertir_registros_validos(registros: List[Any]) -> List[OrderRecord]:
    """Convierte los registros descartando (con aviso) los inválidos."""
    ordenes: List[OrderRecord] = []
    for posicion, registro_crudo in enumerate(registros):
        try:
            ordenes.append(convertir_a_orden(registro_crudo))
        except (ValidationError, AttributeError) as error:
            registro.warning("Registro %d descartado: %s", posicion, error)
    return ordenes


def filtrar_por_simbolo(ordenes: List[OrderRecord], simbolo: Optional[str]) -> List[OrderRecord]:
    if not simbolo:
        return ordenes
    simbolo_normalizado = simbolo.casefold()
    return [orden for orden in ordenes if orden.simbolo.casefold() == simbolo_normalizado]


def filtrar_por_lado(ordenes: List[OrderRecord], lado: Optional[Lado]) -> List[OrderRecord]:
    if not lado:
        return ordenes
    return [orden for orden in ordenes if orden.lado == lado]


def obtener_marca_temporal_comparable(orden: OrderRecord) -> datetime:
    """Las marcas sin zona horaria se tratan como UTC solo para poder ordenarlas."""
    marca = orden.timestamp
    return marca if marca.tzinfo else marca.replace(tzinfo=timezone.utc)


def ordenar_mas_recientes_primero(ordenes: List[OrderRecord]) -> List[OrderRecord]:
    return sorted(ordenes, key=obtener_marca_temporal_comparable, reverse=True)


def seleccionar_pagina(ordenes: List[OrderRecord], pagina: int, tamano: int) -> List[OrderRecord]:
    inicio = (pagina - 1) * tamano
    return ordenes[inicio:inicio + tamano]


def calcular_total_paginas(total: int, tamano: int) -> int:
    return max(1, math.ceil(total / tamano))


def construir_lote(ordenes: List[OrderRecord], pagina: int, tamano: int) -> LoteOrdenes:
    total_paginas = calcular_total_paginas(len(ordenes), tamano)
    return LoteOrdenes(
        total=len(ordenes),
        ordenes=seleccionar_pagina(ordenes, pagina, tamano),
        has_more=pagina < total_paginas,
        pagina=pagina,
        pagina_total=total_paginas,
    )


def cargar_ordenes(
    pagina: int = 1,
    tamano: int = TAMANO_POR_DEFECTO,
    simbolo: Optional[str] = None,
    lado: Optional[Lado] = None,
    ruta: Optional[Path] = None,
) -> LoteOrdenes:
    """Lee, filtra, ordena y pagina las órdenes del logger."""
    contenido = leer_contenido_json(ruta or obtener_ruta_operaciones())
    ordenes = convertir_registros_validos(extraer_registros(contenido))
    ordenes = filtrar_por_lado(filtrar_por_simbolo(ordenes, simbolo), lado)
    return construir_lote(ordenar_mas_recientes_primero(ordenes), pagina, tamano)
