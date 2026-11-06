"""Rutas de almacenamiento aisladas para los perfiles PAPER de HRM."""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

registro = logging.getLogger(__name__)

PERFIL_PERSISTENTE = "persistente"
PERFIL_LIMPIO = "limpia"
_PERFIL_ACTIVO = PERFIL_PERSISTENTE


def configurar_perfil_datos(perfil: str) -> None:
    """Selecciona el perfil PAPER que utilizarán los componentes del proceso."""
    global _PERFIL_ACTIVO
    if perfil not in {PERFIL_PERSISTENTE, PERFIL_LIMPIO}:
        raise ValueError(f"Perfil de datos no válido: {perfil}")
    _PERFIL_ACTIVO = perfil


def obtener_perfil_datos() -> str:
    """Devuelve el perfil PAPER activo en este proceso."""
    return _PERFIL_ACTIVO


def obtener_directorio_paper(perfil: Optional[str] = None, raiz: Path = Path(".")) -> Path:
    """Devuelve el directorio raíz de datos del perfil PAPER indicado."""
    perfil_seleccionado = perfil or obtener_perfil_datos()
    _validar_perfil(perfil_seleccionado)
    return raiz / "data" / "paper" / perfil_seleccionado


def obtener_directorio_operaciones(perfil: Optional[str] = None, raiz: Path = Path(".")) -> Path:
    """Devuelve el directorio de operaciones del perfil PAPER indicado."""
    return obtener_directorio_paper(perfil, raiz) / "paper_trades"


def obtener_ruta_cartera(perfil: Optional[str] = None, raiz: Path = Path("."), modo: str = "simulated") -> Path:
    """Devuelve la ruta de cartera correspondiente al modo operativo."""
    if modo == "live":
        return raiz / "persistent_state" / "portfolio_state_live.json"
    return obtener_directorio_paper(perfil, raiz) / "portfolio_state.json"


def obtener_ruta_log_cartera(perfil: Optional[str] = None, raiz: Path = Path(".")) -> Path:
    """Devuelve la ruta del CSV de evolución de cartera del perfil PAPER."""
    return obtener_directorio_paper(perfil, raiz) / "portfolio_log.csv"


def migrar_datos_legacy_a_persistente(raiz: Path = Path(".")) -> dict[str, int]:
    """Copia una sola vez datos PAPER antiguos al perfil persistente."""
    marcador = raiz / "persistent_state" / "migracion_paper_perfiles_v1.json"
    if marcador.exists():
        return {"cartera": 0, "operaciones": 0, "log_cartera": 0}

    directorio_destino = obtener_directorio_paper(PERFIL_PERSISTENTE, raiz)
    operaciones_destino = obtener_directorio_operaciones(PERFIL_PERSISTENTE, raiz)
    operaciones_destino.mkdir(parents=True, exist_ok=True)
    cartera_copiada = _migrar_cartera_legacy(raiz, directorio_destino)
    operaciones_copiadas = _migrar_operaciones_legacy(raiz, operaciones_destino)
    log_copiado = _migrar_log_cartera_legacy(raiz, directorio_destino)

    marcador.parent.mkdir(parents=True, exist_ok=True)
    temporal = marcador.with_suffix(".tmp")
    temporal.write_text(
        json.dumps({
            "version": 1,
            "migrado_en": datetime.now(timezone.utc).isoformat(),
            "archivos_copiados": {
                "cartera": cartera_copiada,
                "operaciones": operaciones_copiadas,
                "log_cartera": log_copiado,
            },
        }, indent=2),
        encoding="utf-8",
    )
    temporal.replace(marcador)
    return {
        "cartera": cartera_copiada,
        "operaciones": operaciones_copiadas,
        "log_cartera": log_copiado,
    }


def _migrar_cartera_legacy(raiz: Path, destino: Path) -> int:
    """Migra la cartera antigua solo si pertenece a PAPER y falta el destino."""
    origen = raiz / "persistent_state" / "portfolio_state_live.json"
    nuevo = destino / "portfolio_state.json"
    if not origen.is_file() or nuevo.exists():
        return 0

    contenido = json.loads(origen.read_text(encoding="utf-8"))
    if not isinstance(contenido, dict):
        raise ValueError("El estado legacy de cartera debe ser un objeto JSON")

    modo_origen = str(contenido.get("mode", "")).lower()
    if modo_origen not in {"simulated", "paper"}:
        registro.info(
            "Se omite la migración de cartera: "
            "el estado antiguo no identifica PAPER"
        )
        return 0

    _copiar_archivo_atomico(origen, nuevo)
    registro.info("Cartera PAPER antigua migrada a %s", nuevo)
    return 1


def _migrar_operaciones_legacy(raiz: Path, destino: Path) -> int:
    """Copia al perfil persistente los archivos del directorio PAPER antiguo."""
    origen = raiz / "data" / "paper_trades"
    if not origen.is_dir():
        return 0

    copiadas = 0
    archivos = sorted(origen.iterdir(), key=lambda ruta: ruta.name.lower())
    for archivo in archivos:
        if not archivo.is_file() or archivo.suffix.lower() not in {".json", ".csv"}:
            continue

        nuevo = destino / archivo.name
        if nuevo.exists():
            continue

        _copiar_archivo_atomico(archivo, nuevo)
        copiadas += 1

    return copiadas


def _migrar_log_cartera_legacy(raiz: Path, destino: Path) -> int:
    """Copia el CSV histórico de cartera si todavía no existe en el perfil."""
    origen = raiz / "data" / "portfolios" / "portfolio_log.csv"
    nuevo = destino / "portfolio_log.csv"
    if not origen.is_file() or nuevo.exists():
        return 0

    _copiar_archivo_atomico(origen, nuevo)
    return 1


def _copiar_archivo_atomico(origen: Path, destino: Path) -> None:
    """Copia un archivo mediante un temporal y lo publica de forma atómica."""
    import os
    import tempfile

    destino.parent.mkdir(parents=True, exist_ok=True)
    descriptor, ruta_temporal = tempfile.mkstemp(
        prefix=f".{destino.name}.",
        suffix=".tmp",
        dir=destino.parent,
    )
    os.close(descriptor)
    temporal = Path(ruta_temporal)

    try:
        shutil.copy2(origen, temporal)
        temporal.replace(destino)
    finally:
        temporal.unlink(missing_ok=True)


def limpiar_perfil_paper(perfil: str, raiz: Path = Path(".")) -> int:
    """Elimina únicamente los datos de un perfil PAPER, nunca el perfil opuesto."""
    directorio = obtener_directorio_paper(perfil, raiz)
    if not directorio.exists():
        return 0
    eliminados = 0
    for archivo in directorio.rglob("*"):
        if archivo.is_file():
            archivo.unlink()
            eliminados += 1
    for subdirectorio in sorted(
        (ruta for ruta in directorio.rglob("*") if ruta.is_dir()),
        key=lambda ruta: len(ruta.parts),
        reverse=True,
    ):
        subdirectorio.rmdir()
    directorio.mkdir(parents=True, exist_ok=True)
    return eliminados



def obtener_ruta_marcador_reset(raiz: Path = Path(".")) -> Path:
    """Devuelve la ruta del marcador de reinicio diferido del perfil persistente."""
    import os

    ruta_configurada = os.getenv("HRM_RUTA_RESET_PAPER")
    if ruta_configurada:
        return Path(ruta_configurada)
    return raiz / "persistent_state" / "reset_paper_persistente_pendiente.json"


def solicitar_reset_persistente(raiz: Path = Path(".")) -> None:
    """Crea atómicamente el marcador para reiniciar PAPER persistente al arrancar."""
    ruta = obtener_ruta_marcador_reset(raiz)
    if ruta.exists():
        raise FileExistsError(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp")
    contenido = {
        "version": 1,
        "solicitado_en": datetime.now(timezone.utc).isoformat(),
        "motivo": "reinicio_manual_paper_persistente",
    }
    temporal.write_text(json.dumps(contenido, indent=2), encoding="utf-8")
    temporal.replace(ruta)


def hay_reset_persistente_pendiente(raiz: Path = Path(".")) -> bool:
    """Indica si existe una solicitud de reinicio PAPER pendiente."""
    return obtener_ruta_marcador_reset(raiz).is_file()


def consumir_reset_persistente(raiz: Path = Path(".")) -> None:
    """Elimina el marcador solo después de completar correctamente el reinicio."""
    ruta = obtener_ruta_marcador_reset(raiz)
    if ruta.exists():
        ruta.unlink()


def _validar_perfil(perfil: str) -> None:
    """Valida que el perfil pertenece al conjunto permitido."""
    if perfil not in {PERFIL_PERSISTENTE, PERFIL_LIMPIO}:
        raise ValueError(f"Perfil de datos no válido: {perfil}")
