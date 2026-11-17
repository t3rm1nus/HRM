import ast
from pathlib import Path

import pytest

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]
CARPETAS_EXCLUIDAS = {"DEPRECATED_tests_antiguos solo backuo", "archived", "node_modules", "frontend", "tests"}
NIVELES_DE_LOG = {"debug", "info", "warning", "error", "critical", "exception"}
RECEPTORES_LOGURU = {"logger", "self.logger"}
IMPORTACION_LOGURU = "core.logging"


def _archivos_que_usan_loguru() -> list[Path]:
    archivos = []
    for ruta in RAIZ_REPOSITORIO.rglob("*.py"):
        relativa = ruta.relative_to(RAIZ_REPOSITORIO)
        if CARPETAS_EXCLUIDAS & set(relativa.parts):
            continue
        if f"from {IMPORTACION_LOGURU} import logger" in ruta.read_text(encoding="utf-8", errors="ignore"):
            archivos.append(ruta)
    return archivos


def _es_llamada_de_log(nodo: ast.AST) -> bool:
    return (
        isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and nodo.func.attr in NIVELES_DE_LOG
        and ast.unparse(nodo.func.value) in RECEPTORES_LOGURU
    )


def _usa_formato_porcentaje(llamada: ast.Call) -> bool:
    primero = llamada.args[0] if llamada.args else None
    es_texto = isinstance(primero, ast.Constant) and isinstance(primero.value, str)
    return es_texto and "%" in primero.value and len(llamada.args) > 1


def _usa_exc_info(llamada: ast.Call) -> bool:
    return any(clave.arg == "exc_info" for clave in llamada.keywords)


def _llamadas_defectuosas(ruta: Path, es_defectuosa) -> list[str]:
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    return [
        f"{ruta.relative_to(RAIZ_REPOSITORIO)}:{nodo.lineno}"
        for nodo in ast.walk(arbol)
        if _es_llamada_de_log(nodo) and es_defectuosa(nodo)
    ]


def _recopilar(es_defectuosa) -> list[str]:
    defectos: list[str] = []
    for ruta in _archivos_que_usan_loguru():
        defectos.extend(_llamadas_defectuosas(ruta, es_defectuosa))
    return defectos


@pytest.mark.regression
def test_loguru_no_recibe_formato_porcentaje():
    assert _recopilar(_usa_formato_porcentaje) == []


@pytest.mark.regression
def test_loguru_no_recibe_exc_info():
    defectos = _recopilar(_usa_exc_info)
    assert defectos == [], "loguru ignora exc_info; usar logger.exception"
