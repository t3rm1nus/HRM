
"""Regresiones para el comportamiento seguro del diagnóstico manual L1."""

import importlib.util
from pathlib import Path


RAIZ_REPOSITORIO = Path(__file__).resolve().parents[1]
RUTA_DIAGNOSTICO = RAIZ_REPOSITORIO / "test_l1_diagnostic.py"


def cargar_modulo_diagnostico():
    """Importa el script sin ejecutar su punto de entrada manual."""
    especificacion = importlib.util.spec_from_file_location(
        "diagnostico_l1_prueba",
        RUTA_DIAGNOSTICO,
    )

    assert especificacion is not None
    assert especificacion.loader is not None

    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)

    return modulo


def test_importar_diagnostico_no_ejecuta_el_proceso(capsys):
    """Importar el script no debe leer datos ni imprimir el diagnóstico."""
    cargar_modulo_diagnostico()

    salida = capsys.readouterr()

    assert salida.out == ""
    assert salida.err == ""


def test_archivo_de_datos_inexistente_devuelve_error_controlado(
    tmp_path,
    capsys,
):
    """Un CSV ausente debe producir código 1 sin cargar modelos ni pandas."""
    modulo = cargar_modulo_diagnostico()
    ruta_inexistente = tmp_path / "datos_que_no_existen.csv"

    codigo_salida = modulo.ejecutar_diagnostico(ruta_inexistente)

    salida = capsys.readouterr()

    assert codigo_salida == 1
    assert "No se encontró el archivo" in salida.out
