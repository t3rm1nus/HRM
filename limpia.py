#!/usr/bin/env python3
"""Reorganiza la carpeta raíz del repositorio HRM.

Por defecto solo SIMULA: imprime lo que haría sin tocar nada.
Para ejecutar de verdad:  python limpiar_raiz.py --aplicar

- Mueve con `git mv` para conservar el historial.
- Reescribe los imports de los módulos movidos.
- Deja en la raíz: main.py, los .bat, hrm_control.ps1, initial_state.json,
  requirements.txt, pytest.ini, README.md, .env.example y .gitignore.
"""
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
ESTE_ARCHIVO = Path(__file__).resolve()

# módulo antiguo (en la raíz) -> módulo nuevo (con puntos)
MODULOS_A_MOVER = {
    "bootstrap": "system.shutdown_manager",
    "fix_l3_dominance": "l3_strategy.l3_dominance",
    "integration_auto_learning": "auto_learning.integration",
    "auto_learning_system": "auto_learning.auto_learning_system",
    "config_loader": "config.config_loader",
    "execution_gateway": "l1_operational.execution_gateway",
    "runtime_loop": "system.runtime_loop",
}

PAQUETES_NUEVOS = ["auto_learning"]

OTROS_MOVIMIENTOS = {
    "diag_l1_quick.py": "scripts/diagnosticos/diag_l1_quick.py",
    "diagnostico_l1_trades.py": "scripts/diagnosticos/diagnostico_l1_trades.py",
    "requirements-desarrollo.txt": "requirements/desarrollo.txt",
    "requirements-lock.txt": "requirements/lock.txt",
    "requirements-opcional.txt": "requirements/opcional.txt",
}

ARCHIVOS_A_ARCHIVAR = [
    "main_antiguo.py",
    "main_refactor.py",
    "original_main.py",
    "fix.py",
    "fix3.py",
    "write_servicio.py",
]

CARPETAS_IGNORADAS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".cache", ".pytest_cache", "archived",
}

EXTENSIONES_CON_IMPORTS = {".py"}
EXTENSIONES_CON_REFERENCIAS = {".py", ".bat", ".ps1", ".ini", ".yml", ".yaml", ".toml", ".cfg"}

LINEAS_GITIGNORE_OBLIGATORIAS = [
    "__pycache__/", "*.pyc", ".env", ".pytest_cache/", ".cache/",
    ".hrm_runtime/", "logs/", "temp/",
]


# ---------------------------------------------------------------- git / disco

def ejecutar_git(argumentos, simular):
    comando = ["git", *argumentos]
    print("  $", " ".join(f'"{a}"' if " " in a else a for a in comando))
    if not simular:
        subprocess.run(comando, cwd=RAIZ, check=True)


def esta_versionado(ruta_relativa):
    resultado = subprocess.run(
        ["git", "ls-files", "--error-unmatch", ruta_relativa],
        cwd=RAIZ,
        capture_output=True,
    )
    return resultado.returncode == 0


def mover_conservando_historial(origen_relativo, destino_relativo, simular):
    origen = RAIZ / origen_relativo
    destino = RAIZ / destino_relativo
    if not origen.exists():
        print(f"  (omitido, no existe) {origen_relativo}")
        return
    if not simular:
        destino.parent.mkdir(parents=True, exist_ok=True)
    if esta_versionado(origen_relativo):
        ejecutar_git(["mv", origen_relativo, destino_relativo], simular)
    else:
        print(f"  mover (sin versionar) {origen_relativo} -> {destino_relativo}")
        if not simular:
            origen.rename(destino)


def borrar_del_disco(ruta):
    shutil.rmtree(ruta) if ruta.is_dir() else ruta.unlink()


def borrar_ruta(ruta_relativa, simular):
    ruta = RAIZ / ruta_relativa
    if not ruta.exists():
        return
    if esta_versionado(ruta_relativa) or ruta.is_dir():
        ejecutar_git(["rm", "-r", "-f", "--ignore-unmatch", ruta_relativa], simular)
    print(f"  borrar {ruta_relativa}")
    if not simular and ruta.exists():
        borrar_del_disco(ruta)


# --------------------------------------------------------- recorrido del proyecto

def recorrer_proyecto():
    """Recorre todos los archivos podando las carpetas ignoradas (rápido)."""
    for carpeta, subcarpetas, archivos in os.walk(RAIZ):
        subcarpetas[:] = [s for s in subcarpetas if s not in CARPETAS_IGNORADAS]
        for nombre in archivos:
            ruta = Path(carpeta) / nombre
            if ruta.resolve() != ESTE_ARCHIVO:
                yield ruta


def archivos_con_extension(extensiones):
    return (ruta for ruta in recorrer_proyecto() if ruta.suffix in extensiones)


# --------------------------------------------------------------- codificaciones

def codificaciones_candidatas(datos):
    if datos.startswith((b"\xff\xfe", b"\xfe\xff")):
        return ["utf-16"]
    if datos.startswith(b"\xef\xbb\xbf"):
        return ["utf-8-sig"]
    return ["utf-8", "cp1252"]


def decodificar(datos):
    """Devuelve (texto, codificacion) o (None, None) si no se puede leer."""
    for codificacion in codificaciones_candidatas(datos):
        try:
            return datos.decode(codificacion), codificacion
        except UnicodeDecodeError:
            continue
    return None, None


# ------------------------------------------------------------------ movimientos

def ruta_de_modulo(modulo):
    return modulo.replace(".", "/") + ".py"


def asegurar_paquete(carpeta, simular):
    archivo_init = RAIZ / carpeta / "__init__.py"
    if archivo_init.exists():
        return
    print(f"  crear {carpeta}/__init__.py")
    if not simular:
        archivo_init.parent.mkdir(parents=True, exist_ok=True)
        archivo_init.touch()


def mover_modulos_python(simular):
    print("\n[1] Módulos Python de la raíz -> su paquete")
    for carpeta in PAQUETES_NUEVOS:
        asegurar_paquete(carpeta, simular)
    for antiguo, nuevo in MODULOS_A_MOVER.items():
        mover_conservando_historial(ruta_de_modulo(antiguo), ruta_de_modulo(nuevo), simular)


def mover_otros_archivos(simular):
    print("\n[2] Diagnósticos y requirements -> su carpeta")
    for origen, destino in OTROS_MOVIMIENTOS.items():
        mover_conservando_historial(origen, destino, simular)


def mover_tests_de_la_raiz(simular):
    print("\n[3] test_*.py de la raíz -> tests/")
    for prueba in sorted(RAIZ.glob("test_*.py")):
        mover_conservando_historial(prueba.name, f"tests/{prueba.name}", simular)


def archivar_mains_antiguos_y_scripts_puntuales(simular):
    print("\n[4] Mains antiguos y scripts de un solo uso -> archived/")
    for nombre in ARCHIVOS_A_ARCHIVAR:
        mover_conservando_historial(nombre, f"archived/{nombre}", simular)


def calcular_residuos():
    residuos = ["__pycache__", "__init__.py"]
    if not (RAIZ / "package.json").exists():
        residuos.append("package-lock.json")
    return residuos


def eliminar_residuos(simular):
    print("\n[5] Residuos")
    for residuo in calcular_residuos():
        borrar_ruta(residuo, simular)


# ---------------------------------------------------------------------- imports

def construir_patron_import(modulo_antiguo):
    return re.compile(
        rf"^(?P<sangria>[ \t]*)(?P<palabra>from|import)[ \t]+{re.escape(modulo_antiguo)}\b(?!\.)",
        re.MULTILINE,
    )


def construir_reemplazo(modulo_antiguo, modulo_nuevo):
    def reemplazar(coincidencia):
        sangria = coincidencia.group("sangria")
        if coincidencia.group("palabra") == "from":
            return f"{sangria}from {modulo_nuevo}"
        return f"{sangria}import {modulo_nuevo} as {modulo_antiguo}"

    return reemplazar


def reescribir_imports_en_texto(texto):
    for antiguo, nuevo in MODULOS_A_MOVER.items():
        patron = construir_patron_import(antiguo)
        texto = patron.sub(construir_reemplazo(antiguo, nuevo), texto)
    return texto


def reescribir_imports_en_archivo(archivo, simular):
    original, codificacion = decodificar(archivo.read_bytes())
    if original is None:
        print(f"  AVISO: no se pudo leer {archivo.relative_to(RAIZ)}, se omite")
        return
    nuevo = reescribir_imports_en_texto(original)
    if nuevo == original:
        return
    print(f"  imports actualizados en {archivo.relative_to(RAIZ)}")
    if not simular:
        archivo.write_bytes(nuevo.encode(codificacion))


def actualizar_imports(simular):
    print("\n[6] Imports de los módulos movidos")
    for archivo in archivos_con_extension(EXTENSIONES_CON_IMPORTS):
        reescribir_imports_en_archivo(archivo, simular)


# ------------------------------------------------------------------ comprobaciones

def asegurar_gitignore(simular):
    print("\n[7] .gitignore")
    ruta = RAIZ / ".gitignore"
    existentes = ruta.read_text(encoding="utf-8").splitlines() if ruta.exists() else []
    faltantes = [l for l in LINEAS_GITIGNORE_OBLIGATORIAS if l not in existentes]
    for linea in faltantes:
        print(f"  añadir '{linea}'")
    if faltantes and not simular:
        ruta.write_text("\n".join([*existentes, *faltantes]) + "\n", encoding="utf-8")


def buscar_homonimos(modulo):
    nombre = f"{modulo}.py"
    return [r for r in recorrer_proyecto() if r.name == nombre and r.parent != RAIZ]


def avisar_modulos_homonimos():
    print("\n[8] REVISAR EN EL DIFF: módulos con homónimo en otra carpeta")
    for modulo in MODULOS_A_MOVER:
        for homonimo in buscar_homonimos(modulo):
            print(f"  '{modulo}' existe también en {homonimo.relative_to(RAIZ)}")


def nombres_de_archivos_movidos():
    origenes = [ruta_de_modulo(m) for m in MODULOS_A_MOVER]
    origenes += [*OTROS_MOVIMIENTOS, *ARCHIVOS_A_ARCHIVAR]
    return sorted({Path(origen).name for origen in origenes})


def contiene_texto(archivo, texto_buscado):
    texto, _ = decodificar(archivo.read_bytes())
    return texto is not None and texto_buscado in texto


def avisar_referencias_a_archivos_movidos():
    print("\n[9] REVISAR A MANO: scripts, .bat y configs que nombran archivos movidos")
    nombres = nombres_de_archivos_movidos()
    for archivo in archivos_con_extension(EXTENSIONES_CON_REFERENCIAS):
        for nombre in nombres:
            if contiene_texto(archivo, nombre):
                print(f"  '{nombre}' aparece en {archivo.relative_to(RAIZ)}")


def mostrar_contenido_de_la_raiz():
    print("\n[10] Contenido actual de la raíz")
    for elemento in sorted(RAIZ.iterdir(), key=lambda e: (e.is_file(), e.name)):
        if elemento.name != ".git":
            print("  ", elemento.name + ("/" if elemento.is_dir() else ""))


# -------------------------------------------------------------------------- main

def leer_argumentos():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--aplicar", action="store_true", help="ejecuta los cambios")
    return analizador.parse_args()


def main():
    simular = not leer_argumentos().aplicar
    print("MODO SIMULACIÓN (usa --aplicar para ejecutar)" if simular else "APLICANDO CAMBIOS")
    mover_modulos_python(simular)
    mover_otros_archivos(simular)
    mover_tests_de_la_raiz(simular)
    archivar_mains_antiguos_y_scripts_puntuales(simular)
    eliminar_residuos(simular)
    actualizar_imports(simular)
    asegurar_gitignore(simular)
    avisar_modulos_homonimos()
    avisar_referencias_a_archivos_movidos()
    mostrar_contenido_de_la_raiz()


if __name__ == "__main__":
    main()