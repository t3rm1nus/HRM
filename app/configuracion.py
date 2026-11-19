# app/configuracion.py - Lectura de la configuración de ejecución

from core.config import get_config


def leer_paper_mode(por_defecto: bool) -> bool:
    try:
        return getattr(get_config("live"), 'PAPER_MODE', por_defecto)
    except Exception:
        return por_defecto


def determinar_modo() -> str:
    return "simulated" if leer_paper_mode(por_defecto=True) else "live"
