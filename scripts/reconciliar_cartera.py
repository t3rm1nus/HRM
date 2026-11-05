"""Diagnóstico de solo lectura: ¿cuadra la cartera guardada con el log de operaciones?

Uso (desde la raíz del repo):
    python scripts/reconciliar_cartera.py

No modifica ningún archivo.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

RUTA_OPERACIONES = Path("data/paper_trades/paper_trades.json")
RUTA_ESTADO_CARTERA = Path("persistent_state/portfolio_state_live.json")
RUTA_ESTADO_INICIAL = Path("initial_state.json")
TOLERANCIA = 1e-9


def leer_json(ruta: Path) -> dict[str, Any] | None:
    if not ruta.exists():
        return None
    with ruta.open(encoding="utf-8") as archivo:
        return json.load(archivo)


def cantidad_con_signo(operacion: dict[str, Any]) -> float:
    cantidad = float(operacion.get("quantity", 0.0))
    return cantidad if str(operacion.get("side", "")).lower() == "buy" else -cantidad


def calcular_posiciones_desde_operaciones(operaciones: list[dict[str, Any]]) -> dict[str, float]:
    posiciones: dict[str, float] = defaultdict(float)
    for operacion in operaciones:
        posiciones[operacion["symbol"]] += cantidad_con_signo(operacion)
    return dict(posiciones)


def leer_posiciones_guardadas(estado: dict[str, Any]) -> dict[str, float]:
    cartera = estado.get("portfolio", {})
    return {
        simbolo: float(datos.get("position", 0.0))
        for simbolo, datos in cartera.items()
        if isinstance(datos, dict) and "position" in datos
    }


def contar_lados(operaciones: list[dict[str, Any]]) -> dict[str, int]:
    conteo: dict[str, int] = defaultdict(int)
    for operacion in operaciones:
        conteo[str(operacion.get("side", "?")).lower()] += 1
    return dict(conteo)


def rango_de_fechas(operaciones: list[dict[str, Any]]) -> tuple[str, str]:
    fechas = sorted(str(operacion.get("timestamp", "")) for operacion in operaciones)
    return (fechas[0], fechas[-1]) if fechas else ("-", "-")


def imprimir_comparacion(desde_log: dict[str, float], guardadas: dict[str, float]) -> None:
    print(f"{'Símbolo':<10}{'Según log':>16}{'Cartera guardada':>20}{'Diferencia':>16}")
    for simbolo in sorted(set(desde_log) | set(guardadas)):
        esperado = desde_log.get(simbolo, 0.0)
        real = guardadas.get(simbolo, 0.0)
        marca = "" if abs(esperado - real) < TOLERANCIA else "  <-- NO CUADRA"
        print(f"{simbolo:<10}{esperado:>16.6f}{real:>20.6f}{real - esperado:>16.6f}{marca}")


def imprimir_resumen_log(operaciones: list[dict[str, Any]]) -> None:
    primera, ultima = rango_de_fechas(operaciones)
    print(f"Operaciones en el log: {len(operaciones)} {contar_lados(operaciones)}")
    print(f"Primera: {primera}\nÚltima:  {ultima}\n")


def imprimir_estado_guardado(estado: dict[str, Any]) -> None:
    print(f"Estado guardado con marca de tiempo: {estado.get('timestamp', 'sin fecha')}")
    print(f"Balance inicial en el estado: {estado.get('initial_balance')}")


def imprimir_capital_inicial_configurado() -> None:
    estado_inicial = leer_json(RUTA_ESTADO_INICIAL) or {}
    print(f"Capital en initial_state.json: {estado_inicial.get('capital_usdt', 'no existe')}\n")


def main() -> None:
    log = leer_json(RUTA_OPERACIONES)
    estado = leer_json(RUTA_ESTADO_CARTERA)
    if log is None or estado is None:
        print(f"Faltan archivos: {RUTA_OPERACIONES} ({log is not None}), {RUTA_ESTADO_CARTERA} ({estado is not None})")
        return

    operaciones = log.get("trades", [])
    imprimir_resumen_log(operaciones)
    imprimir_comparacion(
        calcular_posiciones_desde_operaciones(operaciones),
        leer_posiciones_guardadas(estado),
    )
    print()
    imprimir_estado_guardado(estado)
    imprimir_capital_inicial_configurado()


if __name__ == "__main__":
    main()
