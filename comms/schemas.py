# schemas.py
# typing nativo + pydantic opcional (robusto en runtime)
from typing import Dict, List, TypedDict, Literal, Optional
from dataclasses import dataclass, field
from datetime import datetime

Vers = Literal["v1"]

class Orden(TypedDict):
    id: str
    activo: str
    lado: Literal["buy", "sell"]
    qty: float
    px: Optional[float]  # None si es market
    status: Literal["new", "sent", "filled", "rejected"]

class Riesgo(TypedDict):
    aprobado: bool
    motivo: Optional[str]

class State(TypedDict):
    version: Vers
    ciclo_id: int
    ts: str
    mercado: Dict[str, float]              # precios spot normalizados
    estrategia: Optional[str]
    portfolio: Dict[str, float]            # saldos por activo (notional o unidades, ver punto 2)
    universo: List[str]
    exposicion: Dict[str, float]           # targets relativos o absolutos (definir contrato)
    senales: Dict[str, float]              # señal [-1..1] o tamaño relativo
    ordenes: List[Orden]
    riesgo: Riesgo
    deriva: bool
