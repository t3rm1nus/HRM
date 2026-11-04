"""Servicio de mercado (solo lectura).

Comparte la instancia de `MarketDataManager` que ya usa el motor: no crea una
fuente paralela, no descarga datos por su cuenta y nunca envía órdenes.
Convierte los DataFrames OHLCV en velas y distingue datos actuales de obsoletos.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Protocol

import pandas as pd

from api.esquemas.mercado_velas import (
    ActualizacionMercado,
    EstadoFrescura,
    EstadoMercado,
    LoteVelas,
    Vela,
)
from api.servicios.servicio_websockets import EventoTipo, ServicioWebSockets, servicio_websockets

registro = logging.getLogger(__name__)

MAX_VELAS_LIMIT = 10000
DEFAULT_VELAS_LIMIT = 500
FACTOR_TOLERANCIA_CACHE = 2.0
VALIDEZ_MINIMA_SEGUNDOS = 60.0
COLUMNAS_OHLCV = ("open", "high", "low", "close", "volume")


class GestorMercado(Protocol):
    """Contrato mínimo que se usa del MarketDataManager existente."""

    symbols: list[str]

    def get_stats(self) -> dict[str, Any]: ...


class ErrorMercado(Exception):
    """Error base del servicio de mercado."""


class MercadoNoInicializado(ErrorMercado):
    """El gestor de datos todavía no está registrado."""


class SimboloDesconocido(ErrorMercado):
    """El símbolo pedido no existe en la última instantánea."""


def normalizar_simbolo(simbolo: str) -> str:
    return simbolo.replace("/", "").replace("-", "").strip().upper()


def limitar_cantidad(limite: int) -> int:
    return max(1, min(int(limite), MAX_VELAS_LIMIT))


def _unidad_temporal(serie: pd.Series) -> Optional[str]:
    """Binance entrega milisegundos como números; las fechas no necesitan unidad."""
    es_numerica = pd.api.types.is_integer_dtype(serie) or pd.api.types.is_float_dtype(serie)
    return "ms" if es_numerica else None


def extraer_instantes(datos: pd.DataFrame) -> pd.DatetimeIndex:
    """Obtiene los instantes UTC desde la columna `timestamp` o desde el índice."""
    if "timestamp" in datos.columns:
        serie = datos["timestamp"]
        return pd.DatetimeIndex(pd.to_datetime(serie, utc=True, unit=_unidad_temporal(serie)))
    indice = datos.index
    if isinstance(indice, pd.DatetimeIndex):
        return indice.tz_localize("UTC") if indice.tz is None else indice.tz_convert("UTC")
    return pd.DatetimeIndex(pd.to_datetime(indice, utc=True, unit=_unidad_temporal(pd.Series(indice))))


def tiene_columnas_ohlcv(datos: Any) -> bool:
    return isinstance(datos, pd.DataFrame) and all(c in datos.columns for c in COLUMNAS_OHLCV)


def fila_es_finita(fila: pd.Series) -> bool:
    return bool(pd.to_numeric(fila[list(COLUMNAS_OHLCV)], errors="coerce").notna().all())


def fila_a_vela(instante: pd.Timestamp, fila: pd.Series) -> Vela:
    return Vela(
        instante=instante.isoformat(),
        apertura=float(fila["open"]),
        maximo=float(fila["high"]),
        minimo=float(fila["low"]),
        cierre=float(fila["close"]),
        volumen=float(fila["volume"]),
    )


def dataframe_a_velas(datos: pd.DataFrame, limite: int) -> list[Vela]:
    """Convierte las últimas `limite` filas válidas en velas ordenadas por tiempo."""
    if datos.empty or not tiene_columnas_ohlcv(datos):
        return []
    recorte = datos.tail(limite)
    instantes = extraer_instantes(recorte)
    velas = [
        fila_a_vela(instante, fila)
        for instante, (_, fila) in zip(instantes, recorte.iterrows())
        if fila_es_finita(fila)
    ]
    return sorted(velas, key=lambda vela: vela.instante)


def calcular_frescura(edad: Optional[float], validez: Optional[float]) -> EstadoFrescura:
    if edad is None or validez is None:
        return "sin_datos"
    umbral = max(validez * FACTOR_TOLERANCIA_CACHE, VALIDEZ_MINIMA_SEGUNDOS)
    return "actual" if edad <= umbral else "obsoleto"


class ServicioMercado:
    """Expone velas y estado de la fuente a partir del gestor compartido."""

    def __init__(
        self,
        servicio_eventos: ServicioWebSockets,
        reloj: Callable[[], float] = time.time,
    ) -> None:
        self._servicio_eventos = servicio_eventos
        self._reloj = reloj
        self._gestor: Optional[GestorMercado] = None
        self._instantanea: dict[str, pd.DataFrame] = {}
        self._instantanea_en: Optional[float] = None

    # --- Registro (lo llama el motor; solo guarda referencias) ---
    def registrar_gestor(self, gestor: GestorMercado) -> None:
        self._gestor = gestor

    def registrar_instantanea(self, datos: dict[str, pd.DataFrame]) -> None:
        validos = {s: d for s, d in (datos or {}).items() if tiene_columnas_ohlcv(d)}
        self._instantanea = validos
        self._instantanea_en = self._reloj()

    # --- Consultas ---
    def obtener_estado(self) -> EstadoMercado:
        if self._gestor is None:
            return EstadoMercado(disponible=False, frescura="sin_datos")
        edad, validez, fuente = self._leer_cache()
        return EstadoMercado(
            disponible=True,
            frescura=self._frescura_efectiva(edad, validez),
            fuente=fuente,
            edad_segundos=edad,
            validez_segundos=validez,
            simbolos=self.listar_simbolos(),
        )

    def listar_simbolos(self) -> list[str]:
        return sorted(self._instantanea)

    def obtener_velas(self, simbolo: str, limite: int = DEFAULT_VELAS_LIMIT) -> LoteVelas:
        self._exigir_gestor()
        clave = normalizar_simbolo(simbolo)
        if clave not in self._instantanea:
            raise SimboloDesconocido(f"Sin datos de mercado para {clave}")
        velas = dataframe_a_velas(self._instantanea[clave], limitar_cantidad(limite))
        return LoteVelas(
            simbolo=clave,
            cantidad=len(velas),
            frescura=self.obtener_estado().frescura,
            velas=velas,
        )

    def construir_actualizacion(self) -> Optional[ActualizacionMercado]:
        """Solo la última vela de cada símbolo: nunca el histórico completo."""
        ultimas: dict[str, Vela] = {}
        for simbolo in self.listar_simbolos():
            velas = dataframe_a_velas(self._instantanea[simbolo], 1)
            if velas:
                ultimas[simbolo] = velas[-1]
        if not ultimas:
            return None
        return ActualizacionMercado(frescura=self.obtener_estado().frescura, ultimas_velas=ultimas)

    async def notificar_datos(self, datos: dict[str, pd.DataFrame]) -> None:
        """Punto único para el motor: registra la instantánea y la difunde."""
        try:
            self.registrar_instantanea(datos)
        except Exception:
            registro.exception("No se pudo registrar la instantánea de mercado")
            return
        await self.publicar_actualizacion()

    async def publicar_actualizacion(self) -> None:
        """Difunde por WebSocket; un fallo aquí jamás debe afectar al motor."""
        try:
            actualizacion = self.construir_actualizacion()
            if actualizacion is not None:
                await self._servicio_eventos.publicar(EventoTipo.MERCADO, actualizacion.model_dump())
        except Exception:
            registro.exception("No se pudo publicar la actualización de mercado")

    # --- Internos ---
    def _exigir_gestor(self) -> None:
        if self._gestor is None:
            raise MercadoNoInicializado("El gestor de mercado aún no está inicializado")

    def _leer_cache(self) -> tuple[Optional[float], Optional[float], Optional[str]]:
        try:
            estadisticas = self._gestor.get_stats()  # type: ignore[union-attr]
            cache = estadisticas.get("cache_status", {})
            validez = float(estadisticas.get("cache_valid_seconds", 0)) or None
            if not cache.get("has_cache"):
                return None, validez, None
            return float(cache.get("cache_age_seconds", 0.0)), validez, cache.get("cache_source")
        except Exception:
            registro.exception("No se pudieron leer las estadísticas del gestor")
            return None, None, None

    def _frescura_efectiva(self, edad: Optional[float], validez: Optional[float]) -> EstadoFrescura:
        if not self._instantanea:
            return "sin_datos"
        return calcular_frescura(edad, validez)


servicio_mercado = ServicioMercado(servicio_websockets)
