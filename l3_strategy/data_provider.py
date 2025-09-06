# data_provider.py
# -*- coding: utf-8 -*-
"""
Data Provider centralizado para L3 (HRM)
----------------------------------------
Unifica acceso a:
- Precios (CSV local, Yahoo Finance opcional, mock)
- Macro (FRED opcional, CSV local)
- On-chain (CSV local / hooks para API)
- Salidas de inferencia L3 (sentiment/regime/volatility/portfolio)
- Construcción de dataset consolidado para entrenamiento o inferencia

Estrategia de fallback por fuente:
1) CSV local en data/datos_para_modelos_l3/
2) (opcional) Descarga online (yfinance / FRED) si USE_ONLINE_SOURCES=true
3) Mock sintético estable si no hay datos

Requisitos opcionales:
- yfinance (para descarga de precios)
- fredapi (para FRED)
- .env con USE_ONLINE_SOURCES, FRED_API_KEY, etc. (opcional)

Salida típica:
- Panel de precios MultiIndex (tickers × columnas OHLCV)
- Matriz "Adj Close" por ticker
- Macro mergeable por fecha
- JSONs de inferencia (sentiment/regime/volatility/portfolio/l3_output)
"""

from __future__ import annotations

import os
import json
import math
import importlib
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# =========================
# Configuración y logging
# =========================

BASE_DATA_DIR = os.getenv("HRM_DATA_DIR", "data/datos_para_modelos_l3")
INFER_DIR = os.getenv("HRM_INFER_DIR", "data/datos_inferencia")
PRICES_DIR = BASE_DATA_DIR  # Precios sueltos por ticker (*.csv) o subcarpetas
MACRO_DIR = os.path.join(BASE_DATA_DIR, "macro")
ONCHAIN_DIR = os.path.join(BASE_DATA_DIR, "onchain")

os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(INFER_DIR, exist_ok=True)
os.makedirs(MACRO_DIR, exist_ok=True)
os.makedirs(ONCHAIN_DIR, exist_ok=True)

USE_ONLINE_SOURCES = os.getenv("USE_ONLINE_SOURCES", "false").lower() in {"1", "true", "yes"}
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

from core.logging import logger

logger.info("HRM.L3.data_provider")
if not logger.handlers:
    # Configuración básica si el proyecto no lo configura fuera
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# =========================
# Utilidades internas
# =========================

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _read_csv_smart(path: str) -> pd.DataFrame:
    """
    Lee un CSV con heurística:
    - Usa 'Date' como índice si existe; si no, toma la 1ª columna como índice temporal.
    - Parsea fechas, ordena, elimina duplicados y espacios en headers.
    - Normaliza nombres de columnas clave (OHLCV).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, parse_dates=True)
    except Exception as e:
        raise RuntimeError(f"No se pudo leer CSV en {path}: {e}")

    # Detectar columna fecha
    date_col = None
    for cand in ["Date", "date", "timestamp", "Datetime", "datetime"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # usar la primera columna como fecha si es parseable
        date_col = df.columns[0]

    # Limpiar headers y establecer índice fecha
    df.columns = [str(c).strip() for c in df.columns]
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    except Exception:
        pass
    df = df.dropna(subset=[date_col]).set_index(date_col)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()[~df.index.duplicated(keep="last")]  # CORRECCIÓN: Quitado paréntesis extra

    # Normalizar nombres OHLCV comunes (por si vienen con otros labels)
    rename_map = {
        "AdjClose": "Adj Close",
        "Adj_Close": "Adj Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
        "Vol": "Volume",
        "vol": "Volume",
    }
    df = df.rename(columns=rename_map)

    # Asegurar tipos numéricos en posibles columnas OHLCV
    for c in PRICE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop filas totalmente vacías
    if len(df.columns):
        df = df.dropna(how="all")

    return df


def _try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _generate_mock_price_series(n: int = 365, start_price: float = 100.0, seed: Optional[int] = 42) -> pd.DataFrame:
    """
    Genera un OHLCV sintético diario por  n  días usando random walk con volatilidad moderada.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=n, freq="D")
    rets = rng.normal(loc=0.0005, scale=0.02, size=n)  # drift y sigma
    price = start_price * np.exp(np.cumsum(rets))
    close = price
    open_ = np.concatenate([[price[0]], price[:-1]])
    high = np.maximum(open_, close) * (1 + rng.normal(0.002, 0.003, size=n).clip(-0.005, 0.02))
    low = np.minimum(open_, close) * (1 - rng.normal(0.002, 0.003, size=n).clip(-0.02, 0.005))
    volume = rng.integers(low=1_000, high=100_000, size=n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume},
        index=dates,
    )
    return df


def _slice_dates(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df


# =========================
# Precios / Mercado
# =========================

def get_market_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    prefer_fieldset: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Adj Close", "Volume"),
    save_if_downloaded: bool = True,
) -> pd.DataFrame:
    """
    Devuelve OHLCV del activo 'ticker' con índice temporal.
    Prioriza CSV local en BASE_DATA_DIR. Si no existe:
      - descarga via yfinance (si USE_ONLINE_SOURCES y módulo disponible)
      - genera mock sintético como último recurso
    """
    csv_path = os.path.join(PRICES_DIR, f"{ticker}.csv")
    df: Optional[pd.DataFrame] = None

    # 1) CSV local
    if os.path.exists(csv_path):
        try:
            df = _read_csv_smart(csv_path)
            logger.info(f"[{ticker}] Cargado desde CSV local: {csv_path} ({len(df)} filas)")
        except Exception as e:
            logger.warning(f"[{ticker}] Error leyendo CSV local, intentando otras fuentes: {e}")

    # 2) Online (yfinance)
    if (df is None or df.empty) and USE_ONLINE_SOURCES:
        yf = _try_import("yfinance")
        if yf is not None:
            try:
                logger.info(f"[{ticker}] Descargando de yfinance (interval={interval})...")
                ydf = yf.download(ticker, interval=interval, progress=False)
                if isinstance(ydf, pd.DataFrame) and not ydf.empty:
                    ydf.index = pd.to_datetime(ydf.index)
                    # Asegurar el set de columnas esperado (si faltan, deja las que haya)
                    for col in prefer_fieldset:
                        if col not in ydf.columns and col in ["Adj Close"] and "Close" in ydf.columns:
                            ydf["Adj Close"] = ydf["Close"]
                    df = ydf
                    if save_if_downloaded:
                        try:
                            df.to_csv(csv_path)
                            logger.info(f"[{ticker}] Guardado cache CSV: {csv_path}")
                        except Exception as e:
                            logger.warning(f"[{ticker}] No se pudo guardar cache CSV: {e}")
            except Exception as e:
                logger.warning(f"[{ticker}] Falló yfinance: {e}")
        else:
            logger.info("yfinance no disponible; omitiendo descarga online.")

    # 3) Mock
    if df is None or df.empty:
        logger.warning(f"[{ticker}] Usando MOCK de precios (no hay CSV ni online).")
        df = _generate_mock_price_series(n=365, start_price=100.0 + hash(ticker) % 200)

    # Normalización final + slice de fechas
    df = df.copy()
    # Garantizar columnas OHLCV si existen
    for c in PRICE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_index()
    df = _slice_dates(df, start, end)

    return df


def get_prices_matrix(
    tickers: List[str],
    field: str = "Adj Close",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Devuelve una matriz (fecha × tickers) del campo seleccionado (por defecto 'Adj Close').
    Si un ticker no tiene 'Adj Close', usa 'Close' como fallback.
    """
    matrix = {}
    for t in tickers:
        df = get_market_data(t, start=start, end=end)
        use_field = field if field in df.columns else ("Close" if "Close" in df.columns else None)
        if use_field is None:
            logger.warning(f"[{t}] No se encontró columna '{field}' ni 'Close'. Se omite.")
            continue
        series = df[use_field].rename(t)
        matrix[t] = series

    if not matrix:
        return pd.DataFrame()

    out = pd.concat(matrix.values(), axis=1)
    out.index.name = "Date"
    return out


def build_prices_panel(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Devuelve un panel de precios con columnas MultiIndex:
      nivel 0: ticker
      nivel 1: columna OHLCV
    """
    parts = {}
    for t in tickers:
        df = get_market_data(t, start=start, end=end)
        keep_cols = [c for c in PRICE_COLS if c in df.columns]
        parts[t] = df[keep_cols]

    if not parts:
        return pd.DataFrame()

    panel = pd.concat(parts, axis=1)  # columnas MultiIndex
    panel.index.name = "Date"
    return panel


# =========================
# Macro (FRED / CSV local)
# =========================

@lru_cache(maxsize=16)
def get_macro_data(
    indicators: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con columnas = indicadores macro.
    Orden de resolución:
      1) CSV local en data/datos_para_modelos_l3/macro/{indicator}.csv
      2) (opcional) fredapi si USE_ONLINE_SOURCES y FRED_API_KEY presente
      3) DataFrame vacío si no disponible

    Ejemplos de indicadores FRED: ["FEDFUNDS", "CPIAUCSL", "DGS10", "GS10"]
    """
    if indicators is None:
        indicators = ["FEDFUNDS", "CPIAUCSL", "DGS10"]  # set por defecto

    frames = []
    unresolved = []

    # 1) CSV local
    for ind in indicators:
        local_path = os.path.join(MACRO_DIR, f"{ind}.csv")
        if os.path.exists(local_path):
            try:
                df = _read_csv_smart(local_path)
                # Usar primera columna como valor si no está claro
                if df.shape[1] == 1:
                    s = df.iloc[:, 0].rename(ind)
                else:
                    # heurística: preferir columna 'value' o 'Value' si existe
                    col = "value" if "value" in df.columns else (df.columns[0])
                    s = df[col].rename(ind)
                frames.append(s.to_frame())
                continue
            except Exception as e:
                logger.warning(f"[MACRO {ind}] CSV local inválido: {e}")
        unresolved.append(ind)

    # 2) FRED online opcional
    if unresolved and USE_ONLINE_SOURCES and FRED_API_KEY:
        fredapi = _try_import("fredapi")
        if fredapi is not None:
            try:
                fred = fredapi.Fred(api_key=FRED_API_KEY)
                for ind in list(unresolved):
                    try:
                        s = fred.get_series(ind)
                        if s is not None and len(s):
                            s.index = pd.to_datetime(s.index)
                            frames.append(pd.Series(s, name=ind).to_frame())
                            unresolved.remove(ind)
                            # cache local
                            cache_path = os.path.join(MACRO_DIR, f"{ind}.csv")
                            pd.DataFrame({ind: s}).to_csv(cache_path, index_label="Date")
                            logger.info(f"[MACRO {ind}] Guardado CSV cache: {cache_path}")
                    except Exception as e:
                        logger.warning(f"[MACRO {ind}] FRED error: {e}")
            except Exception as e:
                        logger.warning(f"[MACRO {ind}] FRED error: {e}")
        else:
            logger.info("fredapi no disponible; omitiendo descarga FRED.")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "Date"
    out = _slice_dates(out, start, end)
    return out


# =========================
# On-chain (CSV local / hooks)
# =========================

def get_onchain_data(
    metrics: Optional[List[str]] = None,
    asset: str = "BTC",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame on-chain con columnas=metrics leídas desde CSV local:
    data/datos_para_modelos_l3/onchain/{asset}_{metric}.csv

    Para integrar APIs (p.ej. Glassnode), añadir lógica opcional similar a FRED.
    """
    if metrics is None:
        metrics = ["active_addresses", "nvt_ratio"]  # ejemplo

    frames = []
    for m in metrics:
        path = os.path.join(ONCHAIN_DIR, f"{asset}_{m}.csv")
        if not os.path.exists(path):
            logger.info(f"[ONCHAIN] No existe CSV {path}; omitiendo '{m}'.")
            continue
        try:
            df = _read_csv_smart(path)
            if df.shape[1] == 1:
                s = df.iloc[:, 0].rename(m)
            else:
                col = "value" if "value" in df.columns else df.columns[0]
                s = df[col].rename(m)
            frames.append(s.to_frame())
        except Exception as e:
            logger.warning(f"[ONCHAIN {m}] CSV inválido: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "Date"
    out = _slice_dates(out, start, end)
    return out


# =========================
# Salidas de inferencia L3
# =========================

def _load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error leyendo JSON {path}: {e}")
    return None


def get_sentiment_latest() -> Optional[Dict[str, Any]]:
    return _load_json_safe(os.path.join(INFER_DIR, "sentiment.json"))


def get_regime_latest() -> Optional[Dict[str, Any]]:
    return _load_json_safe(os.path.join(INFER_DIR, "regime_detection.json"))


def get_volatility_latest() -> Optional[Dict[str, Any]]:
    return _load_json_safe(os.path.join(INFER_DIR, "volatility.json"))


def get_portfolio_latest() -> Optional[Dict[str, Any]]:
    # Opcional: puede existir si ya hay pesos estratégicos calculados
    return _load_json_safe(os.path.join(INFER_DIR, "portfolio.json"))


def get_l3_output_bundle() -> Optional[Dict[str, Any]]:
    return _load_json_safe(os.path.join(INFER_DIR, "l3_output.json"))


# =========================
# Dataset consolidado
# =========================

@dataclass
class FullDataset:
    prices_panel: pd.DataFrame
    prices_matrix: pd.DataFrame
    macro: pd.DataFrame
    onchain: pd.DataFrame
    sentiment: Optional[Dict[str, Any]]
    regime: Optional[Dict[str, Any]]
    volatility: Optional[Dict[str, Any]]
    portfolio: Optional[Dict[str, Any]]
    l3_output: Optional[Dict[str, Any]]
    meta: Dict[str, Any]


def get_full_dataset(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_macro: bool = True,
    include_onchain: bool = False,
    onchain_asset: str = "BTC",
    onchain_metrics: Optional[List[str]] = None,
) -> FullDataset:
    """
    Construye un paquete de datos para modelos L3 o inferencia:
    - prices_panel: columnas MultiIndex [(ticker, OHLCV)]
    - prices_matrix: DataFrame 'Adj Close' por ticker
    - macro / onchain: DataFrames mergeables por fecha
    - JSONs de inferencia recientes (sentiment, regime, volatility, portfolio, l3_output)
    """
    # Precios
    prices_panel = build_prices_panel(tickers, start=start, end=end)
    prices_matrix = get_prices_matrix(tickers, field="Adj Close", start=start, end=end)

    # Macro y on-chain
    macro_df = get_macro_data(start=start, end=end) if include_macro else pd.DataFrame()
    onchain_df = (
        get_onchain_data(metrics=onchain_metrics, asset=onchain_asset, start=start, end=end)
        if include_onchain
        else pd.DataFrame()
    )

    # Inferencias L3
    sentiment = get_sentiment_latest()
    regime = get_regime_latest()
    volatility = get_volatility_latest()
    portfolio = get_portfolio_latest()
    l3_output = get_l3_output_bundle()

    meta = {
        "tickers": tickers,
        "start": start,
        "end": end,
        "rows_prices": int(prices_matrix.shape[0]) if prices_matrix is not None else 0,
        "cols_prices": int(prices_matrix.shape[1]) if prices_matrix is not None else 0,
        "use_online_sources": USE_ONLINE_SOURCES,
        "timestamp": datetime.utcnow().isoformat(),
    }

    return FullDataset(
        prices_panel=prices_panel,
        prices_matrix=prices_matrix,
        macro=macro_df,
        onchain=onchain_df,
        sentiment=sentiment,
        regime=regime,
        volatility=volatility,
        portfolio=portfolio,
        l3_output=l3_output,
        meta=meta,
    )


# =========================
# Ejecución directa (smoke test)
# =========================

if __name__ == "__main__":
    # Pequeño test manual
    test_tickers = ["BTC-USD", "ETH-USD", "AAPL"]
    ds = get_full_dataset(test_tickers, include_macro=True, include_onchain=False)

    logger.info(f"Panel precios: {ds.prices_panel.shape} (MultiIndex columns)")
    logger.info(f"Matriz Adj Close: {ds.prices_matrix.shape}")
    logger.info(f"Macro shape: {ds.macro.shape}")
    logger.info(f"On-chain shape: {ds.onchain.shape}")
    logger.info(f"Sentiment JSON: {'ok' if ds.sentiment else 'none'}")
    logger.info(f"Regime JSON: {'ok' if ds.regime else 'none'}")
    logger.info(f"Volatility JSON: {'ok' if ds.volatility else 'none'}")
    logger.info(f"Portfolio JSON: {'ok' if ds.portfolio else 'none'}")
    logger.info(f"L3 bundle JSON: {'ok' if ds.l3_output else 'none'}")
    logger.info(f"Meta: {ds.meta}")