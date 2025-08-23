"""
Utilidades para cargar dataset de BTC y generar features sin ensuciar el código.

Objetivos:
- Tomar datos históricos solo de BTC/USDT (o BTC/USD) con columnas estándar
  requeridas: timestamp (índice), close; opcionales: open, high, low, volume.
- Generar features: precio (delta, EMA, SMA), volumen relativo, momentum (RSI, MACD).
- Multitimeframe: agregar features de 1m + 5m con sufijos.
- Split train/test respetando el orden temporal.

Nota: Se implementa todo en este archivo para evitar dispersión de utilidades.
"""

from __future__ import annotations

from typing import Optional, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# Configuración por defecto de features
DEFAULT_FEATURE_CONFIG: Dict[str, object] = {
    "ema_windows": (10, 20),
    "sma_windows": (10, 20),
    "rsi_window": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "vol_window": 20,
    # Nuevos indicadores
    "adx_window": 14,
    "stoch_k_window": 14,
    "stoch_d_window": 3,
    "bb_window": 20,
    "bb_std": 2.0,
    "atr_window": 14,
}


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], errors="coerce"))
            df = df.drop(columns=["timestamp"])  # mantener limpio
        else:
            raise ValueError("El DataFrame debe tener índice datetime o columna 'timestamp'.")
    return df.sort_index()


def normalize_btc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas a [open, high, low, close, volume] cuando sea posible.
    Si solo existe 'BTC_close', se mapea a 'close'.
    """
    mapping = {}
    if "BTC_close" in df.columns and "close" not in df.columns:
        mapping["BTC_close"] = "close"
    # Si viniese de alguna otra convención, añadir aquí mapeos futuros
    if mapping:
        df = df.rename(columns=mapping)
    return df


def compute_price_features(
    df: pd.DataFrame,
    ema_windows: Iterable[int] = DEFAULT_FEATURE_CONFIG["ema_windows"],
    sma_windows: Iterable[int] = DEFAULT_FEATURE_CONFIG["sma_windows"],
) -> pd.DataFrame:
    if "close" not in df.columns:
        raise ValueError("Se requiere columna 'close' para features de precio.")

    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"].astype(float)
    out["delta_close"] = out["close"].pct_change().fillna(0.0)

    for w in ema_windows:
        out[f"ema_{w}"] = out["close"].ewm(span=w, adjust=False).mean()
    for w in sma_windows:
        out[f"sma_{w}"] = out["close"].rolling(window=w, min_periods=1).mean()
    return out


def compute_volume_features(
    df: pd.DataFrame,
    vol_window: int = int(DEFAULT_FEATURE_CONFIG["vol_window"]),
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        out["volume"] = vol
        rolling_mean = vol.rolling(window=vol_window, min_periods=1).mean()
        out["vol_rel"] = (vol / rolling_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return out

def normalize_eth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas de ETH/USDT a [open, high, low, close, volume].
    Similar a BTC pero para ETH.
    """
    mapping = {}
    if "ETH_close" in df.columns and "close" not in df.columns:
        mapping["ETH_close"] = "close"
    # Mapear otras convenciones si existen
    if mapping:
        df = df.rename(columns=mapping)
    return df


def compute_rsi(df: pd.DataFrame, window: int = int(DEFAULT_FEATURE_CONFIG["rsi_window"])) -> pd.Series:
    close = df["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = (avg_gain / avg_loss).replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)


def compute_macd(
    df: pd.DataFrame,
    fast: int = int(DEFAULT_FEATURE_CONFIG["macd_fast"]),
    slow: int = int(DEFAULT_FEATURE_CONFIG["macd_slow"]),
    signal: int = int(DEFAULT_FEATURE_CONFIG["macd_signal"]),
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rsi"] = compute_rsi(df)
    macd, macd_signal, macd_hist = compute_macd(df)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist
    return out


def compute_adx(df: pd.DataFrame, window: int = int(DEFAULT_FEATURE_CONFIG["adx_window"])) -> pd.Series:
    """Calcula ADX (Wilder) aproximado. Requiere high/low/close."""
    if not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > (high - prev_high).clip(lower=0), 0.0)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(alpha=1 / window, adjust=False).mean()
    return adx


def compute_stochastic(
    df: pd.DataFrame,
    k_window: int = int(DEFAULT_FEATURE_CONFIG["stoch_k_window"]),
    d_window: int = int(DEFAULT_FEATURE_CONFIG["stoch_d_window"]),
) -> pd.DataFrame:
    """Stochastic Oscillator: %K y %D. Requiere high/low/close."""
    out = pd.DataFrame(index=df.index)
    if not {"high", "low", "close"}.issubset(df.columns):
        out["stoch_k"] = np.nan
        out["stoch_d"] = np.nan
        return out
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    stoch_k = 100 * (close - lowest_low) / denom
    stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d
    return out


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume. Requiere close y volume."""
    if not {"close", "volume"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    sign = np.sign(close.diff()).fillna(0.0)
    obv = (sign * volume).cumsum()
    return obv


def compute_bbw(df: pd.DataFrame, window: int = int(DEFAULT_FEATURE_CONFIG["bb_window"]), num_std: float = float(DEFAULT_FEATURE_CONFIG["bb_std"])) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    close = df["close"].astype(float)
    m = close.rolling(window=window, min_periods=window).mean()
    s = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = m + num_std * s
    lower = m - num_std * s
    denom = m.replace(0, np.nan)
    bbw = (upper - lower) / denom
    return bbw


def compute_atr(df: pd.DataFrame, window: int = int(DEFAULT_FEATURE_CONFIG["atr_window"])) -> pd.Series:
    if not {"high", "low", "close"}.issubset(df.columns):
        return pd.Series(index=df.index, dtype=float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()
    return atr


def build_features_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df_1m = ensure_datetime_index(normalize_btc_columns(df_1m.copy()))
    price = compute_price_features(df_1m)
    vol = compute_volume_features(df_1m)
    mom = add_momentum_features(df_1m)

    # Indicadores adicionales
    adx = compute_adx(df_1m)
    stoch = compute_stochastic(df_1m)
    obv = compute_obv(df_1m)
    bbw = compute_bbw(df_1m)
    atr = compute_atr(df_1m)

    feat = pd.concat([price, vol, mom, adx.rename("trend_adx"), stoch.rename(columns={"stoch_k": "momentum_stoch", "stoch_d": "momentum_stoch_signal"}), obv.rename("volume_obv"), bbw.rename("volatility_bbw"), atr.rename("volatility_atr")], axis=1)

    # Alias con nombres solicitados
    sma_fast = int(DEFAULT_FEATURE_CONFIG["sma_windows"][0])
    sma_slow = int(DEFAULT_FEATURE_CONFIG["sma_windows"][1])
    ema_fast = int(DEFAULT_FEATURE_CONFIG["ema_windows"][0])
    ema_slow = int(DEFAULT_FEATURE_CONFIG["ema_windows"][1])
    if f"sma_{sma_fast}" in feat.columns:
        feat["trend_sma_fast"] = feat[f"sma_{sma_fast}"]
    if f"sma_{sma_slow}" in feat.columns:
        feat["trend_sma_slow"] = feat[f"sma_{sma_slow}"]
    if f"ema_{ema_fast}" in feat.columns:
        feat["trend_ema_fast"] = feat[f"ema_{ema_fast}"]
    if f"ema_{ema_slow}" in feat.columns:
        feat["trend_ema_slow"] = feat[f"ema_{ema_slow}"]
    if "macd" in feat.columns:
        feat["trend_macd"] = feat["macd"]
    if "rsi" in feat.columns:
        feat["momentum_rsi"] = feat["rsi"]
    return feat


def resample_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df_1m = ensure_datetime_index(df_1m)
    agg = {c: "last" for c in df_1m.columns}
    if "volume" in df_1m.columns:
        agg["volume"] = "sum"
    df_5m = df_1m.resample("5T").agg(agg)
    return df_5m.dropna(how="all")


def build_features_5m_from_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    df_5m_base = resample_to_5m(df_1m)
    feat_5m = build_features_1m(df_5m_base)
    feat_5m = feat_5m.add_suffix("_5m")
    # Volver a 1m vía forward-fill para alinear con timestamps 1m
    feat_5m_aligned = feat_5m.reindex(df_1m.index, method="ffill")
    return feat_5m_aligned


def build_multitimeframe_features(df_1m: pd.DataFrame, df_5m: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    feat_1m = build_features_1m(df_1m)
    if df_5m is None:
        feat_5m = build_features_5m_from_1m(df_1m)
    else:
        feat_5m = build_features_1m(ensure_datetime_index(normalize_btc_columns(df_5m.copy()))).add_suffix("_5m")
        feat_5m = feat_5m.reindex(feat_1m.index, method="ffill")
    return pd.concat([feat_1m, feat_5m], axis=1)


def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size debe estar entre 0 y 1.")
    df = df.dropna().sort_index()
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split]
    test = df.iloc[split:]
    return train, test


def prepare_btc_features(
    df_1m: pd.DataFrame,
    df_5m: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline compacta: normaliza columnas, genera features 1m + 5m y divide train/test.
    """
    features = build_multitimeframe_features(df_1m=df_1m, df_5m=df_5m)
    return temporal_train_test_split(features, test_size=test_size)

