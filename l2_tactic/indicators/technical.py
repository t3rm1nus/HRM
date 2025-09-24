# l2_tactic/indicators/technical.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Tipado cómodo
MarketDF = pd.DataFrame


@dataclass
class IndicatorWindows:
    rsi: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bbands: int = 20
    bbands_nstd: float = 2.0
    atr: int = 14
    vwap: int = 20           # VWAP rolling de N barras (si no hay sesión definida)
    vol_lookback: int = 30   # p.ej. para realized vol
    roc: int = 10            # momentum (rate of change)
    vol_ratio_lb: int = 20   # media de volumen para volume_ratio


def _validate_ohlcv(df: MarketDF) -> MarketDF:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("El DataFrame debe tener DatetimeIndex")
    # Normalizamos nombres a minúscula por seguridad
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


# --- Indicadores base (pandas puro) ---

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / (loss.replace(0.0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.clip(0, 100)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, window: int = 20, nstd: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    # %B: dónde está el precio relativo a las bandas
    bbp = (close - lower) / (upper - lower)
    return ma, upper, lower, bbp


def atr(df: MarketDF, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Wilder style (EMA con alpha = 1/window)
    return tr.ewm(alpha=1/window, adjust=False).mean()


def vwap(df: MarketDF, window: int = 20) -> pd.Series:
    # Rolling VWAP usando typical price
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    pv_sum = pv.rolling(window).sum()
    vol_sum = df["volume"].rolling(window).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap_roll = pv_sum / vol_sum
    return vwap_roll


def realized_vol(close: pd.Series, window: int = 30, annualize: bool = False, bars_per_year: int = 365*24*60) -> pd.Series:
    # Si tus barras son 1m, bars_per_year puede ser 525600; ajusta si es 1h/1d
    ret = np.log(close).diff()
    rv = ret.rolling(window).std(ddof=0)
    if annualize:
        rv = rv * np.sqrt(bars_per_year)
    return rv


def roc(close: pd.Series, window: int = 10) -> pd.Series:
    # Rate of change (momentum)
    return close.pct_change(periods=window)


# --- Motor de indicadores ---

class TechnicalIndicators:
    @staticmethod
    def compute_all(df: MarketDF, w: IndicatorWindows = IndicatorWindows()) -> MarketDF:
        """
        Devuelve un DataFrame con columnas de indicadores añadidas.
        Requiere columnas: open, high, low, close, volume y DatetimeIndex.
        """
        df = _validate_ohlcv(df).copy()
        close = df["close"]

        # RSI
        df["ti_rsi"] = rsi(close, w.rsi)

        # MACD
        macd_line, signal_line, hist = macd(close, w.macd_fast, w.macd_slow, w.macd_signal)
        df["ti_macd"] = macd_line
        df["ti_macd_signal"] = signal_line
        df["ti_macd_hist"] = hist

        # Bollinger
        ma, upper, lower, bbp = bollinger_bands(close, w.bbands, w.bbands_nstd)
        df["ti_bb_ma"] = ma
        df["ti_bb_upper"] = upper
        df["ti_bb_lower"] = lower
        df["ti_bb_percent"] = bbp  # %B

        # ATR
        df["ti_atr"] = atr(df, w.atr)

        # VWAP (rolling)
        df["ti_vwap"] = vwap(df, w.vwap)
        df["ti_vwap_dev"] = (close / df["ti_vwap"]) - 1.0

        # Realized vol + ROC
        df["ti_realized_vol"] = realized_vol(close, w.vol_lookback, annualize=False)
        df["ti_roc"] = roc(close, w.roc)

        # Features “core” que usa L2
        # - volatility: usamos realized_vol (o ATR normalizado por precio)
        # - volume_ratio: volumen actual / media rolling
        # - price_momentum: ROC
        vol_mean = df["volume"].rolling(w.vol_ratio_lb).mean()
        df["ti_volume_ratio"] = df["volume"] / vol_mean

        # Volatilidad alternativa (ATR normalizado)
        df["ti_atr_norm"] = df["ti_atr"] / close.replace(0.0, np.nan)

        return df

    @staticmethod
    def features_from_df(df: MarketDF, w: IndicatorWindows = IndicatorWindows()) -> Dict[str, float]:
        """
        Devuelve un dict con las 'MarketFeatures' mínimas que consume tu L2:
        - volatility
        - volume_ratio
        - price_momentum
        + algunos extras útiles por si quieres enriquecer el composer.
        """
        dfi = TechnicalIndicators.compute_all(df, w=w)
        last = dfi.iloc[-1]

        # Volatilidad: prioriza realized_vol; si NaN, usa ATR normalizado
        vol = last.get("ti_realized_vol", np.nan)
        if pd.isna(vol):
            vol = last.get("ti_atr_norm", np.nan)

        from .utils import safe_float
        feats = {
            "volatility": safe_float(vol) if pd.notna(vol) else np.nan,
            "volume_ratio": safe_float(last.get("ti_volume_ratio", np.nan)),
            "price_momentum": safe_float(last.get("ti_roc", np.nan)),
            # Extras opcionales:
            "rsi": safe_float(last.get("ti_rsi", np.nan)),
            "macd": safe_float(last.get("ti_macd", np.nan)),
            "macd_signal": safe_float(last.get("ti_macd_signal", np.nan)),
            "macd_hist": safe_float(last.get("ti_macd_hist", np.nan)),
            "bb_percent": safe_float(last.get("ti_bb_percent", np.nan)),
            "vwap_deviation": safe_float(last.get("ti_vwap_dev", np.nan)),
        }
        return feats

    @staticmethod
    def compute_for_universe(
        market: Dict[str, MarketDF],
        w: IndicatorWindows = IndicatorWindows(),
        as_features: bool = True
    ) -> Dict[str, MarketDF | Dict[str, float]]:
        """
        Calcula indicadores para múltiples símbolos:
        - as_features=True: devuelve dict[symbol -> dict de features]
        - as_features=False: devuelve dict[symbol -> DataFrame con columnas ti_*]
        """
        out: Dict[str, MarketDF | Dict[str, float]] = {}
        for symbol, df in market.items():
            if as_features:
                out[symbol] = TechnicalIndicators.features_from_df(df, w=w)
            else:
                out[symbol] = TechnicalIndicators.compute_all(df, w=w)
        return out
