# l2_tactical/technical/patterns.py
from __future__ import annotations
import pandas as pd
import numpy as np

def detect_doji(df: pd.DataFrame, body_pct: float = 0.1) -> pd.Series:
    body = (df["close"] - df["open"]).abs()
    rng  = (df["high"]  - df["low"]).abs()
    return body / rng < body_pct

def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    bull = (df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1))
    engulf = (df["close"] > df["open"].shift(1)) & (df["open"] < df["close"].shift(1))
    return bull & engulf

def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    bear = (df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1))
    engulf = (df["close"] < df["open"].shift(1)) & (df["open"] > df["close"].shift(1))
    return bear & engulf

def detect_head_and_shoulders(df: pd.DataFrame, window: int = 5) -> pd.Series:
    # Simplificado: 3 picos (bajista) o 3 valles (alcista) en `window`
    highs = df["high"].rolling(window).max()
    lows  = df["low"].rolling(window).min()
    # Bajista: pico > valle > pico > valle > pico
    left  = (df["high"] == highs).shift(2)
    center= (df["high"] == highs).shift(1)
    right = (df["high"] == highs)
    return left & center & right

def detect_all(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["doji"]               = detect_doji(df)
    out["bull_engulfing"]     = detect_bullish_engulfing(df)
    out["bear_engulfing"]     = detect_bearish_engulfing(df)
    out["head_shoulders"]     = detect_head_and_shoulders(df)
    return out
