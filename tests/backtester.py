import os
import sys
import ast
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from core.hrm import ciclo_historico  # tu ciclo HRM adaptado a backtest
from core.logger import logger  # logging estructurado

def _resolver_path_csv(path: str) -> str:
    # Si el path no existe, intentar en carpeta data/
    if os.path.isfile(path):
        return path
    data_path = os.path.join("data", path)
    if os.path.isfile(data_path):
        return data_path
    # fallback absoluto relativo a repo raíz
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / path
    if candidate.is_file():
        return str(candidate)
    candidate = repo_root / "data" / path
    return str(candidate)

def cargar_csv(path: str):
    fichero = _resolver_path_csv(path)
    # Intentar parsear timestamp o ts
    try:
        df = pd.read_csv(fichero, parse_dates=["timestamp"], index_col="timestamp")
    except Exception:
        try:
            df = pd.read_csv(fichero, parse_dates=["ts"], index_col="ts")
            df.index.name = "timestamp"
        except Exception:
            df = pd.read_csv(fichero)
            # Si existe 'ts', convertir a datetime y usar como índice
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                df = df.set_index("ts")
                df.index.name = "timestamp"

    # Si no existen columnas BTC_close/ETH_close, derivarlas desde 'mercado'
    if ("BTC_close" not in df.columns or "ETH_close" not in df.columns) and "mercado" in df.columns:
        def extraer_precios(x):
            try:
                m = ast.literal_eval(x) if isinstance(x, str) else (x or {})
                return pd.Series({
                    "BTC_close": float(m.get("BTC", float("nan"))),
                    "ETH_close": float(m.get("ETH", float("nan")))
                })
            except Exception:
                return pd.Series({"BTC_close": float("nan"), "ETH_close": float("nan")})

        precios = df["mercado"].apply(extraer_precios)
        df = pd.concat([df, precios], axis=1)

    # Filtrar filas sin precios válidos
    if "BTC_close" in df.columns and "ETH_close" in df.columns:
        df = df.dropna(subset=["BTC_close", "ETH_close"])

    return df

def run_backtest(df):
    estado_global = {
        "portfolio": {"BTC": 0, "ETH": 0, "USDT": 1000},
        "estrategia": None,
        "universo": ["BTC", "ETH", "USDT"]
    }

    resultados = []

    for timestamp, row in df.iterrows():
        if "BTC_close" in row and "ETH_close" in row:
            datos_mercado = {
                "BTC": float(row["BTC_close"]),
                "ETH": float(row["ETH_close"]),
                "USDT": 1.0
            }
        elif "mercado" in row:
            try:
                m = ast.literal_eval(row["mercado"]) if isinstance(row["mercado"], str) else (row["mercado"] or {})
            except Exception:
                m = {}
            datos_mercado = {
                "BTC": float(m.get("BTC", float("nan"))),
                "ETH": float(m.get("ETH", float("nan"))),
                "USDT": float(m.get("USDT", 1.0))
            }
        else:
            # No hay datos suficientes, saltar
            continue

        estado_global = ciclo_historico(datos_mercado, estado_global)

        # Guardar snapshot
        resultados.append({**estado_global, "timestamp": timestamp})
        logger.info(f"Ciclo {timestamp} completado. Estado: {estado_global}")

    return pd.DataFrame(resultados)

if __name__ == "__main__":
    df = cargar_csv("historico.csv")
    resultados = run_backtest(df)
    resultados.to_csv("resultados_backtest.csv")
    print("Backtest finalizado. Resultados guardados en resultados_backtest.csv")
