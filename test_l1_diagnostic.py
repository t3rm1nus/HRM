
#!/usr/bin/env python3
"""Diagnóstico manual para investigar la generación de señales del modelo L1."""

from pathlib import Path
import sys


def obtener_ruta_datos_predeterminada() -> Path:
    """Devuelve la ruta del conjunto de datos usado por el diagnóstico."""
    raiz_repositorio = Path(__file__).resolve().parent
    return raiz_repositorio / "backtesting" / "data" / "btcusdt_1m.csv"


def ejecutar_diagnostico(ruta_datos=None) -> int:
    """Ejecuta el diagnóstico L1 y devuelve un código de salida."""
    ruta = (
        Path(ruta_datos)
        if ruta_datos is not None
        else obtener_ruta_datos_predeterminada()
    )

    if not ruta.is_file():
        print(f"No se encontró el archivo: {ruta}")
        return 1

    return _ejecutar_con_datos(ruta)


def _ejecutar_con_datos(ruta_datos: Path) -> int:
    """Carga dependencias y ejecuta las comprobaciones sobre el CSV."""
    import pandas as pd

    from l1_operational.models import L1Model, MomentumModel
    from core.technical_indicators import calculate_technical_indicators

    print("=" * 80)
    print("DIAGNÓSTICO L1 MODEL")
    print("=" * 80)

    print(f"\nCargando datos desde: {ruta_datos}")
    df = pd.read_csv(ruta_datos)
    print(f"Datos cargados: {len(df)} filas")
    print(f"Columnas: {list(df.columns)}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    df_test = df.head(100).copy()
    print(f"\nUsando primeras {len(df_test)} filas para prueba")

    if not df_test.empty:
        print(f"Index: {df_test.index[0]} a {df_test.index[-1]}")

    df_test.columns = [col.lower() for col in df_test.columns]
    print(f"Columnas normalizadas: {list(df_test.columns)}")

    columnas_requeridas = ["open", "high", "low", "close", "volume"]
    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in df_test.columns
    ]

    if columnas_faltantes:
        print(f"Columnas faltantes: {columnas_faltantes}")
    else:
        print("Todas las columnas requeridas están presentes")

    _diagnosticar_indicadores(calculate_technical_indicators, df_test)
    _diagnosticar_modelo_momentum(MomentumModel, df_test)
    _diagnosticar_modelo_l1(L1Model, df_test)

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO COMPLETADO")
    print("=" * 80)

    return 0


def _diagnosticar_indicadores(calcular_indicadores, datos) -> None:
    """Muestra el resultado del cálculo de indicadores técnicos."""
    print("\n" + "=" * 80)
    print("PROBANDO: calculate_technical_indicators")
    print("=" * 80)

    try:
        indicadores = calcular_indicadores({"BTCUSDT": datos})
        print("Indicadores calculados")
        print(f"Keys: {list(indicadores.keys())}")

        if "BTCUSDT" not in indicadores:
            print("'BTCUSDT' no está en el resultado de indicadores")
            return

        datos_indicadores = indicadores["BTCUSDT"]

        print(f"Indicadores shape: {datos_indicadores.shape}")
        print(f"Indicadores columns: {list(datos_indicadores.columns)}")
        print("\nPrimeras 3 filas:")
        print(datos_indicadores.head(3).to_string())
        print("\nÚltimas 3 filas:")
        print(datos_indicadores.tail(3).to_string())

    except Exception as error:
        print(f"Error calculando indicadores: {error}")
        import traceback
        traceback.print_exc()


def _diagnosticar_modelo_momentum(clase_modelo, datos) -> None:
    """Muestra la configuración y las señales del modelo Momentum."""
    print("\n" + "=" * 80)
    print("PROBANDO: MomentumModel")
    print("=" * 80)

    modelo = clase_modelo()

    print(f"short_period: {modelo.short_period}")
    print(f"medium_period: {modelo.medium_period}")
    print(f"momentum_threshold: {modelo.momentum_threshold}")

    print("\nValidando datos:")
    print(f"df is None: {datos is None}")
    print(f"df.empty: {datos.empty}")
    print(f"len(df): {len(datos)}")

    columnas_requeridas = ["open", "high", "low", "close", "volume"]
    estan_presentes = all(
        columna in datos.columns
        for columna in columnas_requeridas
    )
    print(f"required cols present: {estan_presentes}")

    try:
        señales = modelo.generate_signals({"BTCUSDT": datos})

        print(
            "MomentumModel.generate_signals() "
            f"retornó {len(señales)} señales"
        )

        for indice, señal in enumerate(señales[:5]):
            print(
                f"Signal {indice}: {señal.direction}, "
                f"conf={señal.confidence:.3f}, "
                f"type={señal.signal_type.value}"
            )

    except Exception as error:
        print(f"Error en MomentumModel: {error}")
        import traceback
        traceback.print_exc()


def _diagnosticar_modelo_l1(clase_modelo, datos) -> None:
    """Muestra las métricas y el reparto de señales del modelo L1."""
    print("\n" + "=" * 80)
    print("PROBANDO: L1Model (completo)")
    print("=" * 80)

    try:
        modelo = clase_modelo()
        resultado = modelo.predict({"BTCUSDT": datos})

        print("L1Model.predict() ejecutado")
        print(f"Total signals: {len(resultado['signals'])}")
        print(f"Metrics: {resultado['metrics']}")

        señales = resultado["signals"]

        compras = sum(
            1 for señal in señales
            if señal.direction == "buy"
        )
        ventas = sum(
            1 for señal in señales
            if señal.direction == "sell"
        )
        mantener = sum(
            1 for señal in señales
            if señal.direction == "hold"
        )

        print(f"BUY={compras}, SELL={ventas}, HOLD={mantener}")

    except Exception as error:
        print(f"Error en L1Model: {error}")
        import traceback
        traceback.print_exc()


def main() -> int:
    """Punto de entrada del diagnóstico ejecutado manualmente."""
    return ejecutar_diagnostico()


if __name__ == "__main__":
    sys.exit(main())
