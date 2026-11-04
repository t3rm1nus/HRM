import asyncio

import pandas as pd
import pytest

from api.servicios.servicio_mercado import (
    MercadoNoInicializado,
    ServicioMercado,
    SimboloDesconocido,
    calcular_frescura,
    dataframe_a_velas,
)
from api.servicios.servicio_websockets import ServicioWebSockets


class GestorFalso:
    symbols = ["BTCUSDT"]

    def __init__(self, edad=5.0, tiene_cache=True, validez=30):
        self._stats = {
            "cache_valid_seconds": validez,
            "cache_status": {"has_cache": tiene_cache, "cache_source": "realtime", "cache_age_seconds": edad},
        }

    def get_stats(self):
        return self._stats


class SocketFalso:
    def __init__(self):
        self.mensajes = []

    async def accept(self): ...
    async def close(self, code=1000): ...
    async def send_text(self, data): self.mensajes.append(data)


def crear_df(filas=3, con_timestamp=True):
    base = pd.Timestamp("2026-11-04T10:00:00Z")
    instantes = [base + pd.Timedelta(minutes=i) for i in range(filas)]
    df = pd.DataFrame({
        "open": [1.0 + i for i in range(filas)], "high": [2.0 + i for i in range(filas)],
        "low": [0.5 + i for i in range(filas)], "close": [1.5 + i for i in range(filas)],
        "volume": [10.0] * filas,
    })
    if con_timestamp:
        df["timestamp"] = [int(i.timestamp() * 1000) for i in instantes]
    else:
        df.index = pd.DatetimeIndex(instantes)
    return df


@pytest.fixture
def servicio():
    return ServicioMercado(ServicioWebSockets())


def test_sin_gestor_el_estado_es_no_disponible(servicio):
    estado = servicio.obtener_estado()
    assert estado.disponible is False and estado.frescura == "sin_datos"


def test_sin_gestor_pedir_velas_lanza_error(servicio):
    with pytest.raises(MercadoNoInicializado):
        servicio.obtener_velas("BTCUSDT")


def test_datos_vacios_dan_sin_datos(servicio):
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({})
    assert servicio.obtener_estado().frescura == "sin_datos"
    assert servicio.listar_simbolos() == []


def test_dataframe_con_timestamp_en_milisegundos():
    velas = dataframe_a_velas(crear_df(3), 10)
    assert len(velas) == 3
    assert velas[0].instante.startswith("2026-11-04T10:00:00")


def test_dataframe_con_indice_datetime_y_limite():
    velas = dataframe_a_velas(crear_df(5, con_timestamp=False), 2)
    assert len(velas) == 2 and velas[-1].cierre == 5.5


def test_filas_con_nan_se_descartan():
    df = crear_df(3)
    df.loc[1, "close"] = float("nan")
    assert len(dataframe_a_velas(df, 10)) == 2


def test_dataframe_sin_columnas_ohlcv_no_se_registra(servicio):
    servicio.registrar_instantanea({"BTCUSDT": pd.DataFrame({"x": [1]})})
    assert servicio.listar_simbolos() == []


def test_cache_actual(servicio):
    servicio.registrar_gestor(GestorFalso(edad=10))
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    assert servicio.obtener_estado().frescura == "actual"


def test_cache_caducada_se_marca_obsoleta(servicio):
    servicio.registrar_gestor(GestorFalso(edad=600))
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    assert servicio.obtener_estado().frescura == "obsoleto"
    assert servicio.obtener_velas("BTCUSDT", 2).frescura == "obsoleto"


def test_proveedor_caido_sin_cache_es_sin_datos(servicio):
    servicio.registrar_gestor(GestorFalso(tiene_cache=False))
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    assert servicio.obtener_estado().frescura == "sin_datos"


def test_error_leyendo_estadisticas_no_rompe(servicio):
    gestor = GestorFalso()
    gestor.get_stats = lambda: (_ for _ in ()).throw(RuntimeError("fallo"))
    servicio.registrar_gestor(gestor)
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    assert servicio.obtener_estado().frescura == "sin_datos"


def test_simbolo_desconocido(servicio):
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    with pytest.raises(SimboloDesconocido):
        servicio.obtener_velas("DOGEUSDT")


def test_simbolo_se_normaliza(servicio):
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    assert servicio.obtener_velas("btc/usdt").simbolo == "BTCUSDT"


def test_calcular_frescura_umbral():
    assert calcular_frescura(None, 30) == "sin_datos"
    assert calcular_frescura(60, 30) == "actual"
    assert calcular_frescura(61, 30) == "obsoleto"


def test_actualizacion_solo_envia_ultima_vela(servicio):
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({"BTCUSDT": crear_df(400)})
    actualizacion = servicio.construir_actualizacion()
    assert list(actualizacion.ultimas_velas) == ["BTCUSDT"]


def test_publicar_difunde_evento_mercado_y_tolera_fallos():
    eventos = ServicioWebSockets()
    servicio = ServicioMercado(eventos)
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    socket = SocketFalso()

    async def escenario():
        await eventos.conectar(socket)
        await servicio.publicar_actualizacion()
        servicio.registrar_instantanea({})
        await servicio.publicar_actualizacion()  # sin datos: no publica ni falla

    asyncio.run(escenario())
    assert len(socket.mensajes) == 1 and '"tipo": "mercado"' in socket.mensajes[0]


def test_notificar_datos_con_entrada_invalida_no_lanza():
    servicio = ServicioMercado(ServicioWebSockets())
    asyncio.run(servicio.notificar_datos(None))
    asyncio.run(servicio.notificar_datos({"BTCUSDT": "no es un dataframe"}))
    assert servicio.listar_simbolos() == []
