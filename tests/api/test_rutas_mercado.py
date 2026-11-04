import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.rutas import rutas_mercado
from api.servicios.servicio_mercado import ServicioMercado
from api.servicios.servicio_websockets import ServicioWebSockets
from tests.api.test_servicio_mercado import GestorFalso, crear_df


@pytest.fixture
def servicio(monkeypatch):
    nuevo = ServicioMercado(ServicioWebSockets())
    monkeypatch.setattr(rutas_mercado, "servicio_mercado", nuevo)
    return nuevo


@pytest.fixture
def cliente(servicio):
    app = FastAPI()
    app.include_router(rutas_mercado.router_mercado)
    return TestClient(app)


def test_estado_antes_de_inicializar(cliente):
    cuerpo = cliente.get("/mercado/estado").json()
    assert cuerpo["disponible"] is False and cuerpo["modo"] == "paper"


def test_velas_antes_de_inicializar_da_503(cliente):
    assert cliente.get("/mercado/velas/BTCUSDT").status_code == 503


def test_velas_correctas(cliente, servicio):
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({"BTCUSDT": crear_df(5)})
    cuerpo = cliente.get("/mercado/velas/BTCUSDT?limite=3").json()
    assert cuerpo["cantidad"] == 3 and cuerpo["frescura"] == "actual"


def test_simbolo_desconocido_da_404(cliente, servicio):
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    assert cliente.get("/mercado/velas/XRPUSDT").status_code == 404


def test_limite_fuera_de_rango_da_422(cliente):
    assert cliente.get("/mercado/velas/BTCUSDT?limite=0").status_code == 422


def test_simbolos(cliente, servicio):
    servicio.registrar_gestor(GestorFalso())
    servicio.registrar_instantanea({"ETHUSDT": crear_df(), "BTCUSDT": crear_df()})
    assert cliente.get("/mercado/simbolos").json() == {"simbolos": ["BTCUSDT", "ETHUSDT"]}


def test_datos_obsoletos_se_declaran(cliente, servicio):
    servicio.registrar_gestor(GestorFalso(edad=900))
    servicio.registrar_instantanea({"BTCUSDT": crear_df()})
    assert cliente.get("/mercado/estado").json()["frescura"] == "obsoleto"
