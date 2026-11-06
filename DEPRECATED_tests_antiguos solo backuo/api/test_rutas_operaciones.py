import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.rutas.rutas_operaciones import router_operaciones
from api.servicios.servicio_operaciones import VARIABLE_ENTORNO_RUTA


@pytest.fixture
def cliente() -> TestClient:
    app = FastAPI()
    app.include_router(router_operaciones)
    return TestClient(app)


@pytest.fixture
def ruta_operaciones(tmp_path, monkeypatch):
    ruta = tmp_path / "paper_trades.json"
    monkeypatch.setenv(VARIABLE_ENTORNO_RUTA, str(ruta))
    return ruta


def crear_trade(order_id: str, simbolo: str = "BTCUSDT", lado: str = "buy") -> dict:
    return {
        "timestamp": "2026-11-02T10:00:00",
        "symbol": simbolo,
        "side": lado,
        "quantity": 0.01,
        "filled_price": 50000.0,
        "commission": 0.5,
        "status": "filled",
        "order_id": order_id,
    }


def test_sin_archivo_responde_200_con_lista_vacia(cliente, ruta_operaciones):
    respuesta = cliente.get("/operaciones")

    assert respuesta.status_code == 200
    assert respuesta.json()["ordenes"] == []


def test_devuelve_ordenes_con_el_contrato_de_lote(cliente, ruta_operaciones):
    ruta_operaciones.write_text(json.dumps({"trades": [crear_trade("a")]}), encoding="utf-8")

    cuerpo = cliente.get("/operaciones").json()

    assert cuerpo["total"] == 1
    assert cuerpo["ordenes"][0]["order_id"] == "a"
    assert cuerpo["ordenes"][0]["metadata"]["ciclo"] is None


def test_json_corrupto_responde_503(cliente, ruta_operaciones):
    ruta_operaciones.write_text("{ roto", encoding="utf-8")

    assert cliente.get("/operaciones").status_code == 503


@pytest.mark.parametrize("consulta", ["tamano=201", "tamano=0", "pagina=0", "lado=hold"])
def test_parametros_fuera_de_rango_responden_422(cliente, ruta_operaciones, consulta):
    assert cliente.get(f"/operaciones?{consulta}").status_code == 422


def test_aplica_filtros_de_la_consulta(cliente, ruta_operaciones):
    ruta_operaciones.write_text(json.dumps({"trades": [
        crear_trade("1", "BTCUSDT", "buy"),
        crear_trade("2", "ETHUSDT", "buy"),
        crear_trade("3", "ETHUSDT", "sell"),
    ]}), encoding="utf-8")

    cuerpo = cliente.get("/operaciones?simbolo=ethusdt&lado=sell").json()

    assert [orden["order_id"] for orden in cuerpo["ordenes"]] == ["3"]
