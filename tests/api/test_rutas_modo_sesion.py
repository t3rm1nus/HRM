import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.rutas.rutas_modo_sesion import router_modo_sesion
from api.servicios import servicio_modo_sesion
from api.servicios.servicio_estado_vivo import servicio_estado_vivo
from storage.perfil_datos import obtener_ruta_marcador_reset
from api.servicios.servicio_modo_sesion import VARIABLE_ENTORNO_RUTA


@pytest.fixture
def cliente() -> TestClient:
    app = FastAPI()
    app.include_router(router_modo_sesion)
    return TestClient(app)


@pytest.fixture(autouse=True)
def aislar_estado(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE_ENTORNO_RUTA, str(tmp_path / "modo_sesion.json"))
    monkeypatch.setenv("HRM_RUTA_RESET_PAPER", str(tmp_path / "reset_paper_pendiente.json"))
    monkeypatch.setattr(servicio_modo_sesion, "registro_modo_activo", servicio_modo_sesion.RegistroModoActivo())
    servicio_estado_vivo.registrar_modo("unknown")


def test_sin_archivo_el_modo_es_persistente(cliente):
    cuerpo = cliente.get("/sesion/modo").json()
    assert cuerpo == {
        "modo_activo": None,
        "modo_configurado": "persistente",
        "reinicio_pendiente": False,
        "modo_operativo": "unknown",
        "reinicio_datos_pendiente": False,
    }


def test_cambiar_modo_lo_guarda_y_lo_devuelve(cliente):
    respuesta = cliente.post("/sesion/modo", json={"modo": "limpia"})
    assert respuesta.status_code == 200
    assert cliente.get("/sesion/modo").json()["modo_configurado"] == "limpia"


def test_modo_invalido_responde_422(cliente):
    assert cliente.post("/sesion/modo", json={"modo": "otro"}).status_code == 422


def test_reinicio_pendiente_si_difiere_del_modo_activo(cliente):
    from api.esquemas.modo_sesion import ModoSesion

    servicio_modo_sesion.registro_modo_activo.fijar(ModoSesion.PERSISTENTE)
    cuerpo = cliente.post("/sesion/modo", json={"modo": "limpia"}).json()
    assert cuerpo["reinicio_pendiente"] is True


def test_archivo_corrupto_vuelve_al_modo_por_defecto(cliente, tmp_path):
    (tmp_path / "modo_sesion.json").write_text("{no es json", encoding="utf-8")
    assert cliente.get("/sesion/modo").json()["modo_configurado"] == "persistente"

def test_reiniciar_datos_programa_borrado_diferido_solo_en_paper_persistente(cliente, tmp_path):
    from api.esquemas.modo_sesion import ModoSesion

    servicio_estado_vivo.registrar_modo("simulated")
    servicio_modo_sesion.registro_modo_activo.fijar(ModoSesion.PERSISTENTE)
    servicio_modo_sesion.guardar_modo_configurado(ModoSesion.PERSISTENTE)

    respuesta = cliente.post("/sesion/reiniciar-datos", json={"confirmacion": "CONFIRMAR"})

    assert respuesta.status_code == 202
    assert respuesta.json()["estado"] == "pendiente"
    assert obtener_ruta_marcador_reset().exists()


def test_reiniciar_datos_rechaza_live(cliente):
    from api.esquemas.modo_sesion import ModoSesion

    servicio_estado_vivo.registrar_modo("live")
    servicio_modo_sesion.registro_modo_activo.fijar(ModoSesion.PERSISTENTE)
    servicio_modo_sesion.guardar_modo_configurado(ModoSesion.PERSISTENTE)

    assert cliente.post("/sesion/reiniciar-datos", json={"confirmacion": "CONFIRMAR"}).status_code == 403


def test_reiniciar_datos_rechaza_modo_limpia(cliente):
    from api.esquemas.modo_sesion import ModoSesion

    servicio_estado_vivo.registrar_modo("simulated")
    servicio_modo_sesion.registro_modo_activo.fijar(ModoSesion.LIMPIA)
    servicio_modo_sesion.guardar_modo_configurado(ModoSesion.LIMPIA)

    assert cliente.post("/sesion/reiniciar-datos", json={"confirmacion": "CONFIRMAR"}).status_code == 409


def test_reiniciar_datos_exige_confirmacion_exacta(cliente):
    from api.esquemas.modo_sesion import ModoSesion

    servicio_estado_vivo.registrar_modo("simulated")
    servicio_modo_sesion.registro_modo_activo.fijar(ModoSesion.PERSISTENTE)
    servicio_modo_sesion.guardar_modo_configurado(ModoSesion.PERSISTENTE)

    assert cliente.post("/sesion/reiniciar-datos", json={"confirmacion": "confirmar"}).status_code == 422


def test_reiniciar_datos_rechaza_solicitud_duplicada(cliente):
    from api.esquemas.modo_sesion import ModoSesion

    servicio_estado_vivo.registrar_modo("simulated")
    servicio_modo_sesion.registro_modo_activo.fijar(ModoSesion.PERSISTENTE)
    servicio_modo_sesion.guardar_modo_configurado(ModoSesion.PERSISTENTE)

    primera = cliente.post("/sesion/reiniciar-datos", json={"confirmacion": "CONFIRMAR"})
    segunda = cliente.post("/sesion/reiniciar-datos", json={"confirmacion": "CONFIRMAR"})

    assert primera.status_code == 202
    assert segunda.status_code == 409
