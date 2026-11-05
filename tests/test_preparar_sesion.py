import json

import pytest

from api.esquemas.modo_sesion import ModoSesion
from api.servicios import servicio_modo_sesion
from api.servicios.servicio_modo_sesion import VARIABLE_ENTORNO_RUTA, guardar_modo_configurado
from storage.perfil_datos import (
    PERFIL_LIMPIO,
    PERFIL_PERSISTENTE,
    obtener_directorio_operaciones,
    obtener_ruta_cartera,
)
from system.preparar_sesion import preparar_sesion


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE_ENTORNO_RUTA, str(tmp_path / "persistent_state" / "modo_sesion.json"))
    monkeypatch.setattr(servicio_modo_sesion, "registro_modo_activo", servicio_modo_sesion.RegistroModoActivo())
    (tmp_path / "persistent_state").mkdir()
    (tmp_path / "data" / "paper_trades").mkdir(parents=True)
    (tmp_path / "persistent_state" / "portfolio_state_live.json").write_text(
        json.dumps({"mode": "simulated", "portfolio": {"USDT": {"free": 123.0}}}),
        encoding="utf-8",
    )
    (tmp_path / "data" / "paper_trades" / "paper_trades.json").write_text(
        json.dumps({"trades": [{"order_id": "legacy-1"}]}), encoding="utf-8"
    )
    return tmp_path


def test_persistente_migra_legacy_y_conserva_historial(raiz):
    guardar_modo_configurado(ModoSesion.PERSISTENTE)

    assert preparar_sesion(es_modo_live=False, raiz=raiz) is ModoSesion.PERSISTENTE

    ruta_cartera = obtener_ruta_cartera(PERFIL_PERSISTENTE, raiz)
    ruta_operaciones = obtener_directorio_operaciones(PERFIL_PERSISTENTE, raiz) / "paper_trades.json"
    assert json.loads(ruta_cartera.read_text(encoding="utf-8"))["portfolio"]["USDT"]["free"] == 123.0
    assert json.loads(ruta_operaciones.read_text(encoding="utf-8"))["trades"][0]["order_id"] == "legacy-1"
    # La migración copia, no destruye los archivos originales.
    assert (raiz / "persistent_state" / "portfolio_state_live.json").exists()
    assert (raiz / "data" / "paper_trades" / "paper_trades.json").exists()


def test_limpia_usa_almacenamiento_independiente_y_no_borra_persistente(raiz):
    guardar_modo_configurado(ModoSesion.PERSISTENTE)
    preparar_sesion(es_modo_live=False, raiz=raiz)
    cartera_persistente = obtener_ruta_cartera(PERFIL_PERSISTENTE, raiz)
    operaciones_persistentes = obtener_directorio_operaciones(PERFIL_PERSISTENTE, raiz) / "paper_trades.json"

    guardar_modo_configurado(ModoSesion.LIMPIA)
    assert preparar_sesion(es_modo_live=False, raiz=raiz) is ModoSesion.LIMPIA

    ruta_cartera_limpia = obtener_ruta_cartera(PERFIL_LIMPIO, raiz)
    ruta_operaciones_limpias = obtener_directorio_operaciones(PERFIL_LIMPIO, raiz) / "paper_trades.json"
    assert cartera_persistente.exists()
    assert operaciones_persistentes.exists()
    assert not ruta_cartera_limpia.exists()
    assert not ruta_operaciones_limpias.exists()
    assert (raiz / "persistent_state" / "modo_sesion.json").exists()


def test_limpia_borra_solo_su_perfil_en_siguientes_arranques(raiz):
    guardar_modo_configurado(ModoSesion.PERSISTENTE)
    preparar_sesion(es_modo_live=False, raiz=raiz)
    operaciones_persistentes = obtener_directorio_operaciones(PERFIL_PERSISTENTE, raiz) / "paper_trades.json"

    ruta_limpia = obtener_directorio_operaciones(PERFIL_LIMPIO, raiz)
    ruta_limpia.mkdir(parents=True)
    (ruta_limpia / "paper_trades.json").write_text('{"trades": [1]}', encoding="utf-8")

    guardar_modo_configurado(ModoSesion.LIMPIA)
    preparar_sesion(es_modo_live=False, raiz=raiz)

    assert operaciones_persistentes.exists()
    assert not (ruta_limpia / "paper_trades.json").exists()


def test_limpia_nunca_borra_ni_migra_en_live(raiz):
    guardar_modo_configurado(ModoSesion.LIMPIA)
    assert preparar_sesion(es_modo_live=True, raiz=raiz) is ModoSesion.PERSISTENTE
    assert (raiz / "persistent_state" / "portfolio_state_live.json").exists()
    assert (raiz / "data" / "paper_trades" / "paper_trades.json").exists()


def test_persistente_no_sobrescribe_datos_ya_migrados(raiz):
    destino = obtener_ruta_cartera(PERFIL_PERSISTENTE, raiz)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"mode": "simulated", "portfolio": {"USDT": {"free": 999.0}}}), encoding="utf-8")

    guardar_modo_configurado(ModoSesion.PERSISTENTE)
    preparar_sesion(es_modo_live=False, raiz=raiz)

    assert json.loads(destino.read_text(encoding="utf-8"))["portfolio"]["USDT"]["free"] == 999.0


def test_perfil_activo_se_actualiza_en_el_proceso(raiz):
    guardar_modo_configurado(ModoSesion.LIMPIA)
    preparar_sesion(es_modo_live=False, raiz=raiz)
    from storage.perfil_datos import obtener_perfil_datos

    assert obtener_perfil_datos() == PERFIL_LIMPIO


def test_reset_persistente_pendiente_se_ejecuta_al_siguiente_arranque(raiz, monkeypatch):
    from storage.perfil_datos import solicitar_reset_persistente, obtener_ruta_marcador_reset

    guardar_modo_configurado(ModoSesion.PERSISTENTE)
    preparar_sesion(es_modo_live=False, raiz=raiz)
    cartera_persistente = obtener_ruta_cartera(PERFIL_PERSISTENTE, raiz)
    operaciones_persistentes = obtener_directorio_operaciones(PERFIL_PERSISTENTE, raiz) / "paper_trades.json"
    cartera_limpia = obtener_ruta_cartera(PERFIL_LIMPIO, raiz)
    cartera_limpia.parent.mkdir(parents=True, exist_ok=True)
    cartera_limpia.write_text('{"perfil": "limpia"}', encoding="utf-8")

    monkeypatch.setenv("HRM_RUTA_RESET_PAPER", str(raiz / "persistent_state" / "reset_paper_pendiente.json"))
    solicitar_reset_persistente()
    assert cartera_persistente.exists()
    assert operaciones_persistentes.exists()

    preparar_sesion(es_modo_live=False, raiz=raiz)

    assert not cartera_persistente.exists()
    assert not operaciones_persistentes.exists()
    assert cartera_limpia.exists()
    assert not obtener_ruta_marcador_reset(raiz).exists()
    assert servicio_modo_sesion.leer_modo_configurado() is ModoSesion.PERSISTENTE

    # El siguiente arranque no debe volver a importar los ficheros legacy que
    # se conservaron como copia de seguridad durante la migración.
    preparar_sesion(es_modo_live=False, raiz=raiz)
    assert not cartera_persistente.exists()
    assert not operaciones_persistentes.exists()


def test_reset_persistente_no_se_consumira_si_se_arranca_en_limpia(raiz, monkeypatch):
    from storage.perfil_datos import solicitar_reset_persistente, obtener_ruta_marcador_reset

    monkeypatch.setenv("HRM_RUTA_RESET_PAPER", str(raiz / "persistent_state" / "reset_paper_pendiente.json"))
    solicitar_reset_persistente()
    guardar_modo_configurado(ModoSesion.LIMPIA)

    preparar_sesion(es_modo_live=False, raiz=raiz)

    assert obtener_ruta_marcador_reset(raiz).exists()
    assert servicio_modo_sesion.leer_modo_configurado() is ModoSesion.LIMPIA


def test_live_no_consume_reset_paper_pendiente(raiz, monkeypatch):
    from storage.perfil_datos import solicitar_reset_persistente, obtener_ruta_marcador_reset

    monkeypatch.setenv("HRM_RUTA_RESET_PAPER", str(raiz / "persistent_state" / "reset_paper_pendiente.json"))
    solicitar_reset_persistente()
    guardar_modo_configurado(ModoSesion.PERSISTENTE)

    preparar_sesion(es_modo_live=True, raiz=raiz)

    assert obtener_ruta_marcador_reset(raiz).exists()
    assert (raiz / "persistent_state" / "portfolio_state_live.json").exists()
