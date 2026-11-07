import json

import pytest

import storage.perfil_datos as perfiles
import system.preparar_sesion as preparador
from api.esquemas.modo_sesion import ModoSesion


@pytest.fixture(autouse=True)
def restaurar_estado_global(monkeypatch):
    """Aísla el perfil y el modo global entre pruebas."""
    modo_activo_previo = preparador.servicio_modo_sesion.registro_modo_activo.obtener()
    monkeypatch.setattr(perfiles, "_PERFIL_ACTIVO", perfiles.PERFIL_PERSISTENTE)
    monkeypatch.setattr(
        preparador.servicio_modo_sesion.registro_modo_activo,
        "_modo",
        modo_activo_previo,
    )
    monkeypatch.setattr(
        preparador.servicio_modo_sesion,
        "leer_modo_configurado",
        lambda: ModoSesion.PERSISTENTE,
    )


@pytest.mark.unit
def test_modo_limpia_se_conserva_en_paper():
    assert preparador.decidir_modo_efectivo(ModoSesion.LIMPIA, False) is ModoSesion.LIMPIA


@pytest.mark.unit
def test_modo_limpia_se_convierte_a_persistente_en_live():
    assert preparador.decidir_modo_efectivo(ModoSesion.LIMPIA, True) is ModoSesion.PERSISTENTE


@pytest.mark.unit
def test_modo_persistente_se_conserva_en_live():
    assert preparador.decidir_modo_efectivo(ModoSesion.PERSISTENTE, True) is ModoSesion.PERSISTENTE


@pytest.mark.integration
def test_preparar_paper_persistente_conserva_cartera_e_historial(tmp_path):
    directorio = perfiles.obtener_directorio_paper(perfiles.PERFIL_PERSISTENTE, tmp_path)
    operaciones = perfiles.obtener_directorio_operaciones(perfiles.PERFIL_PERSISTENTE, tmp_path)
    operaciones.mkdir(parents=True)
    cartera = directorio / "portfolio_state.json"
    historial = operaciones / "trade_001.json"
    cartera.write_text('{"balance": 1250}', encoding="utf-8")
    historial.write_text('{"symbol": "BTC/USDT"}', encoding="utf-8")

    modo = preparador.preparar_sesion(False, tmp_path)

    assert modo is ModoSesion.PERSISTENTE
    assert perfiles.obtener_perfil_datos() == perfiles.PERFIL_PERSISTENTE
    assert json.loads(cartera.read_text(encoding="utf-8"))["balance"] == 1250
    assert json.loads(historial.read_text(encoding="utf-8"))["symbol"] == "BTC/USDT"


@pytest.mark.integration
def test_preparar_paper_limpia_no_borra_perfil_persistente(tmp_path, monkeypatch):
    persistente = perfiles.obtener_directorio_paper(perfiles.PERFIL_PERSISTENTE, tmp_path)
    limpia = perfiles.obtener_directorio_paper(perfiles.PERFIL_LIMPIO, tmp_path)
    persistente.mkdir(parents=True)
    limpia.mkdir(parents=True)
    archivo_persistente = persistente / "portfolio_state.json"
    archivo_limpio = limpia / "portfolio_state.json"
    archivo_persistente.write_text("cartera persistente", encoding="utf-8")
    archivo_limpio.write_text("cartera temporal", encoding="utf-8")
    monkeypatch.setattr(
        preparador.servicio_modo_sesion,
        "leer_modo_configurado",
        lambda: ModoSesion.LIMPIA,
    )

    modo = preparador.preparar_sesion(False, tmp_path)

    assert modo is ModoSesion.LIMPIA
    assert perfiles.obtener_perfil_datos() == perfiles.PERFIL_LIMPIO
    assert archivo_persistente.read_text(encoding="utf-8") == "cartera persistente"
    assert limpia.is_dir()
    assert list(limpia.iterdir()) == []


@pytest.mark.integration
def test_preparar_live_no_migra_ni_limpia_datos_paper(tmp_path, monkeypatch):
    persistente = perfiles.obtener_directorio_paper(perfiles.PERFIL_PERSISTENTE, tmp_path)
    limpia = perfiles.obtener_directorio_paper(perfiles.PERFIL_LIMPIO, tmp_path)
    persistente.mkdir(parents=True)
    limpia.mkdir(parents=True)
    archivo_persistente = persistente / "portfolio_state.json"
    archivo_limpio = limpia / "portfolio_state.json"
    archivo_persistente.write_text("persistente intacto", encoding="utf-8")
    archivo_limpio.write_text("limpia intacto", encoding="utf-8")
    monkeypatch.setattr(
        preparador.servicio_modo_sesion,
        "leer_modo_configurado",
        lambda: ModoSesion.LIMPIA,
    )
    monkeypatch.setattr(
        preparador,
        "migrar_datos_legacy_a_persistente",
        lambda raiz: pytest.fail("LIVE no debe ejecutar la migración PAPER"),
    )

    modo = preparador.preparar_sesion(True, tmp_path)

    assert modo is ModoSesion.PERSISTENTE
    assert perfiles.obtener_perfil_datos() == perfiles.PERFIL_PERSISTENTE
    assert archivo_persistente.read_text(encoding="utf-8") == "persistente intacto"
    assert archivo_limpio.read_text(encoding="utf-8") == "limpia intacto"
    assert not (tmp_path / "persistent_state" / "migracion_paper_perfiles_v1.json").exists()


@pytest.mark.regression
@pytest.mark.integration
def test_fallo_de_migracion_no_cambia_perfil_activo_ni_crea_marcador(tmp_path, monkeypatch):
    estado_legacy = tmp_path / "persistent_state" / "portfolio_state_live.json"
    estado_legacy.parent.mkdir(parents=True)
    estado_legacy.write_text(json.dumps({"mode": "paper", "balance": 250}), encoding="utf-8")
    perfiles.configurar_perfil_datos(perfiles.PERFIL_LIMPIO)

    def fallar_copia(*args, **kwargs):
        raise OSError("fallo de copia durante el arranque")

    monkeypatch.setattr(perfiles.shutil, "copy2", fallar_copia)

    with pytest.raises(OSError, match="fallo de copia durante el arranque"):
        preparador.preparar_sesion(False, tmp_path)

    marcador = tmp_path / "persistent_state" / "migracion_paper_perfiles_v1.json"
    destino = perfiles.obtener_directorio_paper(perfiles.PERFIL_PERSISTENTE, tmp_path)
    assert perfiles.obtener_perfil_datos() == perfiles.PERFIL_LIMPIO
    assert not marcador.exists()
    assert not (destino / "portfolio_state.json").exists()
    assert list(destino.glob("*.tmp")) == []


@pytest.mark.regression
@pytest.mark.integration
def test_reset_persistente_diferido_se_consume_tras_limpiar_solo_persistente(
    tmp_path, monkeypatch
):
    persistente = perfiles.obtener_directorio_paper(perfiles.PERFIL_PERSISTENTE, tmp_path)
    limpia = perfiles.obtener_directorio_paper(perfiles.PERFIL_LIMPIO, tmp_path)
    persistente.mkdir(parents=True)
    limpia.mkdir(parents=True)
    archivo_persistente = persistente / "portfolio_state.json"
    archivo_limpio = limpia / "portfolio_state.json"
    archivo_persistente.write_text("borrar", encoding="utf-8")
    archivo_limpio.write_text("conservar", encoding="utf-8")
    marcador = tmp_path / "persistent_state" / "reset_paper_persistente_pendiente.json"
    monkeypatch.setenv("HRM_RUTA_RESET_PAPER", str(marcador))
    perfiles.solicitar_reset_persistente(tmp_path)

    preparador.preparar_sesion(False, tmp_path)

    assert perfiles.obtener_perfil_datos() == perfiles.PERFIL_PERSISTENTE
    assert list(persistente.iterdir()) == []
    assert archivo_limpio.read_text(encoding="utf-8") == "conservar"
    assert not marcador.exists()
