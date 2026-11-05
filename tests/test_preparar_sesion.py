import pytest

from api.esquemas.modo_sesion import ModoSesion
from api.servicios import servicio_modo_sesion
from api.servicios.servicio_modo_sesion import VARIABLE_ENTORNO_RUTA, guardar_modo_configurado
from system.preparar_sesion import preparar_sesion


@pytest.fixture
def raiz(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE_ENTORNO_RUTA, str(tmp_path / "persistent_state" / "modo_sesion.json"))
    monkeypatch.setattr(servicio_modo_sesion, "registro_modo_activo", servicio_modo_sesion.RegistroModoActivo())
    (tmp_path / "persistent_state").mkdir()
    (tmp_path / "data" / "paper_trades").mkdir(parents=True)
    (tmp_path / "persistent_state" / "portfolio_state_live.json").write_text("{}")
    (tmp_path / "data" / "paper_trades" / "paper_trades.json").write_text("{}")
    return tmp_path


def existen_los_datos(raiz) -> bool:
    return (
        (raiz / "persistent_state" / "portfolio_state_live.json").exists()
        and (raiz / "data" / "paper_trades" / "paper_trades.json").exists()
    )


def test_persistente_conserva_el_estado(raiz):
    guardar_modo_configurado(ModoSesion.PERSISTENTE)
    assert preparar_sesion(es_modo_live=False, raiz=raiz) is ModoSesion.PERSISTENTE
    assert existen_los_datos(raiz)


def test_limpia_borra_cartera_y_operaciones_pero_no_el_modo(raiz):
    guardar_modo_configurado(ModoSesion.LIMPIA)
    assert preparar_sesion(es_modo_live=False, raiz=raiz) is ModoSesion.LIMPIA
    assert not (raiz / "persistent_state" / "portfolio_state_live.json").exists()
    assert not (raiz / "data" / "paper_trades" / "paper_trades.json").exists()
    assert (raiz / "persistent_state" / "modo_sesion.json").exists()


def test_limpia_nunca_borra_en_live(raiz):
    guardar_modo_configurado(ModoSesion.LIMPIA)
    assert preparar_sesion(es_modo_live=True, raiz=raiz) is ModoSesion.PERSISTENTE
    assert existen_los_datos(raiz)


def test_registra_el_modo_activo(raiz):
    guardar_modo_configurado(ModoSesion.LIMPIA)
    preparar_sesion(es_modo_live=False, raiz=raiz)
    assert servicio_modo_sesion.registro_modo_activo.obtener() is ModoSesion.LIMPIA