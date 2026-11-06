import pytest

from api.esquemas.modo_sesion import ModoSesion
from system.preparar_sesion import decidir_modo_efectivo


@pytest.mark.unit
def test_modo_limpia_se_conserva_en_paper():
    assert decidir_modo_efectivo(ModoSesion.LIMPIA, False) is ModoSesion.LIMPIA


@pytest.mark.unit
def test_modo_limpia_se_convierte_a_persistente_en_live():
    assert decidir_modo_efectivo(ModoSesion.LIMPIA, True) is ModoSesion.PERSISTENTE


@pytest.mark.unit
def test_modo_persistente_se_conserva_en_live():
    assert decidir_modo_efectivo(ModoSesion.PERSISTENTE, True) is ModoSesion.PERSISTENTE
