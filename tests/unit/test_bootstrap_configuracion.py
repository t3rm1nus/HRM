import pytest

from comms.config import config as configuracion_real
from system.bootstrap import admite_consulta_por_clave, bootstrap_simulated_exchange


@pytest.mark.regression
def test_acepta_la_configuracion_real_que_entrega_main():
    assert admite_consulta_por_clave(configuracion_real) is True


@pytest.mark.unit
def test_acepta_un_diccionario():
    assert admite_consulta_por_clave({"paper_mode": True}) is True


@pytest.mark.unit
@pytest.mark.parametrize("valor_invalido", [None, 5, "texto", [1, 2]])
def test_rechaza_objetos_sin_metodo_get(valor_invalido):
    assert admite_consulta_por_clave(valor_invalido) is False


@pytest.mark.regression
def test_bootstrap_rechaza_configuracion_sin_get():
    with pytest.raises(TypeError, match="get"):
        bootstrap_simulated_exchange(object())
