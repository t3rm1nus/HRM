import pytest

from l1_operational.simulated_exchange_client import SimulatedExchangeClient


@pytest.fixture(autouse=True)
def reiniciar_exchange_antes_y_despues():
    # Aísla el singleton PAPER entre pruebas.
    SimulatedExchangeClient.cleanup()
    yield
    SimulatedExchangeClient.cleanup()
