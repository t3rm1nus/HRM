from core.portfolio_manager import PortfolioManager
from l1_operational.simulated_exchange_client import SimulatedExchangeClient


class ClienteSimuladoFalso:
    def __init__(self):
        self.balances = {}

    def set_balance(self, activo, cantidad):
        self.balances[activo] = cantidad


def test_restaurar_balances_persistidos_en_cliente_paper(monkeypatch):
    gestor = PortfolioManager.__new__(PortfolioManager)
    gestor.portfolio = {
        "BTCUSDT": {"position": 0.012},
        "ETHUSDT": {"position": 0.35},
        "USDT": {"free": 1250.50},
    }
    monkeypatch.setattr(gestor, "load_from_json", lambda: True)
    cliente = ClienteSimuladoFalso()

    assert gestor.restaurar_balances_en_cliente_simulado(cliente) is True
    assert cliente.balances == {"BTC": 0.012, "ETH": 0.35, "USDT": 1250.50}


def test_no_restaura_si_no_hay_estado_persistido(monkeypatch):
    gestor = PortfolioManager.__new__(PortfolioManager)
    gestor.portfolio = {}
    monkeypatch.setattr(gestor, "load_from_json", lambda: False)
    cliente = ClienteSimuladoFalso()

    assert gestor.restaurar_balances_en_cliente_simulado(cliente) is False
    assert cliente.balances == {}


def test_order_id_es_unico_y_trade_id_sigue_si_endiente():
    SimulatedExchangeClient.reset_instance()
    cliente = SimulatedExchangeClient(
        initial_balances={"BTC": 0.0, "ETH": 0.0, "USDT": 1000.0},
        fee=0.0,
        slippage=0.0,
    )
    primera = cliente.execute_order("BTCUSDT", "BUY", 0.001, 50000.0)
    segunda = cliente.execute_order("BTCUSDT", "BUY", 0.001, 50000.0)

    assert primera["trade_id"] == 1
    assert segunda["trade_id"] == 2
    assert primera["order_id"] != segunda["order_id"]
    assert primera["order_id"].startswith("simulated_")
    assert segunda["order_id"].startswith("simulated_")
    SimulatedExchangeClient.reset_instance()
