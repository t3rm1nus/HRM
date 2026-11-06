import pytest

from l1_operational.simulated_exchange_client import SimulatedExchangeClient


@pytest.fixture
def exchange():
    return SimulatedExchangeClient.get_instance(
        initial_balances={"USDT": 10_000.0, "BTC": 0.0, "ETH": 0.0},
        fee=0.001,
        slippage=0.0005,
    )


@pytest.mark.unit
def test_crea_exchange_con_balances_iniciales(exchange):
    assert exchange.get_balances() == {"USDT": 10_000.0, "BTC": 0.0, "ETH": 0.0}


@pytest.mark.unit
def test_requiere_balances_al_crear_el_singleton():
    with pytest.raises(RuntimeError):
        SimulatedExchangeClient.get_instance()


@pytest.mark.unit
def test_compra_aplica_slippage_comision_y_actualiza_balances(exchange):
    trade = exchange.execute_order("BTCUSDT", "BUY", 0.1, 100.0)
    precio = 100.0 * (1 + 0.0005)
    coste = 0.1 * precio
    comision = coste * 0.001

    assert trade["execution_price"] == pytest.approx(precio)
    assert trade["fee"] == pytest.approx(comision)
    assert exchange.get_balances()["BTC"] == pytest.approx(0.1)
    assert exchange.get_balances()["USDT"] == pytest.approx(10_000.0 - coste - comision)


@pytest.mark.unit
def test_venta_actualiza_balances_y_registra_trade(exchange):
    exchange.set_balance("BTC", 0.5)
    trade = exchange.execute_order("BTCUSDT", "SELL", 0.2, 100.0)
    precio = 100.0 * (1 - 0.0005)
    coste = 0.2 * precio
    comision = coste * 0.001

    assert trade["side"] == "SELL"
    assert exchange.get_balances()["BTC"] == pytest.approx(0.3)
    assert exchange.get_balances()["USDT"] == pytest.approx(10_000.0 + coste - comision)
    assert len(exchange.get_trade_history()) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cantidad", "precio", "lado"),
    [(0, 100.0, "BUY"), (-1, 100.0, "BUY"), (1, 0, "BUY"),
     (1, -10, "BUY"), (1, 100.0, "HOLD")],
)
def test_rechaza_ordenes_con_parametros_invalidos(exchange, cantidad, precio, lado):
    with pytest.raises(ValueError):
        exchange.execute_order("BTCUSDT", lado, cantidad, precio)


@pytest.mark.unit
def test_rechaza_compra_sin_usdt_suficiente(exchange):
    with pytest.raises(ValueError, match="Fondos insuficientes"):
        exchange.execute_order("BTCUSDT", "BUY", 1_000, 100.0)


@pytest.mark.unit
def test_rechaza_venta_sin_activo_suficiente(exchange):
    with pytest.raises(ValueError, match="Balance insuficiente"):
        exchange.execute_order("BTCUSDT", "SELL", 0.01, 100.0)


@pytest.mark.unit
def test_rechaza_simbolos_que_no_terminan_en_usdt(exchange):
    with pytest.raises(ValueError, match="Símbolo no soportado"):
        exchange.execute_order("BTCEUR", "BUY", 0.01, 100.0)


@pytest.mark.unit
def test_historial_devuelve_copia(exchange):
    exchange.execute_order("BTCUSDT", "BUY", 0.01, 100.0)
    historial = exchange.get_trade_history()
    historial.clear()
    assert len(exchange.get_trade_history()) == 1


@pytest.mark.unit
def test_force_reset_restaura_balances_y_vacia_historial(exchange):
    exchange.execute_order("BTCUSDT", "BUY", 0.01, 100.0)
    exchange.force_reset({"USDT": 500.0, "BTC": 0.0, "ETH": 0.0})

    assert exchange.get_balances() == {"USDT": 500.0, "BTC": 0.0, "ETH": 0.0}
    assert exchange.get_trade_history() == []
    assert exchange.get_order_history() == []
