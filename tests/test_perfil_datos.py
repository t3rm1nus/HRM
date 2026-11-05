import json

from api.servicios import servicio_operaciones
from storage import perfil_datos
from storage.paper_trade_logger import get_paper_logger, reiniciar_loggers_paper


def test_rutas_de_cartera_y_operaciones_son_independientes(tmp_path):
    cartera_persistente = perfil_datos.obtener_ruta_cartera("persistente", tmp_path)
    cartera_limpia = perfil_datos.obtener_ruta_cartera("limpia", tmp_path)
    operaciones_persistentes = perfil_datos.obtener_directorio_operaciones("persistente", tmp_path)
    operaciones_limpias = perfil_datos.obtener_directorio_operaciones("limpia", tmp_path)

    assert cartera_persistente != cartera_limpia
    assert operaciones_persistentes != operaciones_limpias
    assert "persistente" in str(cartera_persistente)
    assert "limpia" in str(cartera_limpia)


def test_logger_separado_por_perfil(tmp_path, monkeypatch):
    reiniciar_loggers_paper()
    monkeypatch.chdir(tmp_path)

    perfil_datos.configurar_perfil_datos("persistente")
    logger_persistente = get_paper_logger()
    logger_persistente.log_paper_trade({
        "status": "filled", "symbol": "BTCUSDT", "side": "buy",
        "quantity": 0.01, "filled_price": 50000, "commission": 0.1,
        "order_id": "persistente-1",
    })

    perfil_datos.configurar_perfil_datos("limpia")
    logger_limpio = get_paper_logger()
    assert logger_limpio.base_dir != logger_persistente.base_dir
    assert logger_limpio.trades == []
    assert len(logger_persistente.trades) == 1

    perfil_datos.configurar_perfil_datos("persistente")
    assert get_paper_logger() is logger_persistente
    reiniciar_loggers_paper()


def test_servicio_operaciones_usa_perfil_activo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(servicio_operaciones.VARIABLE_ENTORNO_RUTA, raising=False)

    perfil_datos.configurar_perfil_datos("persistente")
    ruta_persistente = servicio_operaciones.obtener_ruta_operaciones()
    perfil_datos.configurar_perfil_datos("limpia")
    ruta_limpia = servicio_operaciones.obtener_ruta_operaciones()

    assert ruta_persistente != ruta_limpia
    assert ruta_persistente.parts[-4:-2] == ("paper", "persistente")
    assert ruta_limpia.parts[-4:-2] == ("paper", "limpia")


def test_portfolio_manager_restaura_perfil_y_hidrata_exchange(tmp_path, monkeypatch):
    import asyncio
    from core.portfolio_manager import PortfolioManager

    monkeypatch.chdir(tmp_path)
    perfil_datos.configurar_perfil_datos("persistente")
    ruta = perfil_datos.obtener_ruta_cartera("persistente")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({
        "mode": "simulated",
        "initial_balance": 500.0,
        "portfolio": {
            "USDT": {"free": 123.0},
            "BTCUSDT": {"position": 0.01, "free": 0.01},
            "ETHUSDT": {"position": 0.02, "free": 0.02},
            "total": 500.0,
        },
    }), encoding="utf-8")

    class ExchangeSimuladoFalso:
        def __init__(self):
            self.balances = {}

        def force_reset(self, balances):
            self.balances = balances.copy()

    PortfolioManager.reset_instance()
    exchange = ExchangeSimuladoFalso()
    manager = PortfolioManager(exchange_client=exchange, mode="simulated", initial_balance=500.0)
    asyncio.run(manager.initialize_async())

    assert manager._estado_paper_restaurado is True
    assert exchange.balances == {"BTC": 0.01, "ETH": 0.02, "USDT": 123.0}
    assert manager.portfolio["USDT"]["free"] == 123.0
    assert manager.save_to_json() is True
    assert ruta.exists()
    PortfolioManager.reset_instance()
    reiniciar_loggers_paper()


def test_limpieza_tecnica_no_borra_historial_paper(tmp_path, monkeypatch):
    from system.system_cleanup import filesystem_cleanup

    monkeypatch.chdir(tmp_path)
    historial = tmp_path / "data" / "paper" / "persistente" / "paper_trades" / "paper_trades.json"
    historial.parent.mkdir(parents=True)
    historial.write_text('{"trades": [1]}', encoding="utf-8")

    resultado = filesystem_cleanup()

    assert resultado["success"] is True
    assert json.loads(historial.read_text(encoding="utf-8"))["trades"] == [1]
