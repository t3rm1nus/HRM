import json

import pytest

from api.servicios.servicio_operaciones import (
    ErrorLecturaOperaciones,
    calcular_total_paginas,
    cargar_ordenes,
    convertir_a_orden,
    convertir_registros_validos,
    extraer_registros,
    leer_contenido_json,
)


def crear_registro(**cambios) -> dict:
    registro = {
        "timestamp": "2026-11-02T10:00:00",
        "symbol": "BTCUSDT",
        "side": "buy",
        "quantity": 0.01,
        "price": 50000.0,
        "filled_price": 50010.0,
        "commission": 0.5,
        "pnl": 0.0,
        "status": "filled",
        "order_id": "paper_1",
        "cycle_id": 7,
        "strategy": "paper",
        "reason": "señal L2",
        "market_conditions": {"BTCUSDT_price": 50000.0},
    }
    registro.update(cambios)
    return registro


def escribir_archivo(ruta, trades) -> None:
    ruta.write_text(json.dumps({"trades": trades}), encoding="utf-8")


def test_convierte_registro_del_logger_a_orden():
    orden = convertir_a_orden(crear_registro(side="BUY"))

    assert orden.order_id == "paper_1"
    assert orden.simbolo == "BTCUSDT"
    assert orden.lado == "buy"
    assert orden.precio_llenado == 50010.0
    assert orden.metadata == {
        "ciclo": 7,
        "motivo": "señal L2",
        "condiciones_mercado": {"BTCUSDT_price": 50000.0},
    }


def test_descarta_registros_invalidos_y_conserva_los_validos(caplog):
    registros = [crear_registro(), crear_registro(order_id=None), "no es un dict"]

    ordenes = convertir_registros_validos(registros)

    assert [orden.order_id for orden in ordenes] == ["paper_1"]
    assert caplog.text.count("descartado") == 2


def test_sin_archivo_devuelve_lote_vacio(tmp_path):
    lote = cargar_ordenes(ruta=tmp_path / "no_existe.json")

    assert (lote.total, lote.ordenes, lote.has_more, lote.pagina_total) == (0, [], False, 1)


def test_json_corrupto_lanza_error_de_lectura(tmp_path):
    ruta = tmp_path / "roto.json"
    ruta.write_text("{ no es json", encoding="utf-8")

    with pytest.raises(ErrorLecturaOperaciones):
        leer_contenido_json(ruta)


def test_contenido_sin_lista_trades_lanza_error():
    with pytest.raises(ErrorLecturaOperaciones):
        extraer_registros({"otra_cosa": []})


def test_ordena_de_mas_reciente_a_mas_antigua(tmp_path):
    ruta = tmp_path / "trades.json"
    escribir_archivo(ruta, [
        crear_registro(order_id="a", timestamp="2026-11-01T10:00:00"),
        crear_registro(order_id="c", timestamp="2026-11-03T10:00:00"),
        crear_registro(order_id="b", timestamp="2026-11-02T10:00:00"),
    ])

    lote = cargar_ordenes(ruta=ruta)

    assert [orden.order_id for orden in lote.ordenes] == ["c", "b", "a"]


def test_filtra_por_simbolo_sin_distinguir_mayusculas_y_por_lado(tmp_path):
    ruta = tmp_path / "trades.json"
    escribir_archivo(ruta, [
        crear_registro(order_id="1", symbol="BTCUSDT", side="buy"),
        crear_registro(order_id="2", symbol="BTCUSDT", side="sell"),
        crear_registro(order_id="3", symbol="ETHUSDT", side="buy"),
    ])

    lote = cargar_ordenes(simbolo="btcusdt", lado="sell", ruta=ruta)

    assert [orden.order_id for orden in lote.ordenes] == ["2"]


def test_pagina_y_calcula_has_more(tmp_path):
    ruta = tmp_path / "trades.json"
    escribir_archivo(ruta, [
        crear_registro(order_id=str(i), timestamp=f"2026-11-02T10:00:0{i}") for i in range(5)
    ])

    primera = cargar_ordenes(pagina=1, tamano=2, ruta=ruta)
    ultima = cargar_ordenes(pagina=3, tamano=2, ruta=ruta)

    assert (primera.total, primera.pagina_total, primera.has_more) == (5, 3, True)
    assert [orden.order_id for orden in ultima.ordenes] == ["0"]
    assert ultima.has_more is False


def test_calcula_total_paginas_con_minimo_uno():
    assert calcular_total_paginas(0, 50) == 1
    assert calcular_total_paginas(101, 50) == 3
