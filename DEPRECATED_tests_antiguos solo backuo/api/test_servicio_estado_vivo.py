import asyncio
from types import SimpleNamespace

from api.servicios.servicio_estado_vivo import (
    SEGUNDOS_MAXIMOS_SIN_CICLO,
    ServicioEstadoVivo,
    formatear_duracion,
    normalizar_modo,
)


class RelojManual:
    def __init__(self) -> None:
        self.ahora = 1000.0

    def __call__(self) -> float:
        return self.ahora

    def avanzar(self, segundos: float) -> None:
        self.ahora += segundos


class EventosFalsos:
    def __init__(self) -> None:
        self.publicados: list[tuple[str, object]] = []

    async def publicar(self, tipo, datos) -> int:
        self.publicados.append((str(getattr(tipo, "value", tipo)), datos))
        return 1


def crear_cartera_falsa():
    return SimpleNamespace(
        initial_balance=3000.0,
        portfolio={
            "BTCUSDT": {"position": 0.01, "free": 0.01},
            "ETHUSDT": {"position": 0.5, "free": 0.5},
            "USDT": {"free": 1500.0},
            "total": 3100.0,
        },
    )


def crear_servicio():
    reloj, eventos = RelojManual(), EventosFalsos()
    return ServicioEstadoVivo(eventos, reloj), reloj, eventos


def test_formatea_duracion():
    assert formatear_duracion(3725) == "1h 02m 05s"
    assert formatear_duracion(-5) == "0h 00m 00s"


def test_sin_bucle_el_sistema_esta_pausado():
    servicio, _, _ = crear_servicio()
    estado = servicio.obtener_estado_sistema()
    assert estado == {
        "status": "paused",
        "trading_active": False,
        "cycle_count": 0,
        "uptime": "no disponible",
        "mode": "unknown",
    }


def test_con_bucle_activo_el_sistema_esta_en_marcha():
    servicio, reloj, _ = crear_servicio()
    servicio.registrar_inicio_bucle()
    reloj.avanzar(65)
    servicio.registrar_ciclo(7)
    estado = servicio.obtener_estado_sistema()
    assert estado["status"] == "running"
    assert estado["trading_active"] is True
    assert estado["cycle_count"] == 7
    assert estado["uptime"] == "0h 01m 05s"


def test_un_bucle_sin_ciclos_recientes_se_considera_pausado():
    servicio, reloj, _ = crear_servicio()
    servicio.registrar_inicio_bucle()
    servicio.registrar_ciclo(1)
    reloj.avanzar(SEGUNDOS_MAXIMOS_SIN_CICLO + 1)
    assert servicio.obtener_estado_sistema()["trading_active"] is False


def test_tras_la_parada_el_sistema_esta_pausado():
    servicio, _, _ = crear_servicio()
    servicio.registrar_inicio_bucle()
    servicio.registrar_parada()
    assert servicio.obtener_estado_sistema()["status"] == "paused"


def test_sin_cartera_registrada_no_hay_resumen():
    servicio, _, _ = crear_servicio()
    assert servicio.obtener_resumen_cartera() is None


def test_resumen_de_cartera_lee_balances_y_nav():
    servicio, _, _ = crear_servicio()
    servicio.registrar_cartera(crear_cartera_falsa())
    assert servicio.obtener_resumen_cartera() == {
        "btc_balance": 0.01,
        "eth_balance": 0.5,
        "usdt_balance": 1500.0,
        "total_nav": 3100.0,
        "initial_capital": 3000.0,
    }


def test_resumen_tolera_cartera_incompleta():
    servicio, _, _ = crear_servicio()
    servicio.registrar_cartera(SimpleNamespace(initial_balance=500.0, portfolio={}))
    resumen = servicio.obtener_resumen_cartera()
    assert resumen["btc_balance"] == 0.0 and resumen["total_nav"] == 0.0


def test_notificar_ciclo_publica_estado_y_cartera():
    servicio, _, eventos = crear_servicio()
    servicio.registrar_cartera(crear_cartera_falsa())
    servicio.registrar_inicio_bucle()
    asyncio.run(servicio.notificar_ciclo(3))
    assert [tipo for tipo, _ in eventos.publicados] == ["estado", "cierre"]
    assert eventos.publicados[0][1]["cycle_count"] == 3


def test_un_fallo_al_publicar_no_rompe_el_bucle():
    servicio, _, eventos = crear_servicio()

    async def publicar_roto(tipo, datos):
        raise RuntimeError("cliente caído")

    eventos.publicar = publicar_roto
    asyncio.run(servicio.notificar_ciclo(1))
    assert servicio.obtener_estado_sistema()["cycle_count"] == 1


def test_normaliza_los_modos_internos():
    assert normalizar_modo("simulated") == "paper"
    assert normalizar_modo("live") == "live"
    assert normalizar_modo("LIVE") == "live"
    assert normalizar_modo(None) == "unknown"
    assert normalizar_modo("otro") == "unknown"


def test_el_estado_incluye_el_modo_registrado():
    servicio, _, _ = crear_servicio()
    servicio.registrar_modo("simulated")
    assert servicio.obtener_estado_sistema()["mode"] == "paper"
    servicio.registrar_modo("live")
    assert servicio.obtener_estado_sistema()["mode"] == "live"
