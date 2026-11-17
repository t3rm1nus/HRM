"""Pruebas de concurrencia, ciclo de vida y consulta API de HRM."""
import asyncio
import sys
import threading
from types import ModuleType

import pytest
from fastapi import APIRouter

# Los routers funcionales del proyecto se prueban en sus propias suites.
# Aquí aislamos el endpoint /status real de sus dependencias laterales.
for nombre, atributo in {
    "api.rutas.rutas_operaciones": "router_operaciones",
    "api.rutas.rutas_modo_sesion": "router_modo_sesion",
    "api.rutas.rutas_mercado": "router_mercado",
}.items():
    modulo = ModuleType(nombre)
    setattr(modulo, atributo, APIRouter())
    sys.modules[nombre] = modulo

from system.state_coordinator import StateCoordinator
from api.servicios.servicio_estado_vivo import ServicioEstadoVivo


class RelojFalso:
    def __init__(self):
        self.valor = 100.0

    def __call__(self):
        return self.valor


class EventosFalsos:
    async def publicar(self, *args, **kwargs):
        return 0


@pytest.fixture
def coordinador():
    StateCoordinator.reset_instance()
    instancia = StateCoordinator(mode="simulated")
    yield instancia
    StateCoordinator.reset_instance()


@pytest.fixture
def servicio(coordinador):
    return ServicioEstadoVivo(EventosFalsos(), reloj=RelojFalso(), coordinador=coordinador)


def test_arranque_no_publica_running_hasta_inicio_de_ciclo(coordinador, servicio):
    assert coordinador.iniciar_arranque("prueba", "test")
    servicio.registrar_inicio_bucle()
    estado = servicio.obtener_estado_sistema()
    assert estado["runtime_state"] == "STARTING"
    assert estado["trading_active"] is False
    assert estado["status"] == "starting"

    servicio.registrar_inicio_ciclo(1)
    estado = servicio.obtener_estado_sistema()
    assert estado["runtime_state"] == "RUNNING"
    assert estado["trading_active"] is True
    assert estado["status"] == "running"


def test_parada_ordenada_pasa_por_stopping_y_stopped(coordinador, servicio):
    coordinador.iniciar_arranque("prueba", "test")
    servicio.registrar_inicio_ciclo(1)
    servicio.registrar_parada()
    assert coordinador.obtener_estado_runtime()["state"] == "STOPPING"
    assert not servicio.obtener_estado_sistema()["trading_active"]
    servicio.confirmar_parada()
    assert coordinador.obtener_estado_runtime()["state"] == "STOPPED"


def test_reinicio_desde_stopped(coordinador, servicio):
    coordinador.iniciar_arranque("primero", "test")
    servicio.registrar_inicio_ciclo(1)
    servicio.registrar_ciclo(1)
    servicio.registrar_parada()
    servicio.confirmar_parada()
    assert coordinador.iniciar_arranque("segundo", "test")
    estado = coordinador.obtener_estado_runtime()
    assert estado["state"] == "STARTING"
    assert estado["cycle_count"] == 0


def test_error_durante_arranque_publica_failed_y_permite_reintento(coordinador):
    assert coordinador.iniciar_arranque("arranque", "test")
    coordinador.registrar_fallo(RuntimeError("bootstrap roto"), "bootstrap")
    estado = coordinador.obtener_estado_runtime()
    assert estado["state"] == "FAILED"
    assert not estado["trading_active"]
    assert estado["last_error"] == "bootstrap roto"
    assert coordinador.iniciar_arranque("reintento", "test")


def test_error_durante_ciclo_publica_degraded_y_recupera(coordinador):
    coordinador.iniciar_arranque("arranque", "test")
    coordinador.iniciar_ciclo(1, "runtime")
    coordinador.registrar_error_ciclo(RuntimeError("fallo ciclo"), 1, "runtime")
    assert coordinador.obtener_estado_runtime()["state"] == "DEGRADED"
    assert not coordinador.obtener_estado_runtime()["trading_active"]
    coordinador.iniciar_ciclo(2, "runtime")
    assert coordinador.obtener_estado_runtime()["state"] == "RUNNING"
    coordinador.registrar_ciclo_completado(2, "runtime")
    assert coordinador.obtener_estado_runtime()["cycle_count"] == 2


def test_doble_solicitud_concurrente_solo_adquiere_un_runtime(coordinador):
    barrera = threading.Barrier(3)
    resultados = []
    bloqueo = threading.Lock()

    def solicitar():
        barrera.wait()
        resultado = coordinador.iniciar_arranque("concurrente", "test")
        with bloqueo:
            resultados.append(resultado)

    hilos = [threading.Thread(target=solicitar) for _ in range(2)]
    for hilo in hilos:
        hilo.start()
    barrera.wait()
    for hilo in hilos:
        hilo.join()
    assert sorted(resultados) == [False, True]
    assert coordinador.obtener_estado_runtime()["state"] == "STARTING"


def test_parada_durante_operacion_bloquea_siguientes_ciclos(coordinador):
    coordinador.iniciar_arranque("arranque", "test")
    coordinador.iniciar_ciclo(1, "runtime")
    assert coordinador.obtener_estado_runtime()["cycle_in_progress"]
    coordinador.iniciar_parada("Ctrl+C", "signal_handler")
    assert coordinador.obtener_estado_runtime()["state"] == "STOPPING"
    with pytest.raises(RuntimeError):
        coordinador.iniciar_ciclo(2, "runtime")
    coordinador.finalizar_parada()
    assert coordinador.obtener_estado_runtime()["state"] == "STOPPED"


def test_endpoint_status_consulta_el_mismo_coordinador(coordinador):
    from api.app import get_status
    from api.servicios.servicio_estado_vivo import servicio_estado_vivo

    servicio_estado_vivo.registrar_coordinador(coordinador)
    coordinador.iniciar_arranque("api test", "test")
    servicio_estado_vivo.registrar_inicio_bucle()
    respuesta = asyncio.run(get_status())
    assert respuesta.status == "starting"
    assert respuesta.trading_active is False

    servicio_estado_vivo.registrar_inicio_ciclo(1)
    respuesta = asyncio.run(get_status())
    assert respuesta.status == "running"
    assert respuesta.trading_active is True
    assert respuesta.runtime_state == "RUNNING"
    servicio_estado_vivo.registrar_coordinador(None)


def test_historial_registra_fecha_motivo_y_componente(coordinador):
    coordinador.iniciar_arranque("arranque", "bootstrap")
    coordinador.iniciar_ciclo(1, "main")
    evento = coordinador.obtener_estado_runtime()["history"][-1]
    assert evento["timestamp"]
    assert evento["reason"] == "Comienza el ciclo operativo 1"
    assert evento["component"] == "main"
    assert evento["state"] == "RUNNING"


def test_interfaz_legacy_de_state_coordinator_se_conserva(coordinador):
    coordinador.update_state({
        "system_state_type": "ACTIVE",
        "system_state_reason": "PAPER local",
    })
    coordinador.update_state("market_data", "healthy", {"symbols": 2})
    estado = coordinador.get_state("current")
    assert estado["system_state_type"] == "ACTIVE"
    assert estado["system_state_reason"] == "PAPER local"
    assert coordinador.get_system_state().components["market_data"]["status"] == "healthy"
