
"""
Bootstrap del exchange simulado y contexto principal de HRM.

La inicialización del exchange se realiza exclusivamente en PAPER.
Las sesiones continuas conservan el estado del singleton.
"""

import copy
import logging
import math
from typing import Any, Dict, Optional

from l1_operational.simulated_exchange_client import (
    SimulatedExchangeClient,
)


log = logging.getLogger(__name__)


def _obtener_balances_configurados(
    config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Obtiene y prepara los balances iniciales de PAPER.

    Prioriza simulated_initial_balances.
    Si no existe USDT, utiliza INITIAL_BALANCE.
    """
    balances_configurados = config.get(
        "simulated_initial_balances",
        {},
    )

    if balances_configurados is None:
        balances_configurados = {}

    if not isinstance(balances_configurados, dict):
        raise TypeError(
            "simulated_initial_balances debe ser un diccionario"
        )

    balances = copy.deepcopy(balances_configurados)

    if not balances:
        capital_inicial = config.get(
            "INITIAL_BALANCE",
            500.0,
        )

        balances = {
            "BTC": 0.0,
            "ETH": 0.0,
            "USDT": capital_inicial,
        }

        log.info(
            "No hay balances PAPER configurados. "
            "Se utiliza INITIAL_BALANCE: %s USDT",
            capital_inicial,
        )

    else:
        balances.setdefault("BTC", 0.0)
        balances.setdefault("ETH", 0.0)

        if "USDT" not in balances:
            capital_inicial = config.get(
                "INITIAL_BALANCE",
                500.0,
            )

            balances["USDT"] = capital_inicial

            log.warning(
                "No se ha configurado USDT en "
                "simulated_initial_balances. "
                "Se utiliza INITIAL_BALANCE: %s USDT",
                capital_inicial,
            )

    return balances


def _obtener_capital_usdt(
    balances: Dict[str, float],
) -> float:
    """
    Valida y devuelve el capital USDT de unos balances.
    """
    try:
        capital = float(balances.get("USDT", 0.0))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "El capital USDT configurado no es válido"
        ) from error

    if not math.isfinite(capital) or capital <= 0:
        raise ValueError(
            "El capital USDT debe ser positivo y finito"
        )

    return capital


def _obtener_balances_sesion_activa() -> Dict[str, float]:
    """
    Recupera los balances del singleton sin modificarlos.
    """
    instancia = SimulatedExchangeClient._instance

    if instancia is None:
        raise RuntimeError(
            "No existe una instancia PAPER activa"
        )

    return instancia.get_balances()


def _aplicar_balances_configurados(
    balances: Dict[str, float],
) -> None:
    """
    Aplica los balances configurados a una instancia recién creada.

    Si el reset falla, detiene el bootstrap.
    No modifica directamente los atributos internos del cliente.
    """
    instancia = SimulatedExchangeClient._instance

    if instancia is None:
        raise RuntimeError(
            "No existe una instancia PAPER para inicializar"
        )

    try:
        instancia.force_reset(
            copy.deepcopy(balances)
        )
    except Exception as error:
        log.exception(
            "No se pudo aplicar el capital inicial PAPER"
        )

        raise RuntimeError(
            "No se puede inicializar el capital PAPER"
        ) from error


def bootstrap_simulated_exchange(
    config: Dict[str, Any],
    force_paper: bool = True,
) -> Optional[SimulatedExchangeClient]:
    """
    Inicializa el exchange simulado y conserva el estado de sesión.

    Una instancia nueva recibe el capital configurado.
    Una sesión continua conserva sus balances actuales.

    Args:
        config: Configuración del sistema.
        force_paper: Fuerza la inicialización PAPER.

    Returns:
        Instancia SimulatedExchangeClient o None si PAPER
        está deshabilitado explícitamente.
    """
    if not isinstance(config, dict):
        raise TypeError(
            "La configuración debe ser un diccionario"
        )

    use_paper = (
        force_paper
        or config.get("paper_mode", False)
    )

    if not use_paper:
        log.info(
            "PAPER deshabilitado. "
            "No se inicializa SimulatedExchangeClient."
        )
        return None

    balances_configurados = _obtener_balances_configurados(
        config
    )

    capital_configurado = _obtener_capital_usdt(
        balances_configurados
    )

    singleton_existia = (
        SimulatedExchangeClient._instance is not None
    )

    recien_inicializado = (
        SimulatedExchangeClient.is_just_initialized()
    )

    log.info("=" * 70)
    log.info("INICIALIZACIÓN DEL EXCHANGE PAPER")
    log.info(
        "Singleton existente: %s",
        singleton_existia,
    )
    log.info(
        "Instancia recién inicializada: %s",
        recien_inicializado,
    )

    if singleton_existia and not recien_inicializado:
        log.info(
            "Sesión continua: se conservan los balances existentes."
        )

        balances_iniciales = (
            _obtener_balances_sesion_activa()
        )

        capital_existente = _obtener_capital_usdt(
            balances_iniciales
        )

        if abs(
            capital_existente - capital_configurado
        ) > 1.0:
            log.warning(
                "Diferencia entre capital configurado y "
                "capital de sesión. Se conserva la sesión. "
                "Existente: %.2f USDT | Configurado: %.2f USDT",
                capital_existente,
                capital_configurado,
            )

        balances_para_instancia = balances_iniciales

    elif singleton_existia and recien_inicializado:
        log.info(
            "Instancia recién creada en este ciclo: "
            "se aplica el capital configurado."
        )

        _aplicar_balances_configurados(
            balances_configurados
        )

        balances_para_instancia = balances_configurados

    else:
        log.info(
            "Sesión nueva: se utiliza el capital configurado."
        )

        balances_para_instancia = balances_configurados

    client = SimulatedExchangeClient.get_instance(
        initial_balances=copy.deepcopy(
            balances_para_instancia
        )
    )

    was_new_instance = (
        SimulatedExchangeClient.mark_cycle_completed()
    )

    balances_actuales = client.get_balances()
    capital_actual = _obtener_capital_usdt(
        balances_actuales
    )

    if was_new_instance:
        if abs(
            capital_actual - capital_configurado
        ) >= 0.01:
            raise RuntimeError(
                "El capital inicial PAPER no coincide "
                "con el capital configurado."
            )

    log.info(
        "Exchange PAPER inicializado correctamente."
    )
    log.info(
        "Identificador de sesión: %s",
        id(client),
    )
    log.info(
        "Instancia nueva: %s",
        was_new_instance,
    )
    log.info(
        "Origen del capital: %s",
        (
            "CONFIGURACIÓN"
            if was_new_instance
            else "SESIÓN EXISTENTE"
        ),
    )
    log.info(
        "BTC: %.8f",
        balances_actuales.get("BTC", 0.0),
    )
    log.info(
        "ETH: %.8f",
        balances_actuales.get("ETH", 0.0),
    )
    log.info(
        "USDT: %.2f",
        capital_actual,
    )
    log.info("=" * 70)

    return client


class SystemBootstrap:
    """
    Inicialización centralizada del contexto del sistema HRM.

    Esta clase mantiene la interfaz de compatibilidad existente.
    La inicialización de los componentes de trading corresponde
    al bootstrap principal del sistema.
    """

    def __init__(self) -> None:
        """
        Inicializa el contexto de componentes.
        """
        self.components: Dict[str, Any] = {}
        self.external_adapter = None

        log.info(
            "SystemBootstrap inicializado"
        )

    def initialize_system(self) -> Any:
        """
        Construye y devuelve el contexto del sistema.

        Returns:
            Contexto con componentes y adaptador externo.
        """
        log.info(
            "Inicializando contexto de componentes HRM..."
        )

        class SystemContext:
            """
            Contexto de compatibilidad de HRM.
            """

            def __init__(
                self,
                components: Dict[str, Any],
                external_adapter: Any,
            ) -> None:
                self.components = components
                self.external_adapter = external_adapter

        system_context = SystemContext(
            components=self.components,
            external_adapter=self.external_adapter,
        )

        log.info(
            "SystemBootstrap completado correctamente"
        )

        return system_context
