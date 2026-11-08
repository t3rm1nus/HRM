
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inicialización principal de HRM.

Durante la fase 1, la ejecución de órdenes está limitada a PAPER.
LIVE y TESTNET permanecen deshabilitados.
"""

import os
import sys
import asyncio
from typing import Any, Dict, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from core.state_manager import (
    initialize_state,
    validate_state_structure,
)
from core.portfolio_manager import PortfolioManager
from core.logging import logger
from core.config import get_config, set_forced_mode
from core.incremental_signal_verifier import (
    get_signal_verifier,
    start_signal_verification,
    stop_signal_verification,
)
from core.trading_metrics import get_trading_metrics

from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l1_operational.order_manager import OrderManager
from l1_operational.simulated_exchange_client import (
    SimulatedExchangeClient,
)

from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l2_tactic.config import L2Config
from l2_tactic.risk_controls.manager import RiskControlManager

from comms.config import config, APAGAR_L3
from comms.message_bus import MessageBus

from sentiment.sentiment_manager import update_sentiment_texts

from system.system_cleanup import perform_full_cleanup
from config_loader import get_initial_balances, get_capital_usd


# Estado de sesión
session_is_fresh = True


def is_session_fresh() -> bool:
    """Indica si la sesión actual es nueva."""
    return session_is_fresh


def mark_session_used() -> None:
    """Marca la sesión actual como utilizada."""
    global session_is_fresh
    session_is_fresh = False


def reset_session() -> bool:
    """Restablece el indicador de sesión nueva."""
    global session_is_fresh
    session_is_fresh = True
    logger.info("Sesión restablecida")
    return True


class HRMBootstrap:
    """Inicializa y conecta los componentes principales de HRM."""

    MODO_PAPER = "paper"
    MODOS_PERMITIDOS = {"paper", "simulated"}

    def __init__(self):
        self.components: Dict[str, Any] = {}
        self.state = None
        self.portfolio_manager = None
        self.order_manager = None
        self.execution_gateway = None
        self.runtime_loop = None
        self._mode = self.MODO_PAPER

    def _validar_modo(self, mode: str) -> None:
        """Impide iniciar el sistema en un modo no permitido."""
        if not isinstance(mode, str):
            raise ValueError("El modo de ejecución debe ser un texto")

        modo = mode.strip().lower()

        if modo not in self.MODOS_PERMITIDOS:
            raise RuntimeError(
                f"Modo '{mode}' bloqueado durante la fase 1. "
                "Solo se permite PAPER."
            )

    def _forzar_modo_paper(self) -> None:
        """Establece PAPER como único modo de ejecución."""
        self._mode = self.MODO_PAPER

        os.environ["BINANCE_MODE"] = "PAPER"
        os.environ["PAPER_MODE"] = "true"
        os.environ["HRM_ENABLE_LIVE_TRADING"] = "false"
        os.environ["USE_TESTNET"] = "false"
        os.environ["BINANCE_TESTNET"] = "false"

        set_forced_mode(self.MODO_PAPER)

    def _get_mode_from_bootstrap(self) -> str:
        """Devuelve el modo fijado por el bootstrap."""
        return self._mode

    async def bootstrap_system(
        self,
        mode: str = MODO_PAPER,
    ) -> Tuple[PortfolioManager, OrderManager]:
        """
        Inicializa el sistema completo.

        La inicialización de ejecución queda limitada a PAPER.
        """
        self._validar_modo(mode)
        self._forzar_modo_paper()

        logger.info("Iniciando HRM en modo PAPER")
        logger.info("SESSION_IS_FRESH: %s", is_session_fresh())

        try:
            await self._perform_system_cleanup()

            env_config = await self._load_configuration(
                self.MODO_PAPER
            )

            self.state = await self._initialize_state(
                env_config
            )

            await self._initialize_core_components(
                env_config
            )

            await self._initialize_l1_components(
                env_config
            )

            await self._initialize_l2_components(
                env_config
            )

            if not APAGAR_L3:
                await self._initialize_l3_components()

            self.portfolio_manager = (
                await self._initialize_portfolio_manager(
                    env_config
                )
            )

            self.order_manager = (
                await self._initialize_order_manager()
            )

            await self._initialize_execution_gateway()

            await self._initialize_runtime_loop()

            await self._start_background_services()

            mark_session_used()

            logger.info(
                "HRM inicializado correctamente en PAPER"
            )

            return (
                self.portfolio_manager,
                self.order_manager,
            )

        except Exception:
            logger.exception(
                "Error durante la inicialización de HRM"
            )
            await self._cleanup_on_failure()
            raise

    async def _perform_system_cleanup(self) -> None:
        """Ejecuta la limpieza previa al arranque."""
        logger.info("Ejecutando limpieza del sistema")

        resultado = perform_full_cleanup(
            mode=self.MODO_PAPER
        )

        if not resultado.get("success", False):
            logger.warning(
                "La limpieza terminó con advertencias"
            )
            return

        logger.info(
            "Limpieza completada: %s archivos y %s directorios",
            resultado.get("deleted_files", 0),
            resultado.get("deleted_dirs", 0),
        )

    async def _load_configuration(
        self,
        mode: str,
    ) -> Dict[str, Any]:
        """Carga la configuración manteniendo PAPER forzado."""
        self._forzar_modo_paper()

        load_dotenv()

        # Se vuelven a fijar después de cargar el entorno.
        self._forzar_modo_paper()

        logger.info("Cargando configuración PAPER")

        env_config = get_config(self.MODO_PAPER)

        if not isinstance(env_config, dict):
            raise RuntimeError(
                "La configuración de HRM no es válida"
            )

        return env_config

    async def _initialize_state(
        self,
        env_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inicializa y valida el estado general."""
        logger.info("Inicializando estado")

        symbols = env_config.get(
            "SYMBOLS",
            ["BTCUSDT", "ETHUSDT"],
        )

        initial_balance = get_capital_usd()

        if initial_balance is None or initial_balance <= 0:
            raise RuntimeError(
                "El capital inicial PAPER no es válido"
            )

        state = initialize_state(
            symbols,
            initial_balance,
        )

        state = validate_state_structure(state)

        logger.info(
            "Estado inicializado. Símbolos: %s. Capital: %s USDT",
            symbols,
            initial_balance,
        )

        return state

    async def _initialize_core_components(
        self,
        env_config: Dict[str, Any],
    ) -> None:
        """Inicializa los componentes centrales."""
        logger.info("Inicializando componentes centrales")

        self.components["trading_metrics"] = (
            get_trading_metrics()
        )

        self.components["signal_verifier"] = (
            get_signal_verifier()
        )

        self.components["message_bus"] = MessageBus()

        self.components["sentiment_manager"] = (
            await self._initialize_sentiment_manager()
        )

    async def _initialize_l1_components(
        self,
        env_config: Dict[str, Any],
    ) -> None:
        """
        Inicializa los componentes operativos.

        BinanceClient se conserva para los datos de mercado.
        La ejecución de órdenes utiliza exclusivamente
        SimulatedExchangeClient.
        """
        logger.info("Inicializando componentes L1")

        self._forzar_modo_paper()

        # Cliente de datos de mercado.
        binance_client = BinanceClient(
            mode=self.MODO_PAPER
        )

        self.components["binance_client"] = (
            binance_client
        )

        # Exchange simulado para las operaciones PAPER.
        initial_balances = get_initial_balances()

        if not isinstance(initial_balances, dict):
            raise RuntimeError(
                "Los balances iniciales PAPER no son válidos"
            )

        if not initial_balances:
            raise RuntimeError(
                "No existen balances iniciales PAPER"
            )

        SimulatedExchangeClient._instance = None
        SimulatedExchangeClient._initialized = False

        simulated_client = SimulatedExchangeClient(
            initial_balances
        )

        balances = simulated_client.get_balances()

        if not balances:
            raise RuntimeError(
                "El exchange simulado no tiene balances"
            )

        if all(float(value) == 0 for value in balances.values()):
            raise RuntimeError(
                "El exchange simulado tiene todos los balances a cero"
            )

        self.components["simulated_client"] = (
            simulated_client
        )

        logger.info(
            "Exchange PAPER inicializado. Estado: %s",
            balances,
        )

        # Cargador de datos de mercado.
        loader = RealTimeDataLoader(config)

        self.components["data_loader"] = loader
        self.components["realtime_loader"] = loader

        from l1_operational.trend_ai import models as l1_models

        logger.info(
            "Modelos L1 cargados: %s",
            list(l1_models.keys()),
        )

    async def _initialize_l2_components(
        self,
        env_config: Dict[str, Any],
    ) -> None:
        """Inicializa los componentes tácticos."""
        logger.info("Inicializando componentes L2")

        os.environ["DISABLE_BTC_ETH_SYNC"] = "true"

        l2_config = L2Config()

        self.components["l2_config"] = l2_config

        l2_processor = L2TacticProcessor(
            l2_config,
            portfolio_manager=None,
            apagar_l3=APAGAR_L3,
        )

        self.components["l2_processor"] = (
            l2_processor
        )

        self.components["risk_manager"] = (
            RiskControlManager(l2_config)
        )

    async def _initialize_l3_components(self) -> None:
        """Inicializa los componentes estratégicos."""
        logger.info("Inicializando componentes L3")

        from l3_strategy.regime_classifier import (
            MarketRegimeClassifier,
        )
        from l3_strategy.decision_maker import make_decision

        self.components["regime_classifier"] = (
            MarketRegimeClassifier()
        )

        self.components["l3_decision_maker"] = (
            make_decision
        )

    async def _initialize_portfolio_manager(
        self,
        env_config: Dict[str, Any],
    ) -> PortfolioManager:
        """Inicializa la cartera exclusivamente en PAPER."""
        logger.info("Inicializando PortfolioManager PAPER")

        simulated_client = self.components.get(
            "simulated_client"
        )

        if simulated_client is None:
            raise RuntimeError(
                "No existe el exchange simulado"
            )

        initial_balance = get_capital_usd()

        if initial_balance is None or initial_balance <= 0:
            raise RuntimeError(
                "El capital inicial de la cartera no es válido"
            )

        portfolio_manager = PortfolioManager(
            mode="simulated",
            initial_balance=initial_balance,
            exchange_client=simulated_client,
            symbols=env_config.get(
                "SYMBOLS",
                ["BTCUSDT", "ETHUSDT"],
            ),
            enable_commissions=env_config.get(
                "ENABLE_COMMISSIONS",
                True,
            ),
            enable_slippage=env_config.get(
                "ENABLE_SLIPPAGE",
                True,
            ),
        )

        await portfolio_manager.initialize_async()

        logger.info(
            "PortfolioManager inicializado en PAPER"
        )

        return portfolio_manager

    async def _initialize_order_manager(self) -> OrderManager:
        """Inicializa el gestor de órdenes en PAPER."""
        logger.info("Inicializando OrderManager PAPER")

        simulated_client = self.components.get(
            "simulated_client"
        )

        if simulated_client is None:
            raise RuntimeError(
                "No existe el exchange simulado"
            )

        order_manager = OrderManager(
            state_manager=self.state,
            portfolio_manager=self.portfolio_manager,
            mode="simulated",
            simulated_client=simulated_client,
        )

        logger.info("OrderManager PAPER inicializado")

        return order_manager

    async def _initialize_execution_gateway(self) -> None:
        """Crea la puerta central de ejecución de órdenes."""
        from execution_gateway import ExecutionGateway

        if self.portfolio_manager is None:
            raise RuntimeError(
                "PortfolioManager no está inicializado"
            )

        if self.order_manager is None:
            raise RuntimeError(
                "OrderManager no está inicializado"
            )

        self.execution_gateway = ExecutionGateway(
            self.portfolio_manager,
            self.order_manager,
        )

        self.components["execution_gateway"] = (
            self.execution_gateway
        )

        register_component_for_shutdown(
            "execution_gateway",
            self.execution_gateway,
        )

        logger.info(
            "ExecutionGateway inicializado en PAPER"
        )

    async def _initialize_runtime_loop(self) -> None:
        """Inicializa el runtime y exige un Gateway operativo."""

        from runtime_loop import HRMRuntimeLoop

        if self.portfolio_manager is None:
            raise RuntimeError(
                "No se puede iniciar RuntimeLoop sin PortfolioManager"
            )

        if self.order_manager is None:
            raise RuntimeError(
                "No se puede iniciar RuntimeLoop sin OrderManager"
            )

        if self.execution_gateway is None:
            raise RuntimeError(
                "No se puede iniciar RuntimeLoop sin ExecutionGateway"
            )

        if not callable(
            getattr(self.execution_gateway, "execute_orders", None)
        ):
            raise RuntimeError(
                "ExecutionGateway no implementa execute_orders"
            )

        runtime_loop = HRMRuntimeLoop(
            self.portfolio_manager,
            self.order_manager,
        )

        runtime_loop.execution_gateway = self.execution_gateway

        if runtime_loop.execution_gateway is not self.execution_gateway:
            raise RuntimeError(
                "No se pudo inyectar ExecutionGateway en RuntimeLoop"
            )

        self.runtime_loop = runtime_loop

        logger.info(
            "RuntimeLoop inicializado correctamente con ExecutionGateway PAPER"
        )
    async def _initialize_sentiment_manager(self) -> Dict[str, Any]:
        """Inicializa el gestor de análisis de sentimiento."""
        logger.info("Inicializando análisis de sentimiento")

        sentiment_texts_cache = []

        try:
            sentiment_texts_cache = (
                await update_sentiment_texts()
            )

            logger.info(
                "Análisis de sentimiento inicializado: %s textos",
                len(sentiment_texts_cache),
            )

        except Exception:
            logger.exception(
                "No se pudo inicializar el análisis de sentimiento"
            )

        return {
            "texts_cache": sentiment_texts_cache,
            "last_update": 0,
        }

    async def _start_background_services(self) -> None:
        """Inicia los servicios en segundo plano."""
        logger.info("Iniciando servicios en segundo plano")

        await start_signal_verification()

        logger.info("Verificación de señales iniciada")

    async def _cleanup_on_failure(self) -> None:
        """Libera los componentes si falla el arranque."""
        logger.info("Limpiando componentes tras fallo")

        try:
            await stop_signal_verification()

        except Exception:
            logger.exception(
                "Error deteniendo la verificación de señales"
            )

        for component in reversed(
            list(self.components.values())
        ):
            metodo_cierre = getattr(
                component,
                "close",
                None,
            )

            if not callable(metodo_cierre):
                continue

            try:
                resultado = metodo_cierre()

                if asyncio.iscoroutine(resultado):
                    await resultado

            except Exception:
                logger.exception(
                    "Error cerrando componente: %s",
                    type(component).__name__,
                )

    def get_runtime_loop(self):
        """Devuelve el bucle de ejecución."""
        return self.runtime_loop

    def get_state(self) -> Dict[str, Any]:
        """Devuelve el estado del sistema."""
        return self.state


async def bootstrap_hrm_system(mode: str = "paper"):
    """Inicializa HRM y devuelve sus componentes principales."""
    bootstrap = HRMBootstrap()

    portfolio_manager, order_manager = (
        await bootstrap.bootstrap_system(mode)
    )

    runtime_loop = bootstrap.get_runtime_loop()

    return (
        portfolio_manager,
        order_manager,
        runtime_loop,
    )


# =============================================================================
# SHUTDOWN GLOBAL
# =============================================================================

_shutdown_components = {
    "market_data_manager": None,
    "realtime_loader": None,
    "exchange_client": None,
    "binance_client": None,
    "data_feed": None,
    "portfolio_manager": None,
    "order_manager": None,
    "order_flow_capture": None,
    "execution_gateway": None,
}


def register_component_for_shutdown(
    name: str,
    component: Any,
) -> None:
    """Registra un componente para el apagado global."""
    _shutdown_components[name] = component

    logger.debug(
        "Componente registrado para shutdown: %s",
        name,
    )


def unregister_component_for_shutdown(
    name: str,
) -> None:
    """Desregistra un componente del apagado global."""
    if name in _shutdown_components:
        _shutdown_components[name] = None

        logger.debug(
            "Componente desregistrado: %s",
            name,
        )


async def _cerrar_componente(
    nombre: str,
    componente: Any,
) -> None:
    """Cierra un componente si dispone de método de cierre."""
    if componente is None:
        return

    metodo_cierre = getattr(
        componente,
        "close",
        None,
    )

    if not callable(metodo_cierre):
        metodo_cierre = getattr(
            componente,
            "close_connection",
            None,
        )

    if not callable(metodo_cierre):
        return

    try:
        resultado = metodo_cierre()

        if asyncio.iscoroutine(resultado):
            await resultado

        logger.info("Componente cerrado: %s", nombre)

    except Exception:
        logger.exception(
            "Error cerrando componente: %s",
            nombre,
        )


async def shutdown() -> None:
    """Cierra los recursos globales de HRM."""
    logger.info("Iniciando shutdown global")

    orden_cierre = [
        "market_data_manager",
        "realtime_loader",
        "exchange_client",
        "binance_client",
        "data_feed",
        "execution_gateway",
        "order_flow_capture",
        "order_manager",
        "portfolio_manager",
    ]

    for nombre in orden_cierre:
        componente = _shutdown_components.get(nombre)

        if nombre == "portfolio_manager" and componente is not None:
            try:
                guardar = getattr(
                    componente,
                    "save_to_json",
                    None,
                )

                if callable(guardar):
                    guardar()

            except Exception:
                logger.exception(
                    "No se pudo guardar el estado de la cartera"
                )

        await _cerrar_componente(
            nombre,
            componente,
        )

    await _close_aiohttp_sessions()
    await _cancel_pending_tasks()

    logger.info("Shutdown global completado")


async def _close_aiohttp_sessions() -> None:
    """Cierra las sesiones aiohttp que continúen abiertas."""
    try:
        import aiohttp

        objetos = (
            list(globals().values())
            + list(_shutdown_components.values())
        )

        sesiones_cerradas = 0

        for objeto in objetos:
            if objeto is None:
                continue

            sesion = getattr(
                objeto,
                "session",
                None,
            )

            if isinstance(sesion, aiohttp.ClientSession):
                if not sesion.closed:
                    await sesion.close()
                    sesiones_cerradas += 1

            if isinstance(objeto, aiohttp.ClientSession):
                if not objeto.closed:
                    await objeto.close()
                    sesiones_cerradas += 1

        if sesiones_cerradas:
            logger.info(
                "Sesiones aiohttp cerradas: %s",
                sesiones_cerradas,
            )

    except Exception:
        logger.exception(
            "Error cerrando sesiones aiohttp"
        )


async def _cancel_pending_tasks() -> None:
    """Cancela las tareas pendientes excepto la tarea actual."""
    try:
        loop = asyncio.get_running_loop()
        tarea_actual = asyncio.current_task()

        tareas_pendientes = [
            tarea
            for tarea in asyncio.all_tasks(loop)
            if tarea is not tarea_actual and not tarea.done()
        ]

        if not tareas_pendientes:
            return

        logger.info(
            "Cancelando tareas pendientes: %s",
            len(tareas_pendientes),
        )

        for tarea in tareas_pendientes:
            tarea.cancel()

        resultados = await asyncio.gather(
            *tareas_pendientes,
            return_exceptions=True,
        )

        canceladas = sum(
            isinstance(
                resultado,
                asyncio.CancelledError,
            )
            for resultado in resultados
        )

        logger.info(
            "Tareas canceladas: %s",
            canceladas,
        )

    except Exception:
        logger.exception(
            "Error cancelando tareas pendientes"
        )


async def shutdown_with_signal_handling() -> None:
    """Ejecuta el apagado global."""
    logger.info("Shutdown con manejo de señales")
    await shutdown()


_shutdown_event = asyncio.Event()


def _signal_handler(signum, frame) -> None:
    """Activa el evento de apagado al recibir una señal."""
    import signal

    nombre_senal = signal.Signals(signum).name

    logger.info(
        "Señal recibida: %s",
        nombre_senal,
    )

    _shutdown_event.set()


def setup_signal_handlers() -> None:
    """Configura los manejadores SIGINT y SIGTERM."""
    import signal

    signal.signal(
        signal.SIGINT,
        _signal_handler,
    )

    signal.signal(
        signal.SIGTERM,
        _signal_handler,
    )

    logger.info(
        "Manejadores SIGINT y SIGTERM configurados"
    )


async def wait_for_shutdown_signal() -> None:
    """Espera hasta recibir una señal de apagado."""
    await _shutdown_event.wait()

    logger.info(
        "Señal de apagado recibida"
    )
