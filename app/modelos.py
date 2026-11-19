# app/modelos.py - Estructuras de datos del sistema

from dataclasses import dataclass

from core.portfolio_manager import PortfolioManager
from core.position_rotator import PositionRotator
from l1_operational.order_manager import OrderManager
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from system.error_recovery_manager import ErrorRecoveryManager
from system.market_data_manager import MarketDataManager
from system.state_coordinator import StateCoordinator
from system.trading_pipeline_manager import TradingPipelineManager


@dataclass
class Sistema:
    """Componentes vivos del sistema de trading."""
    state_coordinator: StateCoordinator
    portfolio_manager: PortfolioManager
    order_manager: OrderManager
    l2_processor: L2TacticProcessor
    market_data_manager: MarketDataManager
    trading_pipeline: TradingPipelineManager
    error_recovery: ErrorRecoveryManager
    position_rotator: PositionRotator
    estado: dict


@dataclass
class Acumulados:
    """Totales de todos los ciclos ejecutados."""
    senales: int = 0
    ordenes: int = 0
    rechazadas: int = 0

    def sumar(self, resultado):
        self.senales += resultado.signals_generated
        self.ordenes += resultado.orders_executed
        self.rechazadas += resultado.orders_rejected
