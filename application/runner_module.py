# Application runner module - orchestration logic extracted from main.py

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.logging import logger
from system.bootstrap import SystemBootstrap, shutdown
from system.state_coordinator import StateCoordinator
from system.error_recovery_manager import ErrorRecoveryManager, RecoveryActionType
from system.market_data_manager import MarketDataManager
from l1_operational.order_flow_capture import create_order_flow_capture
from l1_operational.orderflow_signal_generator import create_order_flow_signal_generator
from l1_operational.data_feed import DataFeed
from l2_tactic.tactical_signal_processor import L2TacticProcessor
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l1_operational.realtime_loader import RealTimeDataLoader
from l1_operational.bus_adapter import BusAdapterAsync
from auto_learning.integration import integrate_with_main_system
from core.incremental_signal_verifier import stop_signal_verification

logger = logging.getLogger(__name__)


class HRMRunner:
    """Orquestador del sistema de trading HRM.
    
    Extrae la lógica de orquestación de main.py manteniendo
    todo el comportamiento funcional intacto.
    """
    
    def __init__(self, paper_mode: Optional[bool] = None) -> None:
        self.paper_mode = paper_mode
        self.bootstrap: Optional[SystemBootstrap] = None
        self.state_coordinator: Optional[StateCoordinator] = None
        self.error_recovery: Optional[ErrorRecoveryManager] = None
        self.market_data_manager: Optional[MarketDataManager] = None
        self.order_flow_capture: Optional[object] = None
        self.order_flow_signal_generator: Optional[object] = None
        self.portfolio_manager = None
        self.shutdown_event = asyncio.Event()
        self.config = get_config()
        self._arranque_rechazado = False
        
    async def initialize(self) -> None:
        """Inicializar todos los componentes del sistema."""
        logger.info("🚀 Inicializando HRM Trading System...")
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        self.bootstrap = SystemBootstrap(paper_mode=self.paper_mode)
        await self.bootstrap.initialize()
        
        self.state_coordinator = StateCoordinator()
        await self.state_coordinator.start()
        
        self.error_recovery = ErrorRecoveryManager()
        
        self.market_data_manager = MarketDataManager()
        await self.market_data_manager.start()
        
        self.order_flow_capture = create_order_flow_capture()
        await self.order_flow_capture.start()
        
        self.order_flow_signal_generator = create_order_flow_signal_generator()
        await self.order_flow_signal_generator.start()
        
        data_feed = DataFeed()
        await data_feed.start()
        
        l2_processor = L2TacticProcessor()
        await l2_processor.start()
        
        self.order_manager = OrderManager()
        
        self.binance_client = BinanceClient(paper_mode=self.paper_mode)
        await self.binance_client.start()
        
        self.realtime_loader = RealTimeDataLoader()
        await self.realtime_loader.start()
        
        self.bus_adapter = BusAdapterAsync()
        await self.bus_adapter.start()
        
        integrate_with_main_system()
        
        logger.info("✅ HRM Trading System initialized successfully")
    
    async def run(self) -> None:
        """Ejecutar el sistema completo."""
        await self.initialize()
        await self.shutdown()
    
    async def shutdown(self) -> None:
        """Detener el sistema de manera limpia."""
        logger.info("🛑 Initiating graceful shutdown...")
        self.shutdown_event.set()
        
        try:
            await asyncio.sleep(0)
        except Exception:
            pass
        
        try:
            await shutdown()
            logger.info("✅ Shutdown global completado exitosamente")
        except Exception as shutdown_error:
            logger.error(f"❌ Error en shutdown global: {shutdown_error}")
        
        if self.state_coordinator is not None:
            from api.servicios.servicio_estado_vivo import servicio_estado_vivo
            servicio_estado_vivo.confirmar_parada()
        
        logger.info("👋 HRM System shutdown complete")


def main_entry_point() -> None:
    """Punto de entrada principal cuando se ejecuta main.py directamente."""
    import asyncio
    
    import os as _os
    paper_mode_str = _os.getenv("HRM_PAPER_MODE")
    paper_mode = None
    if paper_mode_str is not None:
        paper_mode = paper_mode_str.lower() in ("true", "1", "paper")
    
    logger.info("🚀 Iniciando HRM Trading System con API FastAPI integrada...")
    logger.info("=" * 60)
    
    asyncio.run(run_hmr_system(paper_mode=paper_mode))


async def run_hmr_system(paper_mode: Optional[bool] = None) -> None:
    """Función principal de entrada para ejecutar el sistema HRM.
    
    Args:
        paper_mode: Si es True, ejecuta en modo paper trading.
                    None usa la configuración del entorno.
    """
    runner = HRMRunner(paper_mode=paper_mode)
    
    try:
        await runner.initialize()
        
        # El sistema de trading corre hasta que se cierra
        # La API FastAPI se maneja en el punto de entrada de main
        await runner.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"🚨 Fatal error: {e}")
        import traceback
        logger.critical(traceback.format_exc())
    finally:
        await runner.shutdown()