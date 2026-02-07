"""
Coordinador de Estado del Sistema HRM

Gestiona el estado global del sistema y coordina entre componentes.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """Representación del estado del sistema"""
    is_ready: bool = False
    health_status: str = "unknown"
    components: Dict[str, Any] = None
    errors: List[str] = None
    timestamp: float = None

class StateCoordinator:
    """Coordinador de estado del sistema HRM"""

    def __init__(self):
        """Inicializa el coordinador de estado"""
        self.logger = logger
        self.system_state = SystemState(
            is_ready=False,
            health_status="initializing",
            components={},
            errors=[],
            timestamp=None
        )
        self.logger.info("StateCoordinator inicializado")

    def update_state(self, component_name: str, status: str, data: Optional[Dict[str, Any]] = None):
        """Actualiza el estado de un componente específico

        Args:
            component_name: Nombre del componente
            status: Estado del componente (healthy, degraded, unhealthy)
            data: Datos adicionales del componente
        """
        self.logger.debug(f"Actualizando estado de {component_name}: {status}")
        self.system_state.components[component_name] = {
            "status": status,
            "data": data,
            "timestamp": time.time()
        }

    def get_system_state(self) -> SystemState:
        """Obtiene el estado actual del sistema

        Returns:
            SystemState: Estado actual del sistema
        """
        return self.system_state
    
    def get_state(self, state_name: str) -> Any:
        """Obtiene un estado específico del sistema (método compatible para retrocompatibilidad)

        Args:
            state_name: Nombre del estado a obtener

        Returns:
            Any: Estado solicitado
        """
        if state_name == "current":
            # Devolver un diccionario compatible con el código existente
            return {
                "market_data": getattr(self, "_market_data", {}),
                "total_value": getattr(self, "_total_value", 0.0),
                "l3_output": getattr(self, "_l3_output", {
                    'regime': 'neutral',
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strategy_type': 'initial',
                    'timestamp': time.time()
                })
            }
        else:
            self.logger.warning(f"Estado desconocido: {state_name}")
            return None
            
    def update_market_data(self, market_data: Dict[str, Any]):
        """Actualiza los datos de mercado en el estado global"""
        if market_data and isinstance(market_data, dict) and len(market_data) > 0:
            self._market_data = market_data
            self.logger.debug(f"Market data updated: {len(market_data)} symbols")
            
    def update_total_value(self, total_value: float):
        """Actualiza el valor total del portfolio en el estado global"""
        if isinstance(total_value, (int, float)) and total_value >= 0:
            self._total_value = total_value
            self.logger.debug(f"Total value updated: ${total_value:.2f}")
            
    def update_portfolio_state(self, portfolio_state: Dict[str, Any]):
        """Actualiza el estado del portfolio en el estado global (single source of truth)"""
        if portfolio_state and isinstance(portfolio_state, dict):
            self._portfolio_state = portfolio_state
            self.logger.debug(f"Portfolio state updated: {portfolio_state}")
    
    def update_l3_output(self, l3_output: Dict[str, Any]):
        """Actualiza la salida de L3 en el estado global"""
        if l3_output and isinstance(l3_output, dict) and 'regime' in l3_output and 'signal' in l3_output:
            self._l3_output = l3_output
            self.logger.debug(f"L3 output updated: {l3_output.get('regime')} - {l3_output.get('signal')}")
            
    def get_state(self, state_name: str) -> Any:
        """Obtiene un estado específico del sistema (método compatible para retrocompatibilidad)

        Args:
            state_name: Nombre del estado a obtener

        Returns:
            Any: Estado solicitado
        """
        if state_name == "current":
            # Devolver un diccionario compatible con el código existente
            state = {
                "market_data": getattr(self, "_market_data", {}),
                "total_value": getattr(self, "_total_value", 0.0),
                "l3_output": getattr(self, "_l3_output", {
                    'regime': 'neutral',
                    'signal': 'hold',
                    'confidence': 0.5,
                    'strategy_type': 'initial',
                    'timestamp': time.time()
                })
            }
            
            # Add portfolio state if available
            if hasattr(self, '_portfolio_state'):
                state["portfolio"] = self._portfolio_state
                
            return state
        else:
            self.logger.warning(f"Estado desconocido: {state_name}")
            return None

    def is_healthy(self) -> bool:
        """Verifica si el sistema está saludable

        Returns:
            bool: True si el sistema está saludable, False si no
        """
        return self.system_state.health_status == "healthy"

    def mark_ready(self):
        """Marca el sistema como listo para operar"""
        self.system_state.is_ready = True
        self.system_state.health_status = "healthy"
        self.system_state.timestamp = time.time()
        self.logger.info("Sistema marcado como listo para operar")

    def record_error(self, error_msg: str):
        """Registra un error en el sistema

        Args:
            error_msg: Mensaje de error a registrar
        """
        self.logger.error(f"Error registrado: {error_msg}")
        self.system_state.errors.append({
            "message": error_msg,
            "timestamp": time.time()
        })