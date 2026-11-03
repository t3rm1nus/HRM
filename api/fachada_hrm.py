# Intento de importación de componentes internos de HRM
# Si fallan, los componentes quedarán como None y los métodos manejarán esto

try:
    from system.state_coordinator import StateCoordinator
    STATE_COORDINATOR_AVAILABLE = True
except ImportError:
    StateCoordinator = None
    STATE_COORDINATOR_AVAILABLE = False

try:
    from l1_operational.order_manager import OrderManager
    ORDER_MANAGER_AVAILABLE = True
except ImportError:
    OrderManager = None
    ORDER_MANAGER_AVAILABLE = False

try:
    from l1_operational.binance_client import MarketDataManager
    MARKET_DATA_MANAGER_AVAILABLE = True
except ImportError:
    MarketDataManager = None
    MARKET_DATA_MANAGER_AVAILABLE = False

try:
    from auto_learning_system import SelfImprovingTradingSystem
    SELF_IMPROVING_TRADING_SYSTEM_AVAILABLE = True
except ImportError:
    SelfImprovingTradingSystem = None
    SELF_IMPROVING_TRADING_SYSTEM_AVAILABLE = False

try:
    from core.trading_metrics import get_trading_metrics, TradingMetrics
    TRADING_METRICS_AVAILABLE = True
except ImportError:
    TradingMetrics = None
    TRADING_METRICS_AVAILABLE = False

try:
    from core.cycle_metrics import get_cycle_metrics, CycleMetrics
    CYCLE_METRICS_AVAILABLE = True
except ImportError:
    CycleMetrics = None
    CYCLE_METRICS_AVAILABLE = False

try:
    from system.error_recovery_manager import ErrorRecoveryManager
    ERROR_RECOVERY_MANAGER_AVAILABLE = True
except ImportError:
    ErrorRecoveryManager = None
    ERROR_RECOVERY_MANAGER_AVAILABLE = False


# Clase interna para estado "no disponible"
class _EstadoNoDisponible:
    """Clase interna para representar estado 'no disponible'."""
    
    def __init__(self, nombre: str):
        self.nombre = nombre
        self._disponible = False
    
    def __bool__(self):
        return self._disponible
    
    def __repr__(self):
        return f"<EstadoNoDisponible:{self.nombre}>"


def _marcar_propiedades(
    datos: Dict[str, Any],
    balance_check: Optional[Any] = None
) -> Tuple[Dict[str, Any], bool, bool]:
    """
    Marca es_respaldo y es_sintetico en el diccionario de datos.
    
    - es_respaldo: True si balance_check y balance_check.was_fallback_used es True
    - es_sintetico: True si los datos L3 tienen régimen 'unknown_no_sync' o datos sintéticos
    """
    es_respaldo = False
    es_sintetico = False
    
    if balance_check is not None:
        try:
            was_fallback = getattr(balance_check, 'was_fallback_used', False)
            if was_fallback:
                es_respaldo = True
        except Exception:
            pass
    
    # Verificar es_sintetico: régimen unknown_no_sync o datos sintéticos
    if datos is not None:
        regime = datos.get('regime') if isinstance(datos, dict) else None
        if regime == 'unknown_no_sync':
            es_sintetico = True
        # También verificar si vienen datos sintéticos marcados
        if isinstance(datos, dict) and datos.get('fuente') == 'synthetic':
            es_sintetico = True
    
    return datos, es_respaldo, es_sintetico


class FachadaHRM:
    """Fachada de solo lectura a componentes internos de HRM."""
    
    @staticmethod
    def _obtener_si_disponible(
        componente: Any,
        metodo_nombre: str,
        fallback: Any
    ) -> Any:
        """Obtener valor de un componente si está disponible, sino fallback."""
        if componente is not None:
            try:
                metodo = getattr(componente, metodo_nombre, None)
                if metodo is not None and callable(metodo):
                    return metodo()
            except Exception:
                pass
        return fallback
    
    @classmethod
    def _es_singleton_inicializado(cls, componente: Any) -> bool:
        """Verificar si un singleton está inicializado."""
        if componente is None:
            return False
        try:
            return getattr(componente, '_initialized', False)
        except Exception:
            return False
    
    @classmethod
    def _get_cycle_metrics_instance(cls) -> Any:
        if not CYCLE_METRICS_AVAILABLE:
            return None
        try:
            return get_cycle_metrics()
        except Exception:
            return None