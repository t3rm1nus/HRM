from l1_operational.simulated_exchange_client import SimulatedExchangeClient
import logging

log = logging.getLogger(__name__)


def bootstrap_simulated_exchange(config, force_paper: bool = True):
    """
    Bootstrap SimulatedExchangeClient con soporte cleanup.
    
    Args:
        config: Configuración del sistema
        force_paper: Si True, fuerza modo paper ignorando config
    
    Returns:
        SimulatedExchangeClient instance
    """
    # Check if we should use paper mode
    use_paper = force_paper or config.get("paper_mode", False)
    
    if not use_paper:
        log.info("Paper mode disabled - not initializing SimulatedExchangeClient")
        return None

    initial_balances = config.get("simulated_initial_balances", {})

    if not initial_balances:
        log.critical("SIM_BALANCES vacío en bootstrap")
        raise RuntimeError("Paper mode requires initial balances")

    # 🔄 IMPORTANTE: Asegurar que singleton está limpo antes de inicializar
    # Esto permite re-inicialización tras cleanup
    SimulatedExchangeClient._instance = None
    SimulatedExchangeClient._initialized = False

    client = SimulatedExchangeClient.initialize_once(
        initial_balances=initial_balances
    )

    log.info("SIM_INIT_ONCE=True")
    log.info(f"SIM_STATE_ID: {id(client)}")
    log.info(f"SIM_BALANCES: {client.balances}")

    return client


class SystemBootstrap:
    """
    Clase para la inicialización centralizada del sistema HRM.
    Proporciona métodos para configurar y arrancar todos los componentes del sistema.
    """
    
    def __init__(self):
        """
        Inicializa el bootstrap del sistema.
        """
        self.components = {}
        self.external_adapter = None
        log.info("SystemBootstrap initialized")
    
    def initialize_system(self):
        """
        Inicializa todo el sistema HRM.
        
        Returns:
            object: Contexto del sistema con componentes inicializados
        """
        log.info("Initializing HRM system components...")
        
        # Aquí se puede agregar la lógica de inicialización de componentes
        # como PortfolioManager, OrderManager, L2TacticProcessor, etc.
        
        # Crear un objeto de contexto simple para compatibilidad
        class SystemContext:
            def __init__(self, components, external_adapter):
                self.components = components
                self.external_adapter = external_adapter
        
        system_context = SystemContext(self.components, self.external_adapter)
        log.info("SystemBootstrap completed successfully")
        
        return system_context