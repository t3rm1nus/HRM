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

    # ✅ FIXED: Establecer balances iniciales correctos (USDT=3000, BTC=0, ETH=0)
    # Primero intentar leer de config, si no existe usar valores por defecto
    initial_balances = config.get("simulated_initial_balances", {})
    
    # Si no hay balances configurados o están vacíos, usar valores por defecto
    if not initial_balances:
        initial_balances = {
            "BTC": 0.0,
            "ETH": 0.0,
            "USDT": 3000.0
        }
        log.info("🎯 Using default initial balances: USDT=3000, BTC=0, ETH=0")
    else:
        # Asegurar que todos los activos estén presentes
        initial_balances.setdefault("BTC", 0.0)
        initial_balances.setdefault("ETH", 0.0)
        initial_balances.setdefault("USDT", 3000.0)
        log.info(f"🎯 Using configured initial balances: {initial_balances}")

# 🔄 CRITICAL FIX: NEVER reset singleton - maintain state across cycles
    # The singleton pattern ensures state persistence in paper trading mode
    # Removing the reset prevents loss of balances after trades
    
    # Always use get_instance() to ensure singleton pattern
    client = SimulatedExchangeClient.get_instance(initial_balances=initial_balances)

    log.info("=" * 70)
    log.info("🎮 SIMULATED EXCHANGE CLIENT INITIALIZED")
    log.info(f"   SIM_STATE_ID: {id(client)}")
    log.info(f"   BTC:  {client.balances.get('BTC', 0):.6f}")
    log.info(f"   ETH:  {client.balances.get('ETH', 0):.4f}")
    log.info(f"   USDT: ${client.balances.get('USDT', 0):.2f}")
    log.info("=" * 70)

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