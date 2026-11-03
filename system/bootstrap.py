from l1_operational.simulated_exchange_client import SimulatedExchangeClient
import logging

log = logging.getLogger(__name__)


def bootstrap_simulated_exchange(config, force_paper: bool = True):
    """
    Bootstrap SimulatedExchangeClient con soporte cleanup.
    
    Detecta si es una instancia recién creada (sesión nueva) o un singleton
    existente (sesión continua) para decidir de dónde tomar el capital inicial.
    
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

    # Leer capital objetivo desde la configuración
    config_balances = config.get("simulated_initial_balances", {})
    
    # Si no hay balances configurados, usar INITIAL_BALANCE de config
    if not config_balances:
        initial_balance = config.get("INITIAL_BALANCE", 500.0)
        config_balances = {
            "BTC": 0.0,
            "ETH": 0.0,
            "USDT": initial_balance
        }
        log.info(f"🎯 Config no especifica simulated_initial_balances - usando INITIAL_BALANCE: {initial_balance} USDT")
    else:
        # Asegurar que todos los activos estén presentes
        config_balances.setdefault("BTC", 0.0)
        config_balances.setdefault("ETH", 0.0)
        # Validar que USDT venga de config, no hardcodeado
        if "USDT" not in config_balances:
            initial_balance = config.get("INITIAL_BALANCE", 500.0)
            config_balances["USDT"] = initial_balance
            log.warning(f"⚠️ Config no especifica USDT en simulated_initial_balances - usando INITIAL_BALANCE: {initial_balance}")
        log.info(f"🎯 Config especifica balances: {config_balances}")

    # 🔍 DETECTAR: ¿Instancia recién creada o sesión continua?
    singleton_existed_before = SimulatedExchangeClient._instance is not None
    was_just_initialized = SimulatedExchangeClient.is_just_initialized()
    
    log.info("=" * 70)
    log.info("🔍 CAPITAL SOURCE DECISION")
    log.info(f"   Singleton existed before: {singleton_existed_before}")
    log.info(f"   _just_initialized flag: {was_just_initialized}")

    # 🔄 CRITICAL FIX: Lógica de decisión del capital
    if singleton_existed_before and not was_just_initialized:
        # SESIÓN CONTINUA: Respetar el singleton existente
        log.info("📌 PATH TAKEN: SESIÓN CONTINUA")
        log.info("   Razón: Singleton ya existía y no es recién creado")
        log.info("   Acción: Usar capital del singleton (preservar estado)")
        
        existing_balances = SimulatedExchangeClient._instance.get_balances()
        existing_usdt = existing_balances.get('USDT', 0.0)
        config_usdt = config_balances.get('USDT', 0)
        
        if abs(existing_usdt - config_usdt) > 1.0:
            log.info(f"   ⚠️ CAPITAL MISMATCH detectado pero IGNORADO:")
            log.info(f"      - Singleton tiene: {existing_usdt} USDT")
            log.info(f"      - Config dice: {config_usdt} USDT")
            log.info(f"      - Se mantiene el capital del singleton para preservar sesión")
        
        # Usar balances existentes (ignorar config)
        initial_balances = existing_balances
        
    elif singleton_existed_before and was_just_initialized:
        # CASO BORDE: Singleton existe pero fue recién creado en este ciclo
        # Esto no debería pasar, pero lo manejamos por seguridad
        log.info("📌 PATH TAKEN: CASO BORDE (singleton existe + just_initialized)")
        log.info("   Razón: Inconsistencia de estado detectada")
        log.info("   Acción: Aplicar capital de config (instancia recién creada)")
        
        # 🔄 CRITICAL FIX: Realmente aplicar el capital de config al singleton existente
        config_usdt = config_balances.get('USDT', 500.0)
        existing_instance = SimulatedExchangeClient._instance
        
        if existing_instance is not None:
            # Aplicar el balance de config usando force_reset
            try:
                existing_instance.force_reset(config_balances.copy())
                log.info(f"✅ SimClient balance synchronized with config: USDT=${config_usdt:.2f}")
            except Exception as e:
                log.error(f"❌ Error aplicando capital de config: {e}")
                # Fallback: modificar directamente los balances
                existing_instance.balances = config_balances.copy()
                existing_instance.initial_balances = config_balances.copy()
                log.info(f"✅ SimClient balance synchronized (fallback): USDT=${config_usdt:.2f}")
        
        initial_balances = config_balances
        
    else:
        # SESIÓN NUEVA: No hay singleton, aplicar capital de config
        log.info("📌 PATH TAKEN: SESIÓN NUEVA")
        log.info("   Razón: No existe singleton (instancia recién creada)")
        log.info("   Acción: Aplicar capital desde config")
        initial_balances = config_balances

    # 🔄 Obtener instancia singleton (crea nueva si no existe)
    client = SimulatedExchangeClient.get_instance(initial_balances=initial_balances)
    
    # 🔚 Marcar que ya no es recién inicializada (primer ciclo completado)
    was_new_instance = SimulatedExchangeClient.mark_cycle_completed()
    
    # ✅ VERIFICACIÓN EXPLÍCITA: Comparar balance SimClient con config
    sim_balance = client.balances.get('USDT', 0)
    config_balance = config_balances.get('USDT', 0)
    balance_match = abs(sim_balance - config_balance) < 0.01  # Tolerancia de 1 centavo
    
    if balance_match:
        log.info(f"✅ SimClient balance matches config: {sim_balance:.2f} == {config_balance:.2f}")
    else:
        log.error(f"❌ MISMATCH: SimClient {sim_balance:.2f} != Config {config_balance:.2f}")
        log.error(f"   Esta discrepancia es CRÍTICA para trading en vivo")
    
    log.info("=" * 70)
    log.info("🎮 SIMULATED EXCHANGE CLIENT INITIALIZED")
    log.info(f"   SIM_STATE_ID: {id(client)}")
    log.info(f"   Was new instance: {was_new_instance}")
    log.info(f"   Capital source: {'CONFIG (nueva sesión)' if was_new_instance else 'SINGLETON (sesión continua)'}")
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