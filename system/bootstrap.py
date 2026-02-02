"""
Módulo de bootstrap del sistema HRM.

Encapsula toda la inicialización del sistema, incluyendo:
- Limpieza de sesiones anteriores
- Inicialización de componentes críticos
- Coordinación de estado del sistema
- Verificación de salud del sistema
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .models import (
    SystemContext, 
    CleanupResult, 
    ComponentRegistry, 
    HealthStatus
)

# Importaciones de módulos del sistema (a implementar)
try:
    from system.cleanup import SystemCleanup
    from system.state_coordinator import StateCoordinator
    from system.orchestrator import SystemOrchestrator
    from system.external_adapter import ExternalAdapter
    from system.component_extractor import ComponentExtractor
except ImportError as e:
    logging.warning(f"Importación parcial de módulos del sistema: {e}")


class SystemBootstrap:
    """Clase principal para la inicialización del sistema HRM.
    
    Encapsula todo el proceso de bootstrap del sistema, proporcionando
    una interfaz limpia para la inicialización completa del sistema.
    """
    
    def __init__(self):
        """Inicializa el bootstrap con configuración básica."""
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        
    def initialize_system(self) -> SystemContext:
        """Inicializa completamente el sistema HRM.
        
        Realiza todos los pasos necesarios para tener el sistema
        listo para operar, incluyendo limpieza, inicialización de
        componentes y verificación de salud.
        
        Returns:
            SystemContext: Contexto completo del sistema inicializado
            
        Raises:
            RuntimeError: Si la inicialización falla críticamente
        """
        self.logger.info("Iniciando proceso de bootstrap del sistema HRM")
        
        try:
            # Paso 1: Limpieza de sesión anterior
            cleanup_result = self.cleanup_previous_session()
            if not cleanup_result.success:
                self.logger.warning(f"Limpieza parcial exitosa: {cleanup_result.errors}")
            
            # Paso 2: Inicialización de componentes
            component_registry = self.initialize_components()
            
            # Paso 3: Verificación de salud del sistema
            health_status = self.verify_system_health(component_registry)
            
            # Paso 4: Construir contexto del sistema
            context = self._build_system_context(
                component_registry, 
                health_status, 
                cleanup_result
            )
            
            # Paso 5: Validar estado final
            if not context.is_ready:
                self.logger.error("Sistema no está listo para operar")
                context.errors.append("Sistema no está listo para operar")
            
            duration = time.time() - self.start_time
            self.logger.info(
                f"Bootstrap completado en {duration:.2f}s. "
                f"Estado: {health_status.value}, Componentes: {component_registry.registered_count}"
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error crítico en bootstrap: {e}", exc_info=True)
            return self._create_fallback_context(str(e))
    
    def cleanup_previous_session(self) -> CleanupResult:
        """Limpia residuos de sesiones anteriores.
        
        Realiza limpieza de:
        - Archivos temporales
        - Paper trades anteriores
        - Estado persistente inconsistente
        - Recursos del sistema
        
        Returns:
            CleanupResult: Resultado de la operación de limpieza
        """
        self.logger.info("Iniciando limpieza de sesión anterior")
        start_time = time.time()
        
        result = CleanupResult(success=True)
        
        try:
            # Inicializar limpiador del sistema
            system_cleanup = SystemCleanup()
            
            # Limpiar archivos temporales
            temp_files = system_cleanup.cleanup_temp_files()
            result.cleaned_files.extend(temp_files)
            
            # Limpiar paper trades
            paper_trades = system_cleanup.cleanup_paper_trades()
            result.cleaned_files.extend(paper_trades)
            
            # Limpiar estado persistente
            persistent_files = system_cleanup.cleanup_persistent_state()
            result.cleaned_files.extend(persistent_files)
            
            duration = (time.time() - start_time) * 1000
            result.duration_ms = duration
            
            self.logger.info(
                f"Limpieza completada: {len(result.cleaned_files)} archivos limpiados "
                f"en {duration:.2f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Error en limpieza de sesión: {e}")
            result.success = False
            result.errors.append(str(e))
        
        return result
    
    def initialize_components(self) -> ComponentRegistry:
        """Inicializa todos los componentes del sistema.
        
        Procesos incluidos:
        - StateCoordinator
        - SystemOrchestrator  
        - ExternalAdapter
        - ComponentExtractor y registro
        
        Returns:
            ComponentRegistry: Registro de componentes inicializados
        """
        self.logger.info("Iniciando inicialización de componentes")
        registry = ComponentRegistry()
        
        try:
            # Paso 1: Inicializar StateCoordinator
            self.logger.debug("Inicializando StateCoordinator")
            # CRITICAL: StateCoordinator debe ser instanciado solo en bootstrap
            state_coordinator = StateCoordinator()
            registry.components['state_coordinator'] = state_coordinator
            registry.registered_count += 1
            
            # Paso 2: Inicializar SystemOrchestrator
            self.logger.debug("Inicializando SystemOrchestrator")
            orchestrator = SystemOrchestrator(state_coordinator)
            registry.components['orchestrator'] = orchestrator
            registry.registered_count += 1
            
            # Paso 3: Inicializar ExternalAdapter
            self.logger.debug("Inicializando ExternalAdapter")
            external_adapter = ExternalAdapter()
            registry.components['external_adapter'] = external_adapter
            registry.registered_count += 1
            
            # Paso 4: Extraer y registrar componentes adicionales
            self.logger.debug("Extrayendo componentes adicionales")
            extractor = ComponentExtractor()
            additional_components = extractor.extract_components()
            
            for name, component in additional_components.items():
                registry.components[name] = component
                registry.registered_count += 1
            
            self.logger.info(
                f"Componentes inicializados: {registry.registered_count}"
            )
            
        except Exception as e:
            self.logger.error(f"Error en inicialización de componentes: {e}")
            registry.errors.append(str(e))
            registry.success = False
        
        return registry
    
    def verify_system_health(self, registry: ComponentRegistry) -> HealthStatus:
        """Verifica la salud del sistema después de la inicialización.
        
        Realiza comprobaciones de:
        - Componentes críticos disponibles
        - Conexiones externas
        - Estado del coordinador
        - Recursos del sistema
        
        Args:
            registry: Registro de componentes inicializados
            
        Returns:
            HealthStatus: Estado de salud del sistema
        """
        self.logger.info("Verificando salud del sistema")
        
        try:
            # Verificar componentes críticos
            critical_components = ['state_coordinator', 'external_adapter']
            missing_components = [
                comp for comp in critical_components 
                if comp not in registry.components
            ]
            
            if missing_components:
                self.logger.error(f"Componentes críticos faltantes: {missing_components}")
                return HealthStatus.UNHEALTHY
            
            # Verificar conexiones externas
            external_adapter = registry.components.get('external_adapter')
            if external_adapter and not external_adapter.is_connected():
                self.logger.warning("Conexión externa no disponible")
                return HealthStatus.DEGRADED
            
            # Verificar estado del coordinador
            state_coordinator = registry.components.get('state_coordinator')
            if state_coordinator and not state_coordinator.is_healthy():
                self.logger.error("StateCoordinator no está saludable")
                return HealthStatus.UNHEALTHY
            
            # Verificar errores en el registro
            if registry.errors:
                self.logger.warning(f"Errores en componentes: {registry.errors}")
                return HealthStatus.DEGRADED
            
            self.logger.info("Sistema verificado como saludable")
            return HealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Error en verificación de salud: {e}")
            return HealthStatus.UNHEALTHY
    
    def _build_system_context(
        self, 
        registry: ComponentRegistry, 
        health_status: HealthStatus,
        cleanup_result: CleanupResult
    ) -> SystemContext:
        """Construye el contexto del sistema a partir de los resultados.
        
        Args:
            registry: Registro de componentes
            health_status: Estado de salud del sistema
            cleanup_result: Resultado de la limpieza
            
        Returns:
            SystemContext: Contexto completo del sistema
        """
        duration = time.time() - self.start_time
        
        return SystemContext(
            state_coordinator=registry.components.get('state_coordinator'),
            components=registry.components,
            external_adapter=registry.components.get('external_adapter'),
            health_status=health_status,
            initialization_time=duration,
            errors=cleanup_result.errors + registry.errors
        )
    
    def _create_fallback_context(self, error_msg: str) -> SystemContext:
        """Crea un contexto de fallback en caso de fallo crítico.
        
        Args:
            error_msg: Mensaje de error para el contexto
            
        Returns:
            SystemContext: Contexto con estado degradado
        """
        self.logger.warning("Creando contexto de fallback por fallo crítico")
        
        return SystemContext(
            health_status=HealthStatus.UNHEALTHY,
            errors=[error_msg],
            initialization_time=time.time() - self.start_time
        )


# Función de conveniencia para inicialización rápida
def bootstrap_system() -> SystemContext:
    """Función de conveniencia para inicializar el sistema.
    
    Returns:
        SystemContext: Contexto del sistema inicializado
    """
    bootstrap = SystemBootstrap()
    return bootstrap.initialize_system()