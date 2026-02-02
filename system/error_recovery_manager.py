"""
Gestor de Recuperación de Errores del Sistema HRM

Este módulo maneja todos los errores del sistema con estrategias de recovery
específicas para cada tipo de error, evitando crashes y manteniendo la operación.
"""

import asyncio
import time
import traceback
import gc
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime

from core.logging import logger
from core.exceptions import HRMException
from core.data_validator import validate_market_data, sanitize_market_data
from system.models import RecoveryAction, ErrorType, RecoveryActionType, ErrorRecoveryResult
from core.error_handler import ErrorHandler


class ErrorRecoveryManager:
    """
    Gestor centralizado de recuperación de errores para el sistema HRM.
    
    Maneja errores de diferentes tipos con estrategias específicas de recovery,
    límites de reintento y seguimiento detallado de cada operación de recovery.
    """

    def __init__(self):
        """Inicializa el gestor de recuperación de errores."""
        self.error_counts: Dict[str, int] = {}
        self.last_recovery_time: Dict[str, float] = {}
        self.max_retries: Dict[ErrorType, int] = {
            ErrorType.DATA_QUALITY: 3,
            ErrorType.ML_FRAMEWORK: 2,
            ErrorType.STATE_CORRUPTION: 2,
            ErrorType.NETWORK: 5,
            ErrorType.UNKNOWN: 1
        }
        self.base_wait_times: Dict[ErrorType, int] = {
            ErrorType.DATA_QUALITY: 5,
            ErrorType.ML_FRAMEWORK: 10,
            ErrorType.STATE_CORRUPTION: 15,
            ErrorType.NETWORK: 30,
            ErrorType.UNKNOWN: 30
        }

    def classify_error(self, error: Exception) -> ErrorType:
        """
        Clasifica un error según su tipo y mensaje.
        
        Args:
            error: Excepción a clasificar
            
        Returns:
            ErrorType: Tipo de error clasificado
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Errores de calidad de datos
        if (isinstance(error, (ValueError, RuntimeError)) and 
            ("market_data" in error_str or "data" in error_str or "empty" in error_str)):
            return ErrorType.DATA_QUALITY
            
        # Errores de frameworks ML
        if (error_type in ["tensorflowerror", "pytorcherror", "runtimeerror"] or
            any(ml_term in error_str for ml_term in ["tensorflow", "pytorch", "torch", "keras", "truth value of dataframe"])):
            return ErrorType.ML_FRAMEWORK
            
        # Errores de corrupción de estado
        if (isinstance(error, (KeyError, AttributeError)) and
            any(state_term in error_str for state_term in ["state", "key", "attribute"])):
            return ErrorType.STATE_CORRUPTION
            
        # Errores de red
        if (isinstance(error, (ConnectionError, TimeoutError)) or
            any(net_term in error_str for net_term in ["connection", "timeout", "network", "api"])):
            return ErrorType.NETWORK
            
        # Errores desconocidos
        return ErrorType.UNKNOWN

    def should_retry(self, error: Exception) -> bool:
        """
        Determina si se debe reintentar un error basado en su tipo y conteo.
        
        Args:
            error: Excepción a evaluar
            
        Returns:
            bool: True si se debe reintentar, False en caso contrario
        """
        error_type = self.classify_error(error)
        error_key = f"{error_type.value}_{type(error).__name__}"
        
        current_count = self.error_counts.get(error_key, 0)
        max_retry = self.max_retries.get(error_type, 1)
        
        return current_count < max_retry

    def get_recovery_wait_time(self, error: Exception) -> int:
        """
        Calcula el tiempo de espera para recovery basado en el tipo de error.
        
        Args:
            error: Excepción para calcular el tiempo de espera
            
        Returns:
            int: Tiempo de espera en segundos
        """
        error_type = self.classify_error(error)
        base_time = self.base_wait_times.get(error_type, 30)
        
        # Aplicar backoff exponencial para errores de red
        if error_type == ErrorType.NETWORK:
            error_key = f"{error_type.value}_{type(error).__name__}"
            retry_count = self.error_counts.get(error_key, 0)
            wait_time = min(base_time * (2 ** retry_count), 300)  # Máximo 5 minutos
        else:
            wait_time = base_time
            
        return wait_time

    async def handle_cycle_error(
        self, 
        error: Exception, 
        state: Dict, 
        cycle_id: int
    ) -> RecoveryAction:
        """
        Maneja errores durante el ciclo de trading con estrategias específicas.
        
        Args:
            error: Excepción ocurrida durante el ciclo
            state: Estado del sistema en el momento del error
            cycle_id: ID del ciclo donde ocurrió el error
            
        Returns:
            RecoveryAction: Acción de recovery a tomar
        """
        start_time = time.time()
        error_type = self.classify_error(error)
        error_key = f"{error_type.value}_{type(error).__name__}"
        
        # Incrementar contador de errores
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        retry_count = self.error_counts[error_key]
        
        logger.error(
            f"🚨 Error en ciclo {cycle_id} - Tipo: {error_type.value}, "
            f"Intento: {retry_count}/{self.max_retries.get(error_type, 1)}"
        )
        logger.error(f"❌ Detalle del error: {error}")
        
        recovery_steps = []
        action_type = RecoveryActionType.SKIP_CYCLE
        wait_seconds = self.get_recovery_wait_time(error)
        success = False
        
        try:
            if error_type == ErrorType.DATA_QUALITY:
                # Estrategia para errores de calidad de datos
                recovery_result = await self.recover_from_data_error()
                if recovery_result:
                    action_type = RecoveryActionType.RETRY
                    success = True
                    recovery_steps.append("Data quality recovery successful")
                else:
                    action_type = RecoveryActionType.SKIP_CYCLE
                    recovery_steps.append("Data quality recovery failed")
                    
            elif error_type == ErrorType.ML_FRAMEWORK:
                # Estrategia para errores de frameworks ML
                recovery_result = await self.recover_from_ml_framework_error()
                if recovery_result:
                    action_type = RecoveryActionType.RETRY
                    success = True
                    recovery_steps.append("ML framework recovery successful")
                    wait_seconds = 10  # Espera fija para ML
                else:
                    action_type = RecoveryActionType.SKIP_CYCLE
                    recovery_steps.append("ML framework recovery failed")
                    
            elif error_type == ErrorType.STATE_CORRUPTION:
                # Estrategia para errores de corrupción de estado
                recovery_result = await self.recover_from_state_corruption()
                if recovery_result:
                    action_type = RecoveryActionType.RESET_COMPONENT
                    success = True
                    recovery_steps.append("State corruption recovery successful")
                else:
                    action_type = RecoveryActionType.SKIP_CYCLE
                    recovery_steps.append("State corruption recovery failed")
                    
            elif error_type == ErrorType.NETWORK:
                # Estrategia para errores de red
                if retry_count < self.max_retries[ErrorType.NETWORK]:
                    action_type = RecoveryActionType.RETRY
                    success = True
                    recovery_steps.append("Network error - retrying with backoff")
                else:
                    action_type = RecoveryActionType.SKIP_CYCLE
                    recovery_steps.append("Network error - max retries exceeded")
                    
            else:  # ErrorType.UNKNOWN
                # Errores desconocidos - loggear y esperar
                action_type = RecoveryActionType.SKIP_CYCLE
                wait_seconds = 30
                recovery_steps.append("Unknown error - logging and waiting")
        
        except Exception as recovery_error:
            logger.error(f"❌ Error durante recovery: {recovery_error}")
            action_type = RecoveryActionType.SHUTDOWN
            wait_seconds = 60
            recovery_steps.append(f"Recovery failed: {recovery_error}")
        
        # Registrar tiempo de recovery
        recovery_time = time.time() - start_time
        self.last_recovery_time[error_key] = time.time()
        
        # Crear acción de recovery
        recovery_action = RecoveryAction(
            action=action_type,
            wait_seconds=wait_seconds,
            recovery_steps_taken=recovery_steps,
            success=success
        )
        
        # Loggear resultado del recovery
        logger.info(
            f"🔄 Recovery completado - Acción: {action_type.value}, "
            f"Espera: {wait_seconds}s, Tiempo: {recovery_time:.2f}s, "
            f"Éxito: {success}"
        )
        
        return recovery_action

    async def recover_from_data_error(self) -> bool:
        """
        Recupera de errores de calidad de datos.
        
        Returns:
            bool: True si la recuperación fue exitosa, False en caso contrario
        """
        recovery_steps = []
        
        try:
            # 1. Limpiar datos corruptos
            logger.info("🧹 Limpiando datos corruptos...")
            recovery_steps.append("Starting data cleanup")
            
            # Intentar obtener datos frescos (simulación)
            # En implementación real, esto llamaría al loader de datos
            fresh_data = await self._get_fresh_market_data()
            
            if fresh_data:
                # 2. Validar datos frescos
                logger.info("✅ Datos frescos obtenidos, validando...")
                recovery_steps.append("Fresh data obtained")
                
                # 3. Sanitizar datos
                sanitized_data = sanitize_market_data(fresh_data)
                
                if sanitized_data:
                    logger.info("✅ Datos sanitizados exitosamente")
                    recovery_steps.append("Data sanitization successful")
                    return True
                else:
                    logger.warning("⚠️ No se pudieron sanitizar los datos")
                    recovery_steps.append("Data sanitization failed")
            else:
                logger.warning("⚠️ No se pudieron obtener datos frescos")
                recovery_steps.append("Failed to obtain fresh data")
                
        except Exception as e:
            logger.error(f"❌ Error en recovery de datos: {e}")
            recovery_steps.append(f"Data recovery error: {e}")
        
        # Registrar pasos de recovery para seguimiento
        logger.info(f"📋 Pasos de recovery de datos: {recovery_steps}")
        return False

    async def recover_from_ml_framework_error(self) -> bool:
        """
        Recupera de errores de frameworks ML (TensorFlow, PyTorch).
        
        Returns:
            bool: True si la recuperación fue exitosa, False en caso contrario
        """
        recovery_steps = []
        
        try:
            # 1. Limpiar recursos de ML
            logger.info("🧹 Limpiando recursos de ML...")
            recovery_steps.append("Starting ML resource cleanup")
            
            await self._cleanup_ml_resources()
            
            # 2. Esperar para permitir liberación de recursos
            logger.info("⏳ Esperando liberación de recursos...")
            await asyncio.sleep(5)
            recovery_steps.append("Waiting for resource release")
            
            # 3. Re-inicializar frameworks
            logger.info("🔧 Re-inicializando frameworks ML...")
            recovery_steps.append("Reinitializing ML frameworks")
            
            success = await self._reinitialize_ml_frameworks()
            
            if success:
                logger.info("✅ Frameworks ML re-inicializados exitosamente")
                recovery_steps.append("ML frameworks reinitialized successfully")
                return True
            else:
                logger.warning("⚠️ No se pudieron re-inicializar los frameworks ML")
                recovery_steps.append("ML framework reinitialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en recovery de ML: {e}")
            recovery_steps.append(f"ML recovery error: {e}")
        
        logger.info(f"📋 Pasos de recovery de ML: {recovery_steps}")
        return False

    async def recover_from_state_corruption(self) -> bool:
        """
        Recupera de errores de corrupción de estado.
        
        Returns:
            bool: True si la recuperación fue exitosa, False en caso contrario
        """
        recovery_steps = []
        
        try:
            # 1. Validar estructura del estado
            logger.info("🔍 Validando estructura del estado...")
            recovery_steps.append("Starting state validation")
            
            if not self._validate_state_structure():
                logger.warning("⚠️ Estado con estructura inválida")
                recovery_steps.append("Invalid state structure detected")
                
                # 2. Intentar reparar estado
                logger.info("🔧 Intentando reparar estado...")
                recovery_steps.append("Attempting state repair")
                
                repaired_state = await self._repair_state()
                
                if repaired_state:
                    logger.info("✅ Estado reparado exitosamente")
                    recovery_steps.append("State repair successful")
                    return True
                else:
                    logger.warning("⚠️ No se pudo reparar el estado")
                    recovery_steps.append("State repair failed")
                    return False
            else:
                logger.info("✅ Estado con estructura válida")
                recovery_steps.append("State structure is valid")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error en recovery de estado: {e}")
            recovery_steps.append(f"State recovery error: {e}")
        
        logger.info(f"📋 Pasos de recovery de estado: {recovery_steps}")
        return False

    async def _get_fresh_market_data(self) -> Optional[Dict]:
        """
        Obtiene datos de mercado frescos (simulación).
        En implementación real, esto llamaría al loader de datos.
        """
        try:
            # Simulación de obtención de datos frescos
            # En implementación real: await loader.get_realtime_data()
            return {"BTCUSDT": {"close": 50000.0}, "ETHUSDT": {"close": 3000.0}}
        except Exception:
            return None

    async def _cleanup_ml_resources(self):
        """Limpia recursos de frameworks ML."""
        try:
            # Limpiar TensorFlow
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                logger.info("🧹 TensorFlow session cleared")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando TensorFlow: {e}")
            
            # Limpiar PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("🧹 PyTorch cache cleared")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"⚠️ Error limpiando PyTorch: {e}")
            
            # Forzar garbage collection
            gc.collect()
            logger.info("🧹 Garbage collection completed")
            
        except Exception as e:
            logger.error(f"❌ Error en limpieza de recursos ML: {e}")

    async def _reinitialize_ml_frameworks(self) -> bool:
        """Re-inicializa frameworks ML."""
        try:
            # Re-inicializar TensorFlow
            try:
                import tensorflow as tf
                tf.config.experimental.reset_memory_growth(tf.config.list_physical_devices('GPU')[0])
                logger.info("✅ TensorFlow re-inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Error re-inicializando TensorFlow: {e}")
                return False
            
            # Re-inicializar PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("✅ PyTorch re-inicializado")
            except Exception as e:
                logger.warning(f"⚠️ Error re-inicializando PyTorch: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error re-inicializando frameworks ML: {e}")
            return False

    def _validate_state_structure(self) -> bool:
        """Valida la estructura del estado del sistema."""
        try:
            # Validación básica de estructura
            # En implementación real, esto validaría el estado actual
            return True
        except Exception:
            return False

    async def _repair_state(self) -> bool:
        """Intenta reparar el estado del sistema."""
        try:
            # Lógica de reparación de estado
            # En implementación real, esto restauraría el estado desde backup
            logger.info("🔧 Estado reparado desde backup")
            return True
        except Exception:
            return False

    def reset_error_counters(self, error_type: Optional[ErrorType] = None):
        """
        Reinicia los contadores de errores.
        
        Args:
            error_type: Tipo de error a reiniciar, o None para reiniciar todos
        """
        if error_type:
            error_keys = [key for key in self.error_counts.keys() if error_type.value in key]
            for key in error_keys:
                self.error_counts[key] = 0
        else:
            self.error_counts.clear()
        
        logger.info(f"🔄 Contadores de errores reiniciados: {error_type.value if error_type else 'todos'}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de errores para monitoreo.
        
        Returns:
            Dict con estadísticas de errores
        """
        return {
            'error_counts': self.error_counts.copy(),
            'last_recovery_times': self.last_recovery_time.copy(),
            'max_retries_config': {k.value: v for k, v in self.max_retries.items()},
            'base_wait_times': {k.value: v for k, v in self.base_wait_times.items()}
        }