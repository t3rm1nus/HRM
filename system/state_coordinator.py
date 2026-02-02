#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StateCoordinator - Gestión Centralizada del Estado del Sistema HRM

Este módulo implementa el StateCoordinator siguiendo estrictamente el contrato definido.
Responsabilidades permitidas:
- Almacenamiento y gestión de estado
- Validación estructural
- Sincronización entre componentes
- Persistencia defensiva

Responsabilidades prohibidas:
- Interpretación semántica del estado
- Lógica de trading o decisiones de negocio
- Interacción con sistemas externos
- Procesamiento de señales o métricas
"""

import json
import pickle
import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os

logger = logging.getLogger(__name__)


class StateVersion(Enum):
    """Tipos de versiones de estado."""
    CURRENT = "current"
    PREVIOUS = "previous"
    HISTORICAL = "historical"


@dataclass
class ValidationResult:
    """Resultado de validación de estado."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class SyncResult:
    """Resultado de sincronización de estado."""
    success: bool
    synced_components: List[str]
    failed_components: List[str]
    conflicts: List[str]


@dataclass
class ComponentStatus:
    """Estado de sincronización de un componente."""
    last_sync: Optional[datetime]
    version: str
    status: str  # 'synced', 'outdated', 'error'


class StateCoordinator:
    """
    Coordinador centralizado del estado del sistema.
    
    Implementa las responsabilidades permitidas sin interpretar el significado del estado.
    """
    
    # Variable de clase para singleton
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, storage_path: str = "storage/state", max_history: int = 1000):
        """
        Inicializa el StateCoordinator.
        
        Args:
            storage_path: Ruta para almacenamiento persistente
            max_history: Máximo número de versiones históricas a mantener
        """
        # Evitar re-inicialización en singleton
        if StateCoordinator._initialized:
            return
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_history = max_history
        
        # Estado en memoria
        self._state_cache: Dict[str, Dict[str, Any]] = {}
        self._state_versions: List[Dict[str, Any]] = []
        self._component_sync_status: Dict[str, ComponentStatus] = {}
        
        # Esquema básico de validación (estructural)
        self._required_keys = {
            "cycle_id", "market_data", "portfolio", "l3_output"
        }
        
        # Inicializar estado básico
        self._initialize_state()
        
        StateCoordinator._initialized = True
        logger.info("✅ StateCoordinator inicializado (primera y única vez)")
    
    def _initialize_state(self) -> None:
        """Inicializa el estado con valores por defecto."""
        try:
            initial_state = {
                "cycle_id": 0,
                "market_data": {},
                "portfolio": {
                    "btc_balance": 0.0,
                    "eth_balance": 0.0,
                    "usdt_balance": 3000.0,
                    "total_value": 3000.0
                },
                "l3_output": {},
                "l3_decision_cache": None,
                "l3_last_update": 0,
                "l3_previous_regime": None,
                "l3_previous_setup_type": None,
                "signal_execution_logged": False,
                "l3_fallback": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "1.0"
            }
            
            self._state_cache["current"] = initial_state
            self._state_cache["previous"] = initial_state.copy()
            
            # Añadir a historial
            self._state_versions.append(initial_state.copy())
            
            logger.info("✅ Estado inicializado con valores por defecto")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando estado: {e}")
            raise RuntimeError(f"Failed to initialize state: {e}")
    
    def get_state(self, version: str = "current") -> Dict[str, Any]:
        """
        Obtiene una copia del estado.
        
        Args:
            version: Versión del estado ("current", "previous", o ID histórico)
            
        Returns:
            Copia del estado solicitado
        """
        try:
            if version == "current":
                state = self._state_cache.get("current", {})
            elif version == "previous":
                state = self._state_cache.get("previous", {})
            else:
                # Buscar en versiones históricas
                for hist_state in reversed(self._state_versions):
                    if hist_state.get("version") == version:
                        state = hist_state.copy()
                        break
                else:
                    logger.warning(f"⚠️ Versión de estado no encontrada: {version}")
                    state = self._state_cache.get("current", {})
            
            # Devolver copia profunda para evitar modificaciones accidentales
            return json.loads(json.dumps(state))
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo estado {version}: {e}")
            return self._state_cache.get("current", {})
    
    def set_state(self, state: Dict[str, Any], version: str = "current") -> bool:
        """
        Establece el estado actual.
        
        Args:
            state: Nuevo estado a establecer
            version: Versión donde almacenar el estado
            
        Returns:
            True si la operación fue exitosa
        """
        try:
            # Validar estructura antes de establecer
            validation = self.validate_state(state)
            if not validation.is_valid:
                logger.error(f"❌ Estado inválido, no se puede establecer: {validation.errors}")
                return False
            
            # Crear copia para evitar referencias
            state_copy = json.loads(json.dumps(state))
            state_copy["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            if version == "current":
                # Mover current a previous antes de actualizar
                if "current" in self._state_cache:
                    self._state_cache["previous"] = self._state_cache["current"].copy()
                
                self._state_cache["current"] = state_copy
                
                # Añadir a historial si es una actualización significativa
                self._add_to_history(state_copy)
                
            elif version == "previous":
                self._state_cache["previous"] = state_copy
            else:
                # Actualizar versión histórica específica
                for i, hist_state in enumerate(self._state_versions):
                    if hist_state.get("version") == version:
                        self._state_versions[i] = state_copy
                        break
            
            logger.debug(f"✅ Estado establecido: {version}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error estableciendo estado {version}: {e}")
            return False
    
    def update_state(self, updates: Dict[str, Any]) -> bool:
        """
        Actualiza parcialmente el estado actual.
        
        Args:
            updates: Diccionario con actualizaciones a aplicar
            
        Returns:
            True si la operación fue exitosa
        """
        try:
            current_state = self.get_state("current")
            
            # Aplicar actualizaciones recursivamente
            self._deep_update(current_state, updates)
            
            # Validar el estado resultante
            validation = self.validate_state(current_state)
            if not validation.is_valid:
                logger.error(f"❌ Estado resultante inválido: {validation.errors}")
                return False
            
            # Establecer el estado actualizado
            return self.set_state(current_state, "current")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando estado: {e}")
            return False
    
    def _deep_update(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Actualización recursiva de diccionarios."""
        for key, value in updates.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def validate_state(self, state: Dict[str, Any]) -> ValidationResult:
        """
        Valida la estructura del estado sin interpretar su significado.
        
        Args:
            state: Estado a validar
            
        Returns:
            Resultado de validación con errores y advertencias
        """
        errors = []
        warnings = []
        
        try:
            # Validar tipo básico
            if not isinstance(state, dict):
                errors.append("El estado debe ser un diccionario")
                return ValidationResult(False, errors, warnings)
            
            # Validar claves requeridas
            missing_keys = self._required_keys - set(state.keys())
            if missing_keys:
                errors.append(f"Claves requeridas faltantes: {missing_keys}")
            
            # Validar tipos de datos básicos
            for key in ["cycle_id", "l3_last_update"]:
                if key in state and not isinstance(state[key], (int, float)):
                    errors.append(f"Tipo incorrecto para {key}: {type(state[key])}")
            
            for key in ["total_value", "btc_balance", "eth_balance", "usdt_balance"]:
                if key in state and not isinstance(state[key], (int, float)):
                    errors.append(f"Tipo incorrecto para {key}: {type(state[key])}")
            
            # Validar estructura de portfolio
            if "portfolio" in state:
                portfolio = state["portfolio"]
                if not isinstance(portfolio, dict):
                    errors.append("portfolio debe ser un diccionario")
                else:
                    required_portfolio_keys = {"btc_balance", "eth_balance", "usdt_balance", "total_value"}
                    missing_portfolio = required_portfolio_keys - set(portfolio.keys())
                    if missing_portfolio:
                        errors.append(f"Claves requeridas faltantes en portfolio: {missing_portfolio}")
            
            # Validar estructura de market_data
            if "market_data" in state:
                market_data = state["market_data"]
                if not isinstance(market_data, dict):
                    errors.append("market_data debe ser un diccionario")
            
            # Validar estructura de l3_output
            if "l3_output" in state:
                l3_output = state["l3_output"]
                if not isinstance(l3_output, dict):
                    errors.append("l3_output debe ser un diccionario")
            
            # Validar timestamp
            if "timestamp" in state:
                try:
                    datetime.fromisoformat(state["timestamp"].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    warnings.append("Formato de timestamp inválido")
            
            return ValidationResult(len(errors) == 0, errors, warnings)
            
        except Exception as e:
            errors.append(f"Error durante validación: {e}")
            return ValidationResult(False, errors, warnings)
    
    def validate_schema_compatibility(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> bool:
        """
        Valida compatibilidad entre versiones del estado.
        
        Args:
            old_state: Estado anterior
            new_state: Estado nuevo
            
        Returns:
            True si son compatibles
        """
        try:
            old_validation = self.validate_state(old_state)
            new_validation = self.validate_state(new_state)
            
            if not old_validation.is_valid or not new_validation.is_valid:
                return False
            
            # Verificar consistencia de claves requeridas
            old_keys = set(old_state.keys())
            new_keys = set(new_state.keys())
            
            # No deben faltar claves requeridas
            missing_required = self._required_keys - new_keys
            if missing_required:
                logger.warning(f"⚠️ Claves requeridas faltantes en nueva versión: {missing_required}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validando compatibilidad de esquemas: {e}")
            return False
    
    def sync_state(self, target_components: List[str], timeout: float = 30.0) -> SyncResult:
        """
        Sincroniza el estado con componentes objetivo.
        
        Args:
            target_components: Lista de componentes a sincronizar
            timeout: Tiempo máximo de espera en segundos
            
        Returns:
            Resultado de la sincronización
        """
        synced_components = []
        failed_components = []
        conflicts = []
        
        try:
            current_state = self.get_state("current")
            
            for component in target_components:
                try:
                    # Simular sincronización (en implementación real, esto interactuaría con componentes)
                    success = self._sync_component_state(component, current_state, timeout)
                    
                    if success:
                        synced_components.append(component)
                        self._component_sync_status[component] = ComponentStatus(
                            last_sync=datetime.now(timezone.utc),
                            version=current_state.get("version", "unknown"),
                            status="synced"
                        )
                    else:
                        failed_components.append(component)
                        self._component_sync_status[component] = ComponentStatus(
                            last_sync=None,
                            version="unknown",
                            status="error"
                        )
                        
                except Exception as e:
                    logger.error(f"❌ Error sincronizando componente {component}: {e}")
                    failed_components.append(component)
            
            return SyncResult(
                success=len(failed_components) == 0,
                synced_components=synced_components,
                failed_components=failed_components,
                conflicts=conflicts
            )
            
        except Exception as e:
            logger.error(f"❌ Error durante sincronización general: {e}")
            return SyncResult(False, [], target_components, [])
    
    def _sync_component_state(self, component: str, state: Dict[str, Any], timeout: float) -> bool:
        """Sincroniza el estado con un componente específico."""
        try:
            # En implementación real, esto enviaría el estado al componente
            # Por ahora, simulamos una sincronización exitosa
            logger.debug(f"🔄 Estado sincronizado con componente: {component}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sincronizando con {component}: {e}")
            return False
    
    def get_sync_status(self) -> Dict[str, ComponentStatus]:
        """Obtiene el estado de sincronización de todos los componentes."""
        return self._component_sync_status.copy()
    
    def resolve_conflict(self, component_states: Dict[str, Dict[str, Any]], strategy: str) -> Dict[str, Any]:
        """
        Resuelve conflictos entre estados de componentes.
        
        Args:
            component_states: Estados de diferentes componentes
            strategy: Estrategia de resolución ("latest", "majority", "component_priority")
            
        Returns:
            Estado resuelto
        """
        try:
            if strategy == "latest":
                # Tomar el estado con timestamp más reciente
                latest_state = max(
                    component_states.values(),
                    key=lambda s: s.get("timestamp", "1970-01-01T00:00:00")
                )
                return latest_state
            
            elif strategy == "majority":
                # Tomar el estado que más componentes coincidan
                # Simplificación: tomar el primero como referencia
                return list(component_states.values())[0]
            
            elif strategy == "component_priority":
                # Prioridad basada en nombre de componente (L3 > L2 > L1)
                priority_order = ["L3", "L2", "L1"]
                for priority in priority_order:
                    for component, state in component_states.items():
                        if priority in component.upper():
                            return state
                return list(component_states.values())[0]
            
            else:
                logger.warning(f"⚠️ Estrategia de resolución desconocida: {strategy}")
                return list(component_states.values())[0]
                
        except Exception as e:
            logger.error(f"❌ Error resolviendo conflicto: {e}")
            return {}
    
    def save_state(self, name: str, format: str = "json") -> bool:
        """
        Persiste el estado actual.
        
        Args:
            name: Nombre del archivo de persistencia
            format: Formato de persistencia ("json", "pickle")
            
        Returns:
            True si la operación fue exitosa
        """
        try:
            current_state = self.get_state("current")
            
            if format == "json":
                file_path = self.storage_path / f"{name}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_state, f, indent=2, ensure_ascii=False, default=str)
            
            elif format == "pickle":
                file_path = self.storage_path / f"{name}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(current_state, f)
            
            else:
                logger.error(f"❌ Formato de persistencia no soportado: {format}")
                return False
            
            logger.info(f"💾 Estado persistido: {name}.{format}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error persistiendo estado {name}: {e}")
            return False
    
    def load_state(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Carga un estado persistido.
        
        Args:
            name: Nombre del archivo de persistencia
            version: Versión específica a cargar (None para última)
            
        Returns:
            Estado cargado o None si falló
        """
        try:
            # Intentar cargar desde JSON
            json_path = self.storage_path / f"{name}.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Validar el estado cargado
                validation = self.validate_state(state)
                if not validation.is_valid:
                    logger.error(f"❌ Estado cargado inválido: {validation.errors}")
                    return None
                
                logger.info(f"📂 Estado cargado: {name}.json")
                return state
            
            # Intentar cargar desde pickle
            pickle_path = self.storage_path / f"{name}.pkl"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    state = pickle.load(f)
                
                # Validar el estado cargado
                validation = self.validate_state(state)
                if not validation.is_valid:
                    logger.error(f"❌ Estado cargado inválido: {validation.errors}")
                    return None
                
                logger.info(f"📂 Estado cargado: {name}.pkl")
                return state
            
            logger.warning(f"⚠️ Archivo de estado no encontrado: {name}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error cargando estado {name}: {e}")
            return None
    
    def backup_state(self, backup_name: str) -> bool:
        """
        Crea un backup del estado actual.
        
        Args:
            backup_name: Nombre del backup
            
        Returns:
            True si la operación fue exitosa
        """
        try:
            current_state = self.get_state("current")
            
            # Añadir metadata de backup
            backup_state = current_state.copy()
            backup_state["backup_info"] = {
                "backup_name": backup_name,
                "backup_timestamp": datetime.now(timezone.utc).isoformat(),
                "backup_type": "full"
            }
            
            # Generar hash para integridad
            state_json = json.dumps(backup_state, sort_keys=True, default=str)
            backup_state["backup_hash"] = hashlib.sha256(state_json.encode()).hexdigest()
            
            return self.save_state(f"backup_{backup_name}", "json")
            
        except Exception as e:
            logger.error(f"❌ Error creando backup {backup_name}: {e}")
            return False
    
    def restore_state(self, backup_name: str) -> bool:
        """
        Restaura un estado desde backup.
        
        Args:
            backup_name: Nombre del backup a restaurar
            
        Returns:
            True si la operación fue exitosa
        """
        try:
            backup_state = self.load_state(f"backup_{backup_name}")
            if backup_state is None:
                return False
            
            # Verificar integridad del backup
            if "backup_hash" in backup_state:
                state_copy = backup_state.copy()
                del state_copy["backup_hash"]
                state_json = json.dumps(state_copy, sort_keys=True, default=str)
                expected_hash = hashlib.sha256(state_json.encode()).hexdigest()
                
                if backup_state["backup_hash"] != expected_hash:
                    logger.error("❌ Hash de integridad del backup no coincide")
                    return False
            
            # Establecer el estado restaurado
            return self.set_state(backup_state, "current")
            
        except Exception as e:
            logger.error(f"❌ Error restaurando backup {backup_name}: {e}")
            return False
    
    def get_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de versiones del estado.
        
        Args:
            limit: Límite de versiones a retornar
            
        Returns:
            Lista de versiones históricas del estado
        """
        try:
            # Retornar copias para evitar modificaciones accidentales
            history = []
            for state in self._state_versions[-limit:]:
                history.append(json.loads(json.dumps(state)))
            
            return history
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo historial de estado: {e}")
            return []
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Revierte el estado a una versión específica.
        
        Args:
            version_id: ID de la versión a la que revertir
            
        Returns:
            True si la operación fue exitosa
        """
        try:
            # Buscar la versión en el historial
            target_state = None
            for state in self._state_versions:
                if state.get("version") == version_id:
                    target_state = state
                    break
            
            if target_state is None:
                logger.error(f"❌ Versión no encontrada: {version_id}")
                return False
            
            # Validar la versión objetivo
            validation = self.validate_state(target_state)
            if not validation.is_valid:
                logger.error(f"❌ Versión objetivo inválida: {validation.errors}")
                return False
            
            # Realizar rollback
            self._state_cache["previous"] = self._state_cache["current"].copy()
            self._state_cache["current"] = target_state.copy()
            
            logger.info(f"🔄 Rollback exitoso a versión: {version_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error realizando rollback a versión {version_id}: {e}")
            return False
    
    def create_state_snapshot(self, snapshot_name: str) -> bool:
        """
        Crea una instantánea del estado actual.
        
        Args:
            snapshot_name: Nombre de la instantánea
            
        Returns:
            True si la operación fue exitosa
        """
        try:
            current_state = self.get_state("current")
            
            # Añadir metadata de snapshot
            snapshot_state = current_state.copy()
            snapshot_state["snapshot_info"] = {
                "snapshot_name": snapshot_name,
                "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
                "snapshot_type": "instantaneous"
            }
            
            return self.save_state(f"snapshot_{snapshot_name}", "json")
            
        except Exception as e:
            logger.error(f"❌ Error creando snapshot {snapshot_name}: {e}")
            return False
    
    def _add_to_history(self, state: Dict[str, Any]) -> None:
        """Añade una versión al historial manteniendo el límite."""
        try:
            # Crear copia para el historial
            history_state = state.copy()
            
            # Limitar tamaño del historial
            if len(self._state_versions) >= self.max_history:
                self._state_versions.pop(0)
            
            self._state_versions.append(history_state)
            
        except Exception as e:
            logger.error(f"❌ Error añadiendo al historial: {e}")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen del estado actual (sin datos sensibles)."""
        try:
            current_state = self.get_state("current")
            
            summary = {
                "cycle_id": current_state.get("cycle_id", 0),
                "total_symbols": len(current_state.get("market_data", {})),
                "portfolio_value": current_state.get("total_value", 0.0),
                "l3_active": bool(current_state.get("l3_output", {})),
                "l3_fallback": current_state.get("l3_fallback", False),
                "timestamp": current_state.get("timestamp", "unknown"),
                "version": current_state.get("version", "unknown"),
                "sync_status": len(self._component_sync_status)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo resumen de estado: {e}")
            return {}

    def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """
        Obtiene un snapshot del portfolio actual para logging.
        
        Returns:
            Dict con balances y valor total del portfolio
        """
        try:
            current_state = self.get_state("current")
            portfolio = current_state.get("portfolio", {})
            
            snapshot = {
                "btc_balance": portfolio.get("btc_balance", 0.0),
                "eth_balance": portfolio.get("eth_balance", 0.0),
                "usdt_balance": portfolio.get("usdt_balance", 0.0),
                "total_value": portfolio.get("total_value", 0.0),
                "timestamp": current_state.get("timestamp", "unknown")
            }
            
            return snapshot
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo snapshot de portfolio: {e}")
            return {
                "btc_balance": 0.0,
                "eth_balance": 0.0,
                "usdt_balance": 0.0,
                "total_value": 0.0,
                "timestamp": "error"
            }
    
    def cleanup_corrupted_state(self) -> bool:
        """
        Manejo defensivo de estado corrupto.
        
        ⚠️  ESTE MÉTODO NO DEBE EJECUTARSE DENTRO DEL TRADING LOOP
        ⚠️  Solo debe usarse durante la inicialización del sistema
        
        Returns:
            True si se recuperó exitosamente
        """
        # Protección contra ejecución en trading loop
        if hasattr(self.cleanup_corrupted_state, '_in_loop') and self.cleanup_corrupted_state._in_loop:
            logger.error("❌ cleanup_corrupted_state() NO puede ejecutarse dentro del trading loop")
            raise RuntimeError("cleanup_corrupted_state() called from within trading loop - this is prohibited")
        
        try:
            logger.warning("🛡️ Manejando estado corrupto...")
            
            # Intentar cargar desde backup más reciente
            backup_files = list(self.storage_path.glob("backup_*.json"))
            if backup_files:
                latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)
                backup_name = latest_backup.stem.replace("backup_", "")
                
                if self.restore_state(backup_name):
                    logger.info("✅ Estado recuperado desde backup")
                    return True
            
            # Si no hay backup, reiniciar con estado inicial
            self._initialize_state()
            logger.info("✅ Estado reiniciado con valores por defecto")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error manejando estado corrupto: {e}")
            return False
