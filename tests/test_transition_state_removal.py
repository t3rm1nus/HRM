#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test para verificar la eliminación de transition_state y uso de update_state
"""

import logging
import sys
import os
import time
import pandas as pd
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Añadir el path del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.state_manager import transition_system_state, inject_state_coordinator

def test_transition_state_removal():
    """Testea la eliminación de transition_state y uso de update_state."""
    print("🧪 Testeando eliminación de transition_state...")
    
    # Crear un mock de StateCoordinator
    class MockStateCoordinator:
        def __init__(self):
            self.initialized = True
            self.state = {}
        
        def get_state(self, version="current"):
            return self.state.copy()
        
        def update_state(self, updates):
            self.state.update(updates)
            return True
        
        def set_state(self, state, version="current"):
            self.state = state.copy()
            return True
    
    # Inyectar el mock
    mock_coordinator = MockStateCoordinator()
    inject_state_coordinator(mock_coordinator)
    
    # Test 1: transition_system_state debe usar update_state
    print("\n1. Testeando transition_system_state con update_state:")
    
    try:
        # Llamar a transition_system_state
        transition_system_state("BLIND", "market_volatility", {"volatility": 0.8})
        
        # Verificar que el estado se actualizó correctamente
        current_state = mock_coordinator.get_state()
        
        expected_keys = [
            "system_state_type",
            "system_state_reason", 
            "system_state_metadata",
            "system_state_timestamp"
        ]
        
        for key in expected_keys:
            if key not in current_state:
                print(f"❌ Clave {key} no encontrada en el estado")
                return False
        
        # Verificar valores
        if current_state["system_state_type"] != "BLIND":
            print(f"❌ system_state_type incorrecto: {current_state['system_state_type']}")
            return False
        
        if current_state["system_state_reason"] != "market_volatility":
            print(f"❌ system_state_reason incorrecto: {current_state['system_state_reason']}")
            return False
        
        if current_state["system_state_metadata"] != {"volatility": 0.8}:
            print(f"❌ system_state_metadata incorrecto: {current_state['system_state_metadata']}")
            return False
        
        if "system_state_timestamp" not in current_state:
            print("❌ system_state_timestamp no encontrado")
            return False
        
        print("✅ transition_system_state actualizó el estado correctamente con update_state")
        
    except Exception as e:
        print(f"❌ Error en test de transition_system_state: {e}")
        return False
    
    # Test 2: Verificar que no se usa transition_state
    print("\n2. Testeando que no se usa transition_state:")
    
    # Crear un mock que detecte si transition_state es llamado
    class MockStateCoordinatorWithDetection:
        def __init__(self):
            self.initialized = True
            self.state = {}
            self.transition_state_called = False
        
        def get_state(self, version="current"):
            return self.state.copy()
        
        def update_state(self, updates):
            self.state.update(updates)
            return True
        
        def set_state(self, state, version="current"):
            self.state = state.copy()
            return True
        
        def transition_state(self, state_type, reason, metadata):
            self.transition_state_called = True
            raise RuntimeError("transition_state should not be called")
    
    # Inyectar el nuevo mock
    mock_coordinator_with_detection = MockStateCoordinatorWithDetection()
    inject_state_coordinator(mock_coordinator_with_detection)
    
    try:
        # Llamar a transition_system_state
        transition_system_state("NORMAL", "market_stabilized", {"volatility": 0.2})
        
        # Verificar que transition_state no fue llamado
        if mock_coordinator_with_detection.transition_state_called:
            print("❌ transition_state fue llamado, debería haberse eliminado")
            return False
        
        print("✅ transition_state no fue llamado, se usa update_state en su lugar")
        
    except Exception as e:
        print(f"❌ Error en test de detección de transition_state: {e}")
        return False
    
    return True

def main():
    """Ejecuta todos los tests."""
    print("🚀 Iniciando tests de eliminación de transition_state...")
    
    try:
        success = test_transition_state_removal()
        
        if success:
            print("\n🎉 Todos los tests PASARON! La eliminación de transition_state está funcionando correctamente.")
            print("✅ Se usa update_state en lugar de transition_state")
            print("✅ El estado se actualiza correctamente")
            return True
        else:
            print("\n❌ Algunos tests FALLARON. Revisar la eliminación de transition_state.")
            return False
            
    except Exception as e:
        print(f"\n💥 Error durante los tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)