#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test para verificar el mecanismo de protección contra ejecución en trading loop
"""

import logging
import sys
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Añadir el path del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.state_manager import initialize_state, validate_state_structure
from system.state_coordinator import StateCoordinator

def test_protection_mechanism():
    """Testea el mecanismo de protección contra ejecución en trading loop."""
    print("🧪 Testeando mecanismo de protección...")
    
    # Test 1: Ejecución normal (fuera del loop)
    print("\n1. Ejecución normal (fuera del loop):")
    try:
        state = initialize_state(["BTCUSDT", "ETHUSDT"], 3000.0)
        print("✅ initialize_state() ejecutado exitosamente fuera del loop")
    except Exception as e:
        print(f"❌ Error inesperado fuera del loop: {e}")
        return False
    
    # Test 2: Simular ejecución dentro del loop
    print("\n2. Simulando ejecución dentro del loop:")
    try:
        # Activar la protección
        initialize_state._in_loop = True
        validate_state_structure._in_loop = True
        
        # Intentar ejecutar initialize_state
        try:
            state = initialize_state(["BTCUSDT", "ETHUSDT"], 3000.0)
            print("❌ initialize_state() debería haber fallado dentro del loop")
            return False
        except RuntimeError as e:
            if "trading loop" in str(e):
                print("✅ initialize_state() correctamente bloqueado dentro del loop")
            else:
                print(f"❌ Error inesperado: {e}")
                return False
        
        # Intentar ejecutar validate_state_structure
        try:
            state = validate_state_structure({"invalid": "state"})
            print("❌ validate_state_structure() debería haber fallado dentro del loop")
            return False
        except RuntimeError as e:
            if "trading loop" in str(e):
                print("✅ validate_state_structure() correctamente bloqueado dentro del loop")
            else:
                print(f"❌ Error inesperado: {e}")
                return False
        
        # Desactivar la protección
        delattr(initialize_state, '_in_loop')
        delattr(validate_state_structure, '_in_loop')
        
    except Exception as e:
        print(f"❌ Error durante la simulación: {e}")
        return False
    
    # Test 3: Verificar que StateCoordinator también tiene protección
    print("\n3. Verificando protección en StateCoordinator:")
    try:
        # Activar protección
        StateCoordinator.cleanup_corrupted_state._in_loop = True
        
        # Intentar ejecutar cleanup_corrupted_state
        sc = StateCoordinator()
        try:
            result = sc.cleanup_corrupted_state()
            print("❌ cleanup_corrupted_state() debería haber fallado dentro del loop")
            return False
        except RuntimeError as e:
            if "trading loop" in str(e):
                print("✅ cleanup_corrupted_state() correctamente bloqueado dentro del loop")
            else:
                print(f"❌ Error inesperado: {e}")
                return False
        
        # Desactivar protección
        delattr(StateCoordinator.cleanup_corrupted_state, '_in_loop')
        
    except Exception as e:
        print(f"❌ Error durante la verificación de StateCoordinator: {e}")
        return False
    
    return True

def main():
    """Ejecuta todos los tests."""
    print("🚀 Iniciando tests de mecanismo de protección...")
    
    try:
        success = test_protection_mechanism()
        
        if success:
            print("\n🎉 Todos los tests PASARON! El mecanismo de protección está funcionando correctamente.")
            print("✅ Los métodos de inicialización están protegidos contra ejecución en el trading loop")
            return True
        else:
            print("\n❌ Algunos tests FALLARON. Revisar el mecanismo de protección.")
            return False
            
    except Exception as e:
        print(f"\n💥 Error durante los tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)