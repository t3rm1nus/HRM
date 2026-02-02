#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test para verificar el fix de singleton en StateCoordinator
"""

import logging
import sys
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Añadir el path del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from system.state_coordinator import StateCoordinator

def test_singleton_behavior():
    """Testea que StateCoordinator se comporte como singleton"""
    print("🧪 Testeando comportamiento de singleton...")
    
    # Crear primera instancia
    print("Creando primera instancia...")
    sc1 = StateCoordinator()
    print(f"✅ Primera instancia creada: {id(sc1)}")
    
    # Crear segunda instancia
    print("Creando segunda instancia...")
    sc2 = StateCoordinator()
    print(f"✅ Segunda instancia creada: {id(sc2)}")
    
    # Verificar que son la misma instancia
    if id(sc1) == id(sc2):
        print("✅ PASS: Ambas instancias son el mismo objeto (singleton working)")
        return True
    else:
        print("❌ FAIL: Las instancias son diferentes (singleton broken)")
        return False

def test_initialization_logging():
    """Testea que el logging solo ocurra una vez"""
    print("\n🧪 Testeando logging de inicialización...")
    
    # Contar cuántas veces aparece el mensaje de inicialización
    import io
    from contextlib import redirect_stderr
    
    log_capture = io.StringIO()
    
    # Crear instancias y capturar logs
    with redirect_stderr(log_capture):
        sc1 = StateCoordinator()
        sc2 = StateCoordinator()
    
    log_output = log_capture.getvalue()
    init_messages = log_output.count("StateCoordinator inicializado (primera y única vez)")
    
    if init_messages == 1:
        print("✅ PASS: Mensaje de inicialización aparece solo una vez")
        return True
    else:
        print(f"❌ FAIL: Mensaje de inicialización aparece {init_messages} veces")
        return False

def test_state_consistency():
    """Testea que el estado sea consistente entre instancias"""
    print("\n🧪 Testeando consistencia de estado...")
    
    # Crear instancias
    sc1 = StateCoordinator()
    sc2 = StateCoordinator()
    
    # Modificar estado en una instancia
    sc1.update_state({"test_key": "test_value"})
    
    # Verificar que la otra instancia ve el cambio
    state_sc2 = sc2.get_state()
    
    if "test_key" in state_sc2 and state_sc2["test_key"] == "test_value":
        print("✅ PASS: Estado consistente entre instancias")
        return True
    else:
        print("❌ FAIL: Estado inconsistente entre instancias")
        return False

def main():
    """Ejecuta todos los tests"""
    print("🚀 Iniciando tests de fix de singleton...")
    print("=" * 50)
    
    tests = [
        test_singleton_behavior,
        test_initialization_logging,
        test_state_consistency
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ FAIL: Test {test.__name__} falló con excepción: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Resultados de los tests:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Resumen: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ¡Todos los tests pasaron! El fix de singleton está funcionando correctamente.")
        return 0
    else:
        print("💥 Algunos tests fallaron. Revisa el fix de singleton.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)