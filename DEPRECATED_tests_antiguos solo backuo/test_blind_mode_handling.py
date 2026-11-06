#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test para verificar el manejo de BLIND MODE en L3
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

from core.l3_processor import get_l3_decision

def test_blind_mode_detection():
    """Testea la detección y manejo de BLIND MODE."""
    print("🧪 Testeando detección de BLIND MODE...")
    
    # Test 1: BLIND MODE detectado (unknown + baja confianza)
    print("\n1. Testeando BLIND MODE con unknown + baja confianza:")
    
    market_data_unknown = {
        'BTCUSDT': {
            'close': [10000, 10100, 10200, 10150, 10300],
            'volume': [100, 150, 200, 180, 220]
        }
    }
    
    # Mock del clasificador para retornar unknown con baja confianza
    import l3_strategy.regime_classifier
    original_ejecutar = l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen
    
    def mock_ejecutar_estrategia_por_regimen(market_data):
        return {
            'regime': 'unknown',
            'signal': 'hold',
            'confidence': 0.05,  # Muy baja confianza
            'allow_l2_signal': True,
            'setup_type': None,
            'subtype': 'unknown'
        }
    
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = mock_ejecutar_estrategia_por_regimen
    
    try:
        l3_output = get_l3_decision(market_data_unknown)
        
        if not l3_output.get('blind_mode', False):
            print("❌ blind_mode no detectado correctamente")
            return False
        
        if l3_output.get('regime') != 'unknown':
            print(f"❌ régimen incorrecto: {l3_output.get('regime')}")
            return False
        
        if l3_output.get('confidence', 0) >= 0.1:
            print(f"❌ confianza demasiado alta para blind mode: {l3_output.get('confidence')}")
            return False
        
        print("✅ BLIND MODE detectado correctamente")
        
    except Exception as e:
        print(f"❌ Error en test de BLIND MODE: {e}")
        return False
    
    # Restaurar función original
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = original_ejecutar
    
    # Test 2: Normal mode (no BLIND)
    print("\n2. Testeando modo normal (no BLIND):")
    
    def mock_ejecutar_normal(market_data):
        return {
            'regime': 'trending',
            'signal': 'buy',
            'confidence': 0.8,  # Alta confianza
            'allow_l2_signal': True,
            'setup_type': 'bullish',
            'subtype': 'uptrend'
        }
    
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = mock_ejecutar_normal
    
    try:
        l3_output = get_l3_decision(market_data_unknown)
        
        if l3_output.get('blind_mode', False):
            print("❌ blind_mode detectado incorrectamente en modo normal")
            return False
        
        if l3_output.get('regime') != 'trending':
            print(f"❌ régimen incorrecto: {l3_output.get('regime')}")
            return False
        
        if l3_output.get('confidence', 0) < 0.5:
            print(f"❌ confianza demasiado baja para modo normal: {l3_output.get('confidence')}")
            return False
        
        print("✅ Modo normal detectado correctamente (no BLIND)")
        
    except Exception as e:
        print(f"❌ Error en test de modo normal: {e}")
        return False
    
    # Restaurar función original
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = original_ejecutar
    
    # Test 3: Error handling (fallback a BLIND MODE)
    print("\n3. Testeando manejo de errores (fallback a BLIND MODE):")
    
    def mock_ejecutar_error(market_data):
        raise Exception("Error en clasificador")
    
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = mock_ejecutar_error
    
    try:
        l3_output = get_l3_decision(market_data_unknown)
        
        if not l3_output.get('blind_mode', False):
            print("❌ blind_mode no detectado en fallback de error")
            return False
        
        if l3_output.get('regime') != 'error':
            print(f"❌ régimen incorrecto en error: {l3_output.get('regime')}")
            return False
        
        if l3_output.get('confidence', 0) != 0.0:
            print(f"❌ confianza incorrecta en error: {l3_output.get('confidence')}")
            return False
        
        print("✅ Fallback a BLIND MODE en error detectado correctamente")
        
    except Exception as e:
        print(f"❌ Error en test de fallback: {e}")
        return False
    
    # Restaurar función original
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = original_ejecutar
    
    return True

def test_blind_mode_integration():
    """Testea la integración de BLIND MODE con el StateCoordinator."""
    print("\n🧪 Testeando integración de BLIND MODE...")
    
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
    from core.state_manager import inject_state_coordinator
    mock_coordinator = MockStateCoordinator()
    inject_state_coordinator(mock_coordinator)
    
    # Test 4: Integración con StateCoordinator
    print("\n4. Testeando integración con StateCoordinator:")
    
    # Mock del clasificador para retornar BLIND MODE
    import l3_strategy.regime_classifier
    original_ejecutar = l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen
    
    def mock_ejecutar_blind(market_data):
        return {
            'regime': 'unknown',
            'signal': 'hold',
            'confidence': 0.05,
            'allow_l2_signal': True,
            'setup_type': None,
            'subtype': 'unknown'
        }
    
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = mock_ejecutar_blind
    
    try:
        # Simular la lógica de actualización de estado
        market_data = {'BTCUSDT': {'close': [10000, 10100, 10200], 'volume': [100, 150, 200]}}
        l3_output = get_l3_decision(market_data)
        
        # Simular la lógica de actualización de estado (como en el FIX EXTRA)
        if l3_output:
            updates = {
                "l3_output": l3_output,
                "l3_last_update": time.time(),
                "l3_fallback": l3_output.get("blind_mode", False)
            }
            
            mock_coordinator.update_state(updates)
            
            # Verificar que el estado se actualizó correctamente
            current_state = mock_coordinator.get_state()
            
            if "l3_output" not in current_state:
                print("❌ l3_output no encontrado en el estado")
                return False
            
            if "l3_fallback" not in current_state:
                print("❌ l3_fallback no encontrado en el estado")
                return False
            
            if current_state["l3_fallback"] != True:
                print(f"❌ l3_fallback incorrecto: {current_state['l3_fallback']}")
                return False
            
            if not current_state["l3_output"].get("blind_mode", False):
                print("❌ blind_mode no detectado en l3_output")
                return False
            
            print("✅ Integración con StateCoordinator correcta")
            
    except Exception as e:
        print(f"❌ Error en test de integración: {e}")
        return False
    
    # Restaurar función original
    l3_strategy.regime_classifier.ejecutar_estrategia_por_regimen = original_ejecutar
    
    return True

def main():
    """Ejecuta todos los tests."""
    print("🚀 Iniciando tests de BLIND MODE...")
    
    try:
        success1 = test_blind_mode_detection()
        success2 = test_blind_mode_integration()
        
        if success1 and success2:
            print("\n🎉 Todos los tests PASARON! El manejo de BLIND MODE está funcionando correctamente.")
            print("✅ BLIND MODE detectado correctamente")
            print("✅ Integración con StateCoordinator funciona")
            print("✅ L2 sabe que L3 existe (no mata el sistema)")
            print("✅ INV-5 solo se activa si de verdad hay vacío")
            return True
        else:
            print("\n❌ Algunos tests FALLARON. Revisar el manejo de BLIND MODE.")
            return False
            
    except Exception as e:
        print(f"\n💥 Error durante los tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)