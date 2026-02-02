#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test para verificar la preservación de L3 confidence sin intermediarios frágiles
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

from system.trading_pipeline_manager import TradingPipelineManager

async def test_l3_confidence_preservation():
    """Testea la preservación de L3 confidence sin intermediarios frágiles."""
    print("🧪 Testeando preservación de L3 confidence...")
    
    # Crear un mock de StateCoordinator
    class MockStateCoordinator:
        def __init__(self):
            self.initialized = True
        
        def get_state(self, version="current"):
            return {}
        
        def transition_state(self, state_type, reason, metadata=None):
            pass
    
    # Crear un mock de L2 Processor
    class MockL2Processor:
        def generate_signals_conservative(self, market_data, l3_context):
            return []
    
    # Crear un mock de Order Manager
    class MockOrderManager:
        async def generate_orders(self, state, signals):
            return []
        
        async def execute_orders(self, orders):
            return []
        
        async def monitor_and_execute_stop_losses_with_validation(self, market_data, current_positions):
            return []
    
    # Crear un mock de Portfolio Manager
    class MockPortfolioManager:
        async def sync_with_exchange(self):
            return True
        
        async def update_from_orders_async(self, orders, market_data):
            pass
        
        def get_balance(self, symbol):
            return 0.0
        
        def get_total_value(self, market_data):
            return 3000.0
    
    # Crear un mock de Signal Verifier
    class MockSignalVerifier:
        async def submit_signal_for_verification(self, signal, market_data):
            pass
    
    # Crear un mock de Position Rotator
    class MockPositionRotator:
        async def check_and_rotate_positions(self, state, market_data):
            return []
    
    # Crear un mock de Auto Rebalancer
    class MockAutoRebalancer:
        async def check_and_execute_rebalance(self, market_data, l3_decision):
            return []
    
    # Crear instancia del pipeline manager
    pipeline_manager = TradingPipelineManager(
        portfolio_manager=MockPortfolioManager(),
        order_manager=MockOrderManager(),
        l2_processor=MockL2Processor(),
        position_rotator=MockPositionRotator(),
        auto_rebalancer=MockAutoRebalancer(),
        signal_verifier=MockSignalVerifier(),
        state_coordinator=MockStateCoordinator(),
        config={"SYMBOLS": ["BTCUSDT", "ETHUSDT"]}
    )
    
    # Test 1: L3 con alta confianza debe preservarse
    print("\n1. Testeando L3 con alta confianza:")
    state = {
        "version": "1.0",
        "l3_output": {
            'regime': 'bullish',
            'signal': 'buy',
            'confidence': 1.0,
            'strategy_type': 'deepseek',
            'timestamp': datetime.utcnow().isoformat()
        },
        "l3_last_update": time.time(),
        "portfolio": {
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "usdt_balance": 3000.0,
            "total_value": 3000.0
        }
    }
    
    market_data = {
        "BTCUSDT": pd.DataFrame(),
        "ETHUSDT": pd.DataFrame()
    }
    
    try:
        # Simular actualización L3
        l3_decision = await pipeline_manager._update_l3_decision(state, market_data)
        
        # Verificar que la confianza se preserva
        if l3_decision.get('confidence') == 1.0:
            print("✅ L3 confidence=1.0 preservado correctamente")
        else:
            print(f"❌ L3 confidence no preservado: {l3_decision.get('confidence')}")
            return False
        
        # Verificar que el estado se actualizó directamente
        if state["l3_output"].get('confidence') == 1.0:
            print("✅ Estado L3 actualizado directamente sin intermediarios")
        else:
            print(f"❌ Estado L3 no actualizado correctamente: {state['l3_output'].get('confidence')}")
            return False
        
    except Exception as e:
        print(f"❌ Error en test de alta confianza: {e}")
        return False
    
    # Test 2: L3 con baja confianza debe preservarse
    print("\n2. Testeando L3 con baja confianza:")
    state = {
        "version": "1.0",
        "l3_output": {
            'regime': 'bearish',
            'signal': 'sell',
            'confidence': 0.3,
            'strategy_type': 'deepseek',
            'timestamp': datetime.utcnow().isoformat()
        },
        "l3_last_update": time.time(),
        "portfolio": {
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "usdt_balance": 3000.0,
            "total_value": 3000.0
        }
    }
    
    try:
        # Simular actualización L3
        l3_decision = await pipeline_manager._update_l3_decision(state, market_data)
        
        # Verificar que la confianza se preserva
        if l3_decision.get('confidence') == 0.3:
            print("✅ L3 confidence=0.3 preservado correctamente")
        else:
            print(f"❌ L3 confidence no preservado: {l3_decision.get('confidence')}")
            return False
        
        # Verificar que el estado se actualizó directamente
        if state["l3_output"].get('confidence') == 0.3:
            print("✅ Estado L3 actualizado directamente sin intermediarios")
        else:
            print(f"❌ Estado L3 no actualizado correctamente: {state['l3_output'].get('confidence')}")
            return False
        
    except Exception as e:
        print(f"❌ Error en test de baja confianza: {e}")
        return False
    
    # Test 3: L3 deshabilitado debe preservar confidence=0.0
    print("\n3. Testeando L3 deshabilitado:")
    state = {
        "version": "1.0",
        "l3_output": {
            'regime': 'disabled',
            'signal': 'hold',
            'confidence': 0.0,
            'strategy_type': 'l3_disabled',
            'timestamp': datetime.utcnow().isoformat()
        },
        "l3_last_update": time.time(),
        "portfolio": {
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "usdt_balance": 3000.0,
            "total_value": 3000.0
        }
    }
    
    try:
        # Simular actualización L3 con APAGAR_L3=True
        import comms.config
        original_apagar_l3 = comms.config.APAGAR_L3
        comms.config.APAGAR_L3 = True
        
        l3_decision = await pipeline_manager._update_l3_decision(state, market_data)
        
        # Restaurar valor original
        comms.config.APAGAR_L3 = original_apagar_l3
        
        # Verificar que la confianza se preserva
        if l3_decision.get('confidence') == 0.0:
            print("✅ L3 confidence=0.0 preservado correctamente en modo deshabilitado")
        else:
            print(f"❌ L3 confidence no preservado en modo deshabilitado: {l3_decision.get('confidence')}")
            return False
        
        # Verificar que el estado se actualizó directamente
        if state["l3_output"].get('confidence') == 0.0:
            print("✅ Estado L3 actualizado directamente sin intermediarios en modo deshabilitado")
        else:
            print(f"❌ Estado L3 no actualizado correctamente en modo deshabilitado: {state['l3_output'].get('confidence')}")
            return False
        
    except Exception as e:
        print(f"❌ Error en test de L3 deshabilitado: {e}")
        return False
    
    return True

async def main():
    """Ejecuta todos los tests."""
    print("🚀 Iniciando tests de preservación de L3 confidence...")
    
    try:
        success = await test_l3_confidence_preservation()
        
        if success:
            print("\n🎉 Todos los tests PASARON! La preservación de L3 confidence está funcionando correctamente.")
            print("✅ L3 confidence se preserva sin intermediarios frágiles")
            print("✅ Actualizaciones directas al estado sin pérdidas de confianza")
            return True
        else:
            print("\n❌ Algunos tests FALLARON. Revisar la preservación de L3 confidence.")
            return False
            
    except Exception as e:
        print(f"\n💥 Error durante los tests: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
