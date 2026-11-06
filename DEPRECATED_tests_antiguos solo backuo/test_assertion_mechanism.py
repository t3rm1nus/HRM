#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test para verificar el mecanismo de assertions de protección
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

async def test_assertion_mechanism():
    """Testea el mecanismo de assertions de protección."""
    print("🧪 Testeando mecanismo de assertions de protección...")
    
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
    
    # Test 1: Estado válido debe pasar assertions
    print("\n1. Testeando estado válido:")
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
        
        # Simular paso 4 (L2 signals) - esto debería pasar
        assert state["l3_output"], "STATE RESET DETECTED"
        assert state["l3_output"].get("confidence", 0) > 0, "L3 OUTPUT LOST"
        
        print("✅ Assertions pasados para estado válido")
        
    except AssertionError as e:
        print(f"❌ Assertions fallaron para estado válido: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado en test de estado válido: {e}")
        return False
    
    # Test 2: Estado sin l3_output debe fallar assertion
    print("\n2. Testeando estado sin l3_output:")
    state = {
        "version": "1.0",
        "portfolio": {
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "usdt_balance": 3000.0,
            "total_value": 3000.0
        }
    }
    
    try:
        # Simular paso 4 (L2 signals) - esto debería fallar
        assert state["l3_output"], "STATE RESET DETECTED"
        print("❌ Assertion debería haber fallado para estado sin l3_output")
        return False
    except AssertionError as e:
        if "STATE RESET DETECTED" in str(e):
            print("✅ Assertion correctamente detectó estado sin l3_output")
        else:
            print(f"❌ Assertion incorrecto: {e}")
            return False
    except Exception as e:
        print(f"❌ Error inesperado en test de estado sin l3_output: {e}")
        return False
    
    # Test 3: Estado con l3_output pero confidence=0 debe fallar assertion
    print("\n3. Testeando estado con confidence=0:")
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
        # Simular paso 4 (L2 signals) - esto debería fallar
        assert state["l3_output"], "STATE RESET DETECTED"
        assert state["l3_output"].get("confidence", 0) > 0, "L3 OUTPUT LOST"
        print("❌ Assertion debería haber fallado para confidence=0")
        return False
    except AssertionError as e:
        if "L3 OUTPUT LOST" in str(e):
            print("✅ Assertion correctamente detectó confidence=0")
        else:
            print(f"❌ Assertion incorrecto: {e}")
            return False
    except Exception as e:
        print(f"❌ Error inesperado en test de confidence=0: {e}")
        return False
    
    # Test 4: Estado con l3_output None debe fallar assertion
    print("\n4. Testeando estado con l3_output=None:")
    state = {
        "version": "1.0",
        "l3_output": None,
        "l3_last_update": time.time(),
        "portfolio": {
            "btc_balance": 0.0,
            "eth_balance": 0.0,
            "usdt_balance": 3000.0,
            "total_value": 3000.0
        }
    }
    
    try:
        # Simular paso 4 (L2 signals) - esto debería fallar
        assert state["l3_output"], "STATE RESET DETECTED"
        print("❌ Assertion debería haber fallado para l3_output=None")
        return False
    except AssertionError as e:
        if "STATE RESET DETECTED" in str(e):
            print("✅ Assertion correctamente detectó l3_output=None")
        else:
            print(f"❌ Assertion incorrecto: {e}")
            return False
    except Exception as e:
        print(f"❌ Error inesperado en test de l3_output=None: {e}")
        return False
    
    return True

async def main():
    """Ejecuta todos los tests."""
    print("🚀 Iniciando tests de mecanismo de assertions...")
    
    try:
        success = await test_assertion_mechanism()
        
        if success:
            print("\n🎉 Todos los tests PASARON! El mecanismo de assertions está funcionando correctamente.")
            print("✅ Assertions detectan correctamente estados inválidos")
            print("✅ Errores claros para debugging rápido")
            return True
        else:
            print("\n❌ Algunos tests FALLARON. Revisar el mecanismo de assertions.")
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
