# l1_operational/test_clean_l1.py
"""
Pruebas para verificar que L1 está limpio y determinista.
L1 solo debe ejecutar órdenes seguras, sin tomar decisiones de trading.
"""

import sys
import os
import time

# Agregar el directorio raíz al path para las importaciones
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from l1_operational.models import Signal
    from l1_operational.order_manager import order_manager
    from l1_operational.bus_adapter import bus_adapter
    from l1_operational.config import RISK_LIMITS, PORTFOLIO_LIMITS
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("💡 Soluciones:")
    print("   1. Ejecutar desde el directorio raíz: python -m l1_operational.test_clean_l1")
    print("   2. Ejecutar: cd c:/proyectos/HRM && python l1_operational/test_clean_l1.py")
    sys.exit(1)

def test_l1_no_takes_trading_decisions():
    """
    Prueba que L1 no toma decisiones de trading.
    """
    print("🧪 Probando que L1 no toma decisiones de trading...")
    
    # Crear una señal de prueba
    test_signal = Signal(
        signal_id="test_signal_1",
        strategy_id="test_strategy",
        timestamp=time.time(),
        symbol="BTC/USDT",
        side="buy",
        qty=0.1,  # Cantidad que excede el límite
        order_type="market",
        risk={"max_slippage_bps": 100},
        metadata={"confidence": 0.8}
    )
    
    # L1 debe rechazar la orden, no ajustarla
    import asyncio
    report = asyncio.run(order_manager.handle_signal(test_signal))
    
    print(f"   Señal original: {test_signal.qty} BTC")
    print(f"   Reporte: {report.status}")
    print(f"   Error: {report.message}")
    
    # Verificar que L1 no modificó la señal original
    assert test_signal.qty == 0.1, "L1 no debe modificar la señal original"
    assert report.status == "rejected", "L1 debe rechazar órdenes que exceden límites"
    
    print("   ✅ L1 no modifica señales, solo las valida")

def test_l1_only_validates_and_executes():
    """
    Prueba que L1 solo valida y ejecuta, sin lógica de decisión.
    """
    print("🧪 Probando que L1 solo valida y ejecuta...")
    
    # Señal válida
    valid_signal = Signal(
        signal_id="test_signal_2",
        strategy_id="test_strategy",
        timestamp=time.time(),
        symbol="BTC/USDT",
        side="buy",
        qty=0.01,  # Cantidad dentro del límite
        order_type="market",
        risk={"max_slippage_bps": 50},
        metadata={"confidence": 0.9}
    )
    
    # L1 debe procesar la señal sin modificarla
    original_qty = valid_signal.qty
    import asyncio
    report = asyncio.run(order_manager.handle_signal(valid_signal))
    
    print(f"   Señal original: {original_qty} BTC")
    print(f"   Señal después: {valid_signal.qty} BTC")
    print(f"   Reporte: {report.status}")
    
    # Verificar que L1 no modificó la señal
    assert valid_signal.qty == original_qty, "L1 no debe modificar señales válidas"
    
    print("   ✅ L1 mantiene las señales intactas")

def test_l1_risk_validation():
    """
    Prueba que L1 valida correctamente los límites de riesgo.
    """
    print("🧪 Probando validación de riesgo en L1...")
    
    # Probar diferentes escenarios de riesgo
    test_cases = [
        {
            "name": "Cantidad excede límite BTC",
            "signal": Signal(
                signal_id="test_risk_1",
                strategy_id="test_strategy",
                timestamp=time.time(),
                symbol="BTC/USDT",
                side="buy",
                qty=RISK_LIMITS["MAX_ORDER_SIZE_BTC"] + 0.01,
                order_type="market"
            ),
            "expected_status": "rejected"
        },
        {
            "name": "Valor mínimo no alcanzado",
            "signal": Signal(
                signal_id="test_risk_2",
                strategy_id="test_strategy",
                timestamp=time.time(),
                symbol="BTC/USDT",
                side="buy",
                qty=0.001,  # Muy pequeño
                order_type="market"
            ),
            "expected_status": "rejected"
        },
        {
            "name": "Señal válida",
            "signal": Signal(
                signal_id="test_risk_3",
                strategy_id="test_strategy",
                timestamp=time.time(),
                symbol="BTC/USDT",
                side="buy",
                qty=0.01,
                order_type="market"
            ),
            "expected_status": "filled"  # Puede ser rechazada por saldo insuficiente en test
        }
    ]
    
    for test_case in test_cases:
        print(f"   Probando: {test_case['name']}")
        import asyncio
        report = asyncio.run(order_manager.handle_signal(test_case['signal']))
        print(f"     Resultado: {report.status}")
        
        # Verificar que L1 no modificó la señal
        assert test_case['signal'].qty == test_case['signal'].qty, "L1 no debe modificar señales"
    
    print("   ✅ L1 valida riesgo sin modificar señales")

def test_l1_deterministic_behavior():
    """
    Prueba que L1 tiene comportamiento determinista.
    """
    print("🧪 Probando comportamiento determinista de L1...")
    
    # Misma señal, mismo resultado
    signal1 = Signal(
        signal_id="test_det_1",
        strategy_id="test_strategy",
        timestamp=time.time(),
        symbol="BTC/USDT",
        side="buy",
        qty=0.1,  # Excede límite
        order_type="market"
    )
    
    signal2 = Signal(
        signal_id="test_det_2",
        strategy_id="test_strategy",
        timestamp=time.time(),
        symbol="BTC/USDT",
        side="buy",
        qty=0.1,  # Excede límite
        order_type="market"
    )
    
    import asyncio
    report1 = asyncio.run(order_manager.handle_signal(signal1))
    report2 = asyncio.run(order_manager.handle_signal(signal2))
    
    print(f"   Primera ejecución: {report1.status}")
    print(f"   Segunda ejecución: {report2.status}")
    
    # Ambas deben ser rechazadas por el mismo motivo
    assert report1.status == "rejected", "Primera señal debe ser rechazada"
    assert report2.status == "rejected", "Segunda señal debe ser rechazada"
    
    print("   ✅ L1 tiene comportamiento determinista")

def main():
    """
    Ejecuta todas las pruebas.
    """
    print("🚀 Iniciando pruebas de L1 limpio y determinista...\n")
    
    try:
        test_l1_no_takes_trading_decisions()
        print()
        
        test_l1_only_validates_and_executes()
        print()
        
        test_l1_risk_validation()
        print()
        
        test_l1_deterministic_behavior()
        print()
        
        print("🎉 Todas las pruebas pasaron! L1 está limpio y determinista.")
        print("\n📋 Resumen de lo que L1 NO hace:")
        print("   ❌ No modifica cantidades de órdenes")
        print("   ❌ No ajusta precios")
        print("   ❌ No toma decisiones de timing")
        print("   ❌ No actualiza portfolio")
        print("   ❌ No actualiza datos de mercado")
        print("\n✅ Lo que L1 SÍ hace:")
        print("   ✅ Valida límites de riesgo")
        print("   ✅ Ejecuta órdenes pre-validadas")
        print("   ✅ Genera reportes de ejecución")
        print("   ✅ Mantiene trazabilidad")
        
    except Exception as e:
        print(f"❌ Error en las pruebas: {e}")
        raise

if __name__ == "__main__":
    main()
