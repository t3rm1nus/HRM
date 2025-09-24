# hacienda/test_tax_tracker.py
# Pruebas básicas del sistema de seguimiento fiscal

import os
import tempfile
import shutil
from datetime import datetime

from .tax_tracker import TaxTracker

def test_fifo_calculation():
    """Prueba el cálculo FIFO con operaciones simuladas"""
    print("🧪 Probando cálculo FIFO...")

    # Crear directorio temporal para pruebas
    test_dir = tempfile.mkdtemp()

    try:
        # Inicializar TaxTracker
        tracker = TaxTracker(test_dir)

        # Simular operaciones de prueba
        operations = [
            # Compra inicial
            {
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'quantity': 1.0,
                'filled_quantity': 1.0,
                'price': 50000.0,
                'filled_price': 50000.0,
                'commission': 5.0,
                'status': 'filled'
            },
            # Segunda compra a precio más alto
            {
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'quantity': 1.0,
                'filled_quantity': 1.0,
                'price': 60000.0,
                'filled_price': 60000.0,
                'commission': 6.0,
                'status': 'filled'
            },
            # Venta parcial (debe vender primero la posición más antigua)
            {
                'symbol': 'BTCUSDT',
                'side': 'sell',
                'quantity': 1.5,
                'filled_quantity': 1.5,
                'price': 55000.0,
                'filled_price': 55000.0,
                'commission': 8.25,
                'status': 'filled'
            }
        ]

        # Registrar operaciones
        for op in operations:
            tracker.record_operation(op, exchange="TestExchange")

        # Verificar posiciones restantes
        positions = tracker.get_positions_summary()
        print(f"📊 Posiciones finales: {positions}")

        # Verificar cálculos
        btc_position = positions.get('BTCUSDT', {})
        expected_remaining = 0.5  # 2.0 compradas - 1.5 vendidas

        if abs(btc_position.get('total_quantity', 0) - expected_remaining) < 0.001:
            print("✅ Cálculo de posiciones correcto")
        else:
            print(f"❌ Error en cálculo de posiciones. Esperado: {expected_remaining}, Obtenido: {btc_position.get('total_quantity', 0)}")

        # Generar informe fiscal
        tax_report = tracker.generate_tax_report()
        if tax_report:
            print("✅ Informe fiscal generado correctamente")
            print(f"   Ganancias realizadas: ${tax_report.get('realized_gains', 0):,.2f}")
            print(f"   Pérdidas realizadas: ${tax_report.get('realized_losses', 0):,.2f}")
            print(f"   Base imponible: ${tax_report.get('tax_base', 0):,.2f}")
        else:
            print("❌ Error generando informe fiscal")

        # Verificar archivos generados
        expected_files = [
            'operaciones.csv',
            'posiciones_fifo.json',
            'ganancias_realizadas.csv'
        ]

        for filename in expected_files:
            filepath = os.path.join(test_dir, filename)
            if os.path.exists(filepath):
                print(f"✅ Archivo generado: {filename}")
            else:
                print(f"❌ Archivo faltante: {filename}")

        print("🎯 Prueba FIFO completada")

    finally:
        # Limpiar directorio temporal
        shutil.rmtree(test_dir, ignore_errors=True)

def test_tax_utils():
    """Prueba las utilidades fiscales"""
    print("\n🧪 Probando TaxUtils...")

    # Crear directorio temporal
    test_dir = tempfile.mkdtemp()

    try:
        from .tax_utils import TaxUtils

        # Inicializar TaxUtils
        tax_utils = TaxUtils(test_dir)

        # Verificar inicialización
        print("✅ TaxUtils inicializado correctamente")

        # Mostrar consejos fiscales
        advice = tax_utils.get_tax_advice()
        if advice and len(advice) > 100:
            print("✅ Consejos fiscales disponibles")
        else:
            print("❌ Error obteniendo consejos fiscales")

        print("🎯 Prueba TaxUtils completada")

    except Exception as e:
        print(f"❌ Error en TaxUtils: {e}")

    finally:
        # Limpiar
        shutil.rmtree(test_dir, ignore_errors=True)

def run_all_tests():
    """Ejecutar todas las pruebas"""
    print("🚀 Iniciando pruebas del módulo Hacienda\n")

    test_fifo_calculation()
    test_tax_utils()

    print("\n✨ Todas las pruebas completadas")
    print("💡 Revisa los resultados arriba para verificar el funcionamiento")

if __name__ == "__main__":
    run_all_tests()
