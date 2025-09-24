#!/usr/bin/env python3
# hacienda/demo_tax_system.py
# Demostración del sistema de seguimiento fiscal

from hacienda.tax_tracker import TaxTracker
from datetime import datetime
import os

def demo_tax_system():
    """Demostración completa del sistema fiscal"""
    print("🚀 DEMOSTRACIÓN DEL SISTEMA FISCAL ESPAÑOL")
    print("=" * 60)

    # Inicializar el sistema fiscal
    print("\n1. Inicializando TaxTracker...")
    tracker = TaxTracker()

    # Simular operaciones de trading
    print("\n2. Registrando operaciones de ejemplo...")

    operations = [
        {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'quantity': 0.5,
            'filled_quantity': 0.5,
            'price': 45000.0,
            'filled_price': 45000.0,
            'commission': 11.25,  # 0.05% de 45000 * 0.5
            'status': 'filled'
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'quantity': 2.0,
            'filled_quantity': 2.0,
            'price': 3000.0,
            'filled_price': 3000.0,
            'commission': 30.0,  # 0.05% de 3000 * 2
            'status': 'filled'
        },
        {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'quantity': 0.3,
            'filled_quantity': 0.3,
            'price': 47000.0,
            'filled_price': 47000.0,
            'commission': 7.05,  # 0.05% de 47000 * 0.3
            'status': 'filled'
        },
        {
            'symbol': 'BTCUSDT',
            'side': 'sell',
            'quantity': 0.6,
            'filled_quantity': 0.6,
            'price': 52000.0,
            'filled_price': 52000.0,
            'commission': 15.6,  # 0.05% de 52000 * 0.6
            'status': 'filled'
        }
    ]

    # Registrar cada operación
    for i, op in enumerate(operations, 1):
        print(f"   📝 Registrando operación {i}: {op['symbol']} {op['side']} {op['quantity']} @ ${op['price']:,.0f}")
        tracker.record_operation(op, exchange="Binance")

    # Mostrar archivos generados
    print("\n3. Archivos generados:")
    hacienda_files = [
        'operaciones.csv',
        'posiciones_fifo.json',
        'ganancias_realizadas.csv'
    ]

    for filename in hacienda_files:
        filepath = os.path.join('hacienda', filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"   ✅ {filename} ({size} bytes)")
        else:
            print(f"   ❌ {filename} (no generado)")

    # Mostrar resumen de posiciones
    print("\n4. Posiciones actuales:")
    positions = tracker.get_positions_summary()
    for symbol, pos_data in positions.items():
        print(f"   {symbol}: {pos_data['total_quantity']:.4f} @ ${pos_data['avg_price']:,.2f}")

    # Generar informe fiscal
    print("\n5. Generando informe fiscal...")
    tax_report = tracker.generate_tax_report()

    if tax_report:
        print("   📊 Resumen fiscal generado:")
        print(f"      Ganancias realizadas: ${tax_report['realized_gains']:,.2f}")
        print(f"      Pérdidas realizadas: ${tax_report['realized_losses']:,.2f}")
        print(f"      Base imponible: ${tax_report['tax_base']:,.2f}")

        # Archivo de informe también generado
        report_file = f"hacienda/informe_fiscal_{datetime.utcnow().year}.json"
        if os.path.exists(report_file):
            print(f"   ✅ Informe guardado: {report_file}")

    # Mostrar contenido de algunos archivos
    print("\n6. Contenido de archivos generados:")

    # Mostrar operaciones
    operations_file = "hacienda/operaciones.csv"
    if os.path.exists(operations_file):
        print("\n   📄 operaciones.csv (primeras líneas):")
        with open(operations_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:6]  # Primeras 6 líneas
            for line in lines:
                print(f"      {line.strip()}")

    # Mostrar ganancias realizadas
    gains_file = "hacienda/ganancias_realizadas.csv"
    if os.path.exists(gains_file):
        print("\n   📄 ganancias_realizadas.csv:")
        with open(gains_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                print(f"      {line.strip()}")

    print("\n" + "=" * 60)
    print("✨ DEMOSTRACIÓN COMPLETADA")
    print("\n💡 El sistema funciona automáticamente:")
    print("   • Registra TODAS las operaciones de trading")
    print("   • Calcula ganancias/pérdidas por método FIFO")
    print("   • Genera informes para declaración de impuestos")
    print("   • Guarda datos en archivos CSV/JSON para AEAT")

    print("\n🔧 Para usar en producción:")
    print("   python -m hacienda.tax_utils --action summary --year 2024")
    print("   python -m hacienda.tax_utils --action export --year 2024 --format csv")

if __name__ == "__main__":
    demo_tax_system()
